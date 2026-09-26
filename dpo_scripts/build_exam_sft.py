"""Build an exam-style MCQ SFT set (targeted data) for TinyMixtral v3.0.

TRAIN-safe splits only (lm-eval evaluates ceval 'val', cmmlu 'test', mmlu 'test',
arc 'test', openbookqa 'test' -> those are NEVER used here):
  ceval/ceval-exam          dev
  haonan-li/cmmlu           cmmlu_v1_0_1.zip -> dev/*.csv
  cais/mmlu                 all/auxiliary_train
  allenai/ai2_arc           ARC-Challenge / ARC-Easy  train + validation
  allenai/openbookqa        main / additional       train + validation

Anti-bias: options are shuffled and answer letters are balanced across A/B/C/D.
Leakage: the eval splits are never touched; stats.json contains an audit.
Output: data/sft_exam/{train.parquet,dev.parquet,stats.json}
Conversation schema matches data/sft: [{from: human|gpt, value: str}, ...]
"""
import argparse
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

LETTERS = "ABCDEFGH"
ZH = {"ceval", "cmmlu"}


def _mk(question, options, ans_idx, source, subject, rng, target):
    n = len(options)
    order = list(range(n))
    letter = None
    for _ in range(60):
        rng.shuffle(order)
        cand = LETTERS[order.index(ans_idx)]
        if cand == target:
            letter = cand
            break
    if letter is None:                       # fall back to the last shuffle
        letter = LETTERS[order.index(ans_idx)]
    opts = [options[i] for i in order]
    body = question.strip() + "\n" + "\n".join(
        f"{LETTERS[i]}. {o.strip()}" for i, o in enumerate(opts))
    if source in ZH:
        user = body + "\n请逐步分析并选出正确答案，最后一行写「答案：X」。"
        asst = f"答案：{letter}"
    else:
        user = body + "\nChoose the correct option; put the letter on the last line as 'Answer: X'."
        asst = f"答案：{letter}"
    return {"conversations": [{"from": "human", "value": user},
                              {"from": "gpt", "value": asst}],
            "source": source, "subject": subject, "answer": letter, "n_options": n}


def load_ceval(limit, rng):
    api = HfApi()
    files = api.list_repo_files("ceval/ceval-exam", repo_type="dataset")
    subjects = sorted({f.split("/")[0] for f in files if "/" in f})
    out = []
    per = max(1, limit // max(len(subjects), 1))
    for s in subjects:
        try:
            ds = load_dataset("ceval/ceval-exam", s, split="dev")
        except Exception as e:
            print("ceval skip", s, e)
            continue
        for r in ds.select(range(min(per, len(ds)))):
            opts = [r["A"], r["B"], r["C"], r["D"]]
            out.append((s, (r["question"], opts, "ABCD".index(r["answer"].strip().upper()))))
    return out


def load_cmmlu(cache, limit, rng):
    z = hf_hub_download("haonan-li/cmmlu", "cmmlu_v1_0_1.zip",
                        repo_type="dataset", cache_dir=cache)
    out = []
    with zipfile.ZipFile(z) as zf:
        names = [n for n in zf.namelist()
                 if n.endswith(".csv") and n.split("/")[0] in ("dev", "val")]
        for name in sorted(names):
            subj = Path(name).stem
            import csv
            with zf.open(name) as fh:
                for row in csv.DictReader(fh.read().decode("utf-8", "ignore").splitlines()):
                    q = (row.get("Question") or "").strip()
                    opts = [(row.get(k) or "").strip() for k in ("A", "B", "C", "D")]
                    a = (row.get("Answer") or "").strip().upper()
                    if q and all(opts) and a in "ABCD":
                        out.append((subj, (q, opts, "ABCD".index(a))))
    rng.shuffle(out)
    return out[:limit]


def load_mmlu(limit, rng):
    ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx[:limit]:
        r = ds[i]
        ch = list(r["choices"])
        if len(ch) >= 2:
            out.append((r.get("subject", "mmlu"), (r["question"], ch, int(r["answer"]))))
    return out


def _choices(r):
    ch = r.get("choices")
    if not isinstance(ch, dict):
        return None, None
    texts, labels = ch.get("text"), ch.get("label")
    if not isinstance(texts, list) or not isinstance(labels, list):
        return None, None
    if not texts or len(texts) != len(labels):
        return None, None
    return [str(t) for t in texts], [str(l) for l in labels]


def load_arc(limit, rng):
    out = []
    for cfg in ("ARC-Challenge", "ARC-Easy"):
        for split in ("train", "validation"):
            ds = load_dataset("allenai/ai2_arc", cfg, split=split)
            for r in ds:
                texts, labels = _choices(r)
                if texts is None or r["answerKey"] not in labels:
                    continue
                out.append((cfg, (r["question"], texts, labels.index(r["answerKey"]))))
    rng.shuffle(out)
    return out[:limit]


def load_obqa(limit, rng):
    out = []
    for cfg in ("main", "additional"):
        for split in ("train", "validation"):
            ds = load_dataset("allenai/openbookqa", cfg, split=split)
            for r in ds:
                texts, labels = _choices(r)
                if texts is None or r["answerKey"] not in labels:
                    continue
                out.append(("openbookqa", (r["question_stem"], texts,
                                           labels.index(r["answerKey"]))))
    rng.shuffle(out)
    return out[:limit]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/sft_exam")
    p.add_argument("--limit", type=int, default=50000)
    p.add_argument("--dev-size", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cmmlu-cache", default="data/raw/hf")
    args = p.parse_args()

    rng = random.Random(args.seed)
    quota = {"ceval": 0.01, "cmmlu": 0.04, "mmlu": 0.50, "arc": 0.10, "obqa": 0.25}
    raw = defaultdict(list)
    raw["ceval"] = load_ceval(int(args.limit * quota["ceval"]), rng)
    raw["cmmlu"] = load_cmmlu(args.cmmlu_cache, int(args.limit * quota["cmmlu"]), rng)
    raw["mmlu"] = load_mmlu(int(args.limit * quota["mmlu"]), rng)
    raw["arc"] = load_arc(int(args.limit * quota["arc"]), rng)
    raw["obqa"] = load_obqa(int(args.limit * quota["obqa"]), rng)
    for k, v in raw.items():
        print(f"loaded {k}: {len(v)}")

    # dedupe by (question, sorted options)
    seen, items = set(), []
    for src, lst in raw.items():
        for subject, (q, opts, ans) in lst:
            if len(opts) < 2 or ans >= len(opts):
                continue
            key = (q.strip(), tuple(sorted(o.strip() for o in opts)))
            if key in seen:
                continue
            seen.add(key)
            items.append({"source": src, "subject": subject, "q": q,
                          "opts": opts, "ans": ans})
    rng.shuffle(items)
    print("deduped items:", len(items))

    # balance answer letters per option-count group
    counters, rows = Counter(), []
    for it in items:
        n = len(it["opts"])
        target = LETTERS[counters[n] % min(n, len(LETTERS))]
        counters[n] += 1
        rows.append(_mk(it["q"], it["opts"], it["ans"], it["source"],
                        it["subject"], rng, target))

    dev = rows[: args.dev_size]
    train = rows[args.dev_size: args.limit]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, part in (("train", train), ("dev", dev)):
        tbl = pa.Table.from_pylist([{k: r[k] for k in
                                     ("conversations", "source", "subject",
                                      "answer", "n_options")} for r in part])
        pq.write_table(tbl, out / f"{name}.parquet")

    stats = {
        "total": len(rows), "train": len(train), "dev": len(dev),
        "seed": args.seed,
        "by_source": dict(Counter(r["source"] for r in rows)),
        "answer_letters": dict(Counter(r["answer"] for r in rows)),
        "by_n_options": dict(Counter(r["n_options"] for r in rows)),
        "subjects": len({r["subject"] for r in rows}),
        "leak_audit": {"used_splits": {
            "ceval": "dev", "cmmlu": "dev", "mmlu": "auxiliary_train",
            "arc": "train+validation", "obqa": "train+validation"},
            "eval_splits_never_used": {
                "ceval": "val", "cmmlu": "test", "mmlu": "test",
                "arc": "test", "obqa": "test"}},
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
