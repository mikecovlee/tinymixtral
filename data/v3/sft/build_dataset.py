"""Build a large, deduplicated, decontaminated general-instruction SFT mixture.

Design (see docs/SFT_V3_PLAN.md):
  * unified schema: {"conversations": [{"from": "human"|"gpt", "value": str}, ...],
                     "source", "category", "lang", "n_turns", "n_chars"}
  * per-source quotas with a global deterministic shuffle (fixes the row-order
    bias of the original 50k OpenHermes subset)
  * cross-source exact dedup (normalised first human turn) + MinHash near-dup
    (Jaccard >= 0.8) because OpenHermes is a superset of SlimOrca/OpenOrca
  * 10-gram decontamination against the evaluation sets
  * deterministic held-out prompt reserve (default 5k) excluded from training

Scales: v1=200k, v2=1M, v3=3M samples.
Outputs: <out>/shards/<source>-*.parquet, train.parquet, dev.parquet,
         heldout_prompts.parquet, stats.json
"""
import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

# ---------------------------------------------------------------- sources
# name -> (quota share, min/max assistant chars)
SHARES = {
    "tulu3": 0.32,
    "openhermes": 0.16,
    "slimorca": 0.12,
    "openorca": 0.12,
    "ultrachat": 0.11,
    "metamath": 0.06,
    "orcamath": 0.04,
    "omi2": 0.03,
    "squad2": 0.02,
    "trivia": 0.02,
}
MIN_CHARS, MAX_CHARS = 40, 12000

_WS = re.compile(r"\s+")


def _norm(s):
    return _WS.sub(" ", (s or "").strip().lower())


def _msgs_to_conv(msgs):
    if isinstance(msgs, str):
        try:
            msgs = json.loads(msgs)
        except Exception:
            return []
    if not isinstance(msgs, (list, tuple)):
        return []
    conv = []
    for m in msgs:
        if isinstance(m, str):
            try:
                m = json.loads(m)
            except Exception:
                continue
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or m.get("from") or "").lower()
        val = m.get("content") or m.get("value") or ""
        if role in ("system",):
            continue
        if not str(val).strip():
            continue
        conv.append({"from": "human" if role in ("user", "human") else "gpt",
                     "value": str(val).strip()})
    return conv


def _pair(q, a):
    return [{"from": "human", "value": str(q).strip()},
            {"from": "gpt", "value": str(a).strip()}]


def load_source(name, cap, rng):
    """Yield raw unified dicts (no filtering) up to `cap`, reservoir-ish order."""
    if name == "tulu3":
        ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
        for r in ds:
            yield _msgs_to_conv(r.get("messages") or [])
    elif name == "openhermes":
        ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
        for r in ds:
            yield _msgs_to_conv(r.get("conversations") or [])
    elif name == "slimorca":
        ds = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)
        for r in ds:
            yield _msgs_to_conv(r.get("conversations") or [])
    elif name == "openorca":
        ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
        for r in ds:
            sp = r.get("system_prompt") or ""
            q = (sp + "\n\n" + (r.get("question") or "")).strip() if sp else r.get("question")
            yield _pair(q or "", r.get("response") or "")
    elif name == "ultrachat":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        for r in ds:
            yield _msgs_to_conv(r.get("messages") or [])
    elif name == "metamath":
        ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
        for r in ds:
            yield _pair(r.get("query") or "", r.get("response") or "")
    elif name == "orcamath":
        ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train", streaming=True)
        for r in ds:
            yield _pair(r.get("question") or "", r.get("answer") or "")
    elif name == "omi2":
        try:
            ds = load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True)
        except Exception as e:  # config name drift
            print("omi2 skip:", str(e)[:120])
            return
        for r in ds:
            yield _pair(r.get("problem") or "", r.get("generated_solution") or "")
    elif name == "squad2":
        ds = load_dataset("rajpurkar/squad_v2", split="train", streaming=True)
        for r in ds:
            ans = (r.get("answers") or {}).get("text") or []
            if not ans:
                continue
            q = f"Answer the question using the context.\n\nContext: {(r.get('context') or '')[:2500]}\n\nQuestion: {r.get('question') or ''}"
            yield _pair(q, ans[0])
    elif name == "trivia":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train", streaming=True)
        for r in ds:
            a = (r.get("answer") or {}).get("value") or ""
            if a:
                yield _pair(r.get("question") or "", a)
    else:
        raise KeyError(name)


# ---------------------------------------------------------------- helpers
def first_user_hash(conv):
    for m in conv:
        if m["from"] == "human":
            return hashlib.sha1(_norm(m["value"]).encode()).hexdigest()
    return None


def accept(conv):
    if len(conv) < 2:
        return False
    if conv[-1]["from"] != "gpt":
        return False
    asst = conv[-1]["value"]
    if not (MIN_CHARS <= len(asst) <= MAX_CHARS):
        return False
    if not conv[0]["value"].strip():
        return False
    return True


_SRC_DIR = Path("data/sft_src")


def _local(name, patterns, kind):
    d = _SRC_DIR / name
    if not d.exists():
        return None
    files = []
    for pat in patterns:
        files += [str(p) for p in sorted(d.rglob(pat))]
    files = [f for f in files if Path(f).stat().st_size > 0]
    if not files:
        return None
    return load_dataset(kind, data_files=files, split="train", streaming=True)


def load_source2(name, cap, rng):
    """Local-first source loader (pre-downloaded files under data/sft_src/<name>)."""
    if name == "tulu3":
        ds = _local(name, ["*.parquet"], "parquet") or load_dataset(
            "allenai/tulu-3-sft-mixture", split="train", streaming=True)
        for r in ds:
            yield _msgs_to_conv(r.get("messages") or [])
    elif name == "openhermes":
        ds = _local(name, ["*.json"], "json") or load_dataset(
            "teknium/OpenHermes-2.5", split="train", streaming=True)
        for r in ds:
            yield _msgs_to_conv(r.get("conversations") or [])
    elif name == "slimorca":
        ds = _local(name, ["*.jsonl", "*.json"], "json") or load_dataset(
            "Open-Orca/SlimOrca", split="train", streaming=True)
        for r in ds:
            yield _msgs_to_conv(r.get("conversations") or [])
    elif name == "openorca":
        ds = _local(name, ["*.parquet"], "parquet") or load_dataset(
            "Open-Orca/OpenOrca", split="train", streaming=True)
        for r in ds:
            q = (r.get("question") or "").strip()
            yield _pair(q, r.get("response") or "")
    elif name == "ultrachat":
        ds = _local(name, ["*.parquet"], "parquet") or load_dataset(
            "HuggingFaceH4/ultrachat_200k", "default", split="train_sft",
            streaming=True)
        for r in ds:
            yield _msgs_to_conv(r.get("messages") or [])
    elif name == "metamath":
        ds = _local(name, ["*.json", "*.parquet"], "json") or load_dataset(
            "meta-math/MetaMathQA", split="train", streaming=True)
        for r in ds:
            yield _pair(r.get("query") or "", r.get("response") or "")
    elif name == "orcamath":
        ds = _local(name, ["*.parquet"], "parquet") or load_dataset(
            "microsoft/orca-math-word-problems-200k", split="train", streaming=True)
        for r in ds:
            yield _pair(r.get("question") or "", r.get("answer") or "")
    elif name == "omi2":
        ds = _local(name, ["*.parquet"], "parquet")
        if ds is None:
            try:
                ds = load_dataset("nvidia/OpenMathInstruct-2", split="train",
                                  streaming=True)
            except Exception as e:
                print("omi2 skip:", str(e)[:120])
                return
        for r in ds:
            yield _pair(r.get("problem") or "", r.get("generated_solution") or "")
    elif name == "squad2":
        ds = _local(name, ["*.parquet"], "parquet") or load_dataset(
            "rajpurkar/squad_v2", "plain_text", split="train", streaming=True)
        for r in ds:
            ans = (r.get("answers") or {}).get("text") or []
            if not ans:
                continue
            q = (f"Answer the question using the context.\n\nContext: "
                 f"{(r.get('context') or '')[:2500]}\n\nQuestion: "
                 f"{r.get('question') or ''}")
            yield _pair(q, ans[0])
    elif name == "trivia":
        ds = _local(name, ["*.parquet"], "parquet") or load_dataset(
            "mandarjoshi/trivia_qa", "rc.nocontext", split="train", streaming=True)
        for r in ds:
            a = (r.get("answer") or {}).get("value") or ""
            if a:
                yield _pair(r.get("question") or "", a)
    else:
        raise KeyError(name)


_NUM_PERM = 16
_BANDS = 8  # 8 bands x 2 rows
_MAX_SHINGLES = 64  # bounded sketch per doc keeps memory flat at 1M-3M rows


def shingle_arr(text, n=5):
    """Sorted unique uint64 shingle hashes, deterministically sub-sampled to <=64."""
    t = _norm(text)
    if len(t) < n:
        return np.zeros(0, dtype=np.uint64)
    m = len(t) - n + 1
    hs = np.fromiter(
        (int.from_bytes(hashlib.blake2b(t[i:i + n].encode(), digest_size=8).digest(), "little")
         for i in range(m)), dtype=np.uint64, count=m)
    if hs.size > _MAX_SHINGLES:
        hs = hs[np.linspace(0, hs.size - 1, _MAX_SHINGLES).astype(np.int64)]
    return np.unique(hs)


def signature(hs, perms):
    if hs.size == 0:
        return np.zeros(_NUM_PERM, dtype=np.int64)
    return np.min(hs[None, :] ^ perms[:, None], axis=1).astype(np.int64)


def np_jaccard(a, b):
    if a.size == 0 or b.size == 0:
        return 0.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (a.size + b.size - inter)


def egrams(text, n=10):
    w = _norm(text).split()
    return {hashlib.blake2b(" ".join(w[i:i + n]).encode(), digest_size=8).digest()
            for i in range(0, max(len(w) - n + 1, 1))}


def build_eval_grams(enable, extra=None):
    grams = set()
    used = []
    specs = [
        ("openai/gsm8k", "main", "test"),
        ("allenai/ai2_arc", "ARC-Challenge", "test"),
        ("allenai/ai2_arc", "ARC-Easy", "test"),
        ("allenai/openbookqa", "main", "test"),
        ("Rowan/hellaswag", None, "validation"),
        ("baber/piqa", None, "validation"),
    ]
    if enable:
        for repo, cfg, split in specs:
            try:
                ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
                for r in ds:
                    q = r.get("question") or r.get("question_stem") or r.get("goal") or ""
                    if not q and r.get("choices"):
                        q = str(r.get("choices"))[:500]
                    grams |= egrams(str(q), 10)
                used.append(f"{repo}/{cfg or '-'}/{split}")
            except Exception as e:
                print("eval-skip", repo, str(e)[:80])
    if extra:
        ep = Path(extra)
        if ep.exists():
            import pyarrow.parquet as _pq
            for row in _pq.read_table(ep).to_pylist():
                grams |= egrams(str(row.get("prompt") or ""), 10)
            used.append(f"extra:{extra}")
    return grams, used


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/sft")
    ap.add_argument("--scale", choices=["v1", "v2", "v3", "pilot", "200k", "1m", "3m"], default="v1")
    ap.add_argument("--pilot", type=int, default=0, help="override total sample target")
    ap.add_argument("--heldout", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--near-dup", action="store_true", default=True)
    ap.add_argument("--no-near-dup", dest="near_dup", action="store_false")
    ap.add_argument("--decontam", action="store_true", default=True)
    ap.add_argument("--no-decontam", dest="decontam", action="store_false")
    ap.add_argument("--sources", default="",
                    help="comma list of sources to use (default: all)")
    ap.add_argument("--extra-holdout", default="",
                    help="parquet with a 'prompt' column to decontaminate against")
    args = ap.parse_args()

    only = {s.strip() for s in args.sources.split(",") if s.strip()}
    shares = {k: v for k, v in SHARES.items() if not only or k in only}
    if not shares:
        raise SystemExit(f"no sources match {sorted(only)}")
    _tw = sum(shares.values())
    shares = {k: v / _tw for k, v in shares.items()}

    _SCALE_ALIAS = {"200k": "v1", "1m": "v2", "3m": "v3"}
    total = {"pilot": args.pilot or 10000, "v1": 200000, "v2": 1000000, "v3": 3000000}[
        _SCALE_ALIAS.get(args.scale, args.scale)]
    if args.pilot:
        total = args.pilot
    out = Path(args.out_dir)
    shard_dir = out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for _p in shard_dir.glob("*.parquet"):
        _p.unlink()

    rng = random.Random(args.seed)
    seen_first = set()          # exact dedup
    perms = np.asarray([rng.getrandbits(32) for _ in range(_NUM_PERM)], dtype=np.uint64)
    bands = defaultdict(list)   # near-dup LSH
    shingle_arrs = []           # bounded uint64 sketches, parallel to insertion order
    heldout = {}
    stats = Counter()
    src_counts = Counter()

    eval_grams, eval_used = build_eval_grams(args.decontam, args.extra_holdout)
    print("eval grams:", len(eval_grams), "from", eval_used)

    t0 = time.time()
    for name, share in shares.items():
        cap = max(1, int(total * share))
        n_acc = n_dup = n_dec = n_held = 0
        writer = None
        pend = []
        path = shard_dir / f"{name}.parquet"
        for conv in load_source2(name, cap, rng):
            stats["raw"] += 1
            if not accept(conv):
                continue
            h = first_user_hash(conv)
            if h is None:
                continue
            # deterministic held-out reserve (5k unique prompts)
            if (int(h[:8], 16) % 1000) < max(1, int(1000 * args.heldout / max(total, 1))):
                if len(heldout) < args.heldout and h not in heldout:
                    heldout[h] = conv[0]["value"]
                    n_held += 1
                continue
            if h in seen_first:
                n_dup += 1
                continue
            seen_first.add(h)
            # near-dup
            if args.near_dup:
                hs = shingle_arr(conv[0]["value"][:512], 5)
                sig = signature(hs, perms)
                ok = True
                for b in range(_BANDS):
                    key = (b, int(sig[2 * b]), int(sig[2 * b + 1]))
                    for idx in list(bands.get(key, ()))[:8]:  # bounded Jaccard check
                        if np_jaccard(hs, shingle_arrs[idx]) >= 0.8:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    n_dup += 1
                    continue
                for b in range(_BANDS):
                    bands[(b, int(sig[2 * b]), int(sig[2 * b + 1]))].append(len(shingle_arrs))
                shingle_arrs.append(hs)
            # decontamination
            if eval_grams:
                g = egrams(conv[0]["value"], 10)
                if g & eval_grams:
                    n_dec += 1
                    continue
            rec = {"conversations": conv, "source": name, "category": name,
                   "lang": "en", "n_turns": len(conv),
                   "n_chars": sum(len(m["value"]) for m in conv)}
            pend.append(rec)
            n_acc += 1
            if len(pend) >= 2000:
                tbl = pa.Table.from_pylist(pend)
                if writer is None:
                    writer = pq.ParquetWriter(path, tbl.schema)
                writer.write_table(tbl)
                pend = []
                if n_acc % 20000 == 0:
                    print(f"{name}: {n_acc} accepted ({time.time()-t0:.0f}s)",
                          flush=True)
            if n_acc >= cap:
                break
        if pend:
            tbl = pa.Table.from_pylist(pend)
            if writer is None:
                writer = pq.ParquetWriter(path, tbl.schema)
            writer.write_table(tbl)
        if writer is not None:
            writer.close()
        src_counts[name] = n_acc
        print(f"{name}: accepted={n_acc} dup={n_dup} decontam={n_dec} held={n_held} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ------------------------------------------------------------ assemble
    import datasets as hfds
    shards = sorted(str(p) for p in shard_dir.glob("*.parquet")
                    if p.stat().st_size > 0)
    if not shards:
        print("no shards produced")
        return
    ds = hfds.load_dataset("parquet", data_files=shards, split="train")
    ds = ds.shuffle(seed=args.seed)
    n_dev = max(1000, int(0.005 * len(ds)))
    dev = ds.select(range(n_dev))
    train = ds.select(range(n_dev, len(ds)))
    train.to_parquet(str(out / "train.parquet"))
    dev.to_parquet(str(out / "dev.parquet"))
    ho = pa.Table.from_pylist([{"prompt": v} for v in heldout.values()])
    pq.write_table(ho, out / "heldout_prompts.parquet")
    st = {
        "scale": args.scale, "target": total, "train": len(train), "dev": len(dev),
        "heldout_prompts": len(heldout), "seed": args.seed,
        "by_source": dict(src_counts), "raw": stats["raw"],
        "decontam_eval_sets": eval_used,
        "near_dup": args.near_dup, "decontam": args.decontam,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=2))
    print(json.dumps(st, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
