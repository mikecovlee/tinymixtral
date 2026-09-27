# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""Win-rate evaluation of two checkpoints with an external LLM judge.

Stages:
  gen   : greedy-generate one response per held-out prompt for a model (GPU)
  judge : pairwise compare two response files with DeepSeek, both A/B orders
          (position-bias controlled), concurrency 2

Usage:
    python scripts/dpo_eval_judge.py gen   --model publish/8b-sft --out data/dpo/eval_sft.jsonl
    python scripts/dpo_eval_judge.py gen   --model publish/8b-dpo --out data/dpo/eval_dpo.jsonl
    python scripts/dpo_eval_judge.py judge --a data/dpo/eval_sft.jsonl --b data/dpo/eval_dpo.jsonl
"""

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

AUTH = Path.home() / ".local" / "share" / "opencode" / "auth.json"
JUDGE_URL = os.environ.get("JUDGE_URL", "https://api.deepseek.com/v1/chat/completions")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-flash")

SYS = ("You are a strict, impartial evaluator. Compare two responses to a user request "
       "and decide which is better on correctness, helpfulness, instruction-following, "
       "coherence, and absence of repetition. Ignore length and formatting. "
       'Output ONLY JSON: {"winner": "A"|"B"|"tie", "reason": "<short>"}')
TMPL = """# User request
{prompt}

# Response A
{a}

# Response B
{b}

Which response is better? Output JSON only."""


def get_key():
    k = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("JUDGE_API_KEY")
    if k:
        return k
    if AUTH.exists():
        return json.loads(AUTH.read_text(encoding="utf-8"))["deepseek"]["key"]
    raise SystemExit("no judge key: set DEEPSEEK_API_KEY (or ~/.local/share/opencode/auth.json)")


def call_judge(key, prompt, a, b, retries=5, url=None, model=None):
    url = url or JUDGE_URL
    model = model or JUDGE_MODEL
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "temperature": 0.0, "max_tokens": 500,
                      "thinking": {"type": "disabled"},
                      "messages": [{"role": "system", "content": SYS},
                                   {"role": "user", "content": TMPL.format(prompt=prompt, a=a, b=b)}]},
                timeout=180,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"].get("content", "")
            m = re.search(r"\{.*\}", text, flags=re.S)
            if not m:
                last = f"unparseable: {text[:80]}"
                continue
            d = json.loads(m.group(0))
            if d.get("winner") in ("A", "B", "tie"):
                return d["winner"]
            last = f"bad winner: {d.get('winner')}"
        except Exception as e:
            last = str(e)[:150]
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"judge failed: {last}")


def stage_gen(args):
    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
    ds = load_dataset("parquet", data_files=args.prompts, split="train")
    todo = [i for i in range(len(ds)) if ds[i]["id"] not in done]
    print(f"gen {args.model}: {len(todo)} to do", flush=True)
    fout = out.open("a", encoding="utf-8")
    for start in range(0, len(todo), args.batch_size):
        idxs = todo[start:start + args.batch_size]
        batch = [ds[i] for i in idxs]
        msgs = [[{"role": "user", "content": b["prompt"]}] for b in batch]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt", padding=True, return_dict=True).to("cuda")
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        for j, b in enumerate(batch):
            gen = g[j][plen:]
            pos = (gen == tok.eos_token_id).nonzero(as_tuple=True)[0]
            txt = tok.decode(gen[: pos[0]] if len(pos) else gen, skip_special_tokens=True).strip()
            fout.write(json.dumps({"id": b["id"], "prompt": b["prompt"], "response": txt},
                                  ensure_ascii=False) + "\n")
        fout.flush()
        if (start + len(batch)) % 50 < args.batch_size:
            print(f"  {start+len(batch)}/{len(todo)}", flush=True)
    fout.close()
    print(f"done -> {out}", flush=True)


def stage_judge(args):
    A = {json.loads(l)["id"]: json.loads(l) for l in open(args.a, encoding="utf-8")}
    B = {json.loads(l)["id"]: json.loads(l) for l in open(args.b, encoding="utf-8")}
    ids = [i for i in A if i in B and A[i]["response"].strip() and B[i]["response"].strip()]
    print(f"judge pairs: {len(ids)} (A={args.a}, B={args.b})", flush=True)
    key = get_key()
    url = args.base_url or JUDGE_URL
    model = args.judge_model or JUDGE_MODEL
    out = Path(args.out)
    fout = out.open("a", encoding="utf-8")
    n_win = n_tie = n_loss = 0

    def work(i):
        rng = random.Random(i)
        sft, dpo = A[i]["response"], B[i]["response"]
        # order 1: A=sft, B=dpo ; order 2: A=dpo, B=sft
        w1 = call_judge(key, A[i]["prompt"], sft, dpo, url=url, model=model)
        w2 = call_judge(key, A[i]["prompt"], dpo, sft, url=url, model=model)
        # normalize to "who wins: sft/dpo/tie"
        v1 = {"A": "sft", "B": "dpo", "tie": "tie"}[w1]
        v2 = {"A": "dpo", "B": "sft", "tie": "tie"}[w2]
        verdict = v1 if v1 == v2 else "tie"
        return {"id": i, "w1": w1, "w2": w2, "verdict": verdict}

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, i) for i in ids]
        for fut in as_completed(futs):
            r = fut.result()
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            if r["verdict"] == "dpo":
                n_win += 1
            elif r["verdict"] == "sft":
                n_loss += 1
            else:
                n_tie += 1
    fout.close()
    total = n_win + n_tie + n_loss
    dec = n_win + n_loss
    print(f"DPO wins={n_win} ties={n_tie} losses={n_loss} (n={total})")
    print(f"win-rate (excl ties) = {n_win/max(dec,1):.3f} | tie-rate = {n_tie/max(total,1):.3f}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="stage", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--model", required=True)
    g.add_argument("--prompts", default="data/dpo/prompts_heldout.parquet")
    g.add_argument("--out", required=True)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--max-new-tokens", type=int, default=448)
    g.set_defaults(func=stage_gen)

    j = sub.add_parser("judge")
    j.add_argument("--a", required=True)
    j.add_argument("--b", required=True)
    j.add_argument("--out", default="data/dpo/eval_judgments.jsonl")
    j.add_argument("--concurrency", type=int, default=2)
    j.add_argument("--base-url", default=None)
    j.add_argument("--judge-model", default=None)
    j.set_defaults(func=stage_judge)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
