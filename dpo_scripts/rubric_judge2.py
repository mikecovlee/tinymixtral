"""Anchored 0-100 rubric evaluation with an external LLM judge (v2).

Fixes vs v1:
  * explicit 0-100 band anchors (v1's judge collapsed to a coarse 0-11 scale),
  * a `reasoning` dimension,
  * `instruction_following` instead of a `format` score that sat at the ceiling,
  * JSON response mode with a fallback retry without it.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

AUTH = Path(os.environ.get("USERPROFILE", os.path.expanduser("~"))) / ".local" / "share" / "opencode" / "auth.json"
URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-flash"
DIMS = ("correctness", "completeness", "reasoning", "instruction_following")

SYS = (
    "You are a strict, impartial evaluator. Score the assistant answer on four "
    "dimensions with INTEGER scores from 0 to 100. Use the full range and do NOT "
    "cluster near the top.\n"
    "Bands: 0-20 = wrong/harmful or ignores the question; 21-40 = major gaps or "
    "errors; 41-60 = partially correct with notable gaps; 61-80 = correct with "
    "minor gaps; 81-100 = fully correct, complete and well-reasoned (reserve >90 "
    "for excellent answers).\n"
    "Dimensions:\n"
    "correctness: factual/logical correctness of the answer.\n"
    "completeness: does it fully address every part of the question.\n"
    "reasoning: quality and coherence of the explanation/working (0 if none is asked for or needed).\n"
    "instruction_following: does it do what the user asked (length/format/language/etc.).\n"
    'Reply with JSON only: {"correctness":int,"completeness":int,"reasoning":int,'
    '"instruction_following":int,"reason":"<=25 words"}'
)
TMPL = "Question:\n{q}\n\nAssistant answer:\n{a}\n\nScore 0-100 on each dimension."


def get_key():
    return json.load(open(AUTH))["deepseek"]["key"]


def call(key, prompt, answer, retries=4):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": TMPL.format(q=prompt[:4000], a=answer[:4000])}]
    for i in range(retries):
        payload = {"model": MODEL, "temperature": 0, "messages": msgs}
        if i == 0:
            payload["response_format"] = {"type": "json_object"}
        try:
            r = requests.post(URL,
                              headers={"Authorization": f"Bearer {key}",
                                       "Content-Type": "application/json"},
                              json=payload, timeout=90)
            if r.status_code != 200:
                raise RuntimeError(r.text[:200])
            txt = r.json()["choices"][0]["message"]["content"]
            d = json.loads(txt)
            out = {k: int(d[k]) for k in DIMS}
            if not all(0 <= v <= 100 for v in out.values()):
                raise ValueError(f"out of range: {out}")
            return out, str(d.get("reason", ""))[:200]
        except Exception as e:                      # noqa: BLE001
            if i == retries - 1:
                return None, str(e)[:200]
            time.sleep(2 * (i + 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    key = get_key()
    items = [json.loads(l) for l in open(a.responses, encoding="utf-8") if l.strip()]
    if a.limit:
        items = items[:a.limit]
    done = set()
    if a.resume and Path(a.out).exists():
        done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8") if l.strip()}
    todo = [it for it in items if it["id"] not in done]
    print(f"scoring {len(todo)} items from {a.responses}")
    sums = {k: 0 for k in DIMS}
    n = 0
    fails = 0
    with open(a.out, "a" if a.resume else "w", encoding="utf-8") as f, \
            ThreadPoolExecutor(a.concurrency) as ex:
        futs = {ex.submit(call, key, it["prompt"], it.get("response", "")): it for it in todo}
        for i, fu in enumerate(futs):
            it = futs[fu]
            sc, reason = fu.result()
            if sc is None:
                fails += 1
                if fails <= 5:
                    print("fail", it["id"], reason)
                continue
            f.write(json.dumps({"id": it["id"], **sc, "reason": reason},
                               ensure_ascii=False) + "\n")
            f.flush()
            for k in DIMS:
                sums[k] += sc[k]
            n += 1
            if (i + 1) % 100 == 0:
                print(f"{i+1}/{len(todo)}")
    if n:
        print("MEAN", {k: round(v / n, 1) for k, v in sums.items()},
              "overall", round(sum(sums.values()) / (4 * n), 1),
              f"n={n} fails={fails}")


if __name__ == "__main__":
    main()
