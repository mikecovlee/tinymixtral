"""Rubric evaluation of assistant responses with an external LLM judge (MiniMind-style).

Scores each response 0-100 on accuracy / completeness / logic / format and prints
the per-file means. Input jsonl rows must have {id, prompt, response}.
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

AUTH = Path(os.environ.get("USERPROFILE", os.path.expanduser("~"))) / ".local" / "share" / "opencode" / "auth.json"
URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-flash"
SYS = ("You are a strict, impartial evaluator of assistant answers. "
       "Score the answer on four dimensions, each an integer 0-100: "
       "accuracy (factual correctness), completeness (covers the question), "
       "logic (reasoning coherence), format (clarity and instruction-following). "
       'Reply with JSON only: {"accuracy":int,"completeness":int,"logic":int,'
       '"format":int,"reason":"<=20 words"}')
TMPL = "Question:\n{q}\n\nAssistant answer:\n{a}\n\nScore the answer."


def get_key():
    return json.load(open(AUTH))["deepseek"]["key"]


def call(key, prompt, answer, retries=4):
    payload = {"model": MODEL, "temperature": 0,
               "messages": [{"role": "system", "content": SYS},
                            {"role": "user",
                             "content": TMPL.format(q=prompt[:4000], a=answer[:4000])}]}
    for i in range(retries):
        try:
            r = requests.post(URL,
                              headers={"Authorization": f"Bearer {key}",
                                       "Content-Type": "application/json"},
                              json=payload, timeout=90)
            if r.status_code != 200:
                raise RuntimeError(r.text[:200])
            txt = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", txt, re.S)
            d = json.loads(m.group(0))
            sc = {k: int(d[k]) for k in ("accuracy", "completeness", "logic", "format")}
            return sc, str(d.get("reason", ""))[:200]
        except Exception as e:
            if i == retries - 1:
                return None, str(e)[:200]
            time.sleep(2 * (i + 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    key = get_key()
    items = [json.loads(l) for l in open(a.responses, encoding="utf-8") if l.strip()]
    if a.limit:
        items = items[: a.limit]
    done = set()
    if a.resume and Path(a.out).exists():
        done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8") if l.strip()}
    todo = [it for it in items if it["id"] not in done]
    print(f"scoring {len(todo)} items from {a.responses}", flush=True)

    sums = {"accuracy": 0, "completeness": 0, "logic": 0, "format": 0}
    n, fails = 0, 0
    with open(a.out, "a" if a.resume else "w", encoding="utf-8") as f, \
            ThreadPoolExecutor(a.concurrency) as ex:
        futs = {ex.submit(call, key, it.get("prompt", ""), it.get("response", "")): it
                for it in todo}
        for i, fu in enumerate(futs):
            it = futs[fu]
            sc, reason = fu.result()
            if sc is None:
                fails += 1
                print("fail", it["id"], reason, flush=True)
                continue
            f.write(json.dumps({"id": it["id"], **sc, "reason": reason},
                               ensure_ascii=False) + "\n")
            f.flush()
            for k in sums:
                sums[k] += sc[k]
            n += 1
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(todo)}", flush=True)
    if n:
        means = {k: round(v / n, 1) for k, v in sums.items()}
        print("MEAN", means, "overall", round(sum(means.values()) / 4, 1),
              f"n={n} fails={fails}", flush=True)


if __name__ == "__main__":
    main()
