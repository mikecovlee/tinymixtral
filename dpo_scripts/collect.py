"""Collect 8-task 0-shot harness means from evals/harness/* (one- and two-level globs)."""
import glob
import json
import os
import sys

T8 = {
    "hellaswag": "acc_norm", "piqa": "acc", "winogrande": "acc", "arc_easy": "acc",
    "arc_challenge": "acc_norm", "openbookqa": "acc_norm", "boolq": "acc",
    "lambada_openai": "acc",
}
ROOT = "/home/mikecovlee/work/tinymixtral"


def pick(d, m):
    for k in (m, m + ",none"):
        if k in d and isinstance(d[k], (int, float)):
            return d[k]
    return None


def main(pattern):
    files = sorted(set(glob.glob(os.path.join(ROOT, "evals/harness", pattern, "results_*.json"))
                       + glob.glob(os.path.join(ROOT, "evals/harness", pattern, "*", "results_*.json"))))
    for f in files:
        res = json.load(open(f))["results"]
        parts = f.split(os.sep)
        arm = parts[parts.index("harness") + 1]
        vals = [pick(res.get(t, {}), m) for t, m in T8.items()]
        ok = [v for v in vals if v is not None]
        mean = round(sum(ok) / len(ok), 4) if ok else None
        print(f"{arm:16s} {mean}  " + " ".join(
            f"{t}={None if v is None else round(v, 3)}" for t, v in zip(T8, vals)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "*")
