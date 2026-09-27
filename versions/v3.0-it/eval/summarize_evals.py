#!/usr/bin/env python
import argparse
import glob
import json
import os
import re

HARNESS_TASKS = [
    ("hellaswag", "acc_norm,none", "acc,none"),
    ("piqa", "acc_norm,none", "acc,none"),
    ("winogrande", "acc,none", None),
    ("arc_easy", "acc,none", "acc_norm,none"),
    ("arc_challenge", "acc_norm,none", "acc,none"),
    ("openbookqa", "acc_norm,none", "acc,none"),
    ("boolq", "acc,none", None),
    ("lambada_openai", "acc,none", None),
]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pick(results, task, metric):
    if metric is None:
        return None
    return results.get(task, {}).get(metric)


def arm_from(path, kind):
    base = os.path.basename(path)
    m = re.match(rf"{kind}_(.+)\.json$", base)
    return m.group(1) if m else base


def harness_rows(path):
    results = load(path)["results"]
    primary, alt = [], []
    for task, metric, alt_metric in HARNESS_TASKS:
        p = pick(results, task, metric)
        a = pick(results, task, alt_metric)
        if p is not None:
            primary.append(p)
        if a is not None:
            alt.append(a)
    mean = sum(primary) / len(primary) if primary else float("nan")
    mean_alt = sum(alt) / len(alt) if alt else float("nan")
    return results, mean, mean_alt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/eval")
    ap.add_argument("--detailed", action="store_true")
    args = ap.parse_args()

    harness = sorted(glob.glob(os.path.join(args.dir, "harness_*.json")))
    ifeval = sorted(glob.glob(os.path.join(args.dir, "ifeval_*.json")))
    gsm8k = sorted(glob.glob(os.path.join(args.dir, "gsm8k_*.json")))

    print(f"== 8-task harness (canonical: acc_norm for hellaswag/piqa/arc_challenge/openbookqa) ==")
    print(f"{'arm':28s} {'mean':>8s} {'alt_mix':>8s}")
    for path in harness:
        results, mean, mean_alt = harness_rows(path)
        print(f"{arm_from(path,'harness'):28s} {mean:8.4f} {mean_alt:9.4f}")
        if args.detailed:
            for task, metric, _ in HARNESS_TASKS:
                print(f"    {task:16s} {pick(results, task, metric):.5f}")

    print(f"\n== IFEval ==")
    print(f"{'arm':28s} {'prompt_str':>10s} {'inst_str':>9s} {'prompt_lo':>10s} {'inst_lo':>9s}")
    for path in ifeval:
        r = load(path)["results"]["ifeval"]
        print(
            f"{arm_from(path,'ifeval'):28s} "
            f"{r.get('prompt_level_strict_acc,none', float('nan')):10.4f} "
            f"{r.get('inst_level_strict_acc,none', float('nan')):9.4f} "
            f"{r.get('prompt_level_loose_acc,none', float('nan')):10.4f} "
            f"{r.get('inst_level_loose_acc,none', float('nan')):9.4f}"
        )

    print(f"\n== GSM8K ==")
    print(f"{'arm':28s} {'strict':>8s} {'flexible':>9s}")
    for path in gsm8k:
        r = load(path)["results"]["gsm8k"]
        print(
            f"{arm_from(path,'gsm8k'):28s} "
            f"{r.get('exact_match,strict-match', float('nan')):8.4f} "
            f"{r.get('exact_match,flexible-extract', float('nan')):9.4f}"
        )


if __name__ == "__main__":
    main()
