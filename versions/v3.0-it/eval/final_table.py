#!/usr/bin/env python
"""Consolidate per-model eval artifacts (data/eval) into one markdown comparison table.

Usage: python versions/v3.0-it/eval/final_table.py [--dir data/eval] [--ref <ref-model>] [--models <model> ...]
"""
import argparse
import glob
import json
import math
import os

DIMS = ("correctness", "completeness", "reasoning", "instruction_following")
HARNESS = [
    ("hellaswag", "acc_norm"),
    ("piqa", "acc_norm"),
    ("winogrande", "acc"),
    ("arc_easy", "acc"),
    ("arc_challenge", "acc_norm"),
    ("openbookqa", "acc_norm"),
    ("lambada_openai", "acc"),
]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_rubric(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if all(k in d for k in DIMS):
                rows[str(d.get("id"))] = sum(d[k] for k in DIMS) / len(DIMS)
    return rows


def mean_se(v):
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(v) / n
    if n < 2:
        return m, 0.0, n
    var = sum((x - m) ** 2 for x in v) / (n - 1)
    return m, math.sqrt(var / n), n


def paired(a, b):
    ids = sorted(set(a) & set(b), key=lambda s: int(s) if s.isdigit() else s)
    if not ids:
        return float("nan"), float("nan"), float("nan"), 0
    m, se, n = mean_se([a[i] - b[i] for i in ids])
    t = (m / se) if se > 0 and n > 1 else float("nan")
    return m, se, t, n


def harness_mean(d):
    if not d:
        return None
    res = d.get("results", {})
    vals = []
    for task, metric in HARNESS:
        entry = res.get(task)
        if not entry:
            return None
        v = entry.get(metric + ",none")
        if v is None:
            return None
        vals.append(v)
    return sum(vals) / len(vals)


def ifeval_vals(d):
    if not d:
        return None
    t = d.get("results", {}).get("ifeval")
    if not t:
        return None
    return (
        t.get("prompt_level_strict_acc,none"),
        t.get("inst_level_strict_acc,none"),
        t.get("prompt_level_loose_acc,none"),
        t.get("inst_level_loose_acc,none"),
    )


def gsm8k_vals(d):
    if not d:
        return None
    t = d.get("results", {}).get("gsm8k")
    if not t:
        return None
    return t.get("exact_match,strict-match"), t.get("exact_match,flexible-extract")


def fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def discover(directory):
    found = set()
    for pat, pre in (("rubric_*.jsonl", "rubric_"), ("harness_*.json", "harness_")):
        for p in glob.glob(os.path.join(directory, pat)):
            base = os.path.basename(p)[len(pre):]
            found.add(base.rsplit(".jsonl", 1)[0].rsplit(".json", 1)[0])
    return sorted(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/eval")
    ap.add_argument("--ref", default=None, help="reference model name (default: first model)")
    ap.add_argument("--models", nargs="*", default=None, help="model names (default: discover in --dir)")
    args = ap.parse_args()

    models = args.models or discover(args.dir)
    if args.ref and args.ref in models:
        models = [args.ref] + [m for m in models if m != args.ref]
    ref_name = args.ref or (models[0] if models else "")
    rub = {a: load_rubric(os.path.join(args.dir, f"rubric_{a}.jsonl")) for a in models}
    ref_rows = rub.get(ref_name, {})

    print(f"| model | rubric mean | rubric vs {ref_name} (Δ±se, t, n) | IFEval p-str | IFEval i-str | GSM8K strict | GSM8K flex | harness |")
    print("|---|---|---|---|---|---|---|---|")
    for a in models:
        m, se, n = mean_se(list(rub[a].values())) if rub[a] else (float("nan"), float("nan"), 0)
        if a != ref_name and ref_rows and rub[a]:
            dm, dse, t, dn = paired(rub[a], ref_rows)
            rub_vs = f"{dm:+.2f}±{dse:.2f}, t={t:+.1f}, n={dn}"
        else:
            rub_vs = "ref" if a == ref_name else "—"
        ifv = ifeval_vals(load_json(os.path.join(args.dir, f"ifeval_{a}.json")))
        gsv = gsm8k_vals(load_json(os.path.join(args.dir, f"gsm8k_{a}.json")))
        hm = harness_mean(load_json(os.path.join(args.dir, f"harness_{a}.json")))
        print(
            f"| {a} | {m:.2f}±{se:.2f} (n={n}) | {rub_vs} | "
            f"{fmt(ifv[0]) if ifv else '—'} | {fmt(ifv[1]) if ifv else '—'} | "
            f"{fmt(gsv[0], 4) if gsv else '—'} | {fmt(gsv[1]) if gsv else '—'} | {fmt(hm)} |"
        )


if __name__ == "__main__":
    main()
