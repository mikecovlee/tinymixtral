# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""解析 psmux capture-pane 快照里的训练日志，输出 val PPL 对比与 top-k 判据。"""

import argparse
import json
import re
import sys
from pathlib import Path

STEP_RE = re.compile(
    r"step\s+(\d+):\s+loss=([\d.]+)\s+ce=([\d.]+)\s+aux=([\d.]+)"
    r".*util%=\[([\d. ]+)\].*lr=([\d.e+-]+)"
)
EVAL_RE = re.compile(r"\[eval\]\s+step\s+(\d+):\s+val_loss=([\d.]+)\s+val_ppl=([\d.]+)")
TARGET_RE = re.compile(r"Training target:\s+([\d,]+)\s+steps")
CFG_RE = re.compile(r"-Cfg\s+(\S+)")
NAME_RE = re.compile(r"NAME=(\S+)")
MODEL_RE = re.compile(r"Model:\s+(\d+)d/(\d+)L/(\d+)Etop(\d)")


def split_cells(lines):
    cells, current, buf = {}, None, []
    for ln in lines:
        m = NAME_RE.search(ln)
        if m:
            if current:
                cells.setdefault(current, []).extend(buf)
            current, buf = m.group(1), [ln]
            continue
        if current:
            if "run_pilot.ps1" in ln and "-Cfg" in ln:
                continue
            buf.append(ln)
    if current:
        cells.setdefault(current, []).extend(buf)
    return cells


def parse_cell(lines):
    steps, evals = [], []
    target = None
    arch = None
    for ln in lines:
        m = STEP_RE.search(ln)
        if m:
            steps.append({
                "step": int(m.group(1)), "loss": float(m.group(2)),
                "ce": float(m.group(3)), "aux": float(m.group(4)),
                "util": [float(v) for v in m.group(5).split()],
                "lr": float(m.group(6)),
            })
        m = EVAL_RE.search(ln)
        if m:
            evals.append({
                "step": int(m.group(1)), "val_loss": float(m.group(2)),
                "val_ppl": float(m.group(3)),
            })
        m = TARGET_RE.search(ln)
        if m:
            target = int(m.group(1).replace(",", ""))
        m = MODEL_RE.search(ln)
        if m:
            arch = {"hidden": int(m.group(1)), "layers": int(m.group(2)),
                    "experts": int(m.group(3)), "top_k": int(m.group(4))}
    util_min = util_max = None
    if steps:
        last = steps[-1]
        util_min, util_max = min(last["util"]), max(last["util"])
    return {"target_steps": target, "arch": arch, "last_step": steps[-1] if steps else None,
            "n_steps_logged": len(steps), "evals": evals,
            "util_last": {"min": util_min, "max": util_max}}


def verdict(c1, c2, flat_tol=0.01, worse_tol=0.03):
    e1 = {p["step"]: p["val_ppl"] for p in c1["evals"]}
    e2 = {p["step"]: p["val_ppl"] for p in c2["evals"]}
    common = sorted(set(e1) & set(e2))
    if not common:
        return {"status": "incomplete",
                "note": "无同步数 val 点（等两格各出至少 1 个 [eval]）"}
    s = common[-1]
    p1, p2 = e1[s], e2[s]
    rel = (p1 - p2) / p2
    if rel < -flat_tol:
        decision = "better: top-1 val PPL 明显更低，主设计转 top-1"
    elif abs(rel) <= flat_tol:
        decision = "flat: 差距 ≤1%，主设计转 top-1 省算力"
    elif rel > worse_tol:
        decision = "worse: top-1 明显差（>+3%），维持 top-2"
    else:
        decision = "between: 1%-3% 灰区，矩阵加跑 C2 复核"
    complete = (c1["last_step"] and c2["last_step"]
                and c1["last_step"]["step"] >= c1["target_steps"] - 200
                and c2["last_step"]["step"] >= c2["target_steps"] - 200)
    return {"status": "decided" if complete else "provisional",
            "compared_at_step": s,
            "top1_val_ppl": p1, "top2_val_ppl": p2,
            "rel_diff_top1_vs_top2": round(rel, 4), "decision": decision}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--snapshot", help="psmux capture-pane 全量回滚文件（含两格）")
    src.add_argument("--logs", nargs="+", help="每格单独的日志文件（按 top1/top2 顺序）")
    ap.add_argument("--json", metavar="OUT", help="报告另存 JSON")
    args = ap.parse_args()

    if args.snapshot:
        lines = Path(args.snapshot).read_text(errors="replace").splitlines()
        cells = split_cells(lines)
        if not cells:
            print("未找到 NAME= 标记行，快照里没有可解析的训练段", file=sys.stderr)
            sys.exit(2)
    else:
        cells = {}
        for i, path in enumerate(args.logs):
            name = f"cell{i + 1}"
            txt = Path(path).read_text(errors="replace").splitlines()
            m = next((CFG_RE.search(l) for l in txt if CFG_RE.search(l)), None)
            if m:
                name = m.group(1)
            cells[name] = txt

    parsed = {k: parse_cell(v) for k, v in cells.items()}
    for name, c in parsed.items():
        print(f"== {name} ==")
        print(f"  arch={c['arch']} target_steps={c['target_steps']} "
              f"logged_steps={c['n_steps_logged']} eval_points={len(c['evals'])}")
        if c["last_step"]:
            ls = c["last_step"]
            print(f"  last step {ls['step']}: loss={ls['loss']:.4f} ce={ls['ce']:.4f} "
                  f"aux={ls['aux']:.2f} util_min={c['util_last']['min']:.1f}% "
                  f"util_max={c['util_last']['max']:.1f}%")
        for e in c["evals"]:
            print(f"  [eval] step {e['step']:6d}: val_ppl={e['val_ppl']:.3f}")

    rep = {"cells": parsed}
    if len(parsed) >= 2:
        if {"pilot_top1", "pilot_top2"} <= set(parsed):
            a, b = "pilot_top1", "pilot_top2"
        else:
            a, b = list(parsed)[:2]
        rep["compare"] = verdict(parsed[a], parsed[b])
        rep["compare"]["pair"] = f"{a} vs {b}"
        print(f"\n判据[{a} vs {b}]: {json.dumps(rep['compare'], ensure_ascii=False)}")
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        print(f"JSON -> {args.json}")


if __name__ == "__main__":
    main()
