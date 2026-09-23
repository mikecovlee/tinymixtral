# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""构建 pilot 混合数据集：按精确比例交错多源 shard，硬链零拷贝。

用法（两次连续构建，段互不重叠）:
    python scripts/make_blend_shards.py --output data/pretrain/pilot_blend30 \
        --source data/pretrain/fineweb3 --take 7 \
        --source data/pretrain/cosmopedia3 --take 3 --val-take 1
    # 续训新段：--start 跳过上一次已用的 shard（fineweb3 0..6，cosmopedia3 0..3）
    python scripts/make_blend_shards.py --output data/pretrain/pilot_blend30b \
        --source data/pretrain/fineweb3 --start 7 --take 7 \
        --source data/pretrain/cosmopedia3 --start 4 --take 3 --val-take 1

--take N 取源目录第 start..start+N-1 个 train_*.pt（--start 默认 0，可省略）；
--val-take M 取第 start+N..start+N+M-1 个，硬链到 <output>_val/val_*.pt（永不参与训练）。
交错用 Bresenham 累加器，保持目标比例且两目录前缀即为混合样本。复制模式（跨卷）用 --copy。
"""

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path


def list_shards(src: str) -> list:
    files = sorted(glob.glob(os.path.join(src, "train_*.pt")))
    if not files:
        sys.exit(f"ERROR: no train_*.pt in {src}")
    return files


def interleave(entries: list) -> list:
    """按目标比例交错（整数 Bresenham），确定性输出。"""
    weights = {name: weight for name, weight, _ in entries}
    total_w = sum(weights.values())
    pool = {name: list(files) for name, _, files in entries}
    err = {name: 0 for name in weights}
    order = []
    while any(pool[name] for name, _, _ in entries):
        best = None
        for name, _, _ in entries:
            if not pool[name]:
                continue
            err[name] += weights[name]
            if best is None or err[name] > err[best]:
                best = name
        err[best] -= total_w
        order.append(pool[best].pop(0))
    return order


def place(src_file: str, dst_file: Path, copy: bool):
    if dst_file.exists():
        sys.exit(f"ERROR: refusing to overwrite {dst_file}")
    if copy:
        shutil.copy2(src_file, dst_file)
    else:
        os.link(src_file, dst_file)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", required=True)
    p.add_argument("--source", action="append", default=[])
    p.add_argument("--take", action="append", default=[])
    p.add_argument("--val-take", action="append", default=[])
    p.add_argument("--start", action="append", default=[])
    p.add_argument("--copy", action="store_true")
    args = p.parse_args()

    n = len(args.source)
    if not (len(args.take) == len(args.val_take) == n):
        p.error("--source/--take/--val-take counts must match")
    if len(args.start) not in (0, n):
        p.error(f"--start must be omitted or given {n} time(s), once per --source")
    starts = args.start or [0] * n

    entries, val_entries = [], []
    for src, take, vtake, start in zip(args.source, args.take, args.val_take, starts):
        files = list_shards(src)
        take, vtake, start = int(take), int(vtake), int(start)
        if start < 0:
            p.error(f"{src}: --start must be >= 0")
        if start + take + vtake > len(files):
            p.error(f"{src}: need {start + take + vtake} shards "
                    f"(start {start} + take {take} + val-take {vtake}), "
                    f"only {len(files)} available")
        entries.append((Path(src).name, take, files[start:start + take]))
        if vtake:
            val_entries.append((Path(src).name, vtake, files[start + take:start + take + vtake]))

    out = Path(args.output)
    if out.exists():
        sys.exit(f"ERROR: {out} already exists")
    out.mkdir(parents=True)
    seq = interleave(entries)
    for i, f in enumerate(seq):
        place(f, out / f"train_{i:04d}.pt", args.copy)
    print(f"train: {len(seq)} shards -> {out}")

    if val_entries:
        vout = Path(str(out) + "_val")
        if vout.exists():
            sys.exit(f"ERROR: {vout} already exists")
        vout.mkdir(parents=True)
        vi = 0
        for name, cnt, files in val_entries:
            for f in files:
                place(f, vout / f"val_{vi:04d}.pt", args.copy)
                vi += 1
        print(f"val:   {vi} shards -> {vout} (held out, never trained)")


if __name__ == "__main__":
    main()
