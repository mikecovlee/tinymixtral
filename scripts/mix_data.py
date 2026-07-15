#!/usr/bin/env python3
# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""交错合并多个数据集的 tokenized shard，支持加权混合。

用法:
    # 等权重混合
    python scripts/mix_data.py data/fineweb data/cosmopedia --output data/mixed

    # 加权混合（4:1 = 80/20）
    python scripts/mix_data.py data/fineweb data/code --output data/mixed --weights 4 1
"""

import argparse
import shutil
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        description="Interleave tokenized shards from multiple datasets"
    )
    p.add_argument("sources", nargs="+", help="源 tokenized 目录列表")
    p.add_argument("--output", required=True, help="输出目录")
    p.add_argument(
        "--weights", type=int, nargs="+", default=None,
        help="每个源目录的权重（默认等权），如 --weights 4 1 表示 80/20",
    )
    p.add_argument("--symlink", action="store_true", help="使用符号链接而非复制")
    args = p.parse_args()

    if args.weights and len(args.weights) != len(args.sources):
        p.error(f"--weights 数量 ({len(args.weights)}) 必须与源目录数量 ({len(args.sources)}) 一致")
    for w in (args.weights or []):
        if w <= 0:
            p.error("权重必须为正整数")

    weights = args.weights if args.weights else [1] * len(args.sources)

    # 收集各源目录的 shard 列表
    shard_groups = []
    for src in args.sources:
        src_path = Path(src)
        if not src_path.is_dir():
            p.error(f"目录不存在: {src}")
        files = sorted(src_path.glob("train_*.pt"))
        if not files:
            p.error(f"{src} 中没有 train_*.pt shard")
        shard_groups.append(files)

    for src, files, w in zip(args.sources, shard_groups, weights):
        total_tokens = 0
        for f in files:
            total_tokens += f.stat().st_size // 8  # int64 = 8 bytes
        print(f"{src}: {len(files)} shards, ~{total_tokens / 1e6:.0f}M tokens (weight={w})")

    # 交错复制
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    idx = 0
    pointers = [0] * len(shard_groups)
    copy_fn = shutil.copy if not args.symlink else lambda src, dst: dst.symlink_to(
        src.resolve()
    ) or None

    while any(p < len(g) for p, g in zip(pointers, shard_groups)):
        for i, (files, weight) in enumerate(zip(shard_groups, weights)):
            taken = 0
            while taken < weight and pointers[i] < len(files):
                src = files[pointers[i]]
                dst = output / f"train_{idx:04d}.pt"
                copy_fn(str(src), str(dst))
                print(f"  {src.name} -> {dst.name}")
                pointers[i] += 1
                idx += 1
                taken += 1

    print(f"\nDone: {idx} shards -> {output}")
    # 估算总 token 数
    total = sum(f.stat().st_size for f in output.glob("train_*.pt")) // 8
    print(f"Total: ~{total / 1e6:.1f}M tokens")


if __name__ == "__main__":
    main()
