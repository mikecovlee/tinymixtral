#!/usr/bin/env python3
# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""v3 预训练数据准备：main_s1..s4 四段混合池（六源，共 8.05B token）。

配方见 data/v3/README.md（逐段 shard 范围同 versions/v3.0/README.md）。
源池（100M-token shard）: fineweb3 / cosmopedia3 / r5_web / r5_code / r5_math /
r5_wiki / p5_web / p5_synth —— 硬链接零拷贝，跨盘时加 --copy。

用法:
    python data/v3/prepare_pretrain.py --dry-run
    python data/v3/prepare_pretrain.py --segments s1 s2
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PIPELINE = REPO / "data" / "pipeline"

SEGMENTS = {
    "s1": [("fineweb3", 8, 7), ("p5_web", 0, 2), ("r5_web", 0, 4),
           ("cosmopedia3", 0, 3), ("r5_code", 0, 2), ("r5_math", 0, 1), ("r5_wiki", 0, 1)],
    "s2": [("fineweb3", 15, 7), ("p5_web", 2, 2), ("r5_web", 4, 4),
           ("cosmopedia3", 4, 1), ("p5_synth", 0, 1), ("r5_code", 2, 3),
           ("r5_math", 1, 1), ("r5_wiki", 1, 1)],
    "s3": [("fineweb3", 22, 7), ("p5_web", 4, 2), ("r5_web", 8, 4),
           ("p5_synth", 1, 3), ("r5_code", 5, 3), ("r5_math", 2, 1), ("r5_wiki", 2, 2)],
    "s4": [("fineweb3", 29, 7), ("p5_web", 6, 2), ("r5_web", 12, 4),
           ("p5_synth", 4, 2), ("r5_code", 8, 2), ("r5_math", 3, 2), ("r5_wiki", 4, 1)],
}
SEG_TOKENS = {"s1": "2.00B", "s2": "1.94B", "s3": "2.20B", "s4": "1.91B"}


def run(cmd: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(str(c) for c in cmd))
    if not dry_run:
        subprocess.run([str(c) for c in cmd], cwd=REPO, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Build main_s1..s4 blend pools (8.05B tokens total)")
    p.add_argument("--output-root", default="data", help="数据根目录（默认 data/）")
    p.add_argument("--segments", nargs="+", default=["s1", "s2", "s3", "s4"],
                   choices=sorted(SEGMENTS), help="要构建的段（默认全部）")
    p.add_argument("--copy", action="store_true", help="复制而非硬链接（跨卷时用）")
    p.add_argument("--dry-run", action="store_true", help="只打印命令")
    args = p.parse_args()

    root = Path(args.output_root)
    pretrain = root / "pretrain"

    for seg in args.segments:
        cmd: list[str] = [PIPELINE / "make_blend_shards.py",
                          "--output", pretrain / f"main_{seg}"]
        for name, start, take in SEGMENTS[seg]:
            cmd += ["--source", pretrain / name,
                    "--start", start, "--take", take, "--val-take", 0]
        if args.copy:
            cmd.append("--copy")
        run(cmd, args.dry_run)
        print(f"done: {pretrain / f'main_{seg}'} ({SEG_TOKENS[seg]})")


if __name__ == "__main__":
    sys.exit(main())
