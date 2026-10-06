#!/usr/bin/env python3
# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""v3 SFT 数据准备入口：三档指令混合 + polish 子集。

配方见 data/v3/README.md 与 data/v3/sft/ 内各脚本：
  build_dataset.py    —— 3M 档通用指令混合（去重/去污染），产出 200k/1M/3M 档
  prefetch_sources.py —— 源文件本地预取（HTTP 代理下 ~100x 提速）
  sample_subset.py    —— 50k CoT-polish 子集（分层采样）

用法:
    python data/v3/prepare_sft.py --prefetch            # 仅预取源文件
    python data/v3/prepare_sft.py                       # 构建主混合
    python data/v3/prepare_sft.py --polish              # 另建 polish 子集
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SFT = REPO / "data" / "v3" / "sft"


def run(cmd: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(str(c) for c in cmd))
    if not dry_run:
        subprocess.run([str(c) for c in cmd], cwd=REPO, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Build the v3 SFT mixtures (200k/1M/3M + polish)")
    p.add_argument("--prefetch", action="store_true", help="只预取源文件")
    p.add_argument("--polish", action="store_true", help="构建 50k polish 子集")
    p.add_argument("--dry-run", action="store_true", help="只打印命令")
    p.add_argument("extra", nargs=argparse.REMAINDER,
                   help="透传给 sft 脚本的额外参数（-- 后）")
    args = p.parse_args()

    if args.prefetch:
        run([sys.executable, SFT / "prefetch_sources.py", *args.extra], args.dry_run)
        return

    run([sys.executable, SFT / "build_dataset.py", *args.extra], args.dry_run)

    if args.polish:
        run([sys.executable, SFT / "sample_subset.py", *args.extra], args.dry_run)

    print("done: data/v3/sft (see build_dataset.py output dirs)")


if __name__ == "__main__":
    sys.exit(main())
