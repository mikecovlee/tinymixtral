#!/usr/bin/env python3
# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""v2 预训练数据准备：smollm_blend（FineWeb-Edu-10B : Cosmopedia-v2 = 36:4）4.00B token。

配方见 data/v2/README.md（池名 smollm-cosmo-tiny）。v2.0-beta 记录的 89:11
变体（smollm-cosmo-tiny-2）同源不同采样，用 --weights 36 4 改 89 11 即可。

用法:
    python data/v2/prepare_pretrain.py --dry-run
    python data/v2/prepare_pretrain.py --weights 89 11 --output-pool smollm-cosmo-tiny-2
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PIPELINE = REPO / "data" / "pipeline"


def run(cmd: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(str(c) for c in cmd))
    if not dry_run:
        subprocess.run([str(c) for c in cmd], cwd=REPO, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Build smollm_blend (FineWeb-Edu + Cosmopedia-v2, 36:4)")
    p.add_argument("--output-root", default="data", help="数据根目录（默认 data/）")
    p.add_argument("--tokenizer", default="tokenizer/", help="tokenizer 目录")
    p.add_argument("--fineweb-parquet", default="data/raw/fineweb-edu-parquet",
                   help="FineWeb-Edu-10B parquet 目录")
    p.add_argument("--cosmo-parquet", default="data/raw/cosmopedia-v2-parquet",
                   help="Cosmopedia-v2 parquet 目录")
    p.add_argument("--weights", type=int, nargs=2, default=[36, 4],
                   help="混合权重（默认 36 4）")
    p.add_argument("--output-pool", default="smollm_blend", help="输出池名（默认 smollm_blend）")
    p.add_argument("--max-tokens", type=int, default=4_000_000_000, help="池子 token 上限")
    p.add_argument("--dry-run", action="store_true", help="只打印命令")
    args = p.parse_args()

    root = Path(args.output_root)
    fw_tok = root / "pretrain" / "fineweb-edu"
    cosmo_tok = root / "pretrain" / "cosmopedia-v2"
    out = root / "pretrain" / args.output_pool

    run([PIPELINE / "prepare_data_local.py", "--input", args.fineweb_parquet,
         "--tokenizer", args.tokenizer, "--output", fw_tok,
         "--max-tokens", args.max_tokens], args.dry_run)
    run([PIPELINE / "prepare_data_local.py", "--input", args.cosmo_parquet,
         "--tokenizer", args.tokenizer, "--output", cosmo_tok,
         "--max-tokens", args.max_tokens], args.dry_run)
    run([PIPELINE / "mix_data.py", fw_tok, cosmo_tok,
         "--output", out,
         "--weights", str(args.weights[0]), str(args.weights[1])], args.dry_run)

    print(f"done: {out} ({args.weights[0]}:{args.weights[1]}, target {args.max_tokens / 1e9:.2f}B)")


if __name__ == "__main__":
    sys.exit(main())
