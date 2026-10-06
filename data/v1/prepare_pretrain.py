#!/usr/bin/env python3
# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""v1 预训练数据准备：C4-en 4.00B token（c4en 池）。

配方见 data/v1/README.md。在线路径直接从 HuggingFace 流式 tokenize；
离线路径用本地 .jsonl.zst → parquet → tokenize（网络受限时用）。

用法:
    python data/v1/prepare_pretrain.py --dry-run
    python data/v1/prepare_pretrain.py --raw-zst /path/to/c4-zst
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
    p = argparse.ArgumentParser(description="Build the v1 pretrain pool (C4-en, 4.00B tokens)")
    p.add_argument("--output-root", default="data", help="数据根目录（默认 data/）")
    p.add_argument("--tokenizer", default="tokenizer/", help="tokenizer 目录")
    p.add_argument("--raw-zst", default=None,
                   help="本地 C4 .jsonl.zst 目录（离线模式；缺省则从 HF 流式）")
    p.add_argument("--max-tokens", type=int, default=4_000_000_000, help="池子 token 上限")
    p.add_argument("--shard-size", type=int, default=100_000_000, help="每 shard token 数")
    p.add_argument("--dry-run", action="store_true", help="只打印命令")
    args = p.parse_args()

    root = Path(args.output_root)
    out = root / "pretrain" / "c4en"

    if args.raw_zst:
        parquet_dir = root / "raw" / "c4en-parquet"
        run([PIPELINE / "zst_jsonl_to_parquet.py", "--input", args.raw_zst,
             "--output", parquet_dir], args.dry_run)
        run([PIPELINE / "prepare_data_local.py", "--input", parquet_dir,
             "--tokenizer", args.tokenizer, "--output", out,
             "--max-tokens", args.max_tokens, "--shard-size", args.shard_size], args.dry_run)
    else:
        run([PIPELINE / "prepare_data.py", "--dataset", "allenai/c4", "--subset", "en",
             "--tokenizer", args.tokenizer, "--output", out,
             "--max-tokens", args.max_tokens, "--shard-size", args.shard_size], args.dry_run)

    print(f"done: {out} ({args.max_tokens / 1e9:.2f}B tokens)")


if __name__ == "__main__":
    sys.exit(main())
