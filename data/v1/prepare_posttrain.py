#!/usr/bin/env python3
# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""v1 后训练数据准备：Wiki + 抽样 FineWeb-Edu，共 1.03B token（knowledge_blend）。

配方见 data/v1/README.md；比例 Wiki 1.67% / FineWeb-Edu 98.33%（DATA_LICENSES）。
历史构建用旧 CLI `prepare_data_local.py --pools wiki,webtext-sampled`；
现按当前工具等价重建：逐池 tokenize 后 mix_data 加权交错。

用法:
    python data/v1/prepare_posttrain.py --dry-run
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
    p = argparse.ArgumentParser(description="Build the v1 post-train pool (knowledge_blend, 1.03B)")
    p.add_argument("--output-root", default="data", help="数据根目录（默认 data/）")
    p.add_argument("--tokenizer", default="tokenizer/", help="tokenizer 目录")
    p.add_argument("--wiki-parquet", default="data/raw/wiki-parquet", help="Wiki parquet 目录")
    p.add_argument("--webtext-parquet", default="data/raw/webtext-sampled-parquet",
                   help="抽样 FineWeb-Edu parquet 目录")
    p.add_argument("--max-tokens", type=int, default=1_030_000_000, help="池子 token 上限")
    p.add_argument("--dry-run", action="store_true", help="只打印命令")
    args = p.parse_args()

    root = Path(args.output_root)
    wiki_tok = root / "posttrain2" / "wiki"
    webtext_tok = root / "posttrain2" / "webtext-sampled"
    out = root / "posttrain2" / "knowledge_blend"

    run([PIPELINE / "prepare_data_local.py", "--input", args.wiki_parquet,
         "--tokenizer", args.tokenizer, "--output", wiki_tok], args.dry_run)
    run([PIPELINE / "prepare_data_local.py", "--input", args.webtext_parquet,
         "--tokenizer", args.tokenizer, "--output", webtext_tok], args.dry_run)
    run([PIPELINE / "mix_data.py", wiki_tok, webtext_tok,
         "--output", out, "--weights", "1", "59"], args.dry_run)

    print(f"done: {out} (Wiki 1.67% / FineWeb-Edu 98.33%, target {args.max_tokens / 1e9:.2f}B)")


if __name__ == "__main__":
    sys.exit(main())
