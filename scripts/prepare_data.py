#!/usr/bin/env python3
# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""预 tokenize 数据集到本地 .pt 文件，消除训练时的数据加载瓶颈。

用法:
    python scripts/prepare_data.py --dataset allenai/c4 --subset en \
        --tokenizer tokenizer/ --output data/c4/tokenized --max-tokens 4000000000
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.config import TinyMixtralConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="allenai/c4")
    p.add_argument("--subset", default=None, help="数据集 subset/配置名（如 c4 的 en）")
    p.add_argument("--tokenizer", default="tokenizer/")
    p.add_argument("--output", default="data/c4/tokenized")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--max-samples", type=int, default=None,
                        help="最大处理样本数")
    target.add_argument("--max-tokens", type=int, default=None,
                        help="最大写入 token 数")
    p.add_argument("--skip-tokens", type=int, default=0,
                   help="从数据集开头跳过指定 token 数（获取非重叠数据）")
    p.add_argument("--shard-size", type=int, default=100_000_000,
                   help="每个 shard 的 token 数（int64 默认约 800MB）")
    p.add_argument("--force", action="store_true",
                   help="完整替换已有输出目录")
    args = p.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        p.error("max-samples must be positive")
    if args.max_tokens is not None and args.max_tokens <= 0:
        p.error("max-tokens must be positive")
    if args.shard_size <= 0:
        p.error("shard-size must be positive")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not output.is_dir():
        p.error(f"output path is not a directory: {output}")
    stale_staging = sorted(output.parent.glob(f".{output.name}.tmp-*"))
    stale_backups = sorted(output.parent.glob(f".{output.name}.backup-*"))

    if not output.exists() and len(stale_backups) == 1:
        os.replace(stale_backups[0], output)
        stale_backups = []
        print(f"Recovered interrupted replacement: {output}")

    stale_paths = stale_staging + stale_backups
    if stale_paths:
        print("ERROR: stale data preparation directories found:")
        for path in stale_paths:
            print(f"  {path}")
        print("  Verify and remove them before retrying.")
        sys.exit(1)

    if output.exists() and any(output.iterdir()) and not args.force:
        print(f"ERROR: output directory is not empty: {output}")
        print("  Use --force to replace it, or choose a new output directory.")
        sys.exit(1)

    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    backup = output.with_name(f".{output.name}.backup-{os.getpid()}")
    staging.mkdir(parents=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, legacy=False)
        expected_vocab_size = TinyMixtralConfig().vocab_size
        if len(tokenizer) != expected_vocab_size:
            raise ValueError(
                f"Tokenizer vocab size is {len(tokenizer)}, expected {expected_vocab_size}"
            )
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id for document boundaries")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs = {}
        if args.subset:
            load_kwargs["name"] = args.subset
        print(f"Loading {args.dataset}" + (f"/{args.subset}" if args.subset else "") + " (streaming)...")
        ds = load_dataset(args.dataset, split="train", streaming=True, **load_kwargs)

        buf = []
        buf_tokens = 0
        shard_idx = 0
        total_tokens = 0
        count = 0
        t0 = time.time()
        skip_remaining = args.skip_tokens

        for ex in ds:
            text = ex.get("text", "")
            if len(text) < 10:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            ids.append(tokenizer.eos_token_id)

            # 跳过前 skip_tokens 个 token（获取非重叠数据）
            if skip_remaining > 0:
                if len(ids) <= skip_remaining:
                    skip_remaining -= len(ids)
                    continue
                else:
                    ids = ids[skip_remaining:]
                    skip_remaining = 0

            if args.max_tokens is not None:
                remaining = args.max_tokens - total_tokens
                if remaining <= 0:
                    break
                if len(ids) > remaining:
                    ids = ids[:remaining]
                    ids[-1] = tokenizer.eos_token_id
                if not ids:
                    break
            buf.extend(ids)
            buf_tokens += len(ids)
            total_tokens += len(ids)
            count += 1
            if buf_tokens >= args.shard_size:
                tokens = torch.tensor(buf, dtype=torch.long)
                path = staging / f"train_{shard_idx:04d}.pt"
                torch.save(tokens, path)
                print(f"  Saved {path}: {buf_tokens/1e6:.1f}M tokens (sample {count})")
                buf = []
                buf_tokens = 0
                shard_idx += 1
            if count % 200000 == 0:
                print(f"  {count} samples, {total_tokens/1e6:.1f}M tokens, {time.time()-t0:.0f}s")
            if args.max_samples is not None and count >= args.max_samples:
                break
            if args.max_tokens is not None and total_tokens >= args.max_tokens:
                break

        if args.max_tokens is not None and total_tokens != args.max_tokens:
            raise RuntimeError(
                f"Dataset ended after {total_tokens} tokens, before target {args.max_tokens}"
            )

        if buf:
            tokens = torch.tensor(buf, dtype=torch.long)
            path = staging / f"train_{shard_idx:04d}.pt"
            torch.save(tokens, path)
            print(f"  Saved {path}: {buf_tokens/1e6:.1f}M tokens (final)")
            shard_idx += 1

        if shard_idx == 0:
            raise RuntimeError("No usable samples were tokenized")

        if output.exists():
            os.replace(output, backup)
        try:
            os.replace(staging, output)
        except BaseException:
            if backup.exists():
                os.replace(backup, output)
            raise
        if backup.exists():
            shutil.rmtree(backup)

        print(f"Done: {count} samples → {total_tokens/1e6:.1f}M tokens → {shard_idx} shards")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
