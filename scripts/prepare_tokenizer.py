#!/usr/bin/env python3
# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""训练或加载 tokenizer。

用法:
    # 从零训练 SentencePiece tokenizer
    python scripts/prepare_tokenizer.py \
        --train-files data/raw/*.txt \
        --output tokenizer/ \
        --vocab-size 32000

    # 下载并使用已有的 HuggingFace tokenizer
    python scripts/prepare_tokenizer.py \
        --from-hf meta-llama/Llama-2-7b-hf \
        --output tokenizer/
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.config import TinyMixtralConfig  # noqa: E402


def train_sentencepiece(train_files, output_dir, vocab_size=32000, model_prefix="tokenizer"):
    """从文本文件训练 SentencePiece BPE tokenizer。"""
    from tokenizers import SentencePieceBPETokenizer

    os.makedirs(output_dir, exist_ok=True)
    tokenizer = SentencePieceBPETokenizer(unk_token="<unk>")
    tokenizer.train(
        files=train_files,
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<|user|>", "<|assistant|>"],
    )

    print(f"Tokenizer saved to {output_dir}")
    return tokenizer


def save_hf_tokenizer(tokenizer_object, output_dir):
    """将 SentencePiece 模型转换为 HuggingFace 兼容格式。"""
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_object,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tokenizer.save_pretrained(output_dir)
    print(f"HF tokenizer saved to {output_dir}")
    return tokenizer


def from_huggingface(model_id, output_dir, expected_vocab_size):
    """从 HuggingFace Hub 下载 tokenizer。"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # 确保有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if len(tokenizer) != expected_vocab_size:
        raise ValueError(f"Tokenizer vocab size is {len(tokenizer)}, expected {expected_vocab_size}")

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer from {model_id} saved to {output_dir}")
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Prepare tokenizer for TinyMixtral")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train-files", type=str, nargs="+", help="Text files for training")
    source.add_argument("--from-hf", type=str, help="HuggingFace model ID to copy tokenizer from")
    parser.add_argument("--output", type=str, default="tokenizer/", help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    args = parser.parse_args()
    expected_vocab_size = TinyMixtralConfig().vocab_size
    if args.vocab_size != expected_vocab_size:
        parser.error(f"vocab-size must match model config ({expected_vocab_size})")

    if args.from_hf:
        from_huggingface(args.from_hf, args.output, expected_vocab_size)
    else:
        tokenizer_object = train_sentencepiece(args.train_files, args.output, expected_vocab_size)
        tokenizer = save_hf_tokenizer(tokenizer_object, args.output)
        if len(tokenizer) != expected_vocab_size:
            raise ValueError(f"Tokenizer vocab size is {len(tokenizer)}, expected {expected_vocab_size}")


if __name__ == "__main__":
    main()
