#!/usr/bin/env python3
# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""用 HuggingFace AutoModel 加载 TinyMixtral 进行交互对话。

用法:
    python chat_hf.py publish/             # 本地 publish 目录
    python chat_hf.py your-username/tinymixtral-432m  # HF Hub 模型
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
    """自回归生成。"""
    device = next(model.parameters()).device
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    eos_id = tokenizer.eos_token_id
    max_pos = getattr(model.config, "max_position_embeddings", 2048)

    generated = []
    for _ in range(max_new_tokens):
        if inputs.shape[1] > max_pos:
            inputs = inputs[:, -max_pos:]

        with torch.inference_mode():
            out = model(inputs)
            logits = out.logits[:, -1, :].float() / max(temperature, 1e-8)

        if top_p < 1.0:
            sorted_logits, sorted_ids = torch.sort(logits, descending=True, dim=-1)
            cumsum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumsum > top_p
            mask[:, 1:] = mask[:, :-1].clone()
            mask[:, 0] = False
            to_remove = mask.scatter(1, sorted_ids, mask)
            logits = logits.masked_fill(to_remove, -float("inf"))

        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        if eos_id is not None and token.item() == eos_id:
            break
        generated.append(token.item())
        inputs = torch.cat([inputs, token], dim=-1)

    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Load a published TinyMixtral Hugging Face model for chat"
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        default="publish/",
        help="Local publish directory or Hugging Face model id",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Forbid Hugging Face network access",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Generate once and exit instead of entering interactive mode",
    )
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")

    model_path = args.model_path
    print(f"Loading {model_path} ...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    )
    model.eval()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    model = model.to(device)
    nM = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {nM:.0f}M params on {device}")
    print(f"Config: {model.config.hidden_size}d/{model.config.num_hidden_layers}L/{model.config.num_local_experts}E")
    print(f"Model class: {type(model).__name__}")
    print(f"model_type: {model.config.model_type}")

    if args.prompt is not None:
        reply = generate(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(reply)
        return

    print(
        f"\n交互模式 (temp={args.temperature}, top_p={args.top_p}, "
        f"max_tokens={args.max_tokens})"
    )
    print("输入 'quit' 退出\n")
    try:
        while True:
            prompt = input(">>> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("quit", "exit", "q"):
                break
            reply = generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(f"\n{reply}\n")
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
