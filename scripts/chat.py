#!/usr/bin/env python3
"""与训练好的 TinyMixtral 模型交互式对话。

用法:
    python scripts/chat.py --checkpoint checkpoints/run/step_0005000 --tokenizer tokenizer/
    python scripts/chat.py --checkpoint checkpoints/run/step_0010000 --max-tokens 512 --temp 0.8
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.modeling import TinyMixtralForCausalLM
from transformers import AutoTokenizer


def generate(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
    """简单自回归生成，支持 temperature + top-p sampling。"""
    device = next(model.parameters()).device
    device_type = str(device).split(":")[0]
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

    generated = []
    # eos_token_id 可能为 None（某些 tokenizer 只在底层 BPE 中存储）
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.convert_tokens_to_ids("</s>")
    if eos_id is None or eos_id == tokenizer.unk_token_id:
        # 无法可靠检测 EOS，禁用提前停止
        eos_id = None

    max_position = getattr(model.config, "max_position_embeddings", 2048)

    for _ in range(max_new_tokens):
        # 截断到最大位置
        if input_ids.shape[1] > max_position:
            input_ids = input_ids[:, -max_position:]

        with torch.inference_mode():
            with torch.amp.autocast(device_type, dtype=torch.bfloat16):
                logits = model(input_ids)["logits"][:, -1, :].float()

        # Temperature
        logits = logits / max(temperature, 1e-8)

        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
            sorted_mask[:, 0] = False
            indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
            logits[indices_to_remove] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if eos_id is not None and next_token.item() == eos_id:
            break

        generated.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser(description="TinyMixtral 交互式对话")
    p.add_argument("--checkpoint", required=True, help="模型 checkpoint 路径")
    p.add_argument("--tokenizer", default="tokenizer/", help="Tokenizer 路径")
    p.add_argument("--max-tokens", type=int, default=256, help="最大生成长度")
    p.add_argument("--temp", type=float, default=0.7, help="采样温度")
    p.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling 阈值")
    p.add_argument("--device", default="cuda", help="设备 (cuda/cpu)")
    p.add_argument("--prompt", default=None, help="单次生成模式（不进入交互）")
    args = p.parse_args()
    if args.max_tokens <= 0:
        p.error("max-tokens must be positive")
    if args.temp <= 0:
        p.error("temp must be positive")
    if not 0 < args.top_p <= 1:
        p.error("top-p must be in (0, 1]")

    # 加载
    device = torch.device(args.device)
    if not torch.cuda.is_available() and device.type == "cuda":
        print("CUDA not available, falling back to CPU"); device = torch.device("cpu")
    print(f"Loading model from {args.checkpoint} ...")
    model = TinyMixtralForCausalLM.from_pretrained(args.checkpoint)
    model.eval()
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(torch.bfloat16)
    nM = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {nM:.0f}M params on {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, legacy=False)
    if len(tokenizer) != model.config.vocab_size:
        p.error(
            f"tokenizer vocab size is {len(tokenizer)}, "
            f"model expects {model.config.vocab_size}"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 单次模式
    if args.prompt:
        print(f"\n{'─' * 40}\n{args.prompt}\n{'─' * 40}")
        reply = generate(model, tokenizer, args.prompt, args.max_tokens, args.temp, args.top_p)
        print(reply)
        return

    # 交互模式
    print(f"\n交互模式 (temp={args.temp}, top_p={args.top_p}, max_tokens={args.max_tokens})")
    print("输入 'quit' 或 Ctrl+C 退出\n")

    try:
        while True:
            prompt = input(">>> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("quit", "exit", "q"):
                break

            reply = generate(model, tokenizer, prompt, args.max_tokens, args.temp, args.top_p)
            print(f"\n{reply}\n")
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
