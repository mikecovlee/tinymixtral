#!/usr/bin/env python3
"""用 HuggingFace AutoModel 加载 TinyMixtral 进行交互对话。

用法:
    python chat_hf.py publish/             # 本地 publish 目录
    python chat_hf.py your-username/tinymixtral-432m  # HF Hub 模型
"""

import sys
from pathlib import Path

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
    model_path = sys.argv[1] if len(sys.argv) > 1 else "publish/"
    print(f"Loading {model_path} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.bfloat16,
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    nM = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {nM:.0f}M params on {device}")
    print(f"Config: {model.config.hidden_size}d/{model.config.num_hidden_layers}L/{model.config.num_local_experts}E")
    print(f"Model class: {type(model).__name__}")
    print(f"model_type: {model.config.model_type}")

    print(f"\n交互模式 (temp=0.7, top_p=0.9, max_tokens=256)")
    print("输入 'quit' 退出\n")
    try:
        while True:
            prompt = input(">>> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("quit", "exit", "q"):
                break
            reply = generate(model, tokenizer, prompt)
            print(f"\n{reply}\n")
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
