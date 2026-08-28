#!/usr/bin/env python3
"""SAMSum 摘要评测：greedy 生成 + ROUGE-1/2/L。

用法:
    # 0-shot 评测
    python scripts/eval_summarization.py \
        --checkpoint checkpoints/v1b_moe_posttrain/step_0060975_final \
        --output evals/samsum_0shot.json

    # 微调后评测
    python scripts/eval_summarization.py \
        --checkpoint checkpoints/samsum_ft/step_0000540_final \
        --output evals/samsum_ft_final.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="SAMSum summarization evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--dataset", type=str, default="knkarthick/samsum")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--precision", type=str, default="bf16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def load_model(checkpoint_path, device, precision):
    dtype = DTYPE_MAP[precision]
    ckpt = Path(checkpoint_path)
    cfg_file = ckpt / "config.json"
    is_custom = False
    if cfg_file.exists():
        with open(cfg_file) as f:
            cfg = json.load(f)
        is_custom = "num_local_experts" in cfg

    if is_custom:
        from hf.modeling_tinymixtral import TinyMixtralForCausalLM
        model = TinyMixtralForCausalLM.from_pretrained(str(ckpt))
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt), torch_dtype=dtype, trust_remote_code=True)

    model = model.to(device=device, dtype=dtype).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Model: {n/1e6:.1f}M params, dtype={precision}, device={device}", flush=True)
    return model


def load_samsum(split, dataset_name, limit):
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    examples = [(ex["dialogue"], ex["summary"]) for ex in ds
                if ex.get("dialogue") and ex.get("summary")]
    print(f"Loaded {len(examples)} examples ({split} split)")
    return examples


def generate_batch(model, tokenizer, prompts, max_new_tokens, device, batch_size=16):
    results = [""] * len(prompts)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    all_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    max_len = max(len(ids) for ids in all_ids)

    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_ids = all_ids[batch_start:batch_end]
        B = len(batch_ids)

        padded = [[pad_id] * (max_len - len(ids)) + ids for ids in batch_ids]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        cache = None
        active = torch.ones(B, dtype=torch.bool, device=device)
        generated = [[] for _ in range(B)]

        for _ in range(max_new_tokens):
            with torch.inference_mode():
                out = model(input_ids, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_token = out["logits"][:, -1, :].argmax(dim=-1)

            for i in range(B):
                if active[i]:
                    tok = next_token[i].item()
                    if eos_id is not None and tok == eos_id:
                        active[i] = False
                    else:
                        generated[i].append(tok)

            if not active.any():
                break
            input_ids = next_token.unsqueeze(1)

        for i in range(B):
            idx = batch_start + i
            results[idx] = tokenizer.decode(
                generated[i], skip_special_tokens=True).strip()

        done = min(batch_end, len(prompts))
        if done % 50 < batch_size:
            print(f"  {done}/{len(prompts)} generated", flush=True)

    return results


def compute_rouge(predictions, references):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    agg = {m: {"p": 0.0, "r": 0.0, "f": 0.0} for m in scorer.rouge_types}
    n = len(predictions)
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for m in scorer.rouge_types:
            agg[m]["p"] += scores[m].precision
            agg[m]["r"] += scores[m].recall
            agg[m]["f"] += scores[m].fmeasure
    return {m: {k: round(v / n * 100, 2) for k, v in vals.items()}
            for m, vals in agg.items()}


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tok_path = args.tokenizer or args.checkpoint
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tok_path, legacy=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.checkpoint, device, args.precision)
    examples = load_samsum(args.split, args.dataset, args.limit)

    template = "Dialogue:\n{dialogue}\n\nSummary:"
    prompts = [template.format(dialogue=d) for d, _ in examples]
    references = [s for _, s in examples]

    print(f"Generating {len(prompts)} summaries (max_new_tokens={args.max_new_tokens})...",
          flush=True)
    t0 = time.time()
    predictions = generate_batch(
        model, tokenizer, prompts, args.max_new_tokens, device)
    elapsed = time.time() - t0
    print(f"Generation done in {elapsed:.0f}s ({elapsed/len(prompts):.2f}s/example)",
          flush=True)

    if args.verbose:
        for i in range(min(3, len(prompts))):
            print(f"\n  [{i}] Ref: {references[i][:100]}...", flush=True)
            print(f"      Pred: {predictions[i][:100]}...", flush=True)

    rouge = compute_rouge(predictions, references)
    print("\nROUGE (test):", flush=True)
    for m, vals in rouge.items():
        print(f"  {m}: P={vals['p']:.2f}  R={vals['r']:.2f}  F={vals['f']:.2f}", flush=True)

    output = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "num_examples": len(examples),
        "max_new_tokens": args.max_new_tokens,
        "rouge": rouge,
        "time_seconds": round(elapsed, 1),
    }
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
