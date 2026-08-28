#!/usr/bin/env python3
"""SAMSum 对话摘要微调。

用法:
    python scripts/train_summarization.py \
        --checkpoint checkpoints/v1b_moe_posttrain/step_0060975_final \
        --output-dir checkpoints/samsum_ft

格式: "Dialogue:\\n{dialogue}\\n\\nSummary:\\n{summary}</s>"
只监督 summary span（含 EOS），其余 -100 掩码。
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.modeling import TinyMixtralForCausalLM
from scripts.train_utils import make_adamw, make_cosine_schedule


def parse_args():
    p = argparse.ArgumentParser(description="SAMSum fine-tuning")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--tokenizer-path", type=str, default="tokenizer/")
    p.add_argument("--output-dir", type=str, default="checkpoints/samsum_ft")
    p.add_argument("--dataset", type=str, default="knkarthick/samsum")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=60)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def format_and_tokenize(dialogue, summary, tokenizer, max_length):
    prompt = f"Dialogue:\n{dialogue}\n\nSummary:"
    summary_part = f"\n{summary}</s>"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    summary_ids = tokenizer.encode(summary_part, add_special_tokens=False)
    total = len(prompt_ids) + len(summary_ids)
    if total > max_length:
        overflow = total - max_length
        summary_ids = summary_ids[:max(1, len(summary_ids) - overflow)]
    ids = prompt_ids + summary_ids
    labels = [-100] * len(prompt_ids) + summary_ids
    return ids, labels


def pack_sequences(examples, max_length, pad_token_id):
    packed_ids, packed_labels = [], []
    buf_ids, buf_labels = [], []

    def flush():
        nonlocal buf_ids, buf_labels
        pad = max_length - len(buf_ids)
        buf_ids.extend([pad_token_id] * pad)
        buf_labels.extend([-100] * pad)
        packed_ids.append(buf_ids)
        packed_labels.append(buf_labels[1:] + [-100])
        buf_ids, buf_labels = [], []

    for ids, labs in examples:
        if len(ids) > max_length:
            ids, labs = ids[:max_length], labs[:max_length]
        if len(buf_ids) + len(ids) > max_length:
            flush()
        buf_ids.extend(ids)
        buf_labels.extend(labs)
    if buf_ids:
        flush()
    return packed_ids, packed_labels


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print(f"Loading tokenizer from {args.tokenizer_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, legacy=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {args.checkpoint}...", flush=True)
    model = TinyMixtralForCausalLM.from_pretrained(args.checkpoint)
    model = model.to(device=device, dtype=torch.bfloat16).train()
    model.gradient_checkpointing_enable()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

    proxy = os.environ.get("https_proxy") or os.environ.get("http_proxy")
    print(f"Loading {args.dataset} (proxy={proxy})...", flush=True)
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train")

    examples = []
    for i, ex in enumerate(ds):
        if args.max_samples and len(examples) >= args.max_samples:
            break
        dlg = ex.get("dialogue", "")
        summ = ex.get("summary", "")
        if not dlg or not summ:
            continue
        examples.append((dlg, summ))
        if len(examples) % 5000 == 0:
            print(f"  {len(examples)} loaded...", flush=True)
    print(f"Loaded {len(examples)} examples", flush=True)

    print("Tokenizing...", flush=True)
    tokenized = []
    for i, (dlg, summ) in enumerate(examples):
        ids, labs = format_and_tokenize(dlg, summ, tokenizer, args.seq_len)
        if len(ids) >= 10:
            tokenized.append((ids, labs))
    del examples
    avg_len = sum(len(ids) for ids, _ in tokenized) / max(len(tokenized), 1)
    print(f"Tokenized {len(tokenized)} examples (avg {avg_len:.0f} tokens)", flush=True)

    print(f"Packing to {args.seq_len}-token sequences...", flush=True)
    packed_ids, packed_labels = pack_sequences(
        tokenized, args.seq_len, tokenizer.pad_token_id)
    del tokenized
    print(f"{len(packed_ids)} sequences", flush=True)

    all_ids = torch.tensor(packed_ids, dtype=torch.long)
    all_labels = torch.tensor(packed_labels, dtype=torch.long)
    del packed_ids, packed_labels

    num_seqs = all_ids.shape[0]
    steps_per_epoch = math.ceil(num_seqs / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    print(f"Training: {num_seqs} seqs x {args.epochs} epochs = {total_steps} steps",
          flush=True)

    opt = make_adamw(model, args.lr, args.wd)
    sched = make_cosine_schedule(opt, args.warmup_steps, total_steps)

    print("Starting...", flush=True)
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        indices = torch.randperm(num_seqs)
        for bi in range(0, num_seqs, args.batch_size):
            batch_idx = indices[bi:bi + args.batch_size]
            input_ids = all_ids[batch_idx].to(device)
            labels = all_labels[batch_idx].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids, labels=labels)

            loss = out["loss"]
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                print(f"  skip step {step + 1}: non-finite loss", flush=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  step {step:5d}/{total_steps}  loss={loss.item():.4f}  "
                    f"lr={sched.get_last_lr()[0]:.2e}  [{elapsed / 60:.1f}m]",
                    flush=True)

            if step % args.save_every == 0:
                sp = output_dir / f"step_{step:07d}"
                model.save_pretrained(str(sp))
                print(f"  saved {sp}", flush=True)

        print(f"  epoch {epoch + 1}/{args.epochs} done "
              f"({(time.time() - t0) / 60:.1f}m)", flush=True)

    final_path = output_dir / f"step_{step:07d}_final"
    model.save_pretrained(str(final_path))
    elapsed = time.time() - t0
    print(f"Done: {step} steps in {elapsed / 60:.1f}m -> {final_path}", flush=True)


if __name__ == "__main__":
    main()
