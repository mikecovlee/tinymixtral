#!/usr/bin/env python3
# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""Instruction tuning (SFT) for TinyMixtral.

Loads a base checkpoint, formats instruction data with the project's chat
template (<|user|> / <|assistant|>), masks loss on non-assistant tokens,
and fine-tunes with AdamW + cosine LR schedule.

Usage:
    python versions/v3.0-it/train_sft.py --checkpoint checkpoints/knowledge_posttrain/step_0040691_final
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model.modeling import TinyMixtralForCausalLM  # noqa: E402
from scripts.train_utils import (
    make_adamw,
    make_cosine_schedule,
    prune_periodic_checkpoints,
    save_training_state,
)  # noqa: E402


def format_conversation(turns: list) -> tuple:
    parts = []
    spans = []
    for turn in turns:
        role = turn.get("from") or turn.get("role") or ""
        content = turn.get("value") or turn.get("content") or ""
        if role in ("human", "user"):
            parts.append(f"<|user|>\n{content}</s>\n")
        elif role in ("gpt", "assistant"):
            start = sum(len(p) for p in parts)
            parts.append(f"<|assistant|>\n{content}</s>\n")
            end = sum(len(p) for p in parts)
            spans.append((start, end))
        elif role == "system":
            parts.append(f"<|system|>\n{content}</s>\n")
    return "".join(parts), spans


def tokenize_with_mask(full_text, spans, tokenizer):
    encoded = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    labels = [-100] * len(ids)
    for i, (ch_start, ch_end) in enumerate(offsets):
        if ch_start >= ch_end:
            continue
        for as_start, as_end in spans:
            if ch_start >= as_start and ch_end <= as_end:
                labels[i] = ids[i]
                break
            if ch_start < as_end and ch_end > as_start:
                labels[i] = ids[i]
                break
    return ids, labels


def pack_sequences(examples, max_length, pad_token_id):
    packed_ids, packed_labels = [], []
    buf_ids, buf_labels = [], []

    def flush():
        nonlocal buf_ids, buf_labels
        pad = max_length - len(buf_ids)
        buf_ids.extend([pad_token_id] * pad)
        buf_labels.extend([-100] * pad)
        shifted = buf_labels[1:] + [-100]
        packed_ids.append(np.asarray(buf_ids, dtype=np.int32))
        packed_labels.append(np.asarray(shifted, dtype=np.int32))
        buf_ids, buf_labels = [], []

    for ids, labs in examples:
        if len(ids) > max_length:
            ids = ids[:max_length]
            labs = labs[:max_length]
        if len(buf_ids) + len(ids) > max_length:
            flush()
        buf_ids.extend(ids)
        buf_labels.extend(labs)
    if buf_ids:
        flush()
    return packed_ids, packed_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer-path", default="tokenizer/")
    p.add_argument("--output-dir", default="checkpoints/sft")
    p.add_argument("--dataset", default="teknium/OpenHermes-2.5")
    p.add_argument("--max-samples", type=int, default=20000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed; --resume requires it to replay the shuffle order")
    p.add_argument("--resume", default=None,
                   help="checkpoint dir containing training_state.pt")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--keep-last", type=int, default=2,
                   help="periodic checkpoints to keep (0 = keep all; *_final never pruned)")
    args = p.parse_args()

    if str(args.device).startswith("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.grad_accum < 1:
        raise SystemExit("--grad-accum must be >= 1")
    micro_per_step = args.grad_accum
    if args.seed is not None:
        torch.manual_seed(args.seed)
        try:
            torch.cuda.manual_seed_all(args.seed)
        except Exception:
            pass
        random.seed(args.seed)
        np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(f"Device: {device}", flush=True)

    print(f"Loading tokenizer from {args.tokenizer_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, legacy=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    src = args.resume or args.checkpoint
    print(f"Loading model from {src}...", flush=True)
    model = TinyMixtralForCausalLM.from_pretrained(src)
    model = model.to(device).train()
    model.gradient_checkpointing_enable()
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_total:,} params", flush=True)

    print(f"Loading {args.dataset} (max {args.max_samples})...", flush=True)
    if args.dataset.endswith(".parquet"):
        ds = load_dataset("parquet", data_files=args.dataset, split="train",
                          streaming=True)
    else:
        ds = load_dataset(args.dataset, split="train", streaming=True)
    texts, spans_list = [], []
    for ex in ds:
        if len(texts) >= args.max_samples:
            break
        conversations = ex.get("conversations")
        if not conversations:
            continue
        text, spans = format_conversation(conversations)
        if not spans:
            continue
        texts.append(text)
        spans_list.append(spans)
        if len(texts) % 5000 == 0:
            print(f"  {len(texts)} loaded...", flush=True)
    print(f"Loaded {len(texts)} examples", flush=True)

    print("Tokenizing...", flush=True)
    tokenized = []
    for i, (text, spans) in enumerate(zip(texts, spans_list, strict=False)):
        ids, labs = tokenize_with_mask(text, spans, tokenizer)
        if len(ids) >= 10:
            tokenized.append((np.asarray(ids, dtype=np.int32),
                              np.asarray(labs, dtype=np.int32)))
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(texts)}...", flush=True)
    del texts, spans_list
    avg_len = sum(len(ids) for ids, _ in tokenized) / max(len(tokenized), 1)
    print(f"Tokenized {len(tokenized)} examples (avg {avg_len:.0f} tokens)", flush=True)

    print(f"Packing to {args.seq_len}-token sequences...", flush=True)
    packed_ids, packed_labels = pack_sequences(tokenized, args.seq_len, tokenizer.pad_token_id)
    del tokenized
    print(f"{len(packed_ids)} sequences", flush=True)

    all_ids = torch.from_numpy(np.stack(packed_ids)).contiguous()
    all_labels = torch.from_numpy(np.stack(packed_labels)).contiguous()
    del packed_ids, packed_labels

    num_seqs = all_ids.shape[0]
    steps_per_epoch = math.ceil(num_seqs / (args.batch_size * micro_per_step))
    total_steps = steps_per_epoch * args.epochs
    print(f"Training: {num_seqs} seqs x {args.epochs} epochs = {total_steps} steps", flush=True)

    opt = make_adamw(model, args.lr, args.wd)
    sched = make_cosine_schedule(opt, args.warmup_steps, total_steps)

    step = 0
    start_epoch = 0
    skip_micros = 0
    if args.resume:
        if args.seed is None:
            raise SystemExit("--resume requires --seed to replay the shuffle order")
        state_path = Path(args.resume) / "training_state.pt"
        if not state_path.exists():
            raise SystemExit(f"{state_path} not found; cannot resume")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step = int(state["step"])
        start_epoch = int(state["fi"])
        skip_micros = int(state["ptr"])
        print(f"Resuming: step={step} epoch={start_epoch} skip_bi={skip_micros}", flush=True)

    def save_at(sp, epoch, next_bi):
        model.save_pretrained(str(sp))
        save_training_state(
            sp / "training_state.pt", opt, sched, step,
            step * args.batch_size * micro_per_step * args.seq_len,
            args.warmup_steps, total_steps,
            fi=epoch, ptr=next_bi,
            batch_size=args.batch_size, seq_len=args.seq_len,
        )

    print("Starting...", flush=True)
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        if args.seed is not None:
            g = torch.Generator()
            g.manual_seed(args.seed * 100_000 + epoch)
            indices = torch.randperm(num_seqs, generator=g)
        else:
            indices = torch.randperm(num_seqs)
        opt.zero_grad(set_to_none=True)
        micro = 0
        last_loss = None
        for bi in range(0, num_seqs, args.batch_size):
            if epoch == start_epoch and bi < skip_micros:
                continue
            batch_idx = indices[bi : bi + args.batch_size]
            input_ids = all_ids[batch_idx].long().to(device)
            labels = all_labels[batch_idx].long().to(device)

            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids, labels=labels)
            else:
                out = model(input_ids, labels=labels)

            loss = out["loss"]
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                micro = 0
                print(f"  [warn] step {step + 1}: non-finite loss, skipping", flush=True)
                continue

            (loss / micro_per_step).backward()
            last_loss = loss
            micro += 1
            if micro < micro_per_step:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            micro = 0
            step += 1

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  step {step:5d}/{total_steps}  loss={last_loss.item():.4f}  "
                    f"lr={sched.get_last_lr()[0]:.2e}  [{elapsed / 60:.1f}m]",
                    flush=True,
                )

            if step % args.save_every == 0:
                sp = output_dir / f"step_{step:07d}"
                save_at(sp, epoch, bi + args.batch_size)
                print(f"  -> saved {sp}", flush=True)
                if args.keep_last > 0:
                    prune_periodic_checkpoints(output_dir, args.keep_last)
        if micro > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        skip_micros = 0

    final_path = output_dir / f"step_{step:07d}_final"
    save_at(final_path, args.epochs, 0)
    elapsed = time.time() - t0
    print(f"Done: {step} steps in {elapsed / 60:.1f}m -> {final_path}", flush=True)


if __name__ == "__main__":
    main()
