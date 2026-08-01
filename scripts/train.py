#!/usr/bin/env python3
# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""预训练：从零开始，按 token 或 step 目标运行。"""
import argparse
import glob
import math
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from model.config import CPT_ROUTER_RECOMPUTE_MODES, TinyMixtralConfig
from model.modeling import TinyMixtralForCausalLM
from scripts.train_utils import (
    DEFAULT_WSD_DECAY_RATIO,
    SCHEDULE_KINDS,
    TRAINING_FAILURE_POLICIES,
    broadcast_rank0_object, build_code_manifest, build_data_manifest,
    build_tokenizer_manifest,
    check_checkpoint_disk_space, final_save,
    make_adamw, make_cosine_schedule, make_wsd_schedule, new_run_id,
    reject_uninitialized_torchrun_environment, training_loop,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="data/c4/tokenized")
    p.add_argument("--tokenizer-dir", default="tokenizer")
    p.add_argument("--output-dir", default="checkpoints/run")
    p.add_argument("--batch-size", type=int, default=22)
    p.add_argument("--seq-len", type=int, default=1024)
    target = p.add_mutually_exclusive_group()
    target.add_argument("--max-tokens", type=int, default=None,
                        help="训练 token 目标（默认 4,000,000,000）")
    target.add_argument("--max-steps", type=int, default=None,
                        help="训练 step 目标")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--save-every-min", type=int, default=120)
    p.add_argument("--keep-last-checkpoints", type=int, default=5)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument(
        "--schedule",
        default="cosine",
        choices=SCHEDULE_KINDS,
        help="LR schedule: cosine or wsd (Warmup-Stable-Decay)",
    )
    p.add_argument(
        "--bf16-optim",
        action="store_true",
        help=(
            "Use FP32 AdamW update arithmetic while storing moments at each "
            "parameter's dtype; FP32 CPT parameters keep FP32 moments"
        ),
    )
    p.add_argument(
        "--recompute",
        choices=CPT_ROUTER_RECOMPUTE_MODES,
        default="global",
        help="MoE Router recompute policy: follow global, force on, or force off",
    )
    p.add_argument(
        "--failure-policy",
        choices=TRAINING_FAILURE_POLICIES,
        default="fail-stop",
        help=(
            "Training failure recovery: fail-stop avoids per-step full CPU snapshots; "
            "exact-rollback enables same-process replay at high host-memory cost"
        ),
    )
    p.add_argument("--eval-on-save", action="store_true",
                   help="每次保存后同步执行 CPU GLUE 评测")
    args = p.parse_args()
    try:
        reject_uninitialized_torchrun_environment()
    except RuntimeError as exc:
        p.error(str(exc))
    if args.batch_size <= 0 or args.seq_len <= 0:
        p.error("batch-size and seq-len must be positive")
    if args.save_every_min <= 0 or args.log_every <= 0:
        p.error("save-every-min and log-every must be positive")
    if args.keep_last_checkpoints <= 0:
        p.error("keep-last-checkpoints must be positive")
    if args.max_tokens is not None and args.max_tokens <= 0:
        p.error("max-tokens must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        p.error("max-steps must be positive")
    if args.warmup_steps < 0:
        p.error("warmup-steps must be non-negative")
    if not math.isfinite(args.lr) or args.lr <= 0:
        p.error("lr must be finite and positive")
    if not math.isfinite(args.wd) or args.wd < 0:
        p.error("wd must be finite and non-negative")
    cfg = TinyMixtralConfig(cpt_router_recompute=args.recompute)
    if args.seq_len > cfg.max_position_embeddings:
        p.error(
            f"seq-len {args.seq_len} exceeds model limit {cfg.max_position_embeddings}"
        )

    output_dir = Path(args.output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        p.error(f"output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.glob("step_*")):
        p.error(f"output directory already contains checkpoints: {output_dir}; use resume.py or a new directory")
    # ---- 数据 ----
    files = sorted(glob.glob(f"{args.cache_dir}/train_*.pt"))
    if not files:
        print(f"ERROR: no .pt shards in {args.cache_dir}", flush=True); sys.exit(1)
    print(f"Data: {len(files)} shards from {args.cache_dir}", flush=True)
    code_manifest = build_code_manifest()
    data_manifest = build_data_manifest(files)
    tokenizer_manifest = build_tokenizer_manifest(args.tokenizer_dir)
    run_id = broadcast_rank0_object(new_run_id())
    checkpoint_metadata = {
        "run_id": run_id,
        "code_manifest": code_manifest,
        "data_manifest": data_manifest,
        "tokenizer_manifest": tokenizer_manifest,
    }
    print(
        f"Run: {run_id} code={code_manifest['sha256'][:12]} "
        f"data={data_manifest['sha256'][:12]} "
        f"tokenizer={tokenizer_manifest['sha256'][:12]}",
        flush=True,
    )

    # ---- 模型 ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = TinyMixtralForCausalLM(cfg)
    model.gradient_checkpointing_enable()
    model = model.to("cuda").to(torch.bfloat16)
    print(
        "Activation checkpointing: global=on, "
        f"router={model.config.cpt_router_recompute}",
        flush=True,
    )
    print(f"Training failure policy: {args.failure_policy}", flush=True)
    nM = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {cfg.hidden_size}d/{cfg.num_hidden_layers}L/{cfg.num_local_experts}E "
          f"GQA{cfg.num_attention_heads}h/{cfg.num_key_value_heads}kv {nM:.0f}M params", flush=True)
    check_checkpoint_disk_space(model, args.output_dir, args.keep_last_checkpoints)

    # ---- 优化器 + schedule ----
    bs, seq = args.batch_size, args.seq_len
    chunk = (seq + 1) * bs
    tokens_per_step = bs * seq
    target_tokens = args.max_tokens if args.max_tokens is not None else 4_000_000_000
    total_steps = (
        args.max_steps
        if args.max_steps is not None
        else math.ceil(target_tokens / tokens_per_step)
    )
    if total_steps <= 0:
        p.error("training targets must be positive")
    target_tokens = total_steps * tokens_per_step

    optimizer_kind = "bf16_adamw" if args.bf16_optim else "adamw"
    opt = make_adamw(
        model,
        lr=args.lr,
        weight_decay=args.wd,
        bf16_states=args.bf16_optim,
    )
    schedule_kind = args.schedule
    schedule_decay_ratio = (
        DEFAULT_WSD_DECAY_RATIO if schedule_kind == "wsd" else None
    )
    warmup_cap = max(
        total_steps - (2 if schedule_kind == "wsd" else 1),
        0,
    )
    effective_warmup = min(args.warmup_steps, warmup_cap)
    if schedule_kind == "wsd":
        sched = make_wsd_schedule(
            opt,
            effective_warmup,
            total_steps,
            decay_ratio=schedule_decay_ratio,
        )
    else:
        sched = make_cosine_schedule(opt, effective_warmup, total_steps)

    print(f"Training target: {total_steps:,} steps / {target_tokens / 1e9:.3f}B tokens "
          f"bs={bs} seq={seq}", flush=True)
    print(
        f"Training recipe: optimizer={optimizer_kind} schedule={schedule_kind}"
        + (
            f" decay_ratio={schedule_decay_ratio:g}"
            if schedule_decay_ratio is not None
            else ""
        ),
        flush=True,
    )

    # ---- 训练 ----
    schedule_args = {
        "warmup_steps": effective_warmup,
        "total_steps": total_steps,
        "schedule_kind": schedule_kind,
        "schedule_decay_ratio": schedule_decay_ratio,
    }
    step, total_tok, fi, ptr, elapsed = training_loop(
        model, opt, sched, files, fi=0, ptr=0, total_tok=0,
        bs=bs, seq=seq, chunk=chunk,
        output_dir=args.output_dir, max_steps=total_steps,
        save_every_min=args.save_every_min, log_every=args.log_every,
        schedule_args=schedule_args, eval_on_save=args.eval_on_save,
        keep_last_checkpoints=args.keep_last_checkpoints,
        checkpoint_metadata=checkpoint_metadata,
        tokenizer_dir=args.tokenizer_dir,
        failure_policy=args.failure_policy,
        optimizer_kind=optimizer_kind,
    )

    final_save(
        model, opt, sched, args.output_dir, step, total_tok, elapsed,
        fi, ptr, bs, seq, schedule_args, checkpoint_metadata,
        data_files=files,
        tokenizer_dir=args.tokenizer_dir,
        optimizer_kind=optimizer_kind,
    )


if __name__ == "__main__":
    main()
