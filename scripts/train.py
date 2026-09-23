#!/usr/bin/env python3
# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""预训练：从零开始，按 token 或 step 目标运行。"""
import random
import sys, argparse, glob, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from model.config import TinyMixtralConfig
from model.modeling import TinyMixtralForCausalLM
from scripts.train_utils import (
    check_checkpoint_disk_space, final_save, make_adamw, make_cosine_schedule,
    make_val_evaluator, make_wsd_schedule, training_loop,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None,
                   help="模型配置 JSON 文件（默认使用 TinyMixtralConfig 默认值）")
    p.add_argument("--cache-dir", default="data/c4/tokenized")
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
    p.add_argument("--save-every-min", type=int, default=120)
    p.add_argument("--keep-last-checkpoints", type=int, default=5)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--schedule", default="cosine", choices=["cosine", "wsd"],
                   help="LR schedule: cosine or wsd (Warmup-Stable-Decay)")
    p.add_argument("--bf16-optim", action="store_true",
                   help="优化器状态使用 bf16 存储 (节省约 50%% 优化器显存)")
    p.add_argument("--chunked-ce", action="store_true",
                   help="分块计算 CE（不物化全量 fp32 logits，省约 2-3GB 显存）")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子（模型初始化与路由噪声）")
    p.add_argument("--val-dir", default=None,
                   help="val 集目录（内含 val_*.pt，永不参与训练）")
    p.add_argument("--eval-every-steps", type=int, default=2000,
                   help="每多少步在 val 集上算一次 PPL")
    p.add_argument("--eval-max-tokens", type=int, default=2_000_000,
                   help="每次 val 评估消耗的 token 上限")
    args = p.parse_args()
    if args.batch_size <= 0 or args.seq_len <= 0:
        p.error("batch-size and seq-len must be positive")
    if args.save_every_min <= 0 or args.log_every <= 0:
        p.error("save-every-min and log-every must be positive")
    if args.keep_last_checkpoints <= 0:
        p.error("keep-last-checkpoints must be positive")
    if args.config is not None:
        cfg = TinyMixtralConfig.from_json_file(args.config)
        print(f"Config loaded from {args.config}", flush=True)
    else:
        cfg = TinyMixtralConfig()
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

    val_files = sorted(glob.glob(f"{args.val_dir}/val_*.pt")) if args.val_dir else []
    if args.val_dir and not val_files:
        p.error(f"--val-dir {args.val_dir} contains no val_*.pt shards")
    if val_files:
        print(f"Val: {len(val_files)} held-out shards from {args.val_dir}", flush=True)

    # ---- 模型 ----
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = TinyMixtralForCausalLM(cfg)
    model.gradient_checkpointing_enable()
    model.use_chunked_ce = args.chunked_ce
    model = model.to("cuda").to(torch.bfloat16)
    nM = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {cfg.hidden_size}d/{cfg.num_hidden_layers}L/{cfg.num_local_experts}E"
          f"top{cfg.num_experts_per_tok} "
          f"GQA{cfg.num_attention_heads}h/{cfg.num_key_value_heads}kv {nM:.0f}M params "
          f"qk_norm={cfg.use_qk_norm} aux={cfg.router_aux_loss_coef} "
          f"jitter={cfg.router_jitter_noise} seed={args.seed} "
          f"chunked_ce={args.chunked_ce}", flush=True)
    check_checkpoint_disk_space(model, args.output_dir, args.keep_last_checkpoints)

    # ---- 优化器 + schedule ----
    bs, seq = args.batch_size, args.seq_len
    chunk = (seq + 1) * bs
    tokens_per_step = bs * seq
    target_tokens = args.max_tokens if args.max_tokens is not None else 4_000_000_000
    total_steps = args.max_steps or math.ceil(target_tokens / tokens_per_step)
    if total_steps <= 0:
        p.error("training targets must be positive")
    target_tokens = total_steps * tokens_per_step

    opt = make_adamw(model, lr=args.lr, weight_decay=args.wd, bf16_states=args.bf16_optim)
    effective_warmup = min(args.warmup_steps, max(total_steps - 1, 0))
    make_schedule = make_wsd_schedule if args.schedule == "wsd" else make_cosine_schedule
    sched = make_schedule(opt, effective_warmup, total_steps)

    print(f"Training target: {total_steps:,} steps / {target_tokens / 1e9:.3f}B tokens "
          f"bs={bs} seq={seq}", flush=True)

    # ---- 训练 ----
    schedule_args = {"warmup_steps": effective_warmup, "total_steps": total_steps}
    eval_fn = make_val_evaluator(val_files, bs, seq, args.eval_max_tokens, device="cuda")
    step, total_tok, fi, ptr, elapsed = training_loop(
        model, opt, sched, files, fi=0, ptr=0, total_tok=0,
        bs=bs, seq=seq, chunk=chunk,
        output_dir=args.output_dir, max_steps=total_steps,
        save_every_min=args.save_every_min, log_every=args.log_every,
        schedule_args=schedule_args,
        keep_last_checkpoints=args.keep_last_checkpoints,
        eval_fn=eval_fn, eval_every=args.eval_every_steps,
    )

    final_save(
        model, opt, sched, args.output_dir, step, total_tok, elapsed,
        fi, ptr, bs, seq, schedule_args,
    )


if __name__ == "__main__":
    main()
