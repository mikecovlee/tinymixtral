#!/usr/bin/env python3
# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""从 checkpoint 恢复训练，自动沿用原始 token/step 目标。"""
import math, sys, argparse, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from model.modeling import TinyMixtralForCausalLM
from scripts.train_utils import (
    BF16AdamW, check_checkpoint_disk_space, final_save, make_adamw,
    make_cosine_schedule, make_wsd_schedule, training_loop,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", default="checkpoints/run")
    p.add_argument("--cache-dir", default="data/c4/tokenized")
    p.add_argument("--batch-size", type=int, default=22)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--max-tokens", type=int, default=None,
                   help="覆盖 checkpoint 中的训练目标（post-training 用）")
    p.add_argument("--output-dir", default=None,
                   help="输出目录（默认同 checkpoint-dir，post-training 建议指定新目录）")
    p.add_argument("--save-every-min", type=int, default=120)
    p.add_argument("--keep-last-checkpoints", type=int, default=5)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--schedule", default="cosine", choices=["cosine", "wsd"],
                   help="LR schedule: cosine or wsd (Warmup-Stable-Decay)")
    p.add_argument("--bf16-optim", action="store_true",
                   help="优化器状态使用 bf16 存储 (节省约 50%% 优化器显存)")
    args = p.parse_args()
    if args.batch_size <= 0 or args.seq_len <= 0:
        p.error("batch-size and seq-len must be positive")
    if args.save_every_min <= 0 or args.log_every <= 0:
        p.error("save-every-min and log-every must be positive")
    if args.keep_last_checkpoints <= 0:
        p.error("keep-last-checkpoints must be positive")

    # ---- 找最新 checkpoint ----
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_dir():
        p.error(f"checkpoint directory does not exist: {ckpt_dir}")
    required_files = ("config.json", "pytorch_model.bin", "training_state.pt")
    ckpts = sorted([
        d for d in ckpt_dir.iterdir()
        if d.is_dir() and d.name.startswith("step_")
        and all((d / name).is_file() for name in required_files)
    ])
    if not ckpts:
        print(f"No checkpoints in {ckpt_dir}", flush=True); sys.exit(1)
    latest = ckpts[-1]
    step_done = int(latest.name.split("_")[1])
    print(f"Loading {latest} (step {step_done})...", flush=True)

    # ---- 模型 ----
    model = TinyMixtralForCausalLM.from_pretrained(str(latest))
    if args.seq_len > model.config.max_position_embeddings:
        p.error(
            f"seq-len {args.seq_len} exceeds model limit "
            f"{model.config.max_position_embeddings}"
        )
    model.gradient_checkpointing_enable()
    model = model.to("cuda").to(torch.bfloat16)
    nM = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {nM:.0f}M params", flush=True)

    # ---- 优化器 + state ----
    bs, seq = args.batch_size, args.seq_len
    chunk = (seq + 1) * bs

    opt = make_adamw(model, lr=args.lr, weight_decay=args.wd, bf16_states=args.bf16_optim)

    state_path = latest / "training_state.pt"
    total_tok = step_done * bs * seq
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if "opt" in state:
            try:
                opt.load_state_dict(state["opt"])
            except ValueError:
                if len(state["opt"].get("param_groups", [])) != 1:
                    raise
                print("Loading legacy single-group AdamW state", flush=True)
                cls = BF16AdamW if args.bf16_optim else torch.optim.AdamW
                opt = cls(
                    model.parameters(), lr=args.lr, weight_decay=args.wd,
                    betas=(0.9, 0.95),
                )
                opt.load_state_dict(state["opt"])
        if "step" in state:
            step_done = state["step"]
        total_tok = state.get("total_tok", total_tok)
        print(f"Loaded optimizer+scheduler state", flush=True)

        saved_bs = state.get("batch_size")
        saved_seq = state.get("seq_len")
        if saved_bs is not None and (saved_bs != bs or saved_seq != seq):
            p.error(
                f"checkpoint uses batch-size={saved_bs}, seq-len={saved_seq}; "
                f"got batch-size={bs}, seq-len={seq}"
            )
    else:
        state = {}

    # ---- 确定训练目标 ----
    output_dir = args.output_dir or args.checkpoint_dir
    is_posttrain = args.max_tokens is not None

    if is_posttrain:
        warmup = args.warmup_steps
        total_steps = math.ceil(args.max_tokens / (bs * seq))
        tokens_per_step = bs * seq
        target_tokens = total_steps * tokens_per_step
        total_tok = 0
        step_done = 0
        fi, ptr = 0, 0
        print(f"Post-training target: {total_steps:,} steps / ~{target_tokens / 1e9:.2f}B tokens "
              f"lr={args.lr:.1e} warmup={warmup}", flush=True)
    else:
        warmup = state.get("warmup_steps", args.warmup_steps)
        total_steps = state["total_steps"]
        print(f"Target: warmup={warmup} total_steps={total_steps}", flush=True)
        if step_done > total_steps:
            p.error(f"checkpoint step {step_done} exceeds target {total_steps}")
        if step_done == total_steps and latest.name.endswith("_final"):
            print(f"Training already complete: {latest}", flush=True)
            return
        fi, ptr = None, None  # 后面从 checkpoint 或 fallback 计算

    check_checkpoint_disk_space(model, output_dir, args.keep_last_checkpoints)

    if is_posttrain:
        # Post-training: 保留 optimizer 动量，用 CLI --lr 覆盖 checkpoint 中的旧 LR
        for pg in opt.param_groups:
            pg["lr"] = args.lr
            pg["initial_lr"] = args.lr
        make_schedule = make_wsd_schedule if args.schedule == "wsd" else make_cosine_schedule
        sched = make_schedule(opt, warmup, total_steps)
        print(f"Starting fresh {args.schedule} schedule: warmup={warmup} total={total_steps}", flush=True)
    else:
        saved_lrs = [pg["lr"] for pg in opt.param_groups]
        make_schedule = make_wsd_schedule if args.schedule == "wsd" else make_cosine_schedule
        sched = make_schedule(opt, warmup, total_steps)
        for pg, lr in zip(opt.param_groups, saved_lrs):
            pg["lr"] = lr
        if "sched" in state:
            sched.load_state_dict(state["sched"])

    # ---- 计算 shard+ptr ----
    files = sorted(glob.glob(f"{args.cache_dir}/train_*.pt"))
    if not files:
        print(f"ERROR: no .pt shards in {args.cache_dir}", flush=True); sys.exit(1)
    if is_posttrain:
        # 新数据集从头开始，fi=0, ptr=0 already set above
        pass
    elif state_path.exists() and "fi" in state:
        # 优先从 checkpoint 恢复位置
        fi = state["fi"]
        ptr = state["ptr"]
    else:
        # Fallback: 根据 total_tok 计算位置
        total_corpus_tokens = sum(
            len(torch.load(f, weights_only=True, mmap=True)) for f in files
        )
        total_corpus_steps = max(1, total_corpus_tokens // (bs * seq))
        remaining = total_tok % (total_corpus_steps * bs * seq)
        fi, ptr = 0, 0
        for i, f in enumerate(files):
            sz = len(torch.load(f, weights_only=True, mmap=True))
            steps_in_shard = sz // chunk
            if remaining >= steps_in_shard * bs * seq:
                remaining -= steps_in_shard * bs * seq
            else:
                fi, ptr = i, (remaining // (bs * seq)) * chunk
                break

    print(f"Resume: step={step_done} shard={fi}/{len(files)} ptr={ptr/1e6:.1f}M", flush=True)

    # ---- 训练 ----
    schedule_args = {"warmup_steps": warmup, "total_steps": total_steps}
    step, total_tok, fi, ptr, elapsed = training_loop(
        model, opt, sched, files, fi=fi, ptr=ptr, total_tok=total_tok,
        bs=bs, seq=seq, chunk=chunk,
        output_dir=output_dir, max_steps=total_steps,
        save_every_min=args.save_every_min, log_every=args.log_every,
        step_start=step_done, schedule_args=schedule_args,
        keep_last_checkpoints=args.keep_last_checkpoints,
    )

    final_save(
        model, opt, sched, output_dir, step, total_tok, elapsed,
        fi, ptr, bs, seq, schedule_args,
    )


if __name__ == "__main__":
    main()
