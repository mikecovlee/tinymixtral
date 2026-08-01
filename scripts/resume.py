#!/usr/bin/env python3
# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""从严格 CPT checkpoint 恢复训练，沿用原始 token/step 目标。"""
import sys, argparse, glob, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from model.config import CPT_ROUTER_RECOMPUTE_MODES
from model.modeling import TinyMixtralForCausalLM
from scripts.train_utils import (
    _is_link_or_reparse_point,
    SCHEDULE_KINDS,
    TRAINING_FAILURE_POLICIES,
    build_code_manifest, build_data_manifest, build_tokenizer_manifest,
    check_checkpoint_disk_space, final_save,
    distributed_world_size, make_adamw, make_cosine_schedule, make_wsd_schedule,
    reject_uninitialized_torchrun_environment, restore_rank_rng_state,
    training_loop, validate_training_state, verify_checkpoint_file_hashes,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", default="checkpoints/run")
    p.add_argument("--cache-dir", default="data/c4/tokenized")
    p.add_argument("--tokenizer-dir", default="tokenizer")
    p.add_argument("--batch-size", type=int, default=22)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=None,
                   help="strict CPT resume 不支持覆盖；必须沿用 checkpoint")
    p.add_argument("--wd", type=float, default=None,
                   help="strict CPT resume 不支持覆盖；必须沿用 checkpoint")
    p.add_argument("--warmup-steps", type=int, default=None,
                   help="strict CPT resume 不支持覆盖；必须沿用 checkpoint")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="首版 CPT resume 禁用；不得重置全局 step")
    p.add_argument("--output-dir", default=None,
                   help="输出目录（默认同 checkpoint-dir）")
    p.add_argument("--save-every-min", type=int, default=120)
    p.add_argument("--keep-last-checkpoints", type=int, default=5)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument(
        "--schedule",
        choices=SCHEDULE_KINDS,
        default=None,
        help="Assert the checkpoint schedule kind; omitted means inherit it",
    )
    p.add_argument(
        "--bf16-optim",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Assert the checkpoint optimizer kind; omitted means inherit it",
    )
    p.add_argument(
        "--recompute",
        choices=CPT_ROUTER_RECOMPUTE_MODES,
        default=None,
        help="Override MoE Router recompute policy; default uses checkpoint config",
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
    if args.max_tokens is not None:
        p.error(
            "--max-tokens is disabled for first-version CPT resume because it "
            "would reset the training step while preserving committed CPT state. "
            "Resume the checkpoint's original target instead."
        )
    requested_overrides = [
        name for name, value in (
            ("--lr", args.lr),
            ("--wd", args.wd),
            ("--warmup-steps", args.warmup_steps),
        ) if value is not None
    ]
    if requested_overrides:
        p.error(
            "strict CPT resume must use the optimizer and schedule stored in the "
            "checkpoint; unsupported overrides: " + ", ".join(requested_overrides)
        )

    # ---- 找最新 checkpoint ----
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_dir():
        p.error(f"checkpoint directory does not exist: {ckpt_dir}")
    if _is_link_or_reparse_point(ckpt_dir):
        p.error(f"checkpoint directory must not be a link or reparse point: {ckpt_dir}")
    required_files = ("config.json", "pytorch_model.bin", "training_state.pt")
    checkpoint_pattern = re.compile(r"^step_(\d+)(?:_final)?$")
    ckpts = []
    for directory in ckpt_dir.iterdir():
        match = checkpoint_pattern.fullmatch(directory.name)
        if match is not None and _is_link_or_reparse_point(directory):
            p.error(
                "checkpoint candidates must not be links or reparse points: "
                f"{directory}"
            )
        required_paths = [directory / name for name in required_files]
        redirected_required = [
            path for path in required_paths if _is_link_or_reparse_point(path)
        ]
        if match is not None and redirected_required:
            p.error(
                "checkpoint files must not be links or reparse points: "
                + ", ".join(str(path) for path in redirected_required)
            )
        if (
            directory.is_dir()
            and match is not None
            and all(path.is_file() for path in required_paths)
        ):
            ckpts.append((int(match.group(1)), directory.name.endswith("_final"), directory))
    ckpts.sort(key=lambda item: (item[0], item[1]))
    if not ckpts:
        print(f"No checkpoints in {ckpt_dir}", flush=True); sys.exit(1)
    step_done, _, latest = ckpts[-1]
    print(f"Loading {latest} (step {step_done})...", flush=True)

    state_path = latest / "training_state.pt"
    try:
        if _is_link_or_reparse_point(state_path) or not state_path.is_file():
            raise RuntimeError(
                "training_state.pt must be a regular local file before loading: "
                f"{state_path}"
            )
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        validate_training_state(
            state,
            checkpoint_step=step_done,
            expected_world_size=distributed_world_size(),
        )
        verify_checkpoint_file_hashes(latest, state)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot resume {latest}: this is not a valid first-version CPT "
            f"training checkpoint ({exc})"
        ) from exc

    optimizer_kind = state["optimizer_kind"]
    schedule_kind = state["schedule_kind"]
    schedule_decay_ratio = state["schedule_decay_ratio"]
    if args.schedule is not None and args.schedule != schedule_kind:
        p.error(
            f"checkpoint uses schedule={schedule_kind}; got --schedule "
            f"{args.schedule}"
        )
    checkpoint_uses_bf16_optimizer = optimizer_kind == "bf16_adamw"
    if (
        args.bf16_optim is not None
        and args.bf16_optim != checkpoint_uses_bf16_optimizer
    ):
        requested_optimizer = (
            "bf16_adamw" if args.bf16_optim else "adamw"
        )
        p.error(
            f"checkpoint uses optimizer={optimizer_kind}; explicit CLI requested "
            f"{requested_optimizer}"
        )

    # ---- 代码、数据与恢复游标身份 ----
    current_code_manifest = build_code_manifest()
    files = sorted(glob.glob(f"{args.cache_dir}/train_*.pt"))
    if not files:
        print(f"ERROR: no .pt shards in {args.cache_dir}", flush=True); sys.exit(1)
    current_data_manifest = build_data_manifest(files)
    current_tokenizer_manifest = build_tokenizer_manifest(args.tokenizer_dir)
    validate_training_state(
        state,
        checkpoint_step=step_done,
        expected_run_id=state["run_id"],
        expected_code_manifest=current_code_manifest,
        expected_data_manifest=current_data_manifest,
        expected_tokenizer_manifest=current_tokenizer_manifest,
        expected_world_size=distributed_world_size(),
    )

    saved_bs = state["batch_size"]
    saved_seq = state["seq_len"]
    if saved_bs != args.batch_size or saved_seq != args.seq_len:
        p.error(
            f"checkpoint uses batch-size={saved_bs}, seq-len={saved_seq}; "
            f"got batch-size={args.batch_size}, seq-len={args.seq_len}"
        )
    bs, seq = saved_bs, saved_seq
    chunk = (seq + 1) * bs
    fi = state["fi"]
    ptr = state["ptr"]
    if fi >= len(files):
        raise RuntimeError(
            f"Checkpoint shard index fi={fi} is outside current dataset with "
            f"{len(files)} shards"
        )
    shard_length = len(torch.load(files[fi], weights_only=True, mmap=True))
    if ptr > shard_length:
        raise RuntimeError(
            f"Checkpoint data pointer ptr={ptr} exceeds shard length={shard_length}"
        )

    # ---- 模型 ----
    model = TinyMixtralForCausalLM.from_pretrained(str(latest))
    if args.recompute is not None:
        model.set_router_recompute(args.recompute)
    if args.seq_len > model.config.max_position_embeddings:
        p.error(
            f"seq-len {args.seq_len} exceeds model limit "
            f"{model.config.max_position_embeddings}"
        )
    model.gradient_checkpointing_enable()
    model = model.to("cuda").to(torch.bfloat16)
    print(
        "Activation checkpointing: global=on, "
        f"router={model.config.cpt_router_recompute}",
        flush=True,
    )
    print(f"Training failure policy: {args.failure_policy}", flush=True)
    validate_training_state(
        state,
        checkpoint_step=step_done,
        model=model,
        expected_run_id=state["run_id"],
        expected_code_manifest=current_code_manifest,
        expected_data_manifest=current_data_manifest,
        expected_tokenizer_manifest=current_tokenizer_manifest,
        expected_world_size=distributed_world_size(),
    )
    nM = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {nM:.0f}M params", flush=True)

    # ---- 优化器 + state ----
    saved_groups = state["opt"]["param_groups"]
    saved_lr = float(saved_groups[0]["lr"])
    saved_weight_decay = max(float(group.get("weight_decay", 0.0)) for group in saved_groups)
    opt = make_adamw(
        model,
        lr=saved_lr,
        weight_decay=saved_weight_decay,
        bf16_states=checkpoint_uses_bf16_optimizer,
    )

    total_tok = state["total_tok"]
    try:
        opt.load_state_dict(state["opt"])
    except (ValueError, KeyError) as exc:
        raise RuntimeError(
            "Checkpoint optimizer state is incompatible with the CPT model. "
            "Legacy linear-router optimizer fallback is intentionally disabled."
        ) from exc
    print(f"Loaded strict CPT optimizer state ({optimizer_kind})", flush=True)

    # ---- 确定训练目标 ----
    output_dir = args.output_dir or args.checkpoint_dir
    if Path(output_dir).resolve() != ckpt_dir.resolve():
        output_path = Path(output_dir)
        if output_path.exists() and any(output_path.glob("step_*")):
            p.error(
                f"separate output directory already contains checkpoints: {output_path}"
            )
    warmup = state["warmup_steps"]
    total_steps = state["total_steps"]
    print(f"Target: warmup={warmup} total_steps={total_steps}", flush=True)
    if step_done > total_steps:
        p.error(f"checkpoint step {step_done} exceeds target {total_steps}")
    if step_done == total_steps and latest.name.endswith("_final"):
        print(f"Training already complete: {latest}", flush=True)
        return

    check_checkpoint_disk_space(model, output_dir, args.keep_last_checkpoints)

    saved_lrs = [pg["lr"] for pg in opt.param_groups]
    if schedule_kind == "wsd":
        sched = make_wsd_schedule(
            opt,
            warmup,
            total_steps,
            decay_ratio=schedule_decay_ratio,
        )
    else:
        sched = make_cosine_schedule(opt, warmup, total_steps)
    for pg, lr in zip(opt.param_groups, saved_lrs):
        pg["lr"] = lr
    try:
        sched.load_state_dict(state["sched"])
    except (ValueError, KeyError) as exc:
        raise RuntimeError(
            "Checkpoint scheduler state is incompatible with strict CPT resume"
        ) from exc

    print(
        f"Restored training recipe: optimizer={optimizer_kind} "
        f"schedule={schedule_kind}"
        + (
            f" decay_ratio={schedule_decay_ratio:g}"
            if schedule_decay_ratio is not None
            else ""
        ),
        flush=True,
    )

    restore_rank_rng_state(state)

    print(f"Resume: step={step_done} shard={fi}/{len(files)} ptr={ptr/1e6:.1f}M", flush=True)

    # ---- 训练 ----
    schedule_args = {
        "warmup_steps": warmup,
        "total_steps": total_steps,
        "schedule_kind": schedule_kind,
        "schedule_decay_ratio": schedule_decay_ratio,
    }
    checkpoint_metadata = {
        "run_id": state["run_id"],
        "code_manifest": state["code_manifest"],
        "data_manifest": state["data_manifest"],
        "tokenizer_manifest": state["tokenizer_manifest"],
    }
    step, total_tok, fi, ptr, elapsed = training_loop(
        model, opt, sched, files, fi=fi, ptr=ptr, total_tok=total_tok,
        bs=bs, seq=seq, chunk=chunk,
        output_dir=output_dir, max_steps=total_steps,
        save_every_min=args.save_every_min, log_every=args.log_every,
        step_start=step_done, schedule_args=schedule_args,
        eval_on_save=args.eval_on_save,
        keep_last_checkpoints=args.keep_last_checkpoints,
        checkpoint_metadata=checkpoint_metadata,
        tokenizer_dir=args.tokenizer_dir,
        failure_policy=args.failure_policy,
        optimizer_kind=optimizer_kind,
    )

    final_save(
        model, opt, sched, output_dir, step, total_tok, elapsed,
        fi, ptr, bs, seq, schedule_args, checkpoint_metadata,
        data_files=files,
        tokenizer_dir=args.tokenizer_dir,
        optimizer_kind=optimizer_kind,
    )


if __name__ == "__main__":
    main()
