# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""train.py 和 resume.py 共享的训练逻辑。"""

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

# ============================================================
# 共享工具
# ============================================================


class BF16AdamW(torch.optim.AdamW):
    """AdamW that stores optimizer states in bfloat16 to save VRAM.

    States (exp_avg, exp_avg_sq) are kept in bf16 between steps and
    cast to fp32 only during the update computation, saving ~50% of
    optimizer state memory at the cost of minor precision loss.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("foreach", False)
        kwargs.setdefault("fused", False)
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros(p.shape, dtype=torch.bfloat16, device=p.device)
                    state["exp_avg_sq"] = torch.zeros(p.shape, dtype=torch.bfloat16, device=p.device)

                state["step"] = int(state["step"]) + 1
                t = state["step"]

                exp_avg = state["exp_avg"].float()
                exp_avg_sq = state["exp_avg_sq"].float()

                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)

                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(exp_avg, denom, value=-step_size)

                state["exp_avg"] = exp_avg.to(torch.bfloat16)
                state["exp_avg_sq"] = exp_avg_sq.to(torch.bfloat16)

        return loss


def make_adamw(model, lr, weight_decay, betas=(0.9, 0.95), bf16_states=False):
    """构建 AdamW：矩阵权重衰减，RMSNorm 等 1D 参数不衰减。

    bf16_states=True 时使用 BF16AdamW，优化器状态存储为 bf16，
    节省约 50% 优化器显存。
    """
    decay = []
    no_decay = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    cls = BF16AdamW if bf16_states else torch.optim.AdamW
    return cls(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
    )


def save_training_state(
    path, opt, sched, step, total_tok, warmup_steps, total_steps, fi=None, ptr=None, batch_size=None, seq_len=None
):
    """保存 optimizer + scheduler + 数据位置状态到文件。"""
    state = {
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "step": step,
        "total_tok": total_tok,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
    }
    if fi is not None:
        state["fi"] = fi
        state["ptr"] = ptr
    if batch_size is not None:
        state["batch_size"] = batch_size
        state["seq_len"] = seq_len
    torch.save(state, path)


def save_checkpoint(
    model,
    opt,
    sched,
    output_dir,
    step,
    total_tok,
    warmup_steps,
    total_steps,
    fi,
    ptr,
    batch_size=None,
    seq_len=None,
    final=False,
):
    """完整写入临时目录后原子发布 checkpoint。"""
    suffix = "_final" if final else ""
    target = Path(output_dir) / f"step_{step:07d}{suffix}"
    temp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    try:
        torch.save(model.state_dict(), temp / "pytorch_model.bin")
        save_training_state(
            temp / "training_state.pt",
            opt,
            sched,
            step,
            total_tok,
            warmup_steps,
            total_steps,
            fi,
            ptr,
            batch_size,
            seq_len,
        )
        model.config.save_pretrained(str(temp))
        os.replace(temp, target)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return target


def prune_periodic_checkpoints(output_dir, keep_last):
    """仅保留最近的周期 checkpoint；final checkpoint 永不删除。"""
    checkpoints = sorted(path for path in Path(output_dir).glob("step_*") if path.is_dir() and not path.name.endswith("_final"))
    for path in checkpoints[:-keep_last]:
        shutil.rmtree(path)


def check_checkpoint_disk_space(model, output_dir, keep_last):
    """确认磁盘可容纳保留的周期 checkpoint、final 和一次原子临时写入。"""
    output_path = Path(output_dir)
    probe_path = output_path
    while not probe_path.exists():
        probe_path = probe_path.parent

    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters() if parameter.requires_grad
    )
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    checkpoint_bytes = int((3 * parameter_bytes + buffer_bytes) * 1.15)
    checkpoint_bytes = max(checkpoint_bytes, 64 * 1024**2)
    target_bytes = checkpoint_bytes * (keep_last + 2)
    existing_bytes = (
        sum(
            file.stat().st_size
            for checkpoint in output_path.glob("step_*")
            if checkpoint.is_dir()
            for file in checkpoint.rglob("*")
            if file.is_file()
        )
        if output_path.exists()
        else 0
    )
    additional_bytes = max(checkpoint_bytes, target_bytes - existing_bytes)
    free_bytes = shutil.disk_usage(probe_path).free
    if free_bytes < additional_bytes:
        raise RuntimeError(
            f"Insufficient checkpoint disk space: need about "
            f"{additional_bytes / 1024**3:.1f} GiB more, "
            f"only {free_bytes / 1024**3:.1f} GiB free at {probe_path}"
        )
    print(
        f"Checkpoint disk preflight: ~{checkpoint_bytes / 1024**3:.1f} GiB each, " f"{free_bytes / 1024**3:.1f} GiB free",
        flush=True,
    )


def make_cosine_schedule(opt, warmup_steps, total_steps):
    """创建 cosine LR schedule with linear warmup，返回 LambdaLR。"""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = min(max(warmup_steps, 0), max(total_steps - 1, 0))

    def lr_lambda(s):
        if warmup_steps > 0 and s < warmup_steps:
            return (s + 1) / warmup_steps
        if warmup_steps == 0:
            progress = s / max(1, total_steps - 1)
        else:
            progress = (s - warmup_steps + 1) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def make_wsd_schedule(opt, warmup_steps, total_steps, decay_ratio=0.1):
    """创建 WSD (Warmup-Stable-Decay) LR schedule。

    warmup:  linear 0 → peak  (warmup_steps)
    stable:  constant peak     (warmup_steps → decay_start)
    decay:   linear peak → 0  (decay_start → total_steps)

    Args:
        decay_ratio: fraction of steps in decay phase (default 0.1 = 10%)
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = min(max(warmup_steps, 0), max(total_steps - 2, 0))
    decay_steps = max(1, int(total_steps * decay_ratio))
    # ensure there's at least 1 stable step
    decay_start = max(warmup_steps + 1, total_steps - decay_steps)

    def lr_lambda(s):
        if s < warmup_steps:
            return (s + 1) / max(1, warmup_steps)
        if s < decay_start:
            return 1.0
        progress = (s - decay_start + 1) / max(1, total_steps - decay_start)
        progress = min(max(progress, 0.0), 1.0)
        return 1.0 - progress

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ============================================================
# CPU eval 子进程
# ============================================================


def run_cpu_eval(checkpoint_path, eval_dir):
    """子进程 CPU GLUE eval。"""
    project_root = Path(__file__).parent.parent.parent
    script = project_root / "scripts" / "eval_glue.py"
    output = Path(eval_dir) / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path = project_root / "tokenizer"
    # 删除旧结果，防止子进程失败时误读
    if output.exists():
        output.unlink()
    cmd = [
        sys.executable,
        str(script),
        "--checkpoint",
        checkpoint_path,
        "--tokenizer",
        str(tokenizer_path),
        "--tasks",
        "sst2,mrpc,qnli,rte,cola",
        "--limit",
        "200",
        "--batch-size",
        "2",
        "--max-length",
        "256",
        "--device",
        "cpu",
        "--precision",
        "fp32",
        "--output",
        str(output),
        "--seed",
        "1234",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            stderr_tail = r.stderr.strip()[-500:] if r.stderr else "(empty)"
            print(f"  [eval err] exit={r.returncode} stderr={stderr_tail}", flush=True)
            return None, None  # 失败后不读旧结果
        if output.exists():
            with open(output) as f:
                d = json.load(f)
            return d.get("aggregate", {}).get("mean_score"), d.get("results", {})
    except Exception as e:
        print(f"  [eval err] {e}", flush=True)
    return None, None


def training_loop(
    model,
    opt,
    sched,
    files,
    fi,
    ptr,
    total_tok,
    bs,
    seq,
    chunk,
    output_dir,
    max_steps,
    save_every_min,
    log_every,
    step_start=0,
    schedule_args=None,
    eval_on_save=False,
    keep_last_checkpoints=5,
):
    """按绝对 step 目标训练。

    schedule_args: 可选 dict with warmup_steps, total_steps，用于 checkpoint 恢复。
    """
    os.makedirs(output_dir, exist_ok=True)
    shard = torch.load(files[fi], weights_only=True)
    tok_base = total_tok
    sa = schedule_args or {}
    t0 = time.time()
    last_save = t0
    last_saved_step = None
    step = step_start
    stop_signal = None
    previous_handlers = {}

    def request_stop(signum, _frame):
        nonlocal stop_signal
        if stop_signal is not None:
            raise KeyboardInterrupt
        stop_signal = signum
        print(f"\n  Signal {signum} received; saving after current step...", flush=True)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)

    try:
        while step < max_steps:
            elapsed = time.time() - t0
            hours_elapsed = elapsed / 3600

            # ---- 分片循环 ----
            if ptr + chunk > len(shard):
                ptr = 0
                for _ in range(len(files)):
                    fi = (fi + 1) % len(files)
                    del shard
                    shard = torch.load(files[fi], weights_only=True)
                    if len(shard) >= chunk:
                        break
                else:
                    raise RuntimeError(f"No shard contains at least {chunk} tokens")

            batch = shard[ptr : ptr + chunk]
            if batch.numel() != chunk:
                print(f"  WARN: short read {batch.numel()}/{chunk} shard={fi}", flush=True)
                ptr = 0
                for _ in range(len(files)):
                    fi = (fi + 1) % len(files)
                    del shard
                    shard = torch.load(files[fi], weights_only=True)
                    if len(shard) >= chunk:
                        break
                else:
                    raise RuntimeError(f"No shard contains at least {chunk} tokens")
                continue

            batch = batch.view(bs, seq + 1).to("cuda", non_blocking=True)
            ptr += chunk
            total_tok += bs * seq

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch[:, :-1], labels=batch[:, 1:])
            if not torch.isfinite(out["loss"]):
                opt.zero_grad(set_to_none=True)
                raise FloatingPointError(f"Non-finite loss at step {step + 1}: {out['loss'].item()}")
            out["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                opt.zero_grad(set_to_none=True)
                raise FloatingPointError(f"Non-finite gradient norm at step {step + 1}: {grad_norm.item()}")
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % log_every == 0:
                elapsed = time.time() - t0
                hours_elapsed = elapsed / 3600
                print(
                    f"  step {step:7d}: loss={out['loss'].item():.4f} "
                    f"aux={out['aux_loss'].item():.1f} "
                    f"tok/s={(total_tok - tok_base) / elapsed:.0f} "
                    f"lr={sched.get_last_lr()[0]:.2e} "
                    f"shard={fi}/{len(files)} "
                    f"[{hours_elapsed:.1f}h]",
                    flush=True,
                )

            # ---- 保存 + eval ----
            if time.time() - last_save > save_every_min * 60:
                d = save_checkpoint(
                    model,
                    opt,
                    sched,
                    output_dir,
                    step,
                    total_tok,
                    sa.get("warmup_steps", 0),
                    sa.get("total_steps", 0),
                    fi,
                    ptr,
                    bs,
                    seq,
                )
                print(f"  -> Saved {d}", flush=True)
                prune_periodic_checkpoints(output_dir, keep_last_checkpoints)
                last_save = time.time()
                last_saved_step = step

                if eval_on_save:
                    eval_dir = f"evals/{Path(output_dir).name}/step_{step:07d}"
                    mean, res = run_cpu_eval(d, eval_dir)
                    if mean is not None and res:
                        parts = [
                            f'{t}={r.get("accuracy", r.get("matthews_correlation", r.get("f1", float("nan")))):.3f}'
                            for t, r in sorted(res.items())
                        ]
                        print(f"  [eval] mean={mean:.4f} | {' '.join(parts)}", flush=True)

            if stop_signal is not None:
                if last_saved_step == step:
                    print("  -> Current step was already checkpointed", flush=True)
                else:
                    d = save_checkpoint(
                        model,
                        opt,
                        sched,
                        output_dir,
                        step,
                        total_tok,
                        sa.get("warmup_steps", 0),
                        sa.get("total_steps", 0),
                        fi,
                        ptr,
                        bs,
                        seq,
                    )
                    print(f"  -> Emergency checkpoint saved: {d}", flush=True)
                    prune_periodic_checkpoints(output_dir, keep_last_checkpoints)
                raise SystemExit(128 + stop_signal)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    elapsed = time.time() - t0
    return step, total_tok, fi, ptr, elapsed


def final_save(model, opt, sched, output_dir, step, total_tok, elapsed, fi, ptr, batch_size, seq_len, schedule_args=None):
    """保存最终 checkpoint。"""
    sa = schedule_args or {}
    d = save_checkpoint(
        model,
        opt,
        sched,
        output_dir,
        step,
        total_tok,
        sa.get("warmup_steps", 0),
        sa.get("total_steps", 0),
        fi,
        ptr,
        batch_size,
        seq_len,
        final=True,
    )
    print(f"\nDone: {step} steps {total_tok / 1e9:.3f}B tokens " f"session={elapsed / 3600:.1f}h → {d}", flush=True)
