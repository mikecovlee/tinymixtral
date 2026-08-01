#!/usr/bin/env python3
"""Bounded local performance and invariant benchmark for the CPT Router.

The mathematical convention is column-vector based:

    q_t in R^{K x 1}, Q in R^{K x T}, B in R^{K x N}, Pi = B^T Q.

The implementation stores tokens as rows and therefore computes
``Pi.T = Q.T @ B``.  This benchmark measures the sequential causal scan while
also checking the probability, gradient, and persistent-state invariants.  It
never commits a congestion-price transaction and constrains every output path
to this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from model.config import TinyMixtralConfig
from model.cpt_router import CPTRouter


CORE_SOURCES = (
    "model/config.py",
    "model/cpt_router.py",
    "scripts/deep_audit_router_benchmark.py",
)


class _GeneralRowPathRouter(CPTRouter):
    """Benchmark-only reference that disables the all-valid fast path."""

    @staticmethod
    def _all_routes_valid(route_valid_mask: torch.Tensor) -> bool:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _source_hashes() -> dict[str, str]:
    return {name: _sha256(REPOSITORY_ROOT / name) for name in CORE_SOURCES}


def _resolve_output(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise ValueError(
            f"output must remain inside {REPOSITORY_ROOT}, got {candidate}"
        ) from error
    return candidate


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _config(args: argparse.Namespace) -> TinyMixtralConfig:
    heads = max(1, args.hidden_size // args.head_dim)
    if heads * args.head_dim != args.hidden_size:
        raise ValueError("hidden-size must be divisible by head-dim")
    key_value_heads = 1
    if heads % key_value_heads != 0:
        raise ValueError("attention-head count must be divisible by key-value heads")
    return TinyMixtralConfig(
        vocab_size=128,
        hidden_size=args.hidden_size,
        num_hidden_layers=1,
        num_attention_heads=heads,
        num_key_value_heads=key_value_heads,
        head_dim=args.head_dim,
        max_position_embeddings=max(args.seq_lengths),
        num_local_experts=args.num_experts,
        num_experts_per_tok=min(2, args.num_experts),
        expert_intermediate_size=max(8, args.hidden_size * 2),
        cpt_num_prototypes=args.num_prototypes,
        cpt_projection_dim=args.projection_dim,
        cpt_rho_beta=0.95,
        cpt_beta_max=0.45,
        cpt_expert_temperature=1.0,
        cpt_state_radius=1.0,
        cpt_eps_z=1e-6,
        cpt_eps_m=1e-6,
        cpt_eps_init=1e-8,
        cpt_capacity_factor=1.25,
        cpt_init_seed=args.seed,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        initializer_range=0.02,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {}
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _assert_output(output, expected_tokens: int) -> dict[str, Any]:
    probabilities = output.probabilities.detach().float()
    q_probabilities = output.q_probabilities.detach().float()
    if probabilities.shape[0] != expected_tokens:
        raise RuntimeError(
            f"Router returned {probabilities.shape[0]} tokens, expected {expected_tokens}"
        )
    if not bool(output.proposal.valid.item()):
        raise RuntimeError("Router marked a benchmark proposal invalid")
    if int(output.proposal.token_count.item()) != expected_tokens:
        raise RuntimeError("Router proposal token_count is inconsistent")
    if not torch.isfinite(probabilities).all() or not torch.isfinite(q_probabilities).all():
        raise RuntimeError("Router returned non-finite probabilities")
    probability_error = (
        float((probabilities.sum(dim=-1) - 1.0).abs().max().item())
        if probabilities.numel()
        else 0.0
    )
    q_error = (
        float((q_probabilities.sum(dim=-1) - 1.0).abs().max().item())
        if q_probabilities.numel()
        else 0.0
    )
    if probability_error > 1e-5 or q_error > 1e-5:
        raise RuntimeError(
            f"Router simplex error is too large: q={q_error}, Pi={probability_error}"
        )
    return {
        "q_simplex_max_abs_error": q_error,
        "pi_simplex_max_abs_error": probability_error,
        "load_mass_error": abs(
            float(output.proposal.load_sum.detach().float().sum().item())
            - expected_tokens
        ),
    }


def _run_once(
    router: CPTRouter,
    hidden: torch.Tensor,
    route_valid_mask: torch.Tensor | None,
    reset_mask: torch.Tensor | None,
    *,
    backward: bool,
    autocast_dtype: torch.dtype | None,
) -> tuple[Any, float]:
    router.zero_grad(set_to_none=True)
    if hidden.grad is not None:
        hidden.grad = None
    context = (
        torch.autocast(hidden.device.type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else torch.autocast(hidden.device.type, enabled=False)
    )
    with context:
        output = router(
            hidden,
            route_valid_mask=route_valid_mask,
            reset_mask=reset_mask,
        )
        objective = (
            output.probabilities.float().square().mean()
            + 0.01 * output.q_probabilities.float().square().mean()
        )
    if backward:
        objective.backward()
    return output, float(objective.detach().item())


def _benchmark_length(
    router: CPTRouter,
    args: argparse.Namespace,
    device: torch.device,
    seq_len: int,
) -> dict[str, Any]:
    backward = seq_len <= args.backward_max_seq
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + seq_len)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    hidden = torch.randn(
        args.batch_size,
        seq_len,
        args.hidden_size,
        generator=generator,
        device=device,
        dtype=dtype,
        requires_grad=backward,
    )
    if args.control_mode == "trusted":
        # This is the canonical production pretraining call: no public
        # padding/packing/reset controls were supplied, so the model forwards
        # ``None`` and the Router may use its trusted dense schedule without a
        # CUDA-to-host mask validation barrier.
        valid = None
        reset = None
    else:
        valid = torch.ones(
            args.batch_size,
            seq_len,
            device=device,
            dtype=torch.bool,
        )
        reset = torch.zeros_like(valid)
        if seq_len:
            reset[:, 0] = True
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None

    for _ in range(args.warmups):
        _run_once(
            router,
            hidden,
            valid,
            reset,
            backward=backward,
            autocast_dtype=autocast_dtype,
        )
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings_ms: list[float] = []
    final_output = None
    objective = None
    for _ in range(args.repeats):
        started = time.perf_counter()
        final_output, objective = _run_once(
            router,
            hidden,
            valid,
            reset,
            backward=backward,
            autocast_dtype=autocast_dtype,
        )
        _synchronize(device)
        timings_ms.append((time.perf_counter() - started) * 1000.0)

    assert final_output is not None and objective is not None
    invariants = _assert_output(
        final_output,
        expected_tokens=args.batch_size * seq_len,
    )
    gradient_summary: dict[str, Any] = {}
    if backward:
        for name in ("projection", "anchors", "energy"):
            gradient = getattr(router, name).grad
            if gradient is None:
                raise RuntimeError(f"Router parameter {name} did not receive a gradient")
            gradient_float = gradient.detach().float()
            if not torch.isfinite(gradient_float).all():
                raise RuntimeError(f"Router parameter {name} has a non-finite gradient")
            gradient_summary[name] = {
                "dtype": str(gradient.dtype),
                "norm": float(gradient_float.norm().item()),
                "max_abs": float(gradient_float.abs().max().item()),
                "nonzero": bool(torch.count_nonzero(gradient_float).item()),
            }

    median_ms = statistics.median(timings_ms)
    tokens = args.batch_size * seq_len
    return {
        "seq_len": seq_len,
        "batch_size": args.batch_size,
        "tokens": tokens,
        "backward": backward,
        "objective": objective,
        "timing_ms": {
            "samples": timings_ms,
            "min": min(timings_ms),
            "mean": statistics.fmean(timings_ms),
            "median": median_ms,
            "p95": _percentile(timings_ms, 0.95),
        },
        "tokens_per_second_median": tokens / (median_ms / 1000.0),
        "invariants": invariants,
        "gradients": gradient_summary,
        "memory": _memory(device),
    }


def _parse_seq_lengths(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("seq-lengths must be comma-separated integers") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("seq-lengths must contain positive integers")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("seq-lengths must not contain duplicates")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seq-lengths", type=_parse_seq_lengths, default=[128, 256, 512, 1024])
    parser.add_argument("--backward-max-seq", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=896)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--num-prototypes", type=int, default=12)
    parser.add_argument("--num-experts", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--control-mode",
        choices=("trusted", "explicit"),
        default="trusted",
        help=(
            "trusted passes None/None like ordinary unpadded pretraining; "
            "explicit passes all-valid and first-position-reset tensors"
        ),
    )
    parser.add_argument("--force-general-path", action="store_true")
    parser.add_argument(
        "--output",
        default="artifacts/deep_audit_20260731/router_benchmark.json",
    )
    args = parser.parse_args()
    if args.backward_max_seq < 0:
        parser.error("--backward-max-seq must be non-negative")
    for name in (
        "batch_size",
        "hidden_size",
        "head_dim",
        "projection_dim",
        "num_prototypes",
        "num_experts",
        "warmups",
        "repeats",
        "threads",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.head_dim % 2:
        parser.error("--head-dim must be even")
    if args.force_general_path and args.control_mode != "explicit":
        parser.error("--force-general-path requires --control-mode explicit")
    return args


def main() -> int:
    args = parse_args()
    output_path = _resolve_output(args.output)
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 support is required for this benchmark")
    device = (
        torch.device("cuda", 0)
        if args.device == "cuda"
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    hashes_before = _source_hashes()
    config = _config(args)
    router_class = _GeneralRowPathRouter if args.force_general_path else CPTRouter
    router = router_class(config, layer_index=0).to(device)
    if device.type == "cuda":
        router = router.to(torch.bfloat16)
    router.train()

    report: dict[str, Any] = {
        "schema": "cpt_router_deep_benchmark_v1",
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "running",
        "repository_root": str(REPOSITORY_ROOT),
        "output": str(output_path),
        "source_hashes_before": hashes_before,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "bf16_supported": (
                torch.cuda.is_bf16_supported() if device.type == "cuda" else None
            ),
            "threads": torch.get_num_threads(),
        },
        "arguments": vars(args) | {"seq_lengths": list(args.seq_lengths)},
        "config": config.to_dict(),
        "results": [],
    }
    _atomic_json(output_path, report)
    started = time.perf_counter()
    try:
        for seq_len in args.seq_lengths:
            result = _benchmark_length(router, args, device, seq_len)
            report["results"].append(result)
            _atomic_json(output_path, report)
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        raise
    finally:
        _synchronize(device)
        report["duration_seconds"] = time.perf_counter() - started
        report["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        report["source_hashes_after"] = _source_hashes()
        report["source_snapshot_stable"] = (
            report["source_hashes_before"] == report["source_hashes_after"]
        )
        _atomic_json(output_path, report)

    print(json.dumps({
        "status": report["status"],
        "output": str(output_path),
        "duration_seconds": report["duration_seconds"],
        "source_snapshot_stable": report["source_snapshot_stable"],
        "results": [
            {
                "seq_len": result["seq_len"],
                "backward": result["backward"],
                "median_ms": result["timing_ms"]["median"],
                "tokens_per_second": result["tokens_per_second_median"],
                "peak_allocated_bytes": result["memory"].get("peak_allocated_bytes"),
            }
            for result in report["results"]
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
