#!/usr/bin/env python3
"""Offline numerical audit for strict-TeX CPT Router v1 preprocessing.

Column-vector convention:

    y_t = P x_t,
    z_t = y_t / max(||y_t||_2, eps_z).

Production tensors are token-major rows, so the equivalent normalization axis
is the final projection dimension.  No projection-coordinate softmax is
allowed between ``P x_t`` and ``z_t``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
import torch.nn.functional as F

from model.config import TinyMixtralConfig
from model.cpt_router import (
    CPT_ROUTER_ALGORITHM_VERSION,
    CPTRouter,
)


CORE_SOURCES = (
    "model/config.py",
    "model/cpt_router.py",
    "scripts/audit_projection_l2_v1.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _source_hashes() -> dict[str, str]:
    return {name: _sha256(REPOSITORY_ROOT / name) for name in CORE_SOURCES}


def _resolve_output(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(WORKSPACE_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"output must stay inside {WORKSPACE_ROOT}, got {candidate}"
        ) from error
    if candidate.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {candidate}")
    return candidate


def _publish_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _config() -> TinyMixtralConfig:
    return TinyMixtralConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        num_local_experts=4,
        num_experts_per_tok=2,
        expert_intermediate_size=24,
        cpt_router_version=1,
        cpt_num_prototypes=4,
        cpt_projection_dim=8,
        cpt_init_seed=20260801,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        initializer_range=0.02,
    )


def _normalize_rows(values: torch.Tensor, eps: float) -> torch.Tensor:
    return values / values.norm(dim=-1, keepdim=True).clamp_min(eps)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().float().flatten()
    right = right.detach().float().flatten()
    denominator = left.norm() * right.norm()
    if denominator == 0:
        return 1.0 if torch.equal(left, right) else 0.0
    return float(torch.dot(left, right).div(denominator).item())


def _column_row_equivalence(router: CPTRouter, seed: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    hidden_rows = torch.randn(3, 17, router.hidden_size, generator=generator)
    projected_rows = F.linear(hidden_rows, router.projection).float()
    z_rows = _normalize_rows(projected_rows, router.eps_z)

    hidden_columns = hidden_rows.reshape(-1, router.hidden_size).T
    projected_columns = router.projection.float() @ hidden_columns
    z_columns = projected_columns / projected_columns.norm(
        dim=0,
        keepdim=True,
    ).clamp_min(router.eps_z)
    delta = z_rows.reshape(-1, router.projection_dim).T - z_columns
    max_abs_delta = float(delta.abs().max().item())
    if max_abs_delta > 2e-6:
        raise RuntimeError("token-major normalization disagrees with column math")
    return {
        "status": "passed",
        "column_formula": "z_t=(P x_t)/max(||P x_t||_2,eps_z)",
        "token_major_formula": "z_rows=projected_rows/norm(dim=-1)",
        "max_abs_delta": max_abs_delta,
    }


def _scale_and_direction_contract(
    router: CPTRouter,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    projected = torch.randn(128, router.projection_dim, generator=generator)
    projected = projected + 0.2 * torch.sign(projected)
    reference = _normalize_rows(projected, router.eps_z)
    scale_residuals = {}
    for scale in (0.125, 0.5, 2.0, 16.0):
        candidate = _normalize_rows(projected * scale, router.eps_z)
        scale_residuals[str(scale)] = float(
            (candidate - reference).abs().max().item()
        )
    sign_residual = float(
        (_normalize_rows(-projected, router.eps_z) + reference)
        .abs()
        .max()
        .item()
    )
    projection_softmax = torch.softmax(projected, dim=-1, dtype=torch.float32)
    wrong_v2_geometry = _normalize_rows(projection_softmax, router.eps_z)
    separation = float((wrong_v2_geometry - reference).abs().max().item())
    if max(scale_residuals.values()) > 2e-6 or sign_residual > 2e-6:
        raise RuntimeError("strict-TeX scale/direction invariance failed")
    if separation <= 1e-3:
        raise RuntimeError("audit did not distinguish v1 from projection softmax")
    return {
        "status": "passed",
        "positive_scale_max_abs_residuals": scale_residuals,
        "sign_reversal_max_abs_residual": sign_residual,
        "projection_softmax_separation_max_abs": separation,
    }


def _epsilon_and_jacobian_contract(router: CPTRouter) -> dict[str, Any]:
    eps = float(router.eps_z)
    ratios = (0.0, 0.25, 0.999, 1.001, 4.0, 100.0)
    norm_records = []
    for ratio in ratios:
        value = torch.zeros(router.projection_dim, dtype=torch.float64)
        value[0] = ratio * eps
        normalized = value / value.norm().clamp_min(eps)
        expected_norm = min(ratio, 1.0)
        observed_norm = float(normalized.norm().item())
        if abs(observed_norm - expected_norm) > 2e-12:
            raise RuntimeError("epsilon branch norm contract failed")
        norm_records.append(
            {
                "input_norm_over_eps": ratio,
                "output_norm": observed_norm,
                "expected_output_norm": expected_norm,
            }
        )

    def normalize(vector: torch.Tensor) -> torch.Tensor:
        return vector / vector.norm().clamp_min(eps)

    small = torch.zeros(router.projection_dim, dtype=torch.float64)
    small[0] = 0.25 * eps
    small_jacobian = torch.autograd.functional.jacobian(normalize, small)
    expected_small = torch.eye(router.projection_dim, dtype=torch.float64) / eps
    small_error = float((small_jacobian - expected_small).abs().max().item())

    generator = torch.Generator().manual_seed(20260801)
    large = torch.randn(
        router.projection_dim,
        generator=generator,
        dtype=torch.float64,
    )
    large = large / large.norm() * (4.0 * eps)
    large_jacobian = torch.autograd.functional.jacobian(normalize, large)
    unit = large / large.norm()
    expected_large = (
        torch.eye(router.projection_dim, dtype=torch.float64)
        - unit[:, None] @ unit[None, :]
    ) / large.norm()
    large_error = float((large_jacobian - expected_large).abs().max().item())
    if small_error > 1e-6 or large_error > 1e-6:
        raise RuntimeError("strict-TeX normalization Jacobian contract failed")
    return {
        "status": "passed",
        "eps_z": eps,
        "piecewise_norms": norm_records,
        "small_branch_jacobian_max_abs_error": small_error,
        "small_branch_operator_norm": 1.0 / eps,
        "large_branch_jacobian_max_abs_error": large_error,
        "risk": "for ||P x||<eps_z, the local Jacobian is I/eps_z",
    }


def _production_contract(
    router: CPTRouter,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(2, 23, router.hidden_size, generator=generator)
    with torch.no_grad():
        reference = router(hidden)
        scaled = router(hidden * 3.0)
        zero = router(torch.zeros(1, 1, router.hidden_size))
    q_scale_delta = float(
        (reference.q_probabilities - scaled.q_probabilities).abs().max().item()
    )
    pi_scale_delta = float(
        (reference.probabilities - scaled.probabilities).abs().max().item()
    )
    uniform = torch.full_like(
        zero.q_probabilities,
        1.0 / router.num_prototypes,
    )
    zero_uniform_delta = float(
        (zero.q_probabilities - uniform).abs().max().item()
    )
    if max(q_scale_delta, pi_scale_delta, zero_uniform_delta) > 3e-6:
        raise RuntimeError("production Router violates strict-TeX behavior")
    if int(router.router_algorithm_version.item()) != 1:
        raise RuntimeError("production Router persistent algorithm identity is not v1")
    return {
        "status": "passed",
        "q_positive_scale_max_abs_delta": q_scale_delta,
        "pi_positive_scale_max_abs_delta": pi_scale_delta,
        "zero_projection_uniform_q_max_abs_delta": zero_uniform_delta,
        "router_algorithm_version": int(
            router.router_algorithm_version.item()
        ),
    }


def _cuda_bf16_parity(config: TinyMixtralConfig, seed: int) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        return {"status": "skipped", "reason": "CUDA BF16 is unavailable"}

    torch.manual_seed(seed)
    cpu_router = CPTRouter(config, layer_index=0).train()
    gpu_router = CPTRouter(config, layer_index=0).cuda().train()
    gpu_router.load_state_dict(cpu_router.state_dict(), strict=True)
    generator = torch.Generator().manual_seed(seed + 1)
    hidden_cpu = torch.randn(
        4,
        64,
        config.hidden_size,
        generator=generator,
        requires_grad=True,
    )
    hidden_gpu = hidden_cpu.detach().cuda().requires_grad_(True)
    route_valid_cpu = torch.rand(4, 64, generator=generator) > 0.18
    route_valid_cpu[:, 0] = True
    reset_cpu = torch.zeros_like(route_valid_cpu)
    reset_cpu[:, 0] = True
    reset_cpu[1, 23] = route_valid_cpu[1, 23]
    coefficient_q = torch.randn(
        int(route_valid_cpu.sum().item()),
        config.cpt_num_prototypes,
        generator=generator,
    )
    coefficient_pi = torch.randn(
        int(route_valid_cpu.sum().item()),
        config.num_local_experts,
        generator=generator,
    )

    output_cpu = cpu_router(
        hidden_cpu,
        route_valid_mask=route_valid_cpu,
        reset_mask=reset_cpu,
    )
    loss_cpu = (
        (output_cpu.q_probabilities * coefficient_q).mean()
        + (output_cpu.probabilities * coefficient_pi).mean()
    )
    loss_cpu.backward()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_gpu = gpu_router(
            hidden_gpu,
            route_valid_mask=route_valid_cpu.cuda(),
            reset_mask=reset_cpu.cuda(),
        )
        loss_gpu = (
            (output_gpu.q_probabilities * coefficient_q.cuda()).mean()
            + (output_gpu.probabilities * coefficient_pi.cuda()).mean()
        )
    loss_gpu.backward()

    projection_cosine = _cosine(
        cpu_router.projection.grad,
        gpu_router.projection.grad.cpu(),
    )
    hidden_cosine = _cosine(hidden_cpu.grad, hidden_gpu.grad.cpu())
    tensors = (
        output_cpu.q_probabilities,
        output_cpu.probabilities,
        output_gpu.q_probabilities,
        output_gpu.probabilities,
        hidden_cpu.grad,
        hidden_gpu.grad,
        cpu_router.projection.grad,
        gpu_router.projection.grad,
    )
    if not all(torch.isfinite(value).all() for value in tensors):
        raise RuntimeError("CPU/CUDA BF16 audit produced non-finite values")
    if projection_cosine < 0.95 or hidden_cosine < 0.95:
        raise RuntimeError("CPU/CUDA BF16 gradient directions diverged")
    return {
        "status": "passed",
        "q_max_abs_delta": float(
            (
                output_gpu.q_probabilities.detach().cpu()
                - output_cpu.q_probabilities.detach()
            )
            .abs()
            .max()
            .item()
        ),
        "pi_max_abs_delta": float(
            (
                output_gpu.probabilities.detach().cpu()
                - output_cpu.probabilities.detach()
            )
            .abs()
            .max()
            .item()
        ),
        "projection_gradient_cosine": projection_cosine,
        "hidden_gradient_cosine": hidden_cosine,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="artifacts/cpt_v1/projection_l2_numeric_audit.json",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    return args


def main() -> int:
    args = parse_args()
    output_path = _resolve_output(args.output)
    torch.set_num_threads(args.threads)
    config = _config()
    if config.cpt_router_version != CPT_ROUTER_ALGORITHM_VERSION:
        raise RuntimeError("config and strict-TeX implementation version diverged")
    router = CPTRouter(config, layer_index=0).eval()
    hashes_before = _source_hashes()
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema": "cpt_projection_l2_v1_numeric_audit_v1",
        "status": "running",
        "arguments": vars(args),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_bf16_supported": (
                torch.cuda.is_bf16_supported()
                if torch.cuda.is_available()
                else False
            ),
        },
        "router_version": config.cpt_router_version,
        "source_hashes_before": hashes_before,
    }
    try:
        report["column_row_equivalence"] = _column_row_equivalence(
            router,
            args.seed,
        )
        report["scale_and_direction"] = _scale_and_direction_contract(
            router,
            args.seed + 1,
        )
        report["epsilon_and_jacobian"] = _epsilon_and_jacobian_contract(router)
        report["production_contract"] = _production_contract(
            router,
            args.seed + 2,
        )
        report["cuda_bf16_parity"] = _cuda_bf16_parity(
            config,
            args.seed + 3,
        )
        report["status"] = "passed"
    finally:
        report["duration_seconds"] = time.perf_counter() - started
        report["source_hashes_after"] = _source_hashes()
        report["source_snapshot_stable"] = (
            report["source_hashes_before"] == report["source_hashes_after"]
        )
        _publish_json(output_path, report)

    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output_path),
                "duration_seconds": report["duration_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
