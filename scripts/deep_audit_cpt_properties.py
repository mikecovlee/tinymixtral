#!/usr/bin/env python3
"""Randomized CPU stress audit for CPT Router math and lifecycle contracts.

This is an offline engineering experiment.  It exercises many legal Router
shapes, masks, reset schedules, continuation boundaries, CPU BF16 autocast,
and gradient paths.  All artifacts are constrained to the repository.  The
mathematical convention is column-vector based, with ``Pi = B^T Q``; the code
stores the equivalent token-major matrix ``Pi.T = Q.T @ B``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
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
    "scripts/deep_audit_cpt_properties.py",
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
        candidate.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise ValueError(
            f"output must stay inside {REPOSITORY_ROOT}, got {candidate}"
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


def _config(
    *,
    hidden_size: int,
    num_experts: int,
    num_prototypes: int,
    projection_dim: int,
    rho_beta: float,
    beta_max: float,
    expert_temperature: float,
    seed: int,
    max_positions: int,
) -> TinyMixtralConfig:
    head_dim = 4
    if hidden_size % head_dim:
        raise ValueError("hidden_size must be divisible by four")
    return TinyMixtralConfig(
        vocab_size=128,
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=hidden_size // head_dim,
        num_key_value_heads=1,
        head_dim=head_dim,
        max_position_embeddings=max_positions,
        num_local_experts=num_experts,
        num_experts_per_tok=min(2, num_experts),
        expert_intermediate_size=hidden_size * 2,
        cpt_num_prototypes=num_prototypes,
        cpt_projection_dim=projection_dim,
        cpt_rho_beta=rho_beta,
        cpt_beta_max=beta_max,
        cpt_expert_temperature=expert_temperature,
        cpt_state_radius=1.0,
        cpt_eps_z=1e-6,
        cpt_eps_m=1e-6,
        cpt_eps_init=1e-8,
        cpt_capacity_factor=1.25,
        cpt_init_seed=seed,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        initializer_range=0.02,
    )


def _make_masks(
    generator: torch.Generator,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    route_valid = torch.rand(batch_size, seq_len, generator=generator) > 0.22
    if seq_len:
        # Alternate dense, left-padded, holey, and all-padding rows.
        route_valid[0] = True
        if batch_size > 1:
            left_pad = min(seq_len, int(torch.randint(0, max(seq_len, 1), (1,), generator=generator)))
            route_valid[1, :left_pad] = False
            if left_pad < seq_len:
                route_valid[1, left_pad:] = True
        if batch_size > 2 and seq_len > 1:
            route_valid[2, ::3] = False
        if batch_size > 3:
            route_valid[3] = False

    first_valid = route_valid & (
        route_valid.to(torch.int64).cumsum(dim=1) == 1
    )
    random_reset = (
        torch.rand(batch_size, seq_len, generator=generator) < 0.08
    ) & route_valid
    reset = first_valid | random_reset
    return route_valid, reset


def _reshape_valid_rows(
    values: torch.Tensor,
    route_valid: torch.Tensor,
) -> list[torch.Tensor]:
    rows: list[torch.Tensor] = []
    offset = 0
    for batch_row in route_valid:
        count = int(batch_row.sum().item())
        rows.append(values[offset : offset + count])
        offset += count
    if offset != values.shape[0]:
        raise RuntimeError("valid-row reconstruction consumed the wrong token count")
    return rows


def _check_output(router: CPTRouter, output, expected_tokens: int) -> dict[str, float]:
    if int(output.proposal.token_count.item()) != expected_tokens:
        raise RuntimeError("proposal token_count does not match the mask")
    if not bool(output.proposal.valid.item()):
        raise RuntimeError("Router marked a legal randomized case invalid")
    q = output.q_probabilities.detach().float()
    probabilities = output.probabilities.detach().float()
    kernel = output.expert_kernel.detach().float()
    tensors = {
        "q": q,
        "probabilities": probabilities,
        "kernel": kernel,
        "state_s": output.sequence_state.state_s.detach().float(),
        "state_nu": output.sequence_state.state_nu.detach().float(),
    }
    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{name} contains a non-finite value")
    if (q < 0).any() or (probabilities < 0).any() or (kernel < 0).any():
        raise RuntimeError("a CPT probability tensor contains a negative value")
    q_error = float((q.sum(dim=-1) - 1).abs().max().item()) if q.numel() else 0.0
    pi_error = (
        float((probabilities.sum(dim=-1) - 1).abs().max().item())
        if probabilities.numel()
        else 0.0
    )
    kernel_error = float((kernel.sum(dim=-1) - 1).abs().max().item())
    if max(q_error, pi_error, kernel_error) > 1e-5:
        raise RuntimeError(
            f"simplex invariant failed: q={q_error}, Pi={pi_error}, B={kernel_error}"
        )
    theory = q @ kernel
    torch.testing.assert_close(probabilities, theory, rtol=1e-6, atol=1e-6)
    state_norm = output.sequence_state.state_s.detach().float().norm(dim=1)
    if (state_norm > router.state_radius + 1e-5).any():
        raise RuntimeError("state_s left its radius constraint")
    nu = output.sequence_state.state_nu.detach().float()
    if (nu < 0).any() or (
        nu.sum(dim=-1, dtype=torch.float64)
        > router.state_nu_upper_bound + router.state_nu_mass_tolerance
    ).any():
        raise RuntimeError("state_nu violated its nonnegative mass bound")
    load_error = abs(
        float(output.proposal.load_sum.detach().float().sum().item())
        - expected_tokens
    )
    if load_error > max(1e-4, 1e-4 * max(expected_tokens, 1)):
        raise RuntimeError("detached soft-load mass does not equal token_count")
    return {
        "q_simplex_error": q_error,
        "pi_simplex_error": pi_error,
        "kernel_simplex_error": kernel_error,
        "load_mass_error": load_error,
        "max_state_norm": float(state_norm.max().item()) if state_norm.numel() else 0.0,
        "max_nu_mass": (
            float(nu.sum(dim=-1).max().item()) if nu.numel() else 0.0
        ),
    }


def _forward(
    router: CPTRouter,
    hidden: torch.Tensor,
    route_valid: torch.Tensor,
    reset: torch.Tensor,
    sequence_state=None,
    *,
    bf16_autocast: bool = False,
):
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16_autocast):
        return router(
            hidden,
            route_valid_mask=route_valid,
            reset_mask=reset,
            sequence_state=sequence_state,
        )


def _compare_full_and_chunked(
    router: CPTRouter,
    hidden: torch.Tensor,
    route_valid: torch.Tensor,
    reset: torch.Tensor,
    split: int,
    *,
    bf16_autocast: bool,
) -> dict[str, float]:
    full = _forward(
        router,
        hidden,
        route_valid,
        reset,
        bf16_autocast=bf16_autocast,
    )
    first = _forward(
        router,
        hidden[:, :split],
        route_valid[:, :split],
        reset[:, :split],
        bf16_autocast=bf16_autocast,
    )
    second = _forward(
        router,
        hidden[:, split:],
        route_valid[:, split:],
        reset[:, split:],
        sequence_state=first.sequence_state,
        bf16_autocast=bf16_autocast,
    )

    full_q_rows = _reshape_valid_rows(full.q_probabilities, route_valid)
    first_q_rows = _reshape_valid_rows(first.q_probabilities, route_valid[:, :split])
    second_q_rows = _reshape_valid_rows(second.q_probabilities, route_valid[:, split:])
    full_pi_rows = _reshape_valid_rows(full.probabilities, route_valid)
    first_pi_rows = _reshape_valid_rows(first.probabilities, route_valid[:, :split])
    second_pi_rows = _reshape_valid_rows(second.probabilities, route_valid[:, split:])

    q_error = 0.0
    pi_error = 0.0
    for full_q, first_q, second_q, full_pi, first_pi, second_pi in zip(
        full_q_rows,
        first_q_rows,
        second_q_rows,
        full_pi_rows,
        first_pi_rows,
        second_pi_rows,
    ):
        chunk_q = torch.cat((first_q, second_q), dim=0)
        chunk_pi = torch.cat((first_pi, second_pi), dim=0)
        torch.testing.assert_close(full_q, chunk_q, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(full_pi, chunk_pi, rtol=2e-5, atol=2e-6)
        if full_q.numel():
            q_error = max(q_error, float((full_q - chunk_q).abs().max().item()))
        if full_pi.numel():
            pi_error = max(pi_error, float((full_pi - chunk_pi).abs().max().item()))
    torch.testing.assert_close(
        full.sequence_state.state_s,
        second.sequence_state.state_s,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        full.sequence_state.state_nu,
        second.sequence_state.state_nu,
        rtol=2e-5,
        atol=2e-6,
    )
    assert torch.equal(
        full.sequence_state.initialized,
        second.sequence_state.initialized,
    )
    return {"q_max_abs_error": q_error, "pi_max_abs_error": pi_error}


def _gradient_chunk_parity(config: TinyMixtralConfig, seed: int) -> dict[str, float]:
    torch.manual_seed(seed)
    full_router = CPTRouter(config, layer_index=0).train()
    chunk_router = CPTRouter(config, layer_index=0).train()
    chunk_router.load_state_dict(copy.deepcopy(full_router.state_dict()), strict=True)
    batch_size = 2
    seq_len = min(config.max_position_embeddings, 17)
    split = max(1, seq_len // 2)
    hidden_full = torch.randn(
        batch_size,
        seq_len,
        config.hidden_size,
        requires_grad=True,
    )
    hidden_chunk = hidden_full.detach().clone().requires_grad_(True)
    valid = torch.ones(batch_size, seq_len, dtype=torch.bool)
    reset = torch.zeros_like(valid)
    reset[:, 0] = True
    if seq_len > 5:
        reset[1, 5] = True

    full = full_router(hidden_full, valid, reset)
    first = chunk_router(hidden_chunk[:, :split], valid[:, :split], reset[:, :split])
    second = chunk_router(
        hidden_chunk[:, split:],
        valid[:, split:],
        reset[:, split:],
        sequence_state=first.sequence_state,
    )
    full_q = full.q_probabilities.reshape(
        batch_size,
        seq_len,
        config.cpt_num_prototypes,
    )
    full_pi = full.probabilities.reshape(
        batch_size,
        seq_len,
        config.num_local_experts,
    )
    chunk_q = torch.cat(
        (
            first.q_probabilities.reshape(
                batch_size,
                split,
                config.cpt_num_prototypes,
            ),
            second.q_probabilities.reshape(
                batch_size,
                seq_len - split,
                config.cpt_num_prototypes,
            ),
        ),
        dim=1,
    )
    chunk_pi = torch.cat(
        (
            first.probabilities.reshape(
                batch_size,
                split,
                config.num_local_experts,
            ),
            second.probabilities.reshape(
                batch_size,
                seq_len - split,
                config.num_local_experts,
            ),
        ),
        dim=1,
    )
    torch.testing.assert_close(full_q, chunk_q, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(full_pi, chunk_pi, rtol=2e-5, atol=2e-6)
    full_loss = full_q.square().mean() + full_pi.square().mean()
    chunk_loss = chunk_q.square().mean() + chunk_pi.square().mean()
    full_loss.backward()
    chunk_loss.backward()

    errors: dict[str, float] = {}
    torch.testing.assert_close(
        hidden_full.grad,
        hidden_chunk.grad,
        rtol=1e-4,
        atol=1e-8,
    )
    errors["hidden"] = float((hidden_full.grad - hidden_chunk.grad).abs().max().item())
    for (full_name, full_parameter), (chunk_name, chunk_parameter) in zip(
        full_router.named_parameters(),
        chunk_router.named_parameters(),
    ):
        assert full_name == chunk_name
        if full_parameter.grad is None or chunk_parameter.grad is None:
            raise RuntimeError(f"missing gradient for {full_name}")
        if not torch.isfinite(full_parameter.grad).all() or not torch.isfinite(
            chunk_parameter.grad
        ).all():
            raise RuntimeError(f"non-finite gradient for {full_name}")
        torch.testing.assert_close(
            full_parameter.grad,
            chunk_parameter.grad,
            rtol=1e-4,
            atol=1e-8,
        )
        errors[full_name] = float(
            (full_parameter.grad - chunk_parameter.grad).abs().max().item()
        )
    return errors


def _causal_and_permutation_checks(config: TinyMixtralConfig, seed: int) -> dict[str, float]:
    torch.manual_seed(seed)
    router = CPTRouter(config, layer_index=0).eval()
    batch_size = 3
    seq_len = min(config.max_position_embeddings, 13)
    prefix = max(1, seq_len // 2)
    hidden = torch.randn(batch_size, seq_len, config.hidden_size)
    changed_future = hidden.clone()
    changed_future[:, prefix:] = torch.randn_like(changed_future[:, prefix:]) * 7
    valid = torch.ones(batch_size, seq_len, dtype=torch.bool)
    reset = torch.zeros_like(valid)
    reset[:, 0] = True
    with torch.no_grad():
        original = router(hidden, valid, reset)
        future_changed = router(changed_future, valid, reset)
    original_q = original.q_probabilities.view(batch_size, seq_len, -1)
    changed_q = future_changed.q_probabilities.view(batch_size, seq_len, -1)
    original_pi = original.probabilities.view(batch_size, seq_len, -1)
    changed_pi = future_changed.probabilities.view(batch_size, seq_len, -1)
    torch.testing.assert_close(
        original_q[:, :prefix], changed_q[:, :prefix], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        original_pi[:, :prefix], changed_pi[:, :prefix], rtol=0.0, atol=0.0
    )

    permutation = torch.tensor([2, 0, 1])
    with torch.no_grad():
        permuted = router(hidden.index_select(0, permutation), valid, reset)
    permuted_q = permuted.q_probabilities.view(batch_size, seq_len, -1)
    permuted_pi = permuted.probabilities.view(batch_size, seq_len, -1)
    torch.testing.assert_close(
        permuted_q,
        original_q.index_select(0, permutation),
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        permuted_pi,
        original_pi.index_select(0, permutation),
        rtol=1e-6,
        atol=1e-7,
    )
    causal_q_error = float(
        (original_q[:, :prefix] - changed_q[:, :prefix]).abs().max().item()
    )
    causal_pi_error = float(
        (original_pi[:, :prefix] - changed_pi[:, :prefix]).abs().max().item()
    )
    permutation_q_error = float(
        (
            permuted_q - original_q.index_select(0, permutation)
        ).abs().max().item()
    )
    permutation_pi_error = float(
        (
            permuted_pi - original_pi.index_select(0, permutation)
        ).abs().max().item()
    )
    return {
        "causal_q_error": causal_q_error,
        "causal_pi_error": causal_pi_error,
        "permutation_q_error": permutation_q_error,
        "permutation_pi_error": permutation_pi_error,
    }


def _invalid_nan_bypass(config: TinyMixtralConfig, seed: int) -> dict[str, float]:
    torch.manual_seed(seed)
    router = CPTRouter(config, layer_index=0).train()
    hidden = torch.randn(2, 12, config.hidden_size)
    valid = torch.tensor(
        [
            [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        ],
        dtype=torch.bool,
    )
    reset = valid & (valid.to(torch.int64).cumsum(dim=1) == 1)
    hidden_nan = hidden.clone()
    hidden_nan[~valid] = float("nan")
    hidden_nan.requires_grad_(True)
    clean = hidden.clone()
    clean[~valid] = 0
    clean.requires_grad_(True)
    output_nan = router(hidden_nan, valid, reset)
    output_clean = router(clean, valid, reset)
    torch.testing.assert_close(
        output_nan.q_probabilities,
        output_clean.q_probabilities,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output_nan.probabilities,
        output_clean.probabilities,
        rtol=0.0,
        atol=0.0,
    )
    loss = output_nan.probabilities.square().mean() + output_nan.q_probabilities.square().mean()
    loss.backward()
    if not torch.isfinite(hidden_nan.grad).all():
        raise RuntimeError("invalid NaN padding produced non-finite hidden gradients")
    invalid_gradient = hidden_nan.grad[~valid]
    if torch.count_nonzero(invalid_gradient).item() != 0:
        raise RuntimeError("invalid positions received Router gradients")
    return {
        "invalid_gradient_max_abs": (
            float(invalid_gradient.abs().max().item())
            if invalid_gradient.numel()
            else 0.0
        )
    }


def _strict_tex_projection_contract(seed: int) -> dict[str, float]:
    """Check x -> P x -> per-column L2 normalization from the TeX source."""
    config = _config(
        hidden_size=8,
        num_experts=3,
        num_prototypes=4,
        projection_dim=4,
        rho_beta=0.95,
        beta_max=0.45,
        expert_temperature=1.0,
        seed=seed,
        max_positions=1,
    )
    router = CPTRouter(config, layer_index=0).eval()
    controlled_anchors = torch.zeros_like(router.anchors)
    controlled_anchors[0, 0] = 1.0
    controlled_anchors[0, 1] = -1.0
    controlled_anchors[1, 2] = 1.0
    controlled_anchors[1, 3] = -1.0
    with torch.no_grad():
        router.anchors.copy_(controlled_anchors)

    desired_projected_columns = torch.tensor(
        [
            [0.0, 3.0, -2.0],
            [0.0, -2.0, 1.0],
            [0.0, 1.0, 0.5],
            [0.0, -0.5, 2.0],
        ],
        dtype=torch.float32,
    )
    hidden_columns = router.projection.float().T @ desired_projected_columns
    hidden = hidden_columns.T.unsqueeze(1)
    with torch.no_grad():
        actual = router(hidden)

    # Column-vector reference: X:[d,T], P X:[d_p,T], normalize each column.
    projected_columns = router.projection.float() @ hidden_columns
    z_columns = projected_columns / projected_columns.norm(
        dim=0,
        keepdim=True,
    ).clamp_min(router.eps_z)
    prototype_columns = controlled_anchors / controlled_anchors.norm(
        dim=0,
        keepdim=True,
    ).clamp_min(router.eps_m)
    expected_q_columns = torch.softmax(
        prototype_columns.T @ z_columns / router.projection_temperature,
        dim=0,
        dtype=torch.float32,
    )
    formula_error = float(
        (actual.q_probabilities.T - expected_q_columns).abs().max().item()
    )
    if formula_error > 2e-6:
        raise RuntimeError(
            "strict TeX projection column formula mismatch: "
            f"max error {formula_error}"
        )

    zero_z_error = float(z_columns[:, 0].abs().max().item())
    zero_q_uniform_error = float(
        (
            expected_q_columns[:, 0]
            - torch.full_like(
                expected_q_columns[:, 0],
                1.0 / config.cpt_num_prototypes,
            )
        )
        .abs()
        .max()
        .item()
    )
    z_norm_error = float(
        (z_columns[:, 1:].norm(dim=0) - 1.0).abs().max().item()
    )
    if zero_z_error != 0.0 or zero_q_uniform_error > 2e-6 or z_norm_error > 2e-6:
        raise RuntimeError("strict TeX projection normalization invariant failed")

    derived_softmax_probabilities = torch.softmax(
        projected_columns,
        dim=0,
        dtype=torch.float32,
    )
    derived_softmax_z = derived_softmax_probabilities / derived_softmax_probabilities.norm(
        dim=0,
        keepdim=True,
    ).clamp_min(router.eps_z)
    derived_softmax_q = torch.softmax(
        prototype_columns.T @ derived_softmax_z / router.projection_temperature,
        dim=0,
        dtype=torch.float32,
    )
    derived_softmax_difference = float(
        (expected_q_columns - derived_softmax_q).abs().max().item()
    )
    if derived_softmax_difference <= 1e-5:
        raise RuntimeError("strict TeX audit did not separate from projection softmax")

    wrong_global_z = projected_columns / projected_columns.norm().clamp_min(router.eps_z)
    wrong_global_q = torch.softmax(
        prototype_columns.T @ wrong_global_z / router.projection_temperature,
        dim=0,
        dtype=torch.float32,
    )
    wrong_global_difference = float(
        (expected_q_columns - wrong_global_q).abs().max().item()
    )
    if wrong_global_difference <= 1e-5:
        raise RuntimeError("per-column normalization audit is not discriminating")

    return {
        "column_formula_max_abs_error": formula_error,
        "zero_projection_z_max_abs_error": zero_z_error,
        "zero_projection_q_uniform_max_abs_error": zero_q_uniform_error,
        "z_unit_norm_max_abs_error": z_norm_error,
        "derived_softmax_max_abs_difference": derived_softmax_difference,
        "wrong_global_normalization_max_abs_difference": wrong_global_difference,
    }


def _long_recurrence(seed: int, seq_len: int) -> dict[str, float]:
    config = _config(
        hidden_size=8,
        num_experts=3,
        num_prototypes=4,
        projection_dim=4,
        rho_beta=0.999,
        beta_max=0.45,
        expert_temperature=1.0,
        seed=seed,
        max_positions=seq_len,
    )
    router = CPTRouter(config, layer_index=0).eval()
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(1, seq_len, config.hidden_size, generator=generator)
    valid = torch.ones(1, seq_len, dtype=torch.bool)
    reset = torch.zeros_like(valid)
    reset[:, 0] = True
    with torch.no_grad():
        output = router(hidden, valid, reset)
    metrics = _check_output(router, output, seq_len)
    theoretical = router.state_nu_upper_bound
    if metrics["max_nu_mass"] > theoretical + router.state_nu_mass_tolerance:
        raise RuntimeError("long recurrence exceeded the theoretical nu bound")
    metrics["theoretical_nu_bound"] = theoretical
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--long-seq", type=int, default=4096)
    parser.add_argument(
        "--output",
        default="artifacts/deep_audit_20260731/cpt_randomized_properties.json",
    )
    args = parser.parse_args()
    if args.cases < 20 or args.cases > 500:
        parser.error("--cases must be between 20 and 500")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.long_seq < 1024 or args.long_seq > 16384:
        parser.error("--long-seq must be between 1024 and 16384")
    return args


def main() -> int:
    args = parse_args()
    output_path = _resolve_output(args.output)
    torch.set_num_threads(args.threads)
    random_source = random.Random(args.seed)
    source_hashes_before = _source_hashes()
    report: dict[str, Any] = {
        "schema": "cpt_randomized_properties_v1",
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "repository_root": str(REPOSITORY_ROOT),
        "output": str(output_path),
        "arguments": vars(args),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
        },
        "source_hashes_before": source_hashes_before,
        "cases": [],
        "special_experiments": {},
    }
    _atomic_json(output_path, report)
    started = time.perf_counter()
    max_errors = {
        "q_simplex_error": 0.0,
        "pi_simplex_error": 0.0,
        "kernel_simplex_error": 0.0,
        "load_mass_error": 0.0,
        "chunk_q_error": 0.0,
        "chunk_pi_error": 0.0,
    }
    try:
        for case_index in range(args.cases):
            case_seed = args.seed + case_index * 7919
            hidden_size = random_source.choice((8, 12, 16, 24, 32))
            num_experts = random_source.choice((2, 3, 4, 6))
            num_prototypes = random_source.choice((2, 3, 4, 8, 12, 17))
            projection_dim = random_source.choice((2, 4, 8, 16, 24, 40))
            rho_beta = random_source.choice((0.0, 0.5, 0.95, 0.999))
            beta_max = random_source.choice((0.0, 0.2, 0.45))
            expert_temperature = random_source.choice((0.1, 1.0, 10.0))
            batch_size = random_source.choice((1, 2, 3, 4))
            seq_len = random_source.choice((1, 2, 3, 7, 16, 31, 64, 129))
            config = _config(
                hidden_size=hidden_size,
                num_experts=num_experts,
                num_prototypes=num_prototypes,
                projection_dim=projection_dim,
                rho_beta=rho_beta,
                beta_max=beta_max,
                expert_temperature=expert_temperature,
                seed=case_seed,
                max_positions=seq_len,
            )
            generator = torch.Generator().manual_seed(case_seed)
            router = CPTRouter(config, layer_index=0).train()
            hidden = torch.randn(
                batch_size,
                seq_len,
                hidden_size,
                generator=generator,
            )
            route_valid, reset = _make_masks(generator, batch_size, seq_len)
            bf16_autocast = case_index % 7 == 0
            with torch.no_grad():
                output = _forward(
                    router,
                    hidden,
                    route_valid,
                    reset,
                    bf16_autocast=bf16_autocast,
                )
                invariants = _check_output(
                    router,
                    output,
                    int(route_valid.sum().item()),
                )
                split = random_source.randint(1, seq_len) if seq_len > 1 else 1
                chunk_errors = _compare_full_and_chunked(
                    router,
                    hidden,
                    route_valid,
                    reset,
                    split,
                    bf16_autocast=bf16_autocast,
                )
            for key in (
                "q_simplex_error",
                "pi_simplex_error",
                "kernel_simplex_error",
                "load_mass_error",
            ):
                max_errors[key] = max(max_errors[key], invariants[key])
            max_errors["chunk_q_error"] = max(
                max_errors["chunk_q_error"], chunk_errors["q_max_abs_error"]
            )
            max_errors["chunk_pi_error"] = max(
                max_errors["chunk_pi_error"], chunk_errors["pi_max_abs_error"]
            )
            case_record: dict[str, Any] = {
                "case": case_index,
                "seed": case_seed,
                "shape": {
                    "batch": batch_size,
                    "seq": seq_len,
                    "hidden": hidden_size,
                    "projection": projection_dim,
                    "prototypes": num_prototypes,
                    "experts": num_experts,
                },
                "rho_beta": rho_beta,
                "beta_max": beta_max,
                "expert_temperature": expert_temperature,
                "valid_tokens": int(route_valid.sum().item()),
                "split": split,
                "bf16_autocast": bf16_autocast,
                "invariants": invariants,
                "chunk_errors": chunk_errors,
            }
            if case_index % 10 == 0:
                case_record["gradient_chunk_parity"] = _gradient_chunk_parity(
                    config,
                    case_seed + 1,
                )
            if case_index % 13 == 0:
                case_record["causal_permutation"] = _causal_and_permutation_checks(
                    config,
                    case_seed + 2,
                )
            report["cases"].append(case_record)
            if (case_index + 1) % 10 == 0:
                _atomic_json(output_path, report)

        stable_config = _config(
            hidden_size=16,
            num_experts=4,
            num_prototypes=4,
            projection_dim=8,
            rho_beta=0.95,
            beta_max=0.45,
            expert_temperature=1.0,
            seed=args.seed + 999,
            max_positions=32,
        )
        report["special_experiments"]["strict_tex_projection_contract"] = (
            _strict_tex_projection_contract(args.seed + 998)
        )
        report["special_experiments"]["invalid_nan_bypass"] = _invalid_nan_bypass(
            stable_config,
            args.seed + 1000,
        )
        report["special_experiments"]["long_recurrence"] = _long_recurrence(
            args.seed + 1001,
            args.long_seq,
        )
        report["max_errors"] = max_errors
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc().splitlines()[-20:],
        }
        raise
    finally:
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
        "cases": len(report["cases"]),
        "duration_seconds": report["duration_seconds"],
        "max_errors": report["max_errors"],
        "source_snapshot_stable": report["source_snapshot_stable"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
