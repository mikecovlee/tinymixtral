#!/usr/bin/env python3
"""Reproducible CPU-only validation for the local CPT-MoE implementation.

The mathematical CPT convention is column-major (for example,
``q_t in R^{K x 1}`` and ``Pi = B^T Q``).  TinyMixtral's implementation is
token-major, so the code-level probability matrix is ``Pi.T = Q.T @ B``.

This script deliberately uses only tiny synthetic data.  It is an engineering
verification, not a claim about language-model quality or long-run training
convergence.  Every generated file is constrained to this repository.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from model.config import TinyMixtralConfig
from model.cpt_router import CPTLayerProposal, CPTRouter
from model.modeling import TinyMixtralForCausalLM
from scripts.train_utils import execute_training_iteration, make_adamw


GRADIENT_CATEGORIES = (
    "projection_P",
    "anchors_A",
    "energy_Theta_C",
    "attention",
    "experts",
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _as_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 and right.numel() == 0:
        return 0.0
    return _as_float((left.detach().float() - right.detach().float()).abs().max())


def _assert_close(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    label: str,
) -> float:
    error = _max_abs(left, right)
    torch.testing.assert_close(left, right, rtol=rtol, atol=atol, msg=label)
    return error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolve_output_root(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise ValueError(
            f"output directory must stay within {REPOSITORY_ROOT}, got {candidate}"
        ) from error
    return candidate


def _tiny_config(seed: int) -> TinyMixtralConfig:
    return TinyMixtralConfig(
        vocab_size=48,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        num_local_experts=4,
        num_experts_per_tok=2,
        expert_intermediate_size=24,
        cpt_num_prototypes=4,
        cpt_projection_dim=8,
        cpt_rho_beta=0.95,
        cpt_beta_max=0.45,
        cpt_expert_temperature=1.0,
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


def _fixed_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 7, 6, 5, 4, 3, 2, 1],
        ],
        dtype=torch.long,
        device="cpu",
    )
    labels = input_ids + 16
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    return input_ids, labels, attention_mask


def _gradient_category(parameter_name: str) -> str | None:
    if ".moe.cpt_router.projection" in parameter_name:
        return "projection_P"
    if ".moe.cpt_router.anchors" in parameter_name:
        return "anchors_A"
    if ".moe.cpt_router.energy" in parameter_name:
        return "energy_Theta_C"
    if ".self_attn." in parameter_name:
        return "attention"
    if any(
        marker in parameter_name
        for marker in (".moe.gate_proj", ".moe.up_proj", ".moe.down_proj")
    ):
        return "experts"
    return None


class _GradientAudit:
    """Capture gradients before strict training clears them."""

    def __init__(self, model: TinyMixtralForCausalLM) -> None:
        self._current: dict[str, dict[str, Any]] | None = None
        self._handles = []
        self._expected_tensors = {category: 0 for category in GRADIENT_CATEGORIES}
        for name, parameter in model.named_parameters():
            category = _gradient_category(name)
            if category is None or not parameter.requires_grad:
                continue
            self._expected_tensors[category] += 1
            self._handles.append(
                parameter.register_hook(self._make_hook(name, category))
            )
        missing = [
            category
            for category, count in self._expected_tensors.items()
            if count == 0
        ]
        if missing:
            raise RuntimeError(f"no parameters found for gradient categories: {missing}")

    def _make_hook(self, name: str, category: str):
        def hook(gradient: torch.Tensor) -> None:
            if self._current is None:
                return
            record = self._current[category]
            detached = gradient.detach().float()
            record["names"].append(name)
            record["tensors_seen"] += 1
            record["finite"] = record["finite"] and bool(
                torch.isfinite(detached).all().item()
            )
            record["nonzero"] = record["nonzero"] or bool(
                torch.count_nonzero(detached).item()
            )
            record["sum_sq"] += _as_float(torch.sum(detached.double().square()))
            if detached.numel():
                record["max_abs"] = max(
                    record["max_abs"],
                    _as_float(detached.abs().max()),
                )

        return hook

    def begin_step(self) -> None:
        if self._current is not None:
            raise RuntimeError("gradient audit step already active")
        self._current = {
            category: {
                "names": [],
                "tensors_seen": 0,
                "finite": True,
                "nonzero": False,
                "sum_sq": 0.0,
                "max_abs": 0.0,
            }
            for category in GRADIENT_CATEGORIES
        }

    def end_step(self) -> dict[str, dict[str, Any]]:
        if self._current is None:
            raise RuntimeError("gradient audit step is not active")
        result: dict[str, dict[str, Any]] = {}
        for category, record in self._current.items():
            if record["tensors_seen"] != self._expected_tensors[category]:
                raise RuntimeError(
                    f"gradient category {category} saw {record['tensors_seen']} tensors; "
                    f"expected {self._expected_tensors[category]}"
                )
            if not record["finite"]:
                raise FloatingPointError(
                    f"non-finite gradient detected in category {category}"
                )
            result[category] = {
                "tensors_seen": record["tensors_seen"],
                "finite": record["finite"],
                "nonzero": record["nonzero"],
                "norm": math.sqrt(record["sum_sq"]),
                "max_abs": record["max_abs"],
            }
        self._current = None
        return result

    @property
    def expected_tensors(self) -> dict[str, int]:
        return dict(self._expected_tensors)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _summarize_gradient_steps(
    steps: list[dict[str, dict[str, Any]]],
    expected_tensors: dict[str, int],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for category in GRADIENT_CATEGORIES:
        category_steps = [step[category] for step in steps]
        norms = [record["norm"] for record in category_steps]
        summary[category] = {
            "expected_tensors_per_step": expected_tensors[category],
            "steps_seen": len(category_steps),
            "all_finite": all(record["finite"] for record in category_steps),
            "nonzero_steps": sum(bool(record["nonzero"]) for record in category_steps),
            "norm_min": min(norms),
            "norm_max": max(norms),
            "norm_last": norms[-1],
            "max_abs_over_steps": max(
                record["max_abs"] for record in category_steps
            ),
        }
    return summary


def _run_overfit(
    config: TinyMixtralConfig,
    *,
    steps: int,
    learning_rate: float,
) -> tuple[
    TinyMixtralForCausalLM,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    dict[str, Any],
]:
    started = time.perf_counter()
    model = TinyMixtralForCausalLM(config).to("cpu").train()
    optimizer = make_adamw(model, lr=learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    input_ids, labels, attention_mask = _fixed_batch()
    gradient_audit = _GradientAudit(model)

    losses: list[float] = []
    grad_norms: list[float] = []
    versions: list[int] = []
    prices: list[list[list[float]]] = []
    gradient_steps: list[dict[str, dict[str, Any]]] = []
    ce_errors: list[float] = []

    try:
        for step_index in range(steps):
            gradient_audit.begin_step()
            output, grad_norm = execute_training_iteration(
                model,
                optimizer,
                scheduler,
                lambda: model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                ),
                max_grad_norm=5.0,
            )
            gradient_steps.append(gradient_audit.end_step())

            if "aux_loss" in output:
                raise RuntimeError("legacy expert load-balancing aux_loss is present")
            ce_loss = F.cross_entropy(
                output["logits"].reshape(-1, config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
            ce_error = _max_abs(output["loss"], ce_loss)
            if ce_error != 0.0:
                raise RuntimeError(
                    f"model loss is not exactly CE at step {step_index}: {ce_error}"
                )

            version = model.get_cpt_state_version()
            if version != step_index + 1:
                raise RuntimeError(
                    f"CPT version {version} does not match accepted step "
                    f"{step_index + 1}"
                )
            layer_prices: list[list[float]] = []
            for layer in model.layers:
                router = layer.moe.cpt_router
                price = router.congestion_price.detach().float()
                if not torch.isfinite(price).all() or (price < 0).any():
                    raise FloatingPointError("invalid persistent congestion price")
                if router.congestion_price.grad is not None:
                    raise RuntimeError("congestion price unexpectedly received a gradient")
                layer_prices.append(price.cpu().tolist())

            losses.append(_as_float(output["loss"]))
            grad_norms.append(_as_float(grad_norm))
            versions.append(version)
            prices.append(layer_prices)
            ce_errors.append(ce_error)
    finally:
        gradient_audit.close()

    gradient_summary = _summarize_gradient_steps(
        gradient_steps,
        gradient_audit.expected_tensors,
    )
    for category, record in gradient_summary.items():
        if not record["all_finite"]:
            raise FloatingPointError(f"non-finite {category} gradients")
        if record["nonzero_steps"] != steps:
            raise RuntimeError(
                f"{category} gradients were zero in "
                f"{steps - record['nonzero_steps']} accepted steps"
            )

    if not losses[-1] < losses[0]:
        raise RuntimeError(
            f"single-batch loss did not decrease: {losses[0]} -> {losses[-1]}"
        )

    result = {
        "status": "passed",
        "steps": steps,
        "learning_rate": learning_rate,
        "batch_size": int(input_ids.shape[0]),
        "sequence_length": int(input_ids.shape[1]),
        "losses": losses,
        "loss_initial": losses[0],
        "loss_final": losses[-1],
        "loss_min": min(losses),
        "loss_reduction": losses[0] - losses[-1],
        "loss_final_over_initial": losses[-1] / losses[0],
        "ce_loss_max_abs_error": max(ce_errors),
        "legacy_aux_loss_absent": True,
        "grad_norms": grad_norms,
        "grad_norm_min": min(grad_norms),
        "grad_norm_max": max(grad_norms),
        "gradient_categories": gradient_summary,
        "state_versions": versions,
        "state_version_final": versions[-1],
        "prices_by_step_layer_expert": prices,
        "prices_final_by_layer": prices[-1],
        "duration_seconds": time.perf_counter() - started,
    }
    return model, optimizer, scheduler, result


def _run_continuation_and_invalid(config: TinyMixtralConfig) -> dict[str, Any]:
    started = time.perf_counter()
    _seed_everything(31_337)
    router = CPTRouter(config, layer_index=0).to("cpu").eval()
    hidden_states = torch.randn(2, 7, config.hidden_size)

    with torch.no_grad():
        full = router(hidden_states)
        first = router(hidden_states[:, :3])
        second = router(
            hidden_states[:, 3:],
            sequence_state=first.sequence_state,
        )
    combined_q = torch.cat(
        (
            first.q_probabilities.view(2, 3, -1),
            second.q_probabilities.view(2, 4, -1),
        ),
        dim=1,
    ).reshape(14, -1)
    combined_pi = torch.cat(
        (
            first.probabilities.view(2, 3, -1),
            second.probabilities.view(2, 4, -1),
        ),
        dim=1,
    ).reshape(14, -1)
    continuation_errors = {
        "q_probabilities": _assert_close(
            full.q_probabilities,
            combined_q,
            label="full/chunk q probabilities",
        ),
        "pi_probabilities": _assert_close(
            full.probabilities,
            combined_pi,
            label="full/chunk Pi probabilities",
        ),
        "state_s": _assert_close(
            full.sequence_state.state_s,
            second.sequence_state.state_s,
            label="full/chunk state S",
        ),
        "state_nu": _assert_close(
            full.sequence_state.state_nu,
            second.sequence_state.state_nu,
            label="full/chunk state nu",
        ),
    }

    continuation_entry_s = first.sequence_state.state_s.clone()
    continuation_entry_nu = first.sequence_state.state_nu.clone()
    with torch.no_grad():
        replay = router(
            hidden_states[:, 3:],
            sequence_state=first.sequence_state,
        )
    replay_errors = {
        "probabilities": _assert_close(
            second.probabilities,
            replay.probabilities,
            label="continuation replay probabilities",
        ),
        "state_s": _assert_close(
            second.sequence_state.state_s,
            replay.sequence_state.state_s,
            label="continuation replay S",
        ),
        "entry_state_s_immutable": _assert_close(
            first.sequence_state.state_s,
            continuation_entry_s,
            label="immutable continuation entry S",
        ),
        "entry_state_nu_immutable": _assert_close(
            first.sequence_state.state_nu,
            continuation_entry_nu,
            label="immutable continuation entry nu",
        ),
    }

    all_padding_hidden = torch.full(
        (2, 3, config.hidden_size),
        float("nan"),
    )
    all_padding_mask = torch.zeros(2, 3, dtype=torch.bool)
    with torch.no_grad():
        all_padding = router(
            all_padding_hidden,
            route_valid_mask=all_padding_mask,
            sequence_state=first.sequence_state,
        )
    all_padding_errors = {
        "state_s": _assert_close(
            first.sequence_state.state_s,
            all_padding.sequence_state.state_s,
            label="all-padding S bypass",
        ),
        "state_nu": _assert_close(
            first.sequence_state.state_nu,
            all_padding.sequence_state.state_nu,
            label="all-padding nu bypass",
        ),
        "initialized": _assert_close(
            first.sequence_state.initialized,
            all_padding.sequence_state.initialized,
            label="all-padding initialized bypass",
        ),
        "load_sum": _max_abs(
            all_padding.proposal.load_sum,
            torch.zeros_like(all_padding.proposal.load_sum),
        ),
    }
    if all_padding.proposal.token_count.item() != 0:
        raise RuntimeError("all-padding input contributed CPT token count")
    if all_padding.probabilities.shape != (0, config.num_local_experts):
        raise RuntimeError("all-padding CPT probabilities have the wrong shape")

    clean_hidden = torch.randn(2, 4, config.hidden_size)
    valid_mask = torch.tensor(
        [[False, True, True, False], [True, False, True, True]],
        dtype=torch.bool,
    )
    nan_hidden = clean_hidden.clone().requires_grad_(True)
    with torch.no_grad():
        nan_hidden[~valid_mask] = float("nan")
    finite_reference = clean_hidden.clone()
    finite_reference[~valid_mask] = 0.0
    router.train()
    router.zero_grad(set_to_none=True)
    nan_output = router(nan_hidden, route_valid_mask=valid_mask)
    with torch.no_grad():
        reference_output = router(
            finite_reference,
            route_valid_mask=valid_mask,
        )
    invalid_errors = {
        "probabilities": _assert_close(
            nan_output.probabilities,
            reference_output.probabilities,
            label="NaN invalid bypass probabilities",
        ),
        "q_probabilities": _assert_close(
            nan_output.q_probabilities,
            reference_output.q_probabilities,
            label="NaN invalid bypass q",
        ),
        "state_s": _assert_close(
            nan_output.sequence_state.state_s,
            reference_output.sequence_state.state_s,
            label="NaN invalid bypass S",
        ),
        "state_nu": _assert_close(
            nan_output.sequence_state.state_nu,
            reference_output.sequence_state.state_nu,
            label="NaN invalid bypass nu",
        ),
    }
    coefficients = torch.linspace(
        -0.7,
        0.9,
        nan_output.probabilities.numel(),
    ).view_as(nan_output.probabilities)
    (nan_output.probabilities * coefficients).sum().backward()
    if nan_hidden.grad is None or not torch.isfinite(nan_hidden.grad).all():
        raise FloatingPointError("NaN-invalid test produced invalid hidden gradients")
    invalid_gradient_max_abs = _as_float(nan_hidden.grad[~valid_mask].abs().max())
    if invalid_gradient_max_abs != 0.0:
        raise RuntimeError("invalid NaN positions received nonzero gradients")
    router_gradient_finite = {}
    for name in ("projection", "anchors", "energy"):
        gradient = getattr(router, name).grad
        finite = gradient is not None and bool(torch.isfinite(gradient).all().item())
        router_gradient_finite[name] = finite
        if not finite:
            raise FloatingPointError(f"invalid router gradient for {name}")

    return {
        "status": "passed",
        "column_vector_equation": "Pi = B^T Q",
        "token_major_code_equation": "Pi.T = Q.T @ B",
        "full_vs_chunk_max_abs_errors": continuation_errors,
        "full_vs_chunk_exact": max(continuation_errors.values()) == 0.0,
        "replay_max_abs_errors": replay_errors,
        "replay_exact_and_entry_immutable": max(replay_errors.values()) == 0.0,
        "all_padding": {
            "max_abs_errors": all_padding_errors,
            "token_count": int(all_padding.proposal.token_count.item()),
            "probability_shape": list(all_padding.probabilities.shape),
            "state_exact": max(all_padding_errors.values()) == 0.0,
        },
        "nan_invalid_bypass": {
            "max_abs_errors": invalid_errors,
            "invalid_hidden_gradient_max_abs": invalid_gradient_max_abs,
            "router_gradient_finite": router_gradient_finite,
            "proposal_valid": bool(nan_output.proposal.valid.item()),
        },
        "duration_seconds": time.perf_counter() - started,
    }


def _run_price_direction(config: TinyMixtralConfig) -> dict[str, Any]:
    started = time.perf_counter()
    _seed_everything(41_337)
    model = TinyMixtralForCausalLM(config).to("cpu").train()
    full_pi = torch.tensor(
        [[0.70, 0.10, 0.10, 0.10]] * 8,
        dtype=torch.float32,
    )
    row_sum_error = _max_abs(
        full_pi.sum(dim=-1),
        torch.ones(full_pi.shape[0]),
    )
    if row_sum_error > 1e-7:
        raise RuntimeError("synthetic full-Pi rows are not normalized")
    load_sum = full_pi.sum(dim=0)
    token_count = full_pi.shape[0]

    proposals = []
    kernels_before = []
    for layer_index, layer in enumerate(model.layers):
        router = layer.moe.cpt_router
        kernels_before.append(router._expert_kernel().detach().clone())
        proposals.append(
            CPTLayerProposal(
                layer_index=layer_index,
                load_sum=load_sum.to(router.congestion_price.device),
                token_count=torch.tensor(token_count, dtype=torch.int64),
                state_version=router.state_version.detach().clone(),
                valid=torch.tensor(True, dtype=torch.bool),
            )
        )
    transaction = model.prepare_cpt_transaction(
        tuple(proposals),
        microbatch_id="synthetic-full-pi",
    )
    model.validate_cpt_transaction(transaction)
    prepared_prices = [
        value.detach().clone() for value in transaction._prepared_prices
    ]
    model.commit_cpt_transaction(transaction)
    kernels_after = [
        layer.moe.cpt_router._expert_kernel().detach().clone()
        for layer in model.layers
    ]

    layers = []
    for layer_index, (layer, before, after, prepared) in enumerate(
        zip(model.layers, kernels_before, kernels_after, prepared_prices)
    ):
        router = layer.moe.cpt_router
        overloaded_delta = after[:, 0] - before[:, 0]
        if not bool((prepared[0] > 0).item()):
            raise RuntimeError("overloaded expert did not receive a positive price")
        if not bool(torch.equal(prepared[1:], torch.zeros_like(prepared[1:]))):
            raise RuntimeError("under-capacity experts unexpectedly received a price")
        if not bool((overloaded_delta < 0).all().item()):
            raise RuntimeError(
                "positive congestion price did not reduce overloaded expert kernel mass"
            )
        layers.append(
            {
                "layer_index": layer_index,
                "prepared_price": prepared.cpu().tolist(),
                "committed_price": router.congestion_price.detach().cpu().tolist(),
                "expert0_kernel_before": before[:, 0].cpu().tolist(),
                "expert0_kernel_after": after[:, 0].cpu().tolist(),
                "expert0_kernel_delta_max": _as_float(overloaded_delta.max()),
                "expert0_kernel_delta_min": _as_float(overloaded_delta.min()),
                "state_version": int(router.state_version.item()),
            }
        )

    return {
        "status": "passed",
        "synthetic_full_pi": full_pi.tolist(),
        "row_sum_max_abs_error": row_sum_error,
        "load_sum": load_sum.tolist(),
        "token_count": token_count,
        "mean_probability": (load_sum / token_count).tolist(),
        "capacity_threshold": config.cpt_capacity_factor
        / config.num_local_experts,
        "layers": layers,
        "direction_check": (
            "expert 0 overload -> lambda_0 increases -> every B[:,0] decreases"
        ),
        "duration_seconds": time.perf_counter() - started,
    }


def _run_activation_checkpoint(config: TinyMixtralConfig) -> dict[str, Any]:
    started = time.perf_counter()
    _seed_everything(51_337)
    regular = TinyMixtralForCausalLM(config).to("cpu").train()
    checkpointed = TinyMixtralForCausalLM(config).to("cpu").train()
    checkpointed.load_state_dict(copy.deepcopy(regular.state_dict()), strict=True)
    checkpointed.gradient_checkpointing_enable()
    input_ids, labels, attention_mask = _fixed_batch()

    regular_output = regular(
        input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    checkpointed_output = checkpointed(
        input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    logits_error = _assert_close(
        regular_output["logits"],
        checkpointed_output["logits"],
        rtol=1e-6,
        atol=1e-7,
        label="activation-checkpoint logits",
    )
    loss_error = _assert_close(
        regular_output["loss"],
        checkpointed_output["loss"],
        rtol=1e-6,
        atol=1e-7,
        label="activation-checkpoint loss",
    )
    proposal_errors = []
    for regular_proposal, checkpointed_proposal in zip(
        regular_output["cpt_transaction"].proposals,
        checkpointed_output["cpt_transaction"].proposals,
    ):
        proposal_errors.append(
            _assert_close(
                regular_proposal.load_sum,
                checkpointed_proposal.load_sum,
                rtol=1e-6,
                atol=1e-7,
                label="activation-checkpoint proposal load",
            )
        )
        if regular_proposal.token_count.item() != checkpointed_proposal.token_count.item():
            raise RuntimeError("activation checkpoint changed proposal token count")

    regular_output["loss"].backward()
    checkpointed_output["loss"].backward()
    gradient_errors = {}
    checkpointed_parameters = dict(checkpointed.named_parameters())
    for name, parameter in regular.named_parameters():
        other = checkpointed_parameters[name]
        if parameter.grad is None or other.grad is None:
            raise RuntimeError(f"activation checkpoint missing gradient for {name}")
        gradient_errors[name] = _assert_close(
            parameter.grad,
            other.grad,
            rtol=1e-5,
            atol=1e-7,
            label=f"activation-checkpoint gradient {name}",
        )
    regular.abort_cpt_transaction(regular_output["cpt_transaction"])
    checkpointed.abort_cpt_transaction(checkpointed_output["cpt_transaction"])

    return {
        "status": "passed",
        "logits_max_abs_error": logits_error,
        "loss_max_abs_error": loss_error,
        "proposal_load_max_abs_error": max(proposal_errors),
        "gradient_max_abs_error": max(gradient_errors.values()),
        "gradient_max_abs_error_by_parameter": gradient_errors,
        "regular_state_version": regular.get_cpt_state_version(),
        "checkpointed_state_version": checkpointed.get_cpt_state_version(),
        "duration_seconds": time.perf_counter() - started,
    }


def _maximum_parameter_error(
    left: TinyMixtralForCausalLM,
    right: TinyMixtralForCausalLM,
) -> tuple[float, dict[str, float]]:
    right_parameters = dict(right.named_parameters())
    errors = {
        name: _max_abs(parameter, right_parameters[name])
        for name, parameter in left.named_parameters()
    }
    return max(errors.values(), default=0.0), errors


def _price_errors(
    left: TinyMixtralForCausalLM,
    right: TinyMixtralForCausalLM,
) -> list[float]:
    return [
        _max_abs(
            left_layer.moe.cpt_router.congestion_price,
            right_layer.moe.cpt_router.congestion_price,
        )
        for left_layer, right_layer in zip(left.layers, right.layers)
    ]


def _run_save_reload_and_resume(
    model: TinyMixtralForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    run_directory: Path,
    *,
    learning_rate: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_ids, labels, attention_mask = _fixed_batch()
    checkpoint_directory = run_directory / "native_checkpoint"
    model.eval()
    with torch.no_grad():
        reference_logits = model(
            input_ids,
            attention_mask=attention_mask,
        )["logits"].detach().clone()
    checkpoint_version = model.get_cpt_state_version()
    checkpoint_prices = [
        layer.moe.cpt_router.congestion_price.detach().clone()
        for layer in model.layers
    ]
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scheduler_state = copy.deepcopy(scheduler.state_dict())
    model.save_pretrained(str(checkpoint_directory))
    restored = TinyMixtralForCausalLM.from_pretrained(
        str(checkpoint_directory)
    ).to("cpu").eval()
    with torch.no_grad():
        restored_logits = restored(
            input_ids,
            attention_mask=attention_mask,
        )["logits"]
    reload_logits_error = _assert_close(
        reference_logits,
        restored_logits,
        label="native save/reload logits",
    )
    reload_price_errors = [
        _max_abs(price, layer.moe.cpt_router.congestion_price)
        for price, layer in zip(checkpoint_prices, restored.layers)
    ]
    if any(error != 0.0 for error in reload_price_errors):
        raise RuntimeError("native save/reload changed CPT prices")
    reload_state_version = restored.get_cpt_state_version()
    if reload_state_version != checkpoint_version:
        raise RuntimeError("native save/reload changed CPT state version")

    model.train()
    restored.train()
    restored_optimizer = make_adamw(
        restored,
        lr=learning_rate,
        weight_decay=0.0,
    )
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer,
        lambda _: 1.0,
    )
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    restored_scheduler.load_state_dict(copy.deepcopy(scheduler_state))

    rng_state = torch.get_rng_state().clone()
    torch.set_rng_state(rng_state.clone())
    original_output, original_grad_norm = execute_training_iteration(
        model,
        optimizer,
        scheduler,
        lambda: model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ),
        max_grad_norm=5.0,
    )
    torch.set_rng_state(rng_state.clone())
    restored_output, restored_grad_norm = execute_training_iteration(
        restored,
        restored_optimizer,
        restored_scheduler,
        lambda: restored(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ),
        max_grad_norm=5.0,
    )

    trajectory_logits_error = _assert_close(
        original_output["logits"],
        restored_output["logits"],
        label="save/reload continuation logits",
    )
    trajectory_loss_error = _assert_close(
        original_output["loss"],
        restored_output["loss"],
        label="save/reload continuation loss",
    )
    grad_norm_error = abs(_as_float(original_grad_norm) - _as_float(restored_grad_norm))
    if grad_norm_error != 0.0:
        raise RuntimeError("save/reload continuation changed gradient norm")
    parameter_max_error, parameter_errors = _maximum_parameter_error(model, restored)
    if parameter_max_error != 0.0:
        raise RuntimeError(
            f"save/reload continuation parameters diverged by {parameter_max_error}"
        )
    trajectory_price_errors = _price_errors(model, restored)
    if any(error != 0.0 for error in trajectory_price_errors):
        raise RuntimeError("save/reload continuation prices diverged")
    if model.get_cpt_state_version() != restored.get_cpt_state_version():
        raise RuntimeError("save/reload continuation versions diverged")

    return {
        "status": "passed",
        "checkpoint_directory": str(checkpoint_directory),
        "checkpoint_state_version": checkpoint_version,
        "reload_state_version": reload_state_version,
        "reload_logits_max_abs_error": reload_logits_error,
        "reload_price_max_abs_errors_by_layer": reload_price_errors,
        "continuation": {
            "performed": True,
            "logits_max_abs_error": trajectory_logits_error,
            "loss_max_abs_error": trajectory_loss_error,
            "grad_norm_abs_error": grad_norm_error,
            "parameter_max_abs_error": parameter_max_error,
            "parameter_max_abs_error_by_name": parameter_errors,
            "price_max_abs_errors_by_layer": trajectory_price_errors,
            "state_version": model.get_cpt_state_version(),
        },
        "duration_seconds": time.perf_counter() - started,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default="artifacts/local_verification",
        help="Repository-relative output root (absolute paths must remain in-repo).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional unique run directory name; existing directories are refused.",
    )
    args = parser.parse_args()
    if not 30 <= args.steps <= 60:
        parser.error("--steps must be between 30 and 60")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    return args


def main() -> int:
    args = _parse_args()
    output_root = _resolve_output_root(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        "cpu_tiny_%Y%m%dT%H%M%S_%fZ"
    )
    run_directory = (output_root / run_name).resolve()
    try:
        run_directory.relative_to(output_root)
    except ValueError as error:
        raise ValueError("run name escapes the selected output root") from error
    run_directory.mkdir(parents=False, exist_ok=False)
    report_path = run_directory / "report.json"

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    _seed_everything(args.seed)
    config = _tiny_config(args.seed)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_name": "tinymixtral-cpt-local-cpu-verification",
        "schema_version": 1,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(REPOSITORY_ROOT),
        "run_directory": str(run_directory),
        "report_path": str(report_path),
        "scope": "CPU-only tiny synthetic engineering verification",
        "limitations": [
            "No GPU experiment is performed by this script.",
            "No formal long training or language-model quality claim is made.",
            "Full-vs-chunk parity is measured at the CPT Router boundary because "
            "the model intentionally has no attention KV-cache continuation path.",
        ],
        "reproducibility": {
            "seed": args.seed,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "torch_deterministic_algorithms": True,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "logical_cpu_count_visible": os.cpu_count(),
            "config": config.to_dict(),
            "fixed_batch": {
                "input_ids": _fixed_batch()[0].tolist(),
                "labels": _fixed_batch()[1].tolist(),
                "attention_mask": _fixed_batch()[2].tolist(),
            },
            "source_sha256": {
                str(path.relative_to(REPOSITORY_ROOT)): _sha256(path)
                for path in (
                    REPOSITORY_ROOT / "model" / "config.py",
                    REPOSITORY_ROOT / "model" / "cpt_router.py",
                    REPOSITORY_ROOT / "model" / "modeling.py",
                    REPOSITORY_ROOT / "scripts" / "train_utils.py",
                    Path(__file__).resolve(),
                )
            },
        },
        "experiments": {},
    }
    _atomic_json_dump(report_path, report)

    try:
        model, optimizer, scheduler, overfit = _run_overfit(
            config,
            steps=args.steps,
            learning_rate=args.learning_rate,
        )
        report["experiments"]["strict_single_batch_overfit"] = overfit
        _atomic_json_dump(report_path, report)

        report["experiments"]["continuation_and_invalid_bypass"] = (
            _run_continuation_and_invalid(config)
        )
        _atomic_json_dump(report_path, report)

        report["experiments"]["manual_full_pi_price_direction"] = (
            _run_price_direction(config)
        )
        _atomic_json_dump(report_path, report)

        report["experiments"]["activation_checkpoint_parity"] = (
            _run_activation_checkpoint(config)
        )
        _atomic_json_dump(report_path, report)

        report["experiments"]["save_reload_and_continuation"] = (
            _run_save_reload_and_resume(
                model,
                optimizer,
                scheduler,
                run_directory,
                learning_rate=args.learning_rate,
            )
        )
        report["status"] = "passed"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["duration_seconds"] = time.perf_counter() - started
        _atomic_json_dump(report_path, report)
    except BaseException as error:
        report["status"] = "failed"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["duration_seconds"] = time.perf_counter() - started
        report["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        _atomic_json_dump(report_path, report)
        raise

    summary = {
        "status": report["status"],
        "report": str(report_path),
        "duration_seconds": report["duration_seconds"],
        "loss_initial": report["experiments"]["strict_single_batch_overfit"][
            "loss_initial"
        ],
        "loss_final": report["experiments"]["strict_single_batch_overfit"][
            "loss_final"
        ],
        "state_version": report["experiments"]["save_reload_and_continuation"][
            "continuation"
        ]["state_version"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
