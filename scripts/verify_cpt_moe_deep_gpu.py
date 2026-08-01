#!/usr/bin/env python3
"""Offline CUDA stress audit for CPT Router numerics and recompute policies."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "deep_audit_20260731"
PROCESS_TEMP = ARTIFACT_ROOT / ".tmp_deep_gpu"
PROCESS_TEMP.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(PROCESS_TEMP)
os.environ["TMP"] = str(PROCESS_TEMP)
os.environ.setdefault("HF_HOME", str(PROCESS_TEMP / "hf_home"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from model.config import TinyMixtralConfig
from model.cpt_router import CPTRouter
from model.modeling import TinyMixtralForCausalLM
from scripts.train_utils import execute_training_iteration


CORE_FILES = (
    "model/config.py",
    "model/cpt_router.py",
    "model/modeling.py",
    "scripts/train_utils.py",
    "scripts/verify_cpt_moe_deep_gpu.py",
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish one immutable evidence file without overwriting another run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite deep GPU evidence: {path}")
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
            )
            handle.flush()
            os.fsync(handle.fileno())
        # Same-directory hard-link publication is atomic and cannot replace an
        # existing report, including when two verification runs finish together.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def source_hashes() -> dict[str, str]:
    return {relative: sha256_file(REPO_ROOT / relative) for relative in CORE_FILES}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def memory_stats() -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "peak_allocated": int(torch.cuda.max_memory_allocated()),
        "peak_reserved": int(torch.cuda.max_memory_reserved()),
        "free": int(free_bytes),
        "total": int(total_bytes),
    }


def finite(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def exception_record(error: BaseException) -> dict[str, Any]:
    return {
        "type": type(error).__name__,
        "message": str(error),
        "is_cuda_oom": isinstance(error, torch.cuda.OutOfMemoryError),
        "traceback_tail": traceback.format_exc().strip().splitlines()[-12:],
    }


def router_config() -> TinyMixtralConfig:
    return TinyMixtralConfig(
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        num_local_experts=6,
        num_experts_per_tok=2,
        expert_intermediate_size=256,
        cpt_num_prototypes=12,
        cpt_projection_dim=64,
        cpt_init_seed=20260731,
        attention_dropout=0.20,
        tie_word_embeddings=True,
        initializer_range=0.02,
    )


def direct_router_numerics() -> dict[str, Any]:
    clean_cuda()
    seed_everything(73101)
    config = router_config()
    router = CPTRouter(config, layer_index=0).to(
        device="cuda",
        dtype=torch.bfloat16,
    ).train()
    hidden = torch.randn(
        3,
        96,
        config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    valid = torch.ones(3, 96, device="cuda", dtype=torch.bool)
    valid[0, :7] = False
    valid[1, 40:45] = False
    reset = torch.zeros_like(valid)
    reset[0, 7] = True
    reset[1, 0] = True
    reset[1, 45] = True
    reset[2, 0] = True

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = router(
            hidden,
            route_valid_mask=valid,
            reset_mask=reset,
        )
        coefficients = torch.linspace(
            -0.75,
            0.95,
            output.probabilities.numel(),
            device="cuda",
            dtype=torch.float32,
        ).reshape_as(output.probabilities)
        objective = (output.probabilities * coefficients).sum()
    objective.backward()
    torch.cuda.synchronize()

    q_error = (output.q_probabilities.sum(dim=-1) - 1).abs().amax()
    b_error = (output.expert_kernel.sum(dim=-1) - 1).abs().amax()
    pi_error = (output.probabilities.sum(dim=-1) - 1).abs().amax()
    gradients = {}
    for name in ("projection", "anchors", "energy"):
        gradient = getattr(router, name).grad
        gradients[name] = {
            "present": gradient is not None,
            "dtype": str(gradient.dtype) if gradient is not None else None,
            "finite": (
                bool(torch.isfinite(gradient).all().item())
                if gradient is not None
                else False
            ),
            "norm": (
                finite(gradient.detach().float().norm().item())
                if gradient is not None
                else None
            ),
        }
    result = {
        "status": "pass",
        "q_dtype": str(output.q_probabilities.dtype),
        "expert_kernel_dtype": str(output.expert_kernel.dtype),
        "pi_dtype": str(output.probabilities.dtype),
        "q_simplex_max_error": finite(q_error.item()),
        "expert_kernel_simplex_max_error": finite(b_error.item()),
        "pi_simplex_max_error": finite(pi_error.item()),
        "proposal_valid": bool(output.proposal.valid.item()),
        "valid_token_count": int(output.proposal.token_count.item()),
        "expected_valid_token_count": int(valid.sum().item()),
        "gradients": gradients,
        "lambda_grad_is_none": router.congestion_price.grad is None,
        "duration_seconds": time.perf_counter() - started,
        "memory": memory_stats(),
    }
    result["status"] = "pass" if all(
        (
            result["q_dtype"] == "torch.float32",
            result["expert_kernel_dtype"] == "torch.float32",
            result["pi_dtype"] == "torch.float32",
            result["proposal_valid"],
            result["valid_token_count"] == result["expected_valid_token_count"],
            result["q_simplex_max_error"] is not None,
            result["q_simplex_max_error"] <= 1e-5,
            result["pi_simplex_max_error"] is not None,
            result["pi_simplex_max_error"] <= 1e-5,
            all(item["present"] and item["finite"] for item in gradients.values()),
            result["lambda_grad_is_none"],
        )
    ) else "fail"
    del output, objective, coefficients, hidden, router
    clean_cuda()
    return result


def clone_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: (
            parameter.grad.detach().float().cpu().clone()
            if parameter.grad is not None
            else None
        )
        for name, parameter in model.named_parameters()
    }


def compare_gradients(
    reference: dict[str, torch.Tensor | None],
    candidate: dict[str, torch.Tensor | None],
) -> dict[str, Any]:
    missing_pattern = []
    max_abs = 0.0
    max_rel = 0.0
    worst_abs = None
    worst_rel = None
    for name in reference:
        left = reference[name]
        right = candidate[name]
        if (left is None) != (right is None):
            missing_pattern.append(name)
            continue
        if left is None:
            continue
        delta = (left - right).abs()
        local_abs = float(delta.max().item()) if delta.numel() else 0.0
        denominator = left.abs().clamp_min(1e-12)
        local_rel = float((delta / denominator).max().item()) if delta.numel() else 0.0
        if local_abs > max_abs:
            max_abs = local_abs
            worst_abs = name
        if local_rel > max_rel:
            max_rel = local_rel
            worst_rel = name
    return {
        "none_pattern_mismatches": missing_pattern,
        "max_abs": max_abs,
        "max_abs_parameter": worst_abs,
        "max_relative": max_rel,
        "max_relative_parameter": worst_rel,
    }


def _one_model_pass(
    model: TinyMixtralForCausalLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    capture: bool,
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    seed_everything(74001)
    rng_before = torch.cuda.get_rng_state().cpu().clone()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    torch.cuda.synchronize()
    forward_finished = time.perf_counter()
    output["loss"].backward()
    torch.cuda.synchronize()
    finished = time.perf_counter()
    rng_after = torch.cuda.get_rng_state().cpu().clone()
    transaction = output["cpt_transaction"]
    proposal_valid = all(
        bool(proposal.valid.item()) for proposal in transaction.proposals
    )
    record = {
        "loss": finite(output["loss"].detach().float().item()),
        "logits": output["logits"].detach().float().cpu().clone() if capture else None,
        "gradients": clone_gradients(model) if capture else None,
        "rng_before": rng_before if capture else None,
        "rng_after": rng_after if capture else None,
        "proposal_valid": proposal_valid,
        "forward_seconds": forward_finished - started,
        "backward_seconds": finished - forward_finished,
        "total_seconds": finished - started,
        "tokens_per_second": (
            int(input_ids.numel()) / (finished - started)
        ),
        "memory": memory_stats(),
    }
    model.abort_cpt_transaction(transaction)
    del output, transaction
    return record


def recompute_matrix() -> dict[str, Any]:
    clean_cuda()
    seed_everything(74000)
    config = router_config()
    template = TinyMixtralForCausalLM(config)
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in template.state_dict().items()
    }
    del template
    generator = torch.Generator(device="cpu").manual_seed(74002)
    tokens = torch.randint(
        0,
        config.vocab_size,
        (2, 129),
        generator=generator,
    )
    input_ids = tokens[:, :-1].cuda()
    labels = tokens[:, 1:].cuda()
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask[0, :11] = False
    labels = labels.clone()
    labels[~attention_mask] = -100

    combinations = [
        (False, "global"),
        (False, "on"),
        (False, "off"),
        (True, "global"),
        (True, "on"),
        (True, "off"),
    ]
    reference = None
    records = []
    for global_checkpointing, router_mode in combinations:
        clean_cuda()
        model = TinyMixtralForCausalLM(config)
        model.load_state_dict(initial_state, strict=True)
        model.to(device="cuda", dtype=torch.bfloat16).train()
        model.set_router_recompute(router_mode)
        if global_checkpointing:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
        versions_before = [
            int(layer.moe.cpt_router.state_version.item())
            for layer in model.layers
        ]
        prices_before = [
            layer.moe.cpt_router.congestion_price.detach().cpu().clone()
            for layer in model.layers
        ]

        warmup = _one_model_pass(
            model,
            input_ids,
            labels,
            attention_mask,
            capture=False,
        )
        model.zero_grad(set_to_none=True)
        measured = _one_model_pass(
            model,
            input_ids,
            labels,
            attention_mask,
            capture=True,
        )
        versions_after = [
            int(layer.moe.cpt_router.state_version.item())
            for layer in model.layers
        ]
        prices_after = [
            layer.moe.cpt_router.congestion_price.detach().cpu().clone()
            for layer in model.layers
        ]
        persistent_state_unchanged = (
            versions_before == versions_after
            and all(
                torch.equal(before, after)
                for before, after in zip(prices_before, prices_after)
            )
        )
        record = {
            "global_checkpointing": global_checkpointing,
            "router_mode": router_mode,
            "effective_router_recompute": (
                router_mode == "on"
                or (router_mode == "global" and global_checkpointing)
            ),
            "warmup_total_seconds": warmup["total_seconds"],
            "loss": measured["loss"],
            "proposal_valid": measured["proposal_valid"],
            "forward_seconds": measured["forward_seconds"],
            "backward_seconds": measured["backward_seconds"],
            "total_seconds": measured["total_seconds"],
            "tokens_per_second": measured["tokens_per_second"],
            "memory": measured["memory"],
            "persistent_state_unchanged_without_commit": persistent_state_unchanged,
        }
        if reference is None:
            reference = measured
            record["vs_reference"] = {
                "loss_abs": 0.0,
                "logits_max_abs": 0.0,
                "gradients": {
                    "none_pattern_mismatches": [],
                    "max_abs": 0.0,
                    "max_relative": 0.0,
                },
                "rng_before_equal": True,
                "rng_after_equal": True,
            }
        else:
            record["vs_reference"] = {
                "loss_abs": abs(measured["loss"] - reference["loss"]),
                "logits_max_abs": finite(
                    (measured["logits"] - reference["logits"]).abs().max().item()
                ),
                "gradients": compare_gradients(
                    reference["gradients"],
                    measured["gradients"],
                ),
                "rng_before_equal": bool(
                    torch.equal(measured["rng_before"], reference["rng_before"])
                ),
                "rng_after_equal": bool(
                    torch.equal(measured["rng_after"], reference["rng_after"])
                ),
            }
        records.append(record)
        del measured, warmup, model
        clean_cuda()

    comparisons_ok = all(
        record["proposal_valid"]
        and record["persistent_state_unchanged_without_commit"]
        and record["vs_reference"]["loss_abs"] == 0.0
        and record["vs_reference"]["logits_max_abs"] == 0.0
        and not record["vs_reference"]["gradients"]["none_pattern_mismatches"]
        and record["vs_reference"]["gradients"]["max_abs"] == 0.0
        and record["vs_reference"]["rng_before_equal"]
        and record["vs_reference"]["rng_after_equal"]
        for record in records
    )
    del initial_state, reference, input_ids, labels, attention_mask, tokens
    clean_cuda()
    return {
        "status": "pass" if comparisons_ok else "fail",
        "batch_size": 2,
        "sequence_length": 128,
        "attention_dropout": config.attention_dropout,
        "records": records,
    }


def router_sequence_scaling() -> dict[str, Any]:
    clean_cuda()
    seed_everything(75000)
    config = router_config()
    router = CPTRouter(config, layer_index=0).to(
        device="cuda",
        dtype=torch.bfloat16,
    ).eval()
    records = []
    for sequence_length in (64, 128, 256, 512):
        hidden = torch.randn(
            4,
            sequence_length,
            config.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        mask = torch.ones(4, sequence_length, device="cuda", dtype=torch.bool)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            warmup = router(hidden, route_valid_mask=mask)
        del warmup
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        iterations = 5
        started = time.perf_counter()
        max_simplex_error = 0.0
        proposal_valid = True
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(iterations):
                output = router(hidden, route_valid_mask=mask)
                max_simplex_error = max(
                    max_simplex_error,
                    float(
                        (output.probabilities.sum(dim=-1) - 1)
                        .abs()
                        .amax()
                        .item()
                    ),
                )
                proposal_valid = proposal_valid and bool(output.proposal.valid.item())
                del output
        torch.cuda.synchronize()
        duration = time.perf_counter() - started
        records.append(
            {
                "batch_size": 4,
                "sequence_length": sequence_length,
                "iterations": iterations,
                "duration_seconds": duration,
                "tokens_per_second": (
                    4 * sequence_length * iterations / duration
                ),
                "proposal_valid": proposal_valid,
                "pi_simplex_max_error": max_simplex_error,
                "memory": memory_stats(),
            }
        )
        del hidden, mask
        clean_cuda()
    del router
    clean_cuda()
    return {
        "status": "pass" if all(
            record["proposal_valid"] and record["pi_simplex_max_error"] <= 1e-5
            for record in records
        ) else "fail",
        "records": records,
    }


def clone_nested(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_nested(item) for item in value)
    return copy.deepcopy(value)


def nested_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def gpu_failure_rollback() -> dict[str, Any]:
    clean_cuda()
    seed_everything(76000)
    config = TinyMixtralConfig(
        vocab_size=64,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        num_local_experts=4,
        num_experts_per_tok=2,
        expert_intermediate_size=128,
        cpt_num_prototypes=8,
        cpt_projection_dim=32,
        cpt_init_seed=76000,
        attention_dropout=0.15,
    )
    model = TinyMixtralForCausalLM(config).to(
        device="cuda",
        dtype=torch.bfloat16,
    ).train()
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    inject_failure = {"enabled": False}

    def fail_after_update(active_optimizer, args, kwargs):
        if inject_failure["enabled"]:
            with torch.no_grad():
                first = active_optimizer.param_groups[0]["params"][0]
                first.view(-1)[0].add_(123.0)
            raise RuntimeError("injected optimizer failure after partial update")

    failure_hook = optimizer.register_step_post_hook(fail_after_update)
    generator = torch.Generator(device="cuda").manual_seed(76001)
    tokens = torch.randint(0, config.vocab_size, (2, 33), generator=generator, device="cuda")
    input_ids = tokens[:, :-1]
    labels = tokens[:, 1:]

    def forward_fn():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(input_ids, labels=labels)

    execute_training_iteration(
        model,
        optimizer,
        scheduler=None,
        forward_fn=forward_fn,
        max_grad_norm=1.0,
    )
    model_before = clone_nested(model.state_dict())
    optimizer_before = clone_nested(optimizer.state_dict())
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state().cpu().clone()
    versions_before = [int(layer.moe.cpt_router.state_version.item()) for layer in model.layers]
    prices_before = [
        layer.moe.cpt_router.congestion_price.detach().cpu().clone()
        for layer in model.layers
    ]

    inject_failure["enabled"] = True
    caught = None
    try:
        execute_training_iteration(
            model,
            optimizer,
            scheduler=None,
            forward_fn=forward_fn,
            max_grad_norm=1.0,
        )
    except RuntimeError as error:
        caught = str(error)
    torch.cuda.synchronize()
    model_after = clone_nested(model.state_dict())
    optimizer_after = clone_nested(optimizer.state_dict())
    versions_after = [int(layer.moe.cpt_router.state_version.item()) for layer in model.layers]
    prices_after = [
        layer.moe.cpt_router.congestion_price.detach().cpu().clone()
        for layer in model.layers
    ]
    result = {
        "caught_expected_error": caught is not None and "injected optimizer failure" in caught,
        "error": caught,
        "model_exactly_restored": nested_equal(model_before, model_after),
        "optimizer_exactly_restored": nested_equal(optimizer_before, optimizer_after),
        "cpu_rng_exactly_restored": torch.equal(torch.get_rng_state(), cpu_rng_before),
        "cuda_rng_exactly_restored": torch.equal(
            torch.cuda.get_rng_state().cpu(),
            cuda_rng_before,
        ),
        "versions_before": versions_before,
        "versions_after": versions_after,
        "versions_not_advanced": versions_before == versions_after,
        "prices_exactly_restored": all(
            torch.equal(before, after)
            for before, after in zip(prices_before, prices_after)
        ),
        "all_gradients_cleared": all(
            parameter.grad is None for parameter in model.parameters()
        ),
        "memory": memory_stats(),
    }
    result["status"] = "pass" if all(
        value
        for key, value in result.items()
        if key in {
            "caught_expected_error",
            "model_exactly_restored",
            "optimizer_exactly_restored",
            "cpu_rng_exactly_restored",
            "cuda_rng_exactly_restored",
            "versions_not_advanced",
            "prices_exactly_restored",
            "all_gradients_cleared",
        }
    ) else "fail"
    failure_hook.remove()
    del model, optimizer, tokens, input_ids, labels
    clean_cuda()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_ROOT / "gpu_recompute_stress_20260731.json",
    )
    args = parser.parse_args()
    output = args.output.resolve(strict=False)
    if not output.is_relative_to(REPO_ROOT.resolve()):
        parser.error("--output must stay inside the repository")
    if output.exists():
        parser.error("--output must name a new evidence file; overwrite is refused")
    args.output = output
    return args


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_name": "cpt_moe_deep_gpu_audit",
        "schema_version": 1,
        "started_at": now_iso(),
        "repository": str(REPO_ROOT),
        "output": str(args.output),
        "source_hashes_before": source_hashes(),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "bf16_supported": (
                torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
            ),
            "device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "experiments": {},
    }
    exit_code = 0
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA BF16 is not supported")
        report["experiments"]["direct_router_numerics"] = direct_router_numerics()
        report["experiments"]["recompute_matrix"] = recompute_matrix()
        report["experiments"]["router_sequence_scaling"] = router_sequence_scaling()
        report["experiments"]["gpu_failure_rollback"] = gpu_failure_rollback()
    except BaseException as error:
        report["fatal_error"] = exception_record(error)
        exit_code = 1
    finally:
        clean_cuda()
        report["finished_at"] = now_iso()
        report["duration_seconds"] = time.perf_counter() - started
        report["source_hashes_after"] = source_hashes()
        report["source_snapshot_stable"] = (
            report["source_hashes_before"] == report["source_hashes_after"]
        )
        statuses = [
            value.get("status")
            for value in report["experiments"].values()
            if isinstance(value, dict)
        ]
        report["overall_status"] = (
            "pass"
            if exit_code == 0
            and statuses
            and all(status == "pass" for status in statuses)
            and report["source_snapshot_stable"]
            else "completed_with_findings" if exit_code == 0 else "fatal_error"
        )
        atomic_write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "overall_status": report["overall_status"],
                "duration_seconds": report["duration_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
