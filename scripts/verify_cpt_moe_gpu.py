#!/usr/bin/env python3
"""Bounded local CUDA verification for the CPT-MoE TinyMixtral model.

This script is intentionally self-contained and offline.  It exercises the
strict optimizer/CPT transaction path on a small BF16 model, performs an
in-memory state-dict round trip, and then probes the repository's default
432M-parameter configuration within an explicit soft time budget.

The generated JSON is evidence, not a claim of convergence or research
quality.  No hardware settings, drivers, or other processes are modified.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "local_verification"
PROCESS_TEMP = ARTIFACT_ROOT / ".tmp_gpu_20260731"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
PROCESS_TEMP.mkdir(parents=True, exist_ok=True)

# Keep process-created temporary/cache files inside the user-authorized tree.
os.environ["TEMP"] = str(PROCESS_TEMP)
os.environ["TMP"] = str(PROCESS_TEMP)
os.environ.setdefault("HF_HOME", str(PROCESS_TEMP / "hf_home"))

sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from model.config import TinyMixtralConfig
from model.modeling import TinyMixtralForCausalLM
from scripts.train_utils import execute_training_iteration, make_adamw


CORE_SOURCE_FILES = (
    "model/config.py",
    "model/cpt_router.py",
    "model/modeling.py",
    "scripts/train_utils.py",
    "scripts/verify_cpt_moe_gpu.py",
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def finite_float(value: Any) -> float | None:
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return normalized if np.isfinite(normalized) else None


def exception_record(error: BaseException) -> dict[str, Any]:
    trace_lines = traceback.format_exc().strip().splitlines()
    return {
        "type": type(error).__name__,
        "message": str(error),
        "is_cuda_oom": isinstance(error, torch.cuda.OutOfMemoryError),
        "traceback_tail": trace_lines[-10:],
    }


def process_memory_stats() -> dict[str, Any]:
    """Return current/peak process memory from the Windows kernel when available."""
    if os.name != "nt":
        return {"supported": False, "platform": os.name}
    try:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("page_fault_count", wintypes.DWORD),
                ("peak_working_set_bytes", ctypes.c_size_t),
                ("working_set_bytes", ctypes.c_size_t),
                ("quota_peak_paged_pool_bytes", ctypes.c_size_t),
                ("quota_paged_pool_bytes", ctypes.c_size_t),
                ("quota_peak_nonpaged_pool_bytes", ctypes.c_size_t),
                ("quota_nonpaged_pool_bytes", ctypes.c_size_t),
                ("pagefile_bytes", ctypes.c_size_t),
                ("peak_pagefile_bytes", ctypes.c_size_t),
                ("private_bytes", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCountersEx),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "supported": True,
            "working_set_bytes": int(counters.working_set_bytes),
            "peak_working_set_bytes": int(counters.peak_working_set_bytes),
            "private_bytes": int(counters.private_bytes),
            "pagefile_bytes": int(counters.pagefile_bytes),
            "peak_pagefile_bytes": int(counters.peak_pagefile_bytes),
            "page_fault_count": int(counters.page_fault_count),
        }
    except BaseException as error:
        return {
            "supported": False,
            "platform": os.name,
            "error": f"{type(error).__name__}: {error}",
        }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish a complete report without replacing old evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite GPU evidence: {path}")
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
        # A same-directory hard link atomically fails if another run published
        # the target first.  Unlike os.replace(), it can never overwrite an
        # existing evidence file.
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
    return {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in CORE_SOURCE_FILES
    }


def run_read_only_command(command: list[str], timeout: int = 15) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except BaseException as error:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(error).__name__}: {error}",
        }


def nvidia_smi_snapshot() -> dict[str, Any]:
    fields = (
        "name,uuid,driver_version,memory.total,memory.used,memory.free,"
        "temperature.gpu,pstate,utilization.gpu,utilization.memory"
    )
    command_result = run_read_only_command(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
    )
    parsed = None
    if command_result["returncode"] == 0 and command_result["stdout"]:
        rows = [row.strip() for row in command_result["stdout"].splitlines() if row.strip()]
        if rows:
            values = [value.strip() for value in rows[0].split(",")]
            names = fields.split(",")
            if len(values) == len(names):
                parsed = dict(zip(names, values))
    return {"command": command_result, "first_gpu": parsed}


def git_snapshot() -> dict[str, Any]:
    return {
        "head": run_read_only_command(["git", "rev-parse", "HEAD"]),
        "branch": run_read_only_command(
            ["git", "branch", "--show-current"]
        ),
        "status_short": run_read_only_command(
            ["git", "status", "--short", "--untracked-files=no"]
        ),
    }


def cuda_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "bf16_supported": (
            torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
        ),
    }
    if not torch.cuda.is_available():
        return snapshot
    device = torch.device("cuda", 0)
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    snapshot.update(
        {
            "device_index": 0,
            "device_name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "multiprocessor_count": properties.multi_processor_count,
            "total_memory_bytes": int(total_bytes),
            "free_memory_bytes": int(free_bytes),
            "process_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "process_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "process_peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "process_peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
        }
    )
    return snapshot


def clean_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def reset_cuda_peak() -> None:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def memory_stats() -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_versions(model: TinyMixtralForCausalLM) -> list[int]:
    return [
        int(layer.moe.cpt_router.state_version.detach().cpu().item())
        for layer in model.layers
    ]


def model_prices(model: TinyMixtralForCausalLM) -> list[list[float]]:
    return [
        [float(value) for value in layer.moe.cpt_router.congestion_price.detach().cpu()]
        for layer in model.layers
    ]


def price_summary(model: TinyMixtralForCausalLM) -> dict[str, Any]:
    prices = [
        layer.moe.cpt_router.congestion_price.detach().float()
        for layer in model.layers
    ]
    flattened = torch.cat(prices) if prices else torch.empty(0)
    return {
        "per_layer": model_prices(model),
        "all_finite": bool(torch.isfinite(flattened).all().item()),
        "all_nonnegative": bool((flattened >= 0).all().item()),
        "min": finite_float(flattened.min().item()) if flattened.numel() else None,
        "max": finite_float(flattened.max().item()) if flattened.numel() else None,
        "mean": finite_float(flattened.mean().item()) if flattened.numel() else None,
        "any_nonzero": bool((flattened != 0).any().item()) if flattened.numel() else False,
    }


def parameter_and_buffer_dtypes(model: TinyMixtralForCausalLM) -> dict[str, Any]:
    parameter_counts: dict[str, int] = {}
    for parameter in model.parameters():
        key = str(parameter.dtype)
        parameter_counts[key] = parameter_counts.get(key, 0) + parameter.numel()
    price_dtypes = sorted(
        {str(layer.moe.cpt_router.congestion_price.dtype) for layer in model.layers}
    )
    version_dtypes = sorted(
        {str(layer.moe.cpt_router.state_version.dtype) for layer in model.layers}
    )
    return {
        "parameter_numel_by_dtype": parameter_counts,
        "congestion_price_dtypes": price_dtypes,
        "state_version_dtypes": version_dtypes,
    }


def tensor_bytes(tensors: Iterable[torch.Tensor]) -> int:
    return int(sum(t.numel() * t.element_size() for t in tensors))


def optimizer_state_tensor_bytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
    return int(total)


class GradientRecorder:
    """Capture finite/norm diagnostics without retaining gradient tensors."""

    def __init__(self, model: TinyMixtralForCausalLM, names: list[str]):
        named = dict(model.named_parameters())
        self.requested = list(names)
        self.missing_at_install = [name for name in names if name not in named]
        self._current: dict[str, dict[str, Any]] = {}
        self._handles = []
        for name in names:
            parameter = named.get(name)
            if parameter is not None:
                self._handles.append(parameter.register_hook(self._hook(name)))

    def _hook(self, name: str):
        def record(gradient: torch.Tensor) -> torch.Tensor:
            detached = gradient.detach().float()
            finite = bool(torch.isfinite(detached).all().item())
            norm = finite_float(torch.linalg.vector_norm(detached).item())
            max_abs = finite_float(detached.abs().max().item())
            previous = self._current.get(name)
            if previous is None:
                self._current[name] = {
                    "calls": 1,
                    "all_finite": finite,
                    "norm_last": norm,
                    "max_abs": max_abs,
                }
            else:
                previous["calls"] += 1
                previous["all_finite"] = previous["all_finite"] and finite
                previous["norm_last"] = norm
                candidates = [value for value in (previous["max_abs"], max_abs) if value is not None]
                previous["max_abs"] = max(candidates) if candidates else None
            return gradient

        return record

    def begin(self) -> None:
        self._current = {}

    def finish(self) -> dict[str, Any]:
        return {
            "requested": list(self.requested),
            "missing_at_install": list(self.missing_at_install),
            "observed": dict(self._current),
            "all_requested_observed": (
                not self.missing_at_install
                and all(name in self._current for name in self.requested)
            ),
            "all_observed_finite": all(
                item["all_finite"] for item in self._current.values()
            ),
        }

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def recomputed_ce_delta(output: dict[str, Any], labels: torch.Tensor) -> float | None:
    with torch.no_grad():
        ce = F.cross_entropy(
            output["logits"].detach().reshape(-1, output["logits"].size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return finite_float((output["loss"].detach().float() - ce.float()).abs().item())


def tiny_config() -> TinyMixtralConfig:
    return TinyMixtralConfig(
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
        cpt_init_seed=20260731,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        initializer_range=0.02,
    )


def tiny_gradient_names() -> list[str]:
    return [
        "embed_tokens.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.moe.gate_proj",
        "layers.0.moe.cpt_router.projection",
        "layers.0.moe.cpt_router.anchors",
        "layers.0.moe.cpt_router.energy",
    ]


def run_tiny_experiment(
    steps: int,
    failure_policy: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    clean_cuda()
    reset_cuda_peak()
    seed_everything(20260731)
    started = time.perf_counter()
    config = tiny_config()
    model = TinyMixtralForCausalLM(config).to(
        device="cuda",
        dtype=torch.bfloat16,
    )
    model.train()
    model.gradient_checkpointing_enable()
    optimizer = make_adamw(model, lr=3e-3, weight_decay=0.0)
    recorder = GradientRecorder(model, tiny_gradient_names())

    generator = torch.Generator(device="cuda").manual_seed(731)
    tokens = torch.randint(
        0,
        config.vocab_size,
        (4, 33),
        generator=generator,
        device="cuda",
    )
    input_ids = tokens[:, :-1]
    labels = tokens[:, 1:]
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    initial_prices = model_prices(model)
    initial_versions = model_versions(model)
    records: list[dict[str, Any]] = []

    try:
        for step_index in range(steps):
            recorder.begin()
            torch.cuda.synchronize()
            step_started = time.perf_counter()
            process_memory_before = process_memory_stats()

            def forward_fn() -> dict[str, Any]:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

            output, grad_norm = execute_training_iteration(
                model,
                optimizer,
                scheduler=None,
                forward_fn=forward_fn,
                max_grad_norm=1.0,
                failure_policy=failure_policy,
            )
            torch.cuda.synchronize()
            transaction = output["cpt_transaction"]
            record = {
                "step": step_index + 1,
                "loss_ce": finite_float(output["loss"].detach().float().item()),
                "loss_minus_recomputed_ce_abs": recomputed_ce_delta(output, labels),
                "grad_norm_before_clip": finite_float(grad_norm.float().item()),
                "duration_seconds": time.perf_counter() - step_started,
                "process_memory_before": process_memory_before,
                "process_memory_after": process_memory_stats(),
                "versions": model_versions(model),
                "price": price_summary(model),
                "transaction_closed": bool(transaction.closed),
                "transaction_aborted": bool(transaction.aborted),
                "key_gradients": recorder.finish(),
            }
            records.append(record)
            del output, transaction, grad_norm
    finally:
        recorder.close()

    final_prices = model_prices(model)
    final_versions = model_versions(model)
    losses = [record["loss_ce"] for record in records]
    gradient_checks = [record["key_gradients"] for record in records]
    lambda_grad_none = all(
        layer.moe.cpt_router.congestion_price.grad is None
        for layer in model.layers
    )
    expected_versions = [steps] * config.num_hidden_layers
    result = {
        "status": "pass" if len(records) == steps else "fail",
        "config": config.to_dict(),
        "device": "cuda:0",
        "parameter_count": model.num_parameters,
        "parameter_bytes": tensor_bytes(model.parameters()),
        "activation_checkpointing_enabled": bool(
            model._use_activation_checkpointing
        ),
        "training_failure_policy": failure_policy,
        "autocast_dtype": "torch.bfloat16",
        "model_and_buffer_dtypes": parameter_and_buffer_dtypes(model),
        "batch_size": int(input_ids.shape[0]),
        "sequence_length": int(input_ids.shape[1]),
        "requested_steps": steps,
        "successful_steps": len(records),
        "initial_versions": initial_versions,
        "final_versions": final_versions,
        "expected_final_versions": expected_versions,
        "version_progression_exact": final_versions == expected_versions,
        "initial_prices": initial_prices,
        "final_price": price_summary(model),
        "prices_changed": initial_prices != final_prices,
        "lambda_has_no_gradient": lambda_grad_none,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_reduced": bool(losses and losses[-1] < losses[0]),
        "loss_reduction_fraction": (
            finite_float((losses[0] - losses[-1]) / losses[0])
            if losses and losses[0]
            else None
        ),
        "all_losses_finite": all(value is not None for value in losses),
        "all_loss_ce_deltas_zero": all(
            record["loss_minus_recomputed_ce_abs"] == 0.0 for record in records
        ),
        "all_key_gradients_observed_and_finite": all(
            check["all_requested_observed"] and check["all_observed_finite"]
            for check in gradient_checks
        ),
        "steps": records,
        "memory": memory_stats(),
        "duration_seconds": time.perf_counter() - started,
    }

    # In-memory state_dict round trip.  This avoids persistent checkpoint
    # clutter while still exercising the native strict load boundary.
    model.eval()
    cpu_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    restored = TinyMixtralForCausalLM(config).to(
        device="cuda",
        dtype=torch.bfloat16,
    )
    load_result = restored.load_state_dict(cpu_state, strict=True)
    restored.eval()
    state_exact = all(
        torch.equal(value.detach().cpu(), cpu_state[name])
        for name, value in restored.state_dict().items()
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        original_output = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        restored_output = restored(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    logits_delta = (
        original_output["logits"].detach().float()
        - restored_output["logits"].detach().float()
    ).abs().max()
    roundtrip = {
        "status": "pass",
        "method": "in_memory_cpu_state_dict_strict_load",
        "state_dict_key_count": len(cpu_state),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "all_state_tensors_exact": state_exact,
        "source_versions": final_versions,
        "restored_versions": model_versions(restored),
        "versions_exact": final_versions == model_versions(restored),
        "source_prices": final_prices,
        "restored_prices": model_prices(restored),
        "prices_exact": final_prices == model_prices(restored),
        "max_abs_logits_delta": finite_float(logits_delta.item()),
        "logits_exact": bool(logits_delta.item() == 0.0),
    }
    roundtrip["status"] = "pass" if all(
        (
            not roundtrip["missing_keys"],
            not roundtrip["unexpected_keys"],
            roundtrip["all_state_tensors_exact"],
            roundtrip["versions_exact"],
            roundtrip["prices_exact"],
            roundtrip["logits_exact"],
        )
    ) else "fail"

    del original_output, restored_output, restored, cpu_state
    del optimizer, model, tokens, input_ids, labels, attention_mask
    clean_cuda()
    return result, roundtrip


def low_memory_adamw(
    model: TinyMixtralForCausalLM,
    lr: float,
) -> torch.optim.AdamW:
    decay: list[torch.Tensor] = []
    no_decay: list[torch.Tensor] = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.0},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        foreach=False,
    )


def default_gradient_names() -> list[str]:
    return [
        "embed_tokens.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.moe.gate_proj",
        "layers.0.moe.cpt_router.projection",
        "layers.0.moe.cpt_router.anchors",
        "layers.0.moe.cpt_router.energy",
    ]


def make_default_batch(
    config: TinyMixtralConfig,
    sequence_length: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tokens = torch.randint(
        0,
        config.vocab_size,
        (1, sequence_length + 1),
        generator=generator,
        device="cuda",
    )
    input_ids = tokens[:, :-1]
    labels = tokens[:, 1:]
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    return input_ids, labels, mask


def run_default_forward(
    model: TinyMixtralForCausalLM,
    config: TinyMixtralConfig,
    sequence_length: int,
) -> tuple[dict[str, Any], tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    input_ids, labels, mask = make_default_batch(
        config,
        sequence_length,
        seed=9000 + sequence_length,
    )
    model.eval()
    reset_cuda_peak()
    started = time.perf_counter()
    process_memory_before = process_memory_stats()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                input_ids,
                attention_mask=mask,
                labels=labels,
            )
        torch.cuda.synchronize()
        record = {
            "status": "pass",
            "sequence_length": sequence_length,
            "loss_ce": finite_float(output["loss"].detach().float().item()),
            "loss_minus_recomputed_ce_abs": recomputed_ce_delta(output, labels),
            "logits_all_finite": bool(
                torch.isfinite(output["logits"].detach()).all().item()
            ),
            "logits_shape": list(output["logits"].shape),
            "transaction_is_none_in_eval": output["cpt_transaction"] is None,
            "duration_seconds": time.perf_counter() - started,
            "memory": memory_stats(),
            "process_memory_before": process_memory_before,
            "process_memory_after": process_memory_stats(),
        }
        del output
        return record, (input_ids, labels, mask)
    except BaseException as error:
        record = {
            "status": "oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "error",
            "sequence_length": sequence_length,
            "duration_seconds": time.perf_counter() - started,
            "error": exception_record(error),
            "memory": memory_stats(),
            "process_memory_before": process_memory_before,
            "process_memory_after": process_memory_stats(),
        }
        del input_ids, labels, mask
        clean_cuda()
        return record, None


def run_default_training_attempt(
    model: TinyMixtralForCausalLM,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    recorder: GradientRecorder,
    sequence_length: int,
    failure_policy: str,
) -> dict[str, Any]:
    input_ids, labels, mask = batch
    model.train()
    recorder.begin()
    reset_cuda_peak()
    started = time.perf_counter()
    process_memory_before = process_memory_stats()
    versions_before = model_versions(model)
    prices_before = model_prices(model)
    try:
        def forward_fn() -> dict[str, Any]:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model(
                    input_ids,
                    attention_mask=mask,
                    labels=labels,
                )

        output, grad_norm = execute_training_iteration(
            model,
            optimizer,
            scheduler=None,
            forward_fn=forward_fn,
            max_grad_norm=1.0,
            failure_policy=failure_policy,
        )
        torch.cuda.synchronize()
        transaction = output["cpt_transaction"]
        record = {
            "status": "pass",
            "sequence_length": sequence_length,
            "training_failure_policy": failure_policy,
            "loss_ce": finite_float(output["loss"].detach().float().item()),
            "loss_minus_recomputed_ce_abs": recomputed_ce_delta(output, labels),
            "grad_norm_before_clip": finite_float(grad_norm.float().item()),
            "versions_before": versions_before,
            "versions_after": model_versions(model),
            "prices_before": prices_before,
            "price_after": price_summary(model),
            "transaction_closed": bool(transaction.closed),
            "transaction_aborted": bool(transaction.aborted),
            "key_gradients": recorder.finish(),
            "lambda_has_no_gradient": all(
                layer.moe.cpt_router.congestion_price.grad is None
                for layer in model.layers
            ),
            "optimizer_state_tensor_bytes": optimizer_state_tensor_bytes(optimizer),
            "duration_seconds": time.perf_counter() - started,
            "memory": memory_stats(),
            "process_memory_before": process_memory_before,
            "process_memory_after": process_memory_stats(),
        }
        del output, grad_norm, transaction
        return record
    except BaseException as error:
        torch.cuda.synchronize()
        record = {
            "status": "oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "error",
            "sequence_length": sequence_length,
            "training_failure_policy": failure_policy,
            "versions_before": versions_before,
            "versions_after_exception": model_versions(model),
            "prices_restored_after_exception": prices_before == model_prices(model),
            "key_gradients_before_exception": recorder.finish(),
            "duration_seconds": time.perf_counter() - started,
            "error": exception_record(error),
            "memory": memory_stats(),
            "process_memory_before": process_memory_before,
            "process_memory_after": process_memory_stats(),
        }
        optimizer.zero_grad(set_to_none=True)
        clean_cuda()
        return record


def classify_default_training_status(
    *,
    successful_steps: int,
    requested_steps: int,
    attempts: list[dict[str, Any]],
    stopped_for_resource_margin: bool,
    stopped_for_soft_budget: bool,
) -> str:
    """Return a status that never hides a failed requested training step."""
    failed_attempts = [
        attempt
        for attempt in attempts
        if attempt.get("status") != "pass"
        and not attempt.get("recovered_by_shorter_sequence_retry", False)
    ]
    if failed_attempts:
        return (
            "training_failed_after_success"
            if successful_steps > 0
            else "training_failed"
        )
    if successful_steps == requested_steps:
        if any(
            attempt.get("recovered_by_shorter_sequence_retry", False)
            for attempt in attempts
        ):
            return "pass_with_shorter_sequence_fallback"
        return "pass"
    if successful_steps > 0 and (
        stopped_for_resource_margin or stopped_for_soft_budget
    ):
        return "partial_resource_budget"
    return "forward_only"


def run_default_experiment(
    max_steps: int,
    script_started: float,
    soft_budget_seconds: float,
    failure_policy: str,
) -> dict[str, Any]:
    clean_cuda()
    reset_cuda_peak()
    seed_everything(20260732)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "status": "not_started",
        "requested_config": "TinyMixtralConfig defaults",
        "requested_batch_size": 1,
        "requested_sequence_length": 16,
        "activation_checkpointing_requested": True,
        "training_failure_policy": failure_policy,
        "max_strict_steps": max_steps,
        "construction": None,
        "forward_attempts": [],
        "training_attempts": [],
        "fail_stop_rebuilds": [],
    }
    model = None
    optimizer = None
    recorder = None
    retained_batch = None
    try:
        config = TinyMixtralConfig()
        construction_started = time.perf_counter()
        model = TinyMixtralForCausalLM(config)
        model = model.to(dtype=torch.bfloat16)
        model = model.to(device="cuda")
        model.gradient_checkpointing_enable()
        torch.cuda.synchronize()
        result["construction"] = {
            "status": "pass",
            "duration_seconds": time.perf_counter() - construction_started,
            "parameter_count": model.num_parameters,
            "parameter_bytes": tensor_bytes(model.parameters()),
            "activation_checkpointing_enabled": bool(
                model._use_activation_checkpointing
            ),
            "model_and_buffer_dtypes": parameter_and_buffer_dtypes(model),
            "initial_versions": model_versions(model),
            "initial_price": price_summary(model),
            "memory": memory_stats(),
        }

        for sequence_length in (16, 8):
            forward_record, batch = run_default_forward(
                model,
                config,
                sequence_length,
            )
            result["forward_attempts"].append(forward_record)
            if batch is not None:
                retained_batch = batch
                break

        if retained_batch is None:
            result["status"] = "forward_failed"
            return result

        optimizer = low_memory_adamw(model, lr=1e-4)
        recorder = GradientRecorder(model, default_gradient_names())
        successful_steps = 0
        sequence_length = int(retained_batch[0].shape[1])
        first_train_batch = retained_batch
        attempted_lengths: set[int] = set()
        fallback_oom_attempt_index: Optional[int] = None

        while successful_steps < max_steps:
            elapsed_total = time.perf_counter() - script_started
            if elapsed_total >= soft_budget_seconds:
                result["training_skipped_due_to_soft_budget"] = True
                break
            attempted_lengths.add(sequence_length)
            attempt = run_default_training_attempt(
                model,
                optimizer,
                first_train_batch,
                recorder,
                sequence_length,
                failure_policy,
            )
            result["training_attempts"].append(attempt)
            if attempt["status"] == "pass":
                if fallback_oom_attempt_index is not None:
                    result["training_attempts"][fallback_oom_attempt_index][
                        "recovered_by_shorter_sequence_retry"
                    ] = True
                    fallback_oom_attempt_index = None
                successful_steps += 1
                # A second strict step is valuable because the rollback snapshot
                # then includes populated AdamW state.  Only attempt it when the
                # first step was comfortably inside the bounded local budget.
                if successful_steps == 1 and max_steps > 1:
                    free_bytes, _ = torch.cuda.mem_get_info()
                    if (
                        attempt["duration_seconds"] > 60.0
                        or time.perf_counter() - script_started > soft_budget_seconds - 75.0
                        or free_bytes < 1_200_000_000
                    ):
                        result["second_step_skipped_for_resource_margin"] = {
                            "first_step_seconds": attempt["duration_seconds"],
                            "free_bytes": int(free_bytes),
                        }
                        break
                continue

            if (
                attempt["status"] == "oom"
                and sequence_length == 16
                and 8 not in attempted_lengths
                and successful_steps == 0
            ):
                fallback_oom_attempt_index = len(result["training_attempts"]) - 1
                del first_train_batch
                retained_batch = None
                if failure_policy == "fail-stop":
                    # A fail-stop exception intentionally poisons the live
                    # training objects.  Reconstruct the deterministic initial
                    # runtime before the shorter-sequence retry; never reuse a
                    # model/optimizer that the policy has declared invalid.
                    rebuild_started = time.perf_counter()
                    recorder.close()
                    recorder = None
                    del optimizer, model
                    optimizer = None
                    model = None
                    clean_cuda()
                    seed_everything(20260732)
                    model = TinyMixtralForCausalLM(config)
                    model = model.to(dtype=torch.bfloat16)
                    model = model.to(device="cuda")
                    model.gradient_checkpointing_enable()
                    optimizer = low_memory_adamw(model, lr=1e-4)
                    recorder = GradientRecorder(
                        model,
                        default_gradient_names(),
                    )
                    torch.cuda.synchronize()
                    result["fail_stop_rebuilds"].append(
                        {
                            "reason": "sequence_length_16_cuda_oom",
                            "retry_sequence_length": 8,
                            "duration_seconds": (
                                time.perf_counter() - rebuild_started
                            ),
                            "versions": model_versions(model),
                            "price": price_summary(model),
                            "optimizer_state_tensor_bytes": (
                                optimizer_state_tensor_bytes(optimizer)
                            ),
                        }
                    )
                first_train_batch = make_default_batch(config, 8, seed=9008)
                sequence_length = 8
                continue
            break

        result["successful_strict_steps"] = successful_steps
        result["final_versions"] = model_versions(model)
        result["final_price"] = price_summary(model)
        result["optimizer_state_tensor_bytes"] = optimizer_state_tensor_bytes(
            optimizer
        )
        result["lambda_has_no_gradient"] = all(
            layer.moe.cpt_router.congestion_price.grad is None
            for layer in model.layers
        )
        result["status"] = classify_default_training_status(
            successful_steps=successful_steps,
            requested_steps=max_steps,
            attempts=result["training_attempts"],
            stopped_for_resource_margin=(
                "second_step_skipped_for_resource_margin" in result
            ),
            stopped_for_soft_budget=bool(
                result.get("training_skipped_due_to_soft_budget", False)
            ),
        )
        return result
    except BaseException as error:
        result["status"] = (
            "construction_oom"
            if isinstance(error, torch.cuda.OutOfMemoryError)
            else "error"
        )
        result["unhandled_error"] = exception_record(error)
        return result
    finally:
        result["duration_seconds"] = time.perf_counter() - started
        if recorder is not None:
            recorder.close()
        del retained_batch, recorder, optimizer, model
        clean_cuda()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_ROOT / "gpu_20260731.json",
    )
    parser.add_argument("--tiny-steps", type=int, default=15)
    parser.add_argument("--default-max-steps", type=int, default=2)
    parser.add_argument("--soft-budget-seconds", type=float, default=285.0)
    parser.add_argument(
        "--failure-policy",
        choices=("exact-rollback", "fail-stop"),
        default="exact-rollback",
        help="training transaction recovery policy exercised by every step",
    )
    args = parser.parse_args()
    if args.tiny_steps < 10 or args.tiny_steps > 20:
        parser.error("--tiny-steps must be between 10 and 20")
    if args.default_max_steps < 1 or args.default_max_steps > 2:
        parser.error("--default-max-steps must be 1 or 2")
    resolved_output = args.output.resolve()
    if not resolved_output.is_relative_to(REPO_ROOT.resolve()):
        parser.error("--output must remain inside the repository")
    if resolved_output.exists():
        parser.error("--output must name a new evidence file; overwrite is refused")
    if args.tiny_steps < 1:
        parser.error("--tiny-steps must be at least 1")
    if args.default_max_steps < 1:
        parser.error("--default-max-steps must be at least 1")
    if args.soft_budget_seconds <= 0:
        parser.error("--soft-budget-seconds must be positive")
    args.output = resolved_output
    return args


def main() -> int:
    args = parse_args()
    script_started = time.perf_counter()
    result: dict[str, Any] = {
        "schema_name": "cpt_moe_local_gpu_verification",
        "schema_version": 1,
        "started_at": now_iso(),
        "repository": str(REPO_ROOT),
        "output": str(args.output),
        "constraints": {
            "offline": True,
            "hardware_or_driver_settings_modified": False,
            "other_processes_terminated": False,
            "formal_training": False,
            "soft_budget_seconds": args.soft_budget_seconds,
            "training_failure_policy": args.failure_policy,
        },
        "git_before": git_snapshot(),
        "source_hashes_before": source_hashes(),
        "environment_before": {
            "cuda": cuda_snapshot(),
            "nvidia_smi": nvidia_smi_snapshot(),
            "python": sys.version,
        },
        "experiment_a_tiny_bf16_strict_training": None,
        "experiment_b_default_432m": None,
        "experiment_c_state_roundtrip": None,
    }
    exit_code = 0
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA device does not report BF16 support")

        tiny_result, roundtrip = run_tiny_experiment(
            args.tiny_steps,
            args.failure_policy,
        )
        result["experiment_a_tiny_bf16_strict_training"] = tiny_result
        result["experiment_c_state_roundtrip"] = roundtrip
        result["experiment_b_default_432m"] = run_default_experiment(
            max_steps=args.default_max_steps,
            script_started=script_started,
            soft_budget_seconds=args.soft_budget_seconds,
            failure_policy=args.failure_policy,
        )
    except BaseException as error:
        result["fatal_error"] = exception_record(error)
        exit_code = 1
    finally:
        clean_cuda()
        result["finished_at"] = now_iso()
        result["duration_seconds"] = time.perf_counter() - script_started
        result["source_hashes_after"] = source_hashes()
        result["source_snapshot_stable"] = (
            result["source_hashes_before"] == result["source_hashes_after"]
        )
        result["environment_after"] = {
            "cuda": cuda_snapshot(),
            "nvidia_smi": nvidia_smi_snapshot(),
        }
        a = result.get("experiment_a_tiny_bf16_strict_training") or {}
        b = result.get("experiment_b_default_432m") or {}
        c = result.get("experiment_c_state_roundtrip") or {}
        if exit_code == 0:
            if (
                a.get("status") == "pass"
                and a.get("all_key_gradients_observed_and_finite")
                and a.get("all_loss_ce_deltas_zero")
                and a.get("version_progression_exact")
                and c.get("status") == "pass"
                and b.get("status") == "pass"
                and result["source_snapshot_stable"]
            ):
                result["overall_status"] = "pass"
            elif (
                a.get("status") == "pass"
                and a.get("all_key_gradients_observed_and_finite")
                and a.get("all_loss_ce_deltas_zero")
                and a.get("version_progression_exact")
                and c.get("status") == "pass"
                and b.get("status") == "pass_with_shorter_sequence_fallback"
                and result["source_snapshot_stable"]
            ):
                result["overall_status"] = "pass_with_default_sequence_fallback"
            elif b.get("status") == "forward_only":
                result["overall_status"] = "partial_default_forward_only"
            else:
                result["overall_status"] = "completed_with_findings"
        else:
            result["overall_status"] = "fatal_error"
        atomic_write_json(args.output, result)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "overall_status": result.get("overall_status"),
                "duration_seconds": result.get("duration_seconds"),
                "tiny_steps": (
                    result.get("experiment_a_tiny_bf16_strict_training") or {}
                ).get("successful_steps"),
                "default_status": (
                    result.get("experiment_b_default_432m") or {}
                ).get("status"),
                "default_strict_steps": (
                    result.get("experiment_b_default_432m") or {}
                ).get("successful_strict_steps"),
            },
            ensure_ascii=False,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
