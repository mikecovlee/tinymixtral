# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""train.py 和 resume.py 共享的训练逻辑。"""

import copy
import hashlib
import math
import numbers
import os
import random
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid
import weakref
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


TRAINING_STATE_SCHEMA_NAME = "tinymixtral-cpt-training-state"
TRAINING_STATE_SCHEMA_VERSION = 5
TRAINING_FAILURE_POLICIES = ("exact-rollback", "fail-stop")
OPTIMIZER_KINDS = ("adamw", "bf16_adamw")
SCHEDULE_KINDS = ("cosine", "wsd")
DEFAULT_WSD_DECAY_RATIO = 0.1
_SCHEDULE_IDENTITIES = weakref.WeakKeyDictionary()

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_TEMP_CHECKPOINT_PATTERN = re.compile(r"^\.step_\d+(?:_final)?\.tmp-(\d+)$")
_PERIODIC_CHECKPOINT_PATTERN = re.compile(r"^step_(\d{7})$")
_CHECKPOINT_REQUIRED_FILES = (
    "config.json",
    "pytorch_model.bin",
    "training_state.pt",
)


class TrainingFailStop(RuntimeError):
    """A failed step poisoned the live state and requires checkpoint recovery."""


def _normalize_failure_policy(policy):
    if not isinstance(policy, str) or policy not in TRAINING_FAILURE_POLICIES:
        raise ValueError(
            "failure_policy must be one of "
            f"{TRAINING_FAILURE_POLICIES}, got {policy!r}"
        )
    return policy


def _training_poison_reason(model):
    return getattr(_unwrap_cpt_model(model), "_cpt_training_poison_reason", None)


def _assert_training_state_usable(model, *, operation):
    reason = _training_poison_reason(model)
    if reason is not None:
        raise TrainingFailStop(
            f"Cannot {operation}: this live training state is poisoned by a prior "
            f"fail-stop event ({reason}). Discard it and resume from the latest "
            "successfully published checkpoint."
        )


def _poison_training_state(model, optimizer, scheduler, error):
    reason = f"{type(error).__name__}: {error}"
    cpt_model = _unwrap_cpt_model(model)
    cpt_model._cpt_training_poison_reason = reason
    optimizer._cpt_training_poison_reason = reason
    if scheduler is not None:
        scheduler._cpt_training_poison_reason = reason
    return reason


def new_run_id():
    """Return a stable identifier that binds all checkpoints from one run."""
    return uuid.uuid4().hex


def _sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _is_link_or_reparse_point(path):
    """Return whether a path can redirect access outside its apparent tree."""
    path = Path(path)
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & reparse_flag)


def _torch_save_fsync(value, path):
    """Serialize a PyTorch payload and durably flush the regular file."""
    with open(path, "wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_file(path):
    """Flush a file produced by an API that owns its file handle."""
    # Windows requires a writable descriptor for ``os.fsync`` even when the
    # file contents are already complete and are only being flushed here.
    with open(path, "r+b") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path):
    """Durably flush directory entries on platforms exposing directory fsync."""
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish_directory(source, target):
    """Atomically publish a complete checkpoint with a durability boundary."""
    source = Path(source)
    target = Path(target)
    if os.name == "nt":
        import ctypes

        movefile_write_through = 0x00000008
        move_file_ex = ctypes.windll.kernel32.MoveFileExW
        move_file_ex.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        move_file_ex.restype = ctypes.c_int
        if not move_file_ex(
            str(source),
            str(target),
            movefile_write_through,
        ):
            raise ctypes.WinError()
        return
    _fsync_directory(source)
    os.replace(source, target)
    _fsync_directory(target.parent)


def verify_checkpoint_file_hashes(checkpoint_dir, state):
    """Bind the serialized model/config files to their training-state payload."""
    checkpoint_path = Path(checkpoint_dir)
    if _is_link_or_reparse_point(checkpoint_path):
        raise RuntimeError(
            f"Checkpoint directory must not be a link or reparse point: {checkpoint_path}"
        )
    model_path = checkpoint_path / "pytorch_model.bin"
    config_path = checkpoint_path / "config.json"
    for path, label in (
        (model_path, "model state"),
        (config_path, "config"),
    ):
        if _is_link_or_reparse_point(path) or not path.is_file():
            raise RuntimeError(
                f"Checkpoint {label} must be a regular local file: {path}"
            )
    actual_model_sha256 = _sha256_file(model_path)
    actual_config_sha256 = _sha256_file(config_path)
    validate_training_state(
        state,
        expected_model_state_sha256=actual_model_sha256,
        expected_config_file_sha256=actual_config_sha256,
    )
    return {
        "model_state_sha256": actual_model_sha256,
        "config_file_sha256": actual_config_sha256,
    }


def _manifest_digest(kind, entries):
    payload = json.dumps(
        {"kind": kind, "entries": entries},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_manifest(kind, named_paths, *, preserve_order=False):
    entries = []
    seen_names = set()
    for order, (name, path) in enumerate(named_paths):
        path = Path(path)
        if name in seen_names:
            raise RuntimeError(f"duplicate {kind} manifest entry: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"{kind} manifest file does not exist: {path}")
        seen_names.add(name)
        entry = {
            "name": str(name).replace("\\", "/"),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        if preserve_order:
            entry["order"] = order
        entries.append(entry)
    if not preserve_order:
        entries.sort(key=lambda entry: entry["name"])
    if not entries:
        raise RuntimeError(f"{kind} manifest must contain at least one file")
    return {
        "kind": kind,
        "algorithm": "sha256",
        "entries": entries,
        "sha256": _manifest_digest(kind, entries),
    }


def build_code_manifest(repo_root=None):
    """Hash the code that defines native CPT training and resume semantics."""
    root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    paths = []
    model_dir = root / "model"
    for path in sorted(model_dir.rglob("*.py")):
        paths.append((path.relative_to(root).as_posix(), path))
    for relative in (
        "scripts/train.py",
        "scripts/train_utils.py",
        "scripts/resume.py",
    ):
        path = root / relative
        paths.append((relative, path))
    return _build_manifest("code", paths)


def build_data_manifest(files):
    """Hash the ordered token-shard set without binding it to an absolute path."""
    paths = [Path(path) for path in files]
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise RuntimeError("data shard filenames must be unique")
    return _build_manifest("data", zip(names, paths), preserve_order=True)


def build_tokenizer_manifest(tokenizer_dir):
    """Hash every tokenizer artifact using paths relative to its directory."""
    root = Path(tokenizer_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"tokenizer directory does not exist: {root}")
    paths = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"tokenizer manifest does not allow symlinks: {path}")
        if path.is_file():
            paths.append((path.relative_to(root).as_posix(), path))
    return _build_manifest("tokenizer", paths)


def canonical_config_hash(config):
    if not hasattr(config, "to_dict"):
        raise TypeError("model config must provide to_dict() for strict checkpointing")
    config_payload = dict(config.to_dict())
    # This changes only activation-memory/compute strategy, so strict resume
    # may safely inherit or override it without changing model identity.
    config_payload.pop("cpt_router_recompute", None)
    payload = json.dumps(
        config_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def capture_rng_state():
    """Capture Python, NumPy, CPU torch and initialized CUDA RNG streams."""
    py_version, py_values, py_gauss = random.getstate()
    np_name, np_values, np_pos, np_has_gauss, np_cached = np.random.get_state()
    cuda_states = []
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_states = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
    return {
        "python": {
            "version": int(py_version),
            "values": list(py_values),
            "gauss": py_gauss,
        },
        "numpy": {
            "bit_generator": str(np_name),
            "values": torch.from_numpy(np_values.copy()),
            "position": int(np_pos),
            "has_gauss": int(np_has_gauss),
            "cached_gaussian": float(np_cached),
        },
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": cuda_states,
    }


def restore_rng_state(rng_state):
    _validate_rng_state(rng_state)
    py_state = rng_state["python"]
    random.setstate(
        (
            py_state["version"],
            tuple(py_state["values"]),
            py_state["gauss"],
        )
    )
    np_state = rng_state["numpy"]
    np.random.set_state(
        (
            np_state["bit_generator"],
            np_state["values"].cpu().numpy().astype(np.uint32, copy=True),
            np_state["position"],
            np_state["has_gauss"],
            np_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(rng_state["torch_cpu"].cpu())
    cuda_states = rng_state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA RNG device count mismatch: checkpoint has "
                f"{len(cuda_states)}, runtime has {torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


def distributed_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def reject_uninitialized_torchrun_environment():
    """Fail before any training side effect when torchrun is not wired end to end."""
    initialized = dist.is_available() and dist.is_initialized()
    runtime_world_size = dist.get_world_size() if initialized else 1
    raw_world_size = os.environ.get("WORLD_SIZE")
    if raw_world_size is None:
        if runtime_world_size > 1:
            raise RuntimeError(
                "End-to-end DDP training entrypoint is not implemented: an "
                f"initialized process group has world size={runtime_world_size}. "
                "Launch train.py/resume.py as a single process until data "
                "sharding, rank-local device selection, DDP wrapping, and "
                "rank-safe cursors are implemented."
            )
        return 1
    if re.fullmatch(r"[1-9]\d*", raw_world_size) is None:
        raise RuntimeError(
            "WORLD_SIZE must be a canonical positive integer, got "
            f"{raw_world_size!r}"
        )
    environment_world_size = int(raw_world_size)
    if not initialized:
        if environment_world_size > 1:
            raise RuntimeError(
                "End-to-end DDP training entrypoint is not implemented: torchrun "
                f"advertises WORLD_SIZE={environment_world_size}, but no process "
                "group was initialized. Launch train.py/resume.py as a single "
                "process until data sharding, device selection, DDP wrapping, and "
                "rank-safe cursors are implemented."
            )
        return environment_world_size
    if runtime_world_size != environment_world_size:
        raise RuntimeError(
            f"WORLD_SIZE={environment_world_size} disagrees with the initialized "
            f"process-group world size={runtime_world_size}"
        )
    if runtime_world_size > 1:
        raise RuntimeError(
            "End-to-end DDP training entrypoint is not implemented: an initialized "
            f"process group has world size={runtime_world_size}. Launch "
            "train.py/resume.py as a single process until data sharding, "
            "rank-local device selection, DDP wrapping, and rank-safe cursors "
            "are implemented."
        )
    return runtime_world_size


def distributed_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process():
    return distributed_rank() == 0


def broadcast_rank0_object(value):
    if distributed_world_size() == 1:
        return value
    payload = [value if is_main_process() else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


# ============================================================
# 共享工具
# ============================================================

class BF16AdamW(torch.optim.AdamW):
    """AdamW that stores optimizer states in bfloat16 to save VRAM.

    States (exp_avg, exp_avg_sq) are kept in bf16 between steps and
    cast to fp32 only during the update computation, saving optimizer
    memory at the cost of minor precision loss. The optimizer kind is
    serialized separately because ``state_dict()`` does not encode the
    Python optimizer class.
    """

    def __init__(self, *args, **kwargs):
        unsupported_flags = {
            "amsgrad": False,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
        }
        for option, supported_value in unsupported_flags.items():
            value = kwargs.get(option, supported_value)
            if value is not supported_value:
                raise ValueError(
                    f"BF16AdamW supports only {option}={supported_value!r}, "
                    f"got {value!r}"
                )
            kwargs[option] = supported_value
        for option in ("foreach", "fused"):
            value = kwargs.get(option)
            if value not in (None, False):
                raise ValueError(f"BF16AdamW does not support {option}={value!r}")
            # Keep the serialized param-group schema aligned with strict
            # non-fused AdamW. This custom step never uses either fast path.
            kwargs[option] = None
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
                    state["step"] = torch.zeros((), dtype=torch.float32)
                    # Ordinary model parameters are BF16, while CPT projection,
                    # anchor, energy and persistent-price parameters remain FP32.
                    # Preserve that mixed-precision boundary in optimizer state.
                    state["exp_avg"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )

                state["step"].add_(1)
                t = int(state["step"].item())

                exp_avg = state["exp_avg"].float()
                exp_avg_sq = state["exp_avg_sq"].float()

                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(exp_avg, denom, value=-step_size)

                state["exp_avg"] = exp_avg.to(p.dtype)
                state["exp_avg_sq"] = exp_avg_sq.to(p.dtype)

        return loss


def _validate_rng_state(rng_state):
    if not isinstance(rng_state, dict):
        raise RuntimeError("rng_state must be a dictionary")
    if set(rng_state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise RuntimeError("rng_state has an unsupported schema")

    py_state = rng_state["python"]
    if not isinstance(py_state, dict) or set(py_state) != {
        "version", "values", "gauss",
    }:
        raise RuntimeError("Python RNG state has an unsupported schema")
    if not isinstance(py_state["version"], int):
        raise RuntimeError("Python RNG version must be an integer")
    if not isinstance(py_state["values"], list) or not py_state["values"]:
        raise RuntimeError("Python RNG values must be a non-empty list")
    if not all(isinstance(value, int) for value in py_state["values"]):
        raise RuntimeError("Python RNG values must be integers")
    if py_state["gauss"] is not None and not math.isfinite(py_state["gauss"]):
        raise RuntimeError("Python RNG Gaussian cache must be finite")

    np_state = rng_state["numpy"]
    if not isinstance(np_state, dict) or set(np_state) != {
        "bit_generator", "values", "position", "has_gauss", "cached_gaussian",
    }:
        raise RuntimeError("NumPy RNG state has an unsupported schema")
    if not isinstance(np_state["bit_generator"], str):
        raise RuntimeError("NumPy RNG bit-generator name must be a string")
    np_values = np_state["values"]
    if not isinstance(np_values, torch.Tensor) or np_values.ndim != 1:
        raise RuntimeError("NumPy RNG values must be a one-dimensional tensor")
    if np_values.dtype != torch.uint32:
        raise RuntimeError("NumPy RNG values must use torch.uint32")
    if not isinstance(np_state["position"], int) or np_state["position"] < 0:
        raise RuntimeError("NumPy RNG position must be a non-negative integer")
    if np_state["has_gauss"] not in (0, 1):
        raise RuntimeError("NumPy RNG has_gauss must be 0 or 1")
    if not math.isfinite(np_state["cached_gaussian"]):
        raise RuntimeError("NumPy RNG Gaussian cache must be finite")

    cpu_state = rng_state["torch_cpu"]
    if (
        not isinstance(cpu_state, torch.Tensor)
        or cpu_state.dtype != torch.uint8
        or cpu_state.ndim != 1
    ):
        raise RuntimeError("torch CPU RNG state must be a one-dimensional uint8 tensor")
    cuda_states = rng_state["torch_cuda"]
    if not isinstance(cuda_states, list):
        raise RuntimeError("torch CUDA RNG states must be a list")
    for state in cuda_states:
        if (
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 1
        ):
            raise RuntimeError(
                "every torch CUDA RNG state must be a one-dimensional uint8 tensor"
            )


def _validate_rank_rng_states(rng_states, rng_world_size):
    if isinstance(rng_world_size, bool) or not isinstance(rng_world_size, int):
        raise RuntimeError("rng_world_size must be an integer")
    if rng_world_size <= 0:
        raise RuntimeError("rng_world_size must be positive")
    if not isinstance(rng_states, list):
        raise RuntimeError("rng_states must be a list")
    if len(rng_states) != rng_world_size:
        raise RuntimeError(
            f"rng_states contains {len(rng_states)} entries, "
            f"expected rng_world_size={rng_world_size}"
        )

    ranks = []
    for entry in rng_states:
        if not isinstance(entry, dict) or set(entry) != {"rank", "state"}:
            raise RuntimeError("every rng_states entry must contain only rank and state")
        rank = entry["rank"]
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise RuntimeError("RNG rank must be an integer")
        if rank < 0 or rank >= rng_world_size:
            raise RuntimeError(
                f"RNG rank {rank} is outside world size {rng_world_size}"
            )
        ranks.append(rank)
        _validate_rng_state(entry["state"])

    if len(ranks) != len(set(ranks)):
        raise RuntimeError(f"rng_states contains duplicate ranks: {ranks}")
    expected_ranks = list(range(rng_world_size))
    if ranks != expected_ranks:
        raise RuntimeError(
            f"rng_states ranks must be complete and ordered as {expected_ranks}; "
            f"got {ranks}"
        )
    return rng_states


def _rng_state_to_wire(rng_state):
    """Convert RNG tensors to pure Python for portable object collectives."""
    _validate_rng_state(rng_state)
    return {
        "python": copy.deepcopy(rng_state["python"]),
        "numpy": {
            "bit_generator": rng_state["numpy"]["bit_generator"],
            "values": rng_state["numpy"]["values"].cpu().tolist(),
            "position": rng_state["numpy"]["position"],
            "has_gauss": rng_state["numpy"]["has_gauss"],
            "cached_gaussian": rng_state["numpy"]["cached_gaussian"],
        },
        "torch_cpu": rng_state["torch_cpu"].cpu().tolist(),
        "torch_cuda": [state.cpu().tolist() for state in rng_state["torch_cuda"]],
    }


def _rng_state_from_wire(wire_state):
    if not isinstance(wire_state, dict) or set(wire_state) != {
        "python", "numpy", "torch_cpu", "torch_cuda",
    }:
        raise RuntimeError("Distributed RNG wire state has an unsupported schema")
    numpy_state = wire_state["numpy"]
    if not isinstance(numpy_state, dict) or set(numpy_state) != {
        "bit_generator", "values", "position", "has_gauss", "cached_gaussian",
    }:
        raise RuntimeError("Distributed NumPy RNG wire state is invalid")
    if not isinstance(wire_state["torch_cpu"], list):
        raise RuntimeError("Distributed torch CPU RNG wire state must be a list")
    if not isinstance(wire_state["torch_cuda"], list) or not all(
        isinstance(state, list) for state in wire_state["torch_cuda"]
    ):
        raise RuntimeError("Distributed torch CUDA RNG wire states must be lists")
    try:
        rng_state = {
            "python": copy.deepcopy(wire_state["python"]),
            "numpy": {
                "bit_generator": numpy_state["bit_generator"],
                "values": torch.tensor(numpy_state["values"], dtype=torch.uint32),
                "position": numpy_state["position"],
                "has_gauss": numpy_state["has_gauss"],
                "cached_gaussian": numpy_state["cached_gaussian"],
            },
            "torch_cpu": torch.tensor(wire_state["torch_cpu"], dtype=torch.uint8),
            "torch_cuda": [
                torch.tensor(state, dtype=torch.uint8)
                for state in wire_state["torch_cuda"]
            ],
        }
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError("Distributed RNG wire state contains invalid values") from exc
    _validate_rng_state(rng_state)
    return rng_state


def restore_rank_rng_state(training_state, rank=None, world_size=None):
    """Restore the RNG stream owned by one rank from a strict checkpoint."""
    if not isinstance(training_state, dict):
        raise RuntimeError("training state must be a dictionary")
    if "rng_world_size" not in training_state or "rng_states" not in training_state:
        raise RuntimeError("training state is missing distributed RNG metadata")
    runtime_world_size = distributed_world_size() if world_size is None else world_size
    runtime_rank = distributed_rank() if rank is None else rank
    _validate_rank_rng_states(
        training_state["rng_states"],
        training_state["rng_world_size"],
    )
    if runtime_world_size != training_state["rng_world_size"]:
        raise RuntimeError(
            f"Runtime world_size={runtime_world_size} disagrees with checkpoint "
            f"rng_world_size={training_state['rng_world_size']}"
        )
    if (
        isinstance(runtime_rank, bool)
        or not isinstance(runtime_rank, int)
        or runtime_rank < 0
        or runtime_rank >= runtime_world_size
    ):
        raise RuntimeError(
            f"Runtime rank={runtime_rank!r} is invalid for world_size={runtime_world_size}"
        )
    restore_rng_state(training_state["rng_states"][runtime_rank]["state"])


def _validate_manifest(manifest, expected_kind):
    if not isinstance(manifest, dict):
        raise RuntimeError(f"{expected_kind} manifest must be a dictionary")
    required = {"kind", "algorithm", "entries", "sha256"}
    if set(manifest) != required:
        raise RuntimeError(f"{expected_kind} manifest has an unsupported schema")
    if manifest["kind"] != expected_kind or manifest["algorithm"] != "sha256":
        raise RuntimeError(f"invalid {expected_kind} manifest identity")
    entries = manifest["entries"]
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"{expected_kind} manifest entries must be non-empty")
    names = []
    entry_fields = {"name", "size", "sha256"}
    if expected_kind == "data":
        entry_fields.add("order")
    for expected_order, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != entry_fields:
            raise RuntimeError(f"invalid {expected_kind} manifest entry")
        name = entry["name"]
        size = entry["size"]
        digest = entry["sha256"]
        if not isinstance(name, str) or not name or "\\" in name:
            raise RuntimeError(f"invalid {expected_kind} manifest filename")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise RuntimeError(f"invalid {expected_kind} manifest file size")
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise RuntimeError(f"invalid {expected_kind} manifest file hash")
        if expected_kind == "data":
            order = entry["order"]
            if (
                isinstance(order, bool)
                or not isinstance(order, int)
                or order != expected_order
            ):
                raise RuntimeError(
                    "data manifest order fields must be contiguous and match "
                    "the stored shard sequence"
                )
        names.append(name)
    if len(names) != len(set(names)):
        raise RuntimeError(f"{expected_kind} manifest names must be unique")
    if expected_kind != "data" and names != sorted(names):
        raise RuntimeError(f"{expected_kind} manifest names must be sorted")
    if manifest["sha256"] != _manifest_digest(expected_kind, entries):
        raise RuntimeError(f"{expected_kind} manifest aggregate hash is invalid")


_FOREACH_NORM_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def _append_nested_numeric_checks(
    checks,
    value,
    path,
    *,
    include_tensors=True,
    include_host=True,
):
    """Append numeric leaves in the same diagnostic order as the legacy walker."""
    if isinstance(value, torch.Tensor):
        if include_tensors and (value.is_floating_point() or value.is_complex()):
            checks.append((value, f"Non-finite numeric state at {path}"))
        return
    if isinstance(value, np.ndarray):
        if include_host and np.issubdtype(value.dtype, np.inexact):
            checks.append((value, f"Non-finite numeric state at {path}"))
        return
    if isinstance(value, (float, np.floating)):
        if include_host:
            checks.append((value, f"Non-finite numeric state at {path}"))
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            _append_nested_numeric_checks(
                checks,
                nested,
                f"{path}.{key}",
                include_tensors=include_tensors,
                include_host=include_host,
            )
        return
    if isinstance(value, (tuple, list)):
        for index, nested in enumerate(value):
            _append_nested_numeric_checks(
                checks,
                nested,
                f"{path}[{index}]",
                include_tensors=include_tensors,
                include_host=include_host,
            )


def _numeric_leaf_is_finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value.detach()).all().item())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    return math.isfinite(float(value))


def _fast_numeric_checks_are_finite(checks):
    """Check all tensor leaves with at most one host synchronization per device.

    CUDA strided tensors use an infinity-norm foreach reduction.  Empty tensors
    are skipped because ``torch._foreach_norm(..., inf)`` has no empty-tensor
    identity.  Complex tensors are viewed as their real components so finite
    real/imaginary parts near the dtype limit cannot overflow through a complex
    magnitude and create a false positive.  Unsupported layouts and dtypes use
    the generic isfinite reduction while sharing the same final device flag.
    """
    tensors_by_device = {}
    seen_tensors = set()
    host_finite = True
    for value, _ in checks:
        if isinstance(value, torch.Tensor):
            tensor_id = id(value)
            if tensor_id in seen_tensors:
                continue
            seen_tensors.add(tensor_id)
            if value.numel() == 0:
                continue
            tensors_by_device.setdefault(value.device, []).append(value.detach())
        elif isinstance(value, np.ndarray):
            host_finite = host_finite and bool(np.isfinite(value).all())
        else:
            host_finite = host_finite and math.isfinite(float(value))

    device_finite = []
    for device, tensors in tensors_by_device.items():
        aggregate = torch.ones((), dtype=torch.bool, device=device)
        generic = []
        foreach_inputs = []
        if device.type == "cuda":
            for tensor in tensors:
                candidate = tensor
                if candidate.is_complex():
                    candidate = torch.view_as_real(candidate.resolve_conj())
                if (
                    type(candidate) is torch.Tensor
                    and candidate.layout == torch.strided
                    and candidate.dtype in _FOREACH_NORM_DTYPES
                ):
                    foreach_inputs.append(candidate)
                else:
                    generic.append(tensor)
        else:
            generic = tensors

        if foreach_inputs:
            try:
                norms = torch._foreach_norm(foreach_inputs, float("inf"))
            except (RuntimeError, TypeError):
                generic.extend(foreach_inputs)
            else:
                aggregate.logical_and_(torch.isfinite(torch.stack(norms)).all())
        for tensor in generic:
            aggregate.logical_and_(torch.isfinite(tensor).all())
        device_finite.append(aggregate)

    # Calling item() only here is the synchronization boundary: once per
    # represented device on the normal path, independent of tensor count.
    tensors_finite = True
    for aggregate in device_finite:
        tensors_finite = bool(aggregate.item()) and tensors_finite
    return host_finite and tensors_finite


def _assert_numeric_checks_finite(checks):
    if _fast_numeric_checks_are_finite(checks):
        return
    # The aggregate gate deliberately carries no paths.  Only the exceptional
    # path pays for per-leaf synchronization so the historical first-error text
    # and traversal order remain unchanged.
    for value, message in checks:
        if not _numeric_leaf_is_finite(value):
            raise FloatingPointError(message)
    raise RuntimeError(
        "Aggregated finite check failed but slow diagnostics found no non-finite value"
    )


def _assert_nested_finite(value, path):
    checks = []
    _append_nested_numeric_checks(checks, value, path)
    _assert_numeric_checks_finite(checks)


def _append_model_optimizer_numeric_checks(checks, model, optimizer):
    cpt_model = _unwrap_cpt_model(model)
    for name, parameter in cpt_model.named_parameters():
        if parameter.is_floating_point() or parameter.is_complex():
            checks.append(
                (
                    parameter,
                    f"Non-finite model parameter after optimizer: {name}",
                )
            )
    for name, buffer in cpt_model.named_buffers():
        if buffer.is_floating_point() or buffer.is_complex():
            checks.append(
                (buffer, f"Non-finite model buffer after optimizer: {name}")
            )
    _append_nested_numeric_checks(checks, optimizer.state_dict(), "optimizer")


def _validate_training_numerics(model, optimizer, scheduler):
    checks = []
    _append_model_optimizer_numeric_checks(checks, model, optimizer)
    if scheduler is not None:
        if not hasattr(scheduler, "state_dict") or not hasattr(scheduler, "load_state_dict"):
            _assert_numeric_checks_finite(checks)
            raise TypeError("scheduler must expose state_dict/load_state_dict")
        try:
            scheduler_state = scheduler.state_dict()
        except BaseException:
            _assert_numeric_checks_finite(checks)
            raise
        _append_nested_numeric_checks(checks, scheduler_state, "scheduler")
        if hasattr(scheduler, "get_last_lr"):
            try:
                scheduler_last_lr = scheduler.get_last_lr()
            except BaseException:
                _assert_numeric_checks_finite(checks)
                raise
            _append_nested_numeric_checks(
                checks,
                scheduler_last_lr,
                "scheduler.last_lr",
            )
    for group_index, group in enumerate(optimizer.param_groups):
        _append_nested_numeric_checks(
            checks,
            group.get("lr"),
            f"optimizer.param_groups[{group_index}].lr",
        )
    _assert_numeric_checks_finite(checks)


def _iter_nested_numeric_tensors(value, path):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            yield path, value
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            yield from _iter_nested_numeric_tensors(nested, f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, nested in enumerate(value):
            yield from _iter_nested_numeric_tensors(nested, f"{path}[{index}]")


def _tensor_mutation_token(path, tensor):
    try:
        version = int(tensor._version)
    except RuntimeError:
        version = None
    return (
        path,
        id(tensor),
        version,
        tensor.device,
        tensor.dtype,
        tensor.layout,
        tuple(tensor.shape),
    )


def _capture_non_scheduler_tensor_guard(model, optimizer):
    """Capture cheap version tokens for state a scheduler must not modify."""
    cpt_model = _unwrap_cpt_model(model)
    tokens = []
    reliable = True
    for name, parameter in cpt_model.named_parameters():
        if parameter.is_floating_point() or parameter.is_complex():
            token = _tensor_mutation_token(f"model.parameter.{name}", parameter)
            reliable = reliable and token[2] is not None
            tokens.append(token)
    for name, buffer in cpt_model.named_buffers():
        if buffer.is_floating_point() or buffer.is_complex():
            token = _tensor_mutation_token(f"model.buffer.{name}", buffer)
            reliable = reliable and token[2] is not None
            tokens.append(token)
    for state_index, state in enumerate(optimizer.state.values()):
        for path, tensor in _iter_nested_numeric_tensors(
            state,
            f"optimizer.state[{state_index}]",
        ):
            token = _tensor_mutation_token(path, tensor)
            reliable = reliable and token[2] is not None
            tokens.append(token)
    return tuple(tokens) if reliable else None


def _validate_post_scheduler_numerics(model, optimizer, scheduler, tensor_guard):
    """Validate scheduler-owned state without rescanning unchanged model state."""
    current_guard = _capture_non_scheduler_tensor_guard(model, optimizer)
    if tensor_guard is None or current_guard != tensor_guard:
        # A scheduler stepped outside its normal mutation domain.  Preserve the
        # old full-check semantics on this exceptional path.
        _validate_training_numerics(model, optimizer, scheduler)
        return

    checks = []
    optimizer_state = optimizer.state_dict()
    # Tensor optimizer state is covered by the unchanged version guard.  Host
    # numeric leaves still need checking because they have no version counter.
    _append_nested_numeric_checks(
        checks,
        optimizer_state,
        "optimizer",
        include_tensors=False,
    )
    # Param-group values are the optimizer fields schedulers are expected to
    # change, including the uncommon tensor-valued learning-rate case.
    _append_nested_numeric_checks(
        checks,
        optimizer_state.get("param_groups", ()),
        "optimizer.param_groups",
        include_host=False,
    )
    if not hasattr(scheduler, "state_dict") or not hasattr(scheduler, "load_state_dict"):
        _assert_numeric_checks_finite(checks)
        raise TypeError("scheduler must expose state_dict/load_state_dict")
    scheduler_state = scheduler.state_dict()
    _append_nested_numeric_checks(checks, scheduler_state, "scheduler")
    if hasattr(scheduler, "get_last_lr"):
        _append_nested_numeric_checks(
            checks,
            scheduler.get_last_lr(),
            "scheduler.last_lr",
        )
    for group_index, group in enumerate(optimizer.param_groups):
        _append_nested_numeric_checks(
            checks,
            group.get("lr"),
            f"optimizer.param_groups[{group_index}].lr",
        )
    _assert_numeric_checks_finite(checks)


def _clone_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _clone_to_cpu(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(nested) for nested in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(nested) for nested in value)
    return copy.deepcopy(value)


def _capture_training_snapshot(model, optimizer, scheduler):
    """Capture the complete applied=0 step-entry state on CPU.

    This intentionally trades host-memory bandwidth for the PDF's strict rule
    that theta, lambda, optimizer, scheduler and step state do not advance when
    an optimizer attempt fails after a partial in-place write.
    """
    _validate_training_numerics(model, optimizer, scheduler)
    cpt_model = _unwrap_cpt_model(model)
    return {
        "model": _clone_to_cpu(cpt_model.state_dict()),
        "optimizer": _clone_to_cpu(optimizer.state_dict()),
        "scheduler": (
            _clone_to_cpu(scheduler.state_dict()) if scheduler is not None else None
        ),
        "rng": capture_rng_state(),
    }


def _restore_training_snapshot(model, optimizer, scheduler, snapshot):
    errors = []
    cpt_model = _unwrap_cpt_model(model)
    try:
        cpt_model.load_state_dict(snapshot["model"], strict=True)
    except BaseException as exc:
        errors.append(f"model rollback failed: {exc}")
    try:
        optimizer.load_state_dict(snapshot["optimizer"])
    except BaseException as exc:
        errors.append(f"optimizer rollback failed: {exc}")
    if scheduler is not None:
        try:
            scheduler.load_state_dict(snapshot["scheduler"])
        except BaseException as exc:
            errors.append(f"scheduler rollback failed: {exc}")
    try:
        restore_rng_state(snapshot["rng"])
    except BaseException as exc:
        errors.append(f"RNG rollback failed: {exc}")
    if errors:
        raise RuntimeError("; ".join(errors))
    _validate_training_numerics(model, optimizer, scheduler)


def capture_training_step_entry(model, optimizer, scheduler=None):
    """Capture a pre-forward snapshot for exact replay of an applied=0 step."""
    return _capture_training_snapshot(model, optimizer, scheduler)


def restore_training_step_entry(model, optimizer, scheduler, snapshot):
    """Restore a previously captured pre-forward step-entry snapshot."""
    _restore_training_snapshot(model, optimizer, scheduler, snapshot)


def _distributed_enabled():
    return distributed_world_size() > 1


def _consensus_device(model):
    if not _distributed_enabled():
        return torch.device("cpu")
    backend = str(dist.get_backend()).lower()
    if "nccl" in backend:
        return next(model.parameters()).device
    return torch.device("cpu")


def _run_consensus_phase(model, phase_name, operation):
    """Run one local phase, then make every rank agree before progressing."""
    local_error = None
    result = None
    try:
        result = operation()
    except BaseException as exc:
        local_error = exc

    if _distributed_enabled():
        success = torch.tensor(
            0 if local_error is not None else 1,
            device=_consensus_device(model),
            dtype=torch.int32,
        )
        dist.all_reduce(success, op=dist.ReduceOp.MIN)
        globally_successful = bool(success.item())
    else:
        globally_successful = local_error is None

    if not globally_successful:
        if local_error is not None:
            raise local_error
        raise RuntimeError(
            f"Training phase {phase_name!r} failed on another distributed rank"
        )
    if local_error is not None:
        raise local_error
    return result


def _run_rank0_coordinated(model, phase_name, operation):
    """Run a side effect only on rank 0 and publish one result to all ranks."""
    if not _distributed_enabled():
        return operation()

    payload = [None]
    result = None
    if distributed_rank() == 0:
        try:
            result = operation()
            payload[0] = {"ok": True, "error_type": None, "message": None}
        except BaseException as exc:
            payload[0] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
    dist.broadcast_object_list(payload, src=0)
    outcome = payload[0]
    if not isinstance(outcome, dict) or set(outcome) != {
        "ok", "error_type", "message",
    }:
        raise RuntimeError(f"Invalid rank-0 outcome for phase {phase_name!r}")
    if not outcome["ok"]:
        raise RuntimeError(
            f"Rank-0 phase {phase_name!r} failed with "
            f"{outcome['error_type']}: {outcome['message']}"
        )
    dist.barrier()
    return result


def _gather_checkpoint_rng_states(model, identity):
    rank = distributed_rank()
    world_size = distributed_world_size()
    local_state = _run_consensus_phase(
        model,
        "checkpoint_rng_capture",
        capture_rng_state,
    )
    local_payload = {
        "rank": rank,
        "state": _rng_state_to_wire(local_state),
        "identity": identity,
    }
    if world_size == 1:
        gathered = [local_payload]
    else:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_payload)

    def validate_gathered():
        if not isinstance(gathered, list) or len(gathered) != world_size:
            raise RuntimeError("Distributed RNG gather returned an invalid payload count")
        rng_states = []
        identities = []
        for expected_rank, payload in enumerate(gathered):
            if not isinstance(payload, dict) or set(payload) != {
                "rank", "state", "identity",
            }:
                raise RuntimeError("Distributed RNG gather returned an invalid payload")
            if payload["rank"] != expected_rank:
                raise RuntimeError(
                    f"Distributed RNG payload rank={payload['rank']} appeared at "
                    f"position {expected_rank}"
                )
            rng_states.append(
                {
                    "rank": expected_rank,
                    "state": _rng_state_from_wire(payload["state"]),
                }
            )
            identities.append(payload["identity"])
        if any(item != identities[0] for item in identities[1:]):
            raise RuntimeError("Checkpoint identity disagrees across distributed ranks")
        _validate_rank_rng_states(rng_states, world_size)
        return rng_states

    return _run_consensus_phase(
        model,
        "checkpoint_rng_validation",
        validate_gathered,
    )


def _broadcast_rank0_flag(model, value):
    if not _distributed_enabled():
        return bool(value)
    flag = torch.tensor(
        1 if (distributed_rank() == 0 and value) else 0,
        device=_consensus_device(model),
        dtype=torch.int32,
    )
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def _synchronize_stop_signal(model, stop_signal):
    if not _distributed_enabled():
        return stop_signal
    signal_value = torch.tensor(
        int(stop_signal or 0),
        device=_consensus_device(model),
        dtype=torch.int32,
    )
    dist.all_reduce(signal_value, op=dist.ReduceOp.MAX)
    synchronized = int(signal_value.item())
    return synchronized if synchronized else None


def _unwrap_cpt_model(model):
    """Return the module that owns the CPT transaction API."""
    required = (
        "validate_cpt_transaction",
        "commit_cpt_transaction",
        "abort_cpt_transaction",
        "get_cpt_state_version",
    )
    if all(hasattr(model, name) for name in required):
        return model
    module = getattr(model, "module", None)
    if module is not None and all(hasattr(module, name) for name in required):
        return module
    missing = [name for name in required if not hasattr(model, name)]
    raise TypeError(
        "Model does not expose the required CPT transaction API: "
        + ", ".join(missing)
    )


def _digest_cpt_learnable_state(router, layer_index):
    digest = hashlib.sha256()
    seen_names = set()
    parameter_count = 0
    named_parameters = getattr(router, "named_parameters", None)
    if not callable(named_parameters):
        raise RuntimeError(
            f"CPT router layer {layer_index} does not expose named_parameters()"
        )
    for name, parameter in named_parameters(recurse=True):
        if not isinstance(name, str) or not name or name in seen_names:
            raise RuntimeError(
                f"CPT router layer {layer_index} has an invalid parameter name {name!r}"
            )
        if not isinstance(parameter, torch.Tensor):
            raise RuntimeError(
                f"CPT router layer {layer_index} parameter {name!r} is not a tensor"
            )
        if parameter.is_meta:
            raise RuntimeError(
                f"CPT router layer {layer_index} parameter {name!r} is still meta"
            )
        if parameter.layout != torch.strided:
            raise RuntimeError(
                f"CPT router layer {layer_index} parameter {name!r} must be strided"
            )
        if parameter.dtype != torch.float32:
            raise RuntimeError(
                f"CPT router layer {layer_index} parameter {name!r} must remain FP32"
            )
        if parameter.grad is not None and parameter.grad.dtype != torch.float32:
            raise RuntimeError(
                f"CPT router layer {layer_index} parameter {name!r} gradient "
                "must remain FP32"
            )
        detached = parameter.detach()
        if not bool(torch.isfinite(detached).all().item()):
            raise FloatingPointError(
                f"CPT router layer {layer_index} parameter {name!r} is non-finite"
            )
        metadata = json.dumps(
            {
                "name": name,
                "dtype": str(detached.dtype),
                "shape": list(detached.shape),
                "requires_grad": bool(parameter.requires_grad),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        raw = (
            detached.contiguous()
            .reshape(-1)
            .cpu()
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        digest.update(len(metadata).to_bytes(8, byteorder="little"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, byteorder="little"))
        digest.update(raw)
        seen_names.add(name)
        parameter_count += 1
    if parameter_count == 0:
        raise RuntimeError(f"CPT router layer {layer_index} has no learnable state")
    return digest.digest()


def _extract_local_cpt_consistency_state(model):
    cpt_model = _unwrap_cpt_model(model)
    routers_fn = getattr(cpt_model, "_cpt_routers", None)
    if not callable(routers_fn):
        raise RuntimeError("CPT model does not expose _cpt_routers()")
    routers = tuple(routers_fn())
    if not routers:
        raise RuntimeError("CPT model must contain at least one router")

    expert_counts = []
    versions = []
    prices = []
    parameter_digests = []
    for layer_index, router in enumerate(routers):
        num_experts = getattr(router, "num_experts", None)
        if isinstance(num_experts, bool) or not isinstance(num_experts, int):
            raise RuntimeError(
                f"CPT router layer {layer_index} num_experts must be an integer"
            )
        if num_experts <= 0:
            raise RuntimeError(
                f"CPT router layer {layer_index} num_experts must be positive"
            )
        price = getattr(router, "congestion_price", None)
        if not isinstance(price, torch.Tensor):
            raise RuntimeError(
                f"CPT router layer {layer_index} congestion_price is not a tensor"
            )
        if price.is_meta:
            raise RuntimeError(
                f"CPT router layer {layer_index} congestion_price is still meta"
            )
        if price.dtype != torch.float32 or price.shape != (num_experts,):
            raise RuntimeError(
                f"CPT router layer {layer_index} congestion_price must be FP32 "
                f"with shape ({num_experts},)"
            )
        if price.requires_grad:
            raise RuntimeError(
                f"CPT router layer {layer_index} congestion_price must be detached"
            )
        detached_price = price.detach()
        if not bool(torch.isfinite(detached_price).all().item()):
            raise FloatingPointError(
                f"CPT router layer {layer_index} congestion_price is non-finite"
            )
        if bool((detached_price < 0).any().item()):
            raise RuntimeError(
                f"CPT router layer {layer_index} congestion_price must be non-negative"
            )

        version = getattr(router, "state_version", None)
        if not isinstance(version, torch.Tensor):
            raise RuntimeError(
                f"CPT router layer {layer_index} state_version is not a tensor"
            )
        if (
            version.is_meta
            or version.dtype != torch.int64
            or version.ndim != 0
            or version.requires_grad
        ):
            raise RuntimeError(
                f"CPT router layer {layer_index} state_version must be a detached "
                "materialized int64 scalar"
            )
        version_value = int(version.detach().item())
        if version_value < 0:
            raise RuntimeError(
                f"CPT router layer {layer_index} state_version must be non-negative"
            )

        expert_counts.append(num_experts)
        versions.append(version_value)
        prices.append(detached_price.float().cpu().clone())
        parameter_digests.append(
            _digest_cpt_learnable_state(router, layer_index)
        )
    return {
        "layer_count": len(routers),
        "expert_counts": expert_counts,
        "versions": versions,
        "prices": prices,
        "parameter_digests": parameter_digests,
    }


def _distributed_min_max(values):
    flattened = values.reshape(-1)
    if flattened.numel() == 0:
        raise RuntimeError("distributed consistency payload must not be empty")
    combined = torch.cat((flattened, -flattened))
    dist.all_reduce(combined, op=dist.ReduceOp.MAX)
    size = flattened.numel()
    maximum = combined[:size].reshape(values.shape)
    minimum = (-combined[size:]).reshape(values.shape)
    return minimum, maximum


def _validate_distributed_cpt_consistency(model, *, context):
    """Require identical active CPT state on every rank before progressing."""
    if not _distributed_enabled():
        return
    device = _consensus_device(model)
    local_error = None
    local_state = None
    try:
        local_state = _extract_local_cpt_consistency_state(model)
    except BaseException as exc:
        local_error = exc

    locally_valid = torch.tensor(
        0 if local_error is not None else 1,
        device=device,
        dtype=torch.int32,
    )
    dist.all_reduce(locally_valid, op=dist.ReduceOp.MIN)
    if not bool(locally_valid.item()):
        if local_error is not None:
            raise RuntimeError(
                f"Invalid local CPT state during {context}: {local_error}"
            ) from local_error
        raise RuntimeError(
            f"Invalid CPT state on another distributed rank during {context}"
        )

    layer_count = torch.tensor(
        [local_state["layer_count"]],
        device=device,
        dtype=torch.int64,
    )
    minimum, maximum = _distributed_min_max(layer_count)
    if not torch.equal(minimum, maximum):
        raise RuntimeError(
            f"CPT layer count disagrees across distributed ranks during {context}"
        )

    expert_counts = torch.tensor(
        local_state["expert_counts"],
        device=device,
        dtype=torch.int64,
    )
    versions = torch.tensor(
        local_state["versions"],
        device=device,
        dtype=torch.int64,
    )
    digest_values = torch.tensor(
        [list(item) for item in local_state["parameter_digests"]],
        device=device,
        dtype=torch.int64,
    )
    integer_payload = torch.cat(
        (expert_counts, versions, digest_values.reshape(-1))
    )
    minimum, maximum = _distributed_min_max(integer_payload)
    layer_count_value = local_state["layer_count"]
    expert_minimum = minimum[:layer_count_value]
    expert_maximum = maximum[:layer_count_value]
    version_start = layer_count_value
    version_end = 2 * layer_count_value
    version_minimum = minimum[version_start:version_end]
    version_maximum = maximum[version_start:version_end]
    digest_minimum = minimum[version_end:].reshape(layer_count_value, 32)
    digest_maximum = maximum[version_end:].reshape(layer_count_value, 32)
    if not torch.equal(expert_minimum, expert_maximum):
        layers = torch.nonzero(
            expert_minimum != expert_maximum
        ).flatten().cpu().tolist()
        raise RuntimeError(
            "CPT expert counts disagree across distributed ranks during "
            f"{context}; layers={layers}"
        )
    if not torch.equal(version_minimum, version_maximum):
        layers = torch.nonzero(
            version_minimum != version_maximum
        ).flatten().cpu().tolist()
        raise RuntimeError(
            "CPT state_version disagrees across distributed ranks during "
            f"{context}; layers={layers}"
        )
    if not torch.equal(digest_minimum, digest_maximum):
        layers = torch.nonzero(
            torch.any(digest_minimum != digest_maximum, dim=1)
        ).flatten().cpu().tolist()
        raise RuntimeError(
            "CPT learnable state digest disagrees across distributed ranks during "
            f"{context}; layers={layers}"
        )

    prices = torch.cat(local_state["prices"]).to(device=device, dtype=torch.float32)
    minimum, maximum = _distributed_min_max(prices)
    if not torch.equal(minimum, maximum):
        flat_indexes = torch.nonzero(minimum != maximum).flatten().cpu().tolist()
        raise RuntimeError(
            "CPT congestion_price disagrees across distributed ranks during "
            f"{context}; flattened_indexes={flat_indexes[:16]}"
        )


def get_cpt_state_version(model):
    """Read and normalize the model-wide committed CPT state version."""
    value = _unwrap_cpt_model(model).get_cpt_state_version()
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError("CPT state version must be a scalar")
        value = value.detach().item()
    elif isinstance(value, (tuple, list)):
        if not value:
            raise RuntimeError("CPT state version list must not be empty")
        normalized = []
        for item in value:
            if isinstance(item, torch.Tensor):
                if item.numel() != 1:
                    raise RuntimeError("Every CPT layer version must be a scalar")
                item = item.detach().item()
            normalized.append(int(item))
        if len(set(normalized)) != 1:
            raise RuntimeError(f"CPT layer versions disagree: {normalized}")
        value = normalized[0]
    if isinstance(value, bool):
        raise RuntimeError("CPT state version must be an integer, not bool")
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid CPT state version: {value!r}") from exc
    if version < 0:
        raise RuntimeError(f"CPT state version must be non-negative, got {version}")
    return version


def _abort_training_step(model, transaction, optimizer, original_error):
    """Abort a pre-optimizer attempt and discard its gradients/proposal."""
    cleanup_errors = []
    if transaction is not None:
        try:
            _unwrap_cpt_model(model).abort_cpt_transaction(transaction)
        except BaseException as exc:  # preserve KeyboardInterrupt/SystemExit context too
            cleanup_errors.append(f"CPT abort failed: {exc}")
    try:
        optimizer.zero_grad(set_to_none=True)
    except BaseException as exc:
        cleanup_errors.append(f"zero_grad failed: {exc}")
    if cleanup_errors:
        raise RuntimeError("; ".join(cleanup_errors)) from original_error


def execute_training_step(
    model,
    output,
    optimizer,
    scheduler=None,
    max_grad_norm=1.0,
    step_entry_snapshot=None,
    failure_policy="exact-rollback",
):
    """Apply one strict all-or-none CPT optimizer transaction.

    The caller owns data/step counters and may advance them only after this
    function returns successfully.  Forward is already complete; its detached
    proposal is supplied through ``output['cpt_transaction']``.
    Production callers should pass a snapshot captured before forward so an
    applied=0 attempt restores the exact pre-forward RNG stream as well.  When
    no snapshot is supplied, this helper captures a parameter/state fallback
    immediately before ``optimizer.step()``; that fallback cannot undo RNG
    consumed by the already-completed forward.
    """
    failure_policy = _normalize_failure_policy(failure_policy)
    _assert_training_state_usable(model, operation="execute another training step")
    transaction = output.get("cpt_transaction") if isinstance(output, dict) else None
    snapshot = step_entry_snapshot
    try:
        def local_preflight():
            if not isinstance(output, dict):
                raise TypeError("Training model output must be a dict")
            if "cpt_transaction" not in output:
                raise RuntimeError("Training output is missing 'cpt_transaction'")
            local_transaction = output["cpt_transaction"]
            if local_transaction is None:
                raise RuntimeError("Training output contains an empty CPT transaction")
            local_loss = output.get("loss")
            if not isinstance(local_loss, torch.Tensor) or local_loss.numel() != 1:
                raise RuntimeError("Training loss must be a scalar tensor")
            if not bool(torch.isfinite(local_loss.detach()).item()):
                raise FloatingPointError(
                    f"Non-finite training loss: {local_loss.detach().item()}"
                )
            local_cpt_model = _unwrap_cpt_model(model)
            return local_transaction, local_loss, local_cpt_model

        transaction, loss, cpt_model = _run_consensus_phase(
            model,
            "loss_and_local_proposal_preflight",
            local_preflight,
        )
        _validate_distributed_cpt_consistency(
            model,
            context="transaction validation",
        )
        _run_consensus_phase(
            model,
            "cpt_transaction_validation",
            lambda: cpt_model.validate_cpt_transaction(transaction),
        )

        def backward_and_clip():
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
                error_if_nonfinite=False,
            )
            local_grad_norm = torch.as_tensor(grad_norm)
            if not bool(torch.isfinite(local_grad_norm.detach()).all().item()):
                raise FloatingPointError(
                    f"Non-finite gradient norm: {local_grad_norm.detach().item()}"
                )
            return local_grad_norm.detach()

        grad_norm_tensor = _run_consensus_phase(
            model,
            "backward_and_gradient_preflight",
            backward_and_clip,
        )

        if snapshot is None and failure_policy == "exact-rollback":
            snapshot = _run_consensus_phase(
                model,
                "step_entry_snapshot",
                lambda: _capture_training_snapshot(model, optimizer, scheduler),
            )

        def optimizer_update():
            optimizer.step()
            _validate_training_numerics(model, optimizer, None)

        _run_consensus_phase(model, "optimizer", optimizer_update)
        if scheduler is not None:
            scheduler_guard = {}

            def scheduler_update():
                scheduler_guard["tensor_versions"] = (
                    _capture_non_scheduler_tensor_guard(model, optimizer)
                )
                scheduler.step()

            _run_consensus_phase(model, "scheduler", scheduler_update)
            _run_consensus_phase(
                model,
                "post_update_finite_check",
                lambda: _validate_post_scheduler_numerics(
                    model,
                    optimizer,
                    scheduler,
                    scheduler_guard["tensor_versions"],
                ),
            )
        _validate_distributed_cpt_consistency(
            model,
            context="pre-commit",
        )
        _run_consensus_phase(
            model,
            "gradient_cleanup",
            lambda: optimizer.zero_grad(set_to_none=True),
        )
        _run_consensus_phase(
            model,
            "cpt_commit",
            lambda: cpt_model.commit_cpt_transaction(transaction),
        )
        return grad_norm_tensor.detach()
    except BaseException as error:
        if failure_policy == "fail-stop":
            cleanup_errors = []
            if transaction is not None:
                try:
                    _unwrap_cpt_model(model).abort_cpt_transaction(transaction)
                except BaseException as exc:
                    cleanup_errors.append(f"CPT abort failed: {exc}")
            try:
                optimizer.zero_grad(set_to_none=True)
            except BaseException as exc:
                cleanup_errors.append(f"zero_grad failed: {exc}")
            reason = _poison_training_state(model, optimizer, scheduler, error)
            cleanup_suffix = (
                f" Cleanup also failed: {'; '.join(cleanup_errors)}."
                if cleanup_errors
                else ""
            )
            raise TrainingFailStop(
                "Training step failed under fail-stop policy; the live model, "
                "optimizer, and scheduler must not be reused or checkpointed. "
                "Resume from the latest successfully published checkpoint. "
                f"Cause: {reason}.{cleanup_suffix}"
            ) from error
        if snapshot is not None:
            def rollback():
                _restore_training_snapshot(model, optimizer, scheduler, snapshot)
                _unwrap_cpt_model(model).abort_cpt_transaction(transaction)
                optimizer.zero_grad(set_to_none=True)

            try:
                _run_consensus_phase(model, "rollback", rollback)
            except BaseException as rollback_error:
                reason = _poison_training_state(
                    model,
                    optimizer,
                    scheduler,
                    rollback_error,
                )
                raise TrainingFailStop(
                    "Training step failed and exact rollback was incomplete; the "
                    "live model, optimizer, and scheduler may be partially restored "
                    "and must not be reused or checkpointed. Resume from the latest "
                    "successfully published checkpoint. "
                    f"Rollback cause: {reason}."
                ) from error
        else:
            _abort_training_step(model, transaction, optimizer, error)
        raise


def execute_training_iteration(
    model,
    optimizer,
    scheduler,
    forward_fn,
    max_grad_norm=1.0,
    failure_policy="exact-rollback",
):
    """Run one strict transaction with exact rollback or fail-stop recovery."""
    failure_policy = _normalize_failure_policy(failure_policy)
    _assert_training_state_usable(model, operation="execute another training iteration")
    snapshot = None
    if failure_policy == "exact-rollback":
        snapshot = _run_consensus_phase(
            model,
            "pre_forward_step_entry_snapshot",
            lambda: capture_training_step_entry(model, optimizer, scheduler),
        )
    output_holder = {}

    def forward_phase():
        output_holder["output"] = forward_fn()
        return output_holder["output"]

    try:
        output = _run_consensus_phase(model, "forward", forward_phase)
    except BaseException as error:
        transaction = None
        local_output = output_holder.get("output")
        if isinstance(local_output, dict):
            transaction = local_output.get("cpt_transaction")

        if failure_policy == "fail-stop":
            cleanup_errors = []
            if transaction is not None:
                try:
                    _unwrap_cpt_model(model).abort_cpt_transaction(transaction)
                except BaseException as exc:
                    cleanup_errors.append(f"CPT abort failed: {exc}")
            try:
                optimizer.zero_grad(set_to_none=True)
            except BaseException as exc:
                cleanup_errors.append(f"zero_grad failed: {exc}")
            reason = _poison_training_state(model, optimizer, scheduler, error)
            cleanup_suffix = (
                f" Cleanup also failed: {'; '.join(cleanup_errors)}."
                if cleanup_errors
                else ""
            )
            raise TrainingFailStop(
                "Forward failed under fail-stop policy; the live model, optimizer, "
                "and scheduler must not be reused or checkpointed. Resume from the "
                "latest successfully published checkpoint. "
                f"Cause: {reason}.{cleanup_suffix}"
            ) from error

        def rollback():
            restore_training_step_entry(model, optimizer, scheduler, snapshot)
            _unwrap_cpt_model(model).abort_cpt_transaction(transaction)
            optimizer.zero_grad(set_to_none=True)

        try:
            _run_consensus_phase(model, "forward_rollback", rollback)
        except BaseException as rollback_error:
            reason = _poison_training_state(
                model,
                optimizer,
                scheduler,
                rollback_error,
            )
            raise TrainingFailStop(
                "Forward failed and exact rollback was incomplete; the live model, "
                "optimizer, and scheduler may be partially restored and must not "
                "be reused or checkpointed. Resume from the latest successfully "
                f"published checkpoint. Rollback cause: {reason}."
            ) from error
        raise

    grad_norm = execute_training_step(
        model,
        output,
        optimizer,
        scheduler,
        max_grad_norm=max_grad_norm,
        step_entry_snapshot=snapshot,
        failure_policy=failure_policy,
    )
    return output, grad_norm


def _finite_real(value, label):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.is_complex():
            raise RuntimeError(f"{label} must be a real scalar")
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise RuntimeError(f"{label} must be a real scalar")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise RuntimeError(f"{label} must be finite")
    return normalized


def _normalize_optimizer_kind(optimizer_kind):
    if not isinstance(optimizer_kind, str) or optimizer_kind not in OPTIMIZER_KINDS:
        raise RuntimeError(
            f"optimizer_kind must be one of {OPTIMIZER_KINDS}, "
            f"got {optimizer_kind!r}"
        )
    return optimizer_kind


def _optimizer_kind_from_instance(optimizer):
    if type(optimizer) is BF16AdamW:
        return "bf16_adamw"
    if type(optimizer) is torch.optim.AdamW:
        return "adamw"
    raise RuntimeError(
        "strict CPT checkpointing supports only AdamW or BF16AdamW, "
        f"got {type(optimizer).__name__}"
    )


def _normalize_schedule_identity(schedule_kind, schedule_decay_ratio):
    if not isinstance(schedule_kind, str) or schedule_kind not in SCHEDULE_KINDS:
        raise RuntimeError(
            f"schedule_kind must be one of {SCHEDULE_KINDS}, "
            f"got {schedule_kind!r}"
        )
    if schedule_kind == "cosine":
        if schedule_decay_ratio is not None:
            raise RuntimeError(
                "schedule_decay_ratio must be None for the cosine schedule"
            )
        return schedule_kind, None
    ratio = _finite_real(schedule_decay_ratio, "schedule_decay_ratio")
    if ratio <= 0 or ratio > 1:
        raise RuntimeError("schedule_decay_ratio must satisfy 0 < value <= 1")
    return schedule_kind, ratio


def _register_schedule_identity(
    scheduler,
    *,
    schedule_kind,
    warmup_steps,
    total_steps,
    schedule_decay_ratio,
):
    schedule_kind, schedule_decay_ratio = _normalize_schedule_identity(
        schedule_kind,
        schedule_decay_ratio,
    )
    identity = {
        "schedule_kind": schedule_kind,
        "warmup_steps": int(warmup_steps),
        "total_steps": int(total_steps),
        "schedule_decay_ratio": schedule_decay_ratio,
    }
    _SCHEDULE_IDENTITIES[scheduler] = identity
    return scheduler


def _scheduler_identity_from_instance(scheduler):
    identity = _SCHEDULE_IDENTITIES.get(scheduler)
    if identity is None:
        raise RuntimeError(
            "strict CPT checkpointing requires a scheduler created by "
            "make_cosine_schedule or make_wsd_schedule"
        )
    return dict(identity)


def _assert_scheduler_identity(
    scheduler,
    *,
    optimizer,
    schedule_kind,
    warmup_steps,
    total_steps,
    schedule_decay_ratio,
):
    if getattr(scheduler, "optimizer", None) is not optimizer:
        raise RuntimeError(
            "declared scheduler is not bound to the live optimizer"
        )
    schedule_kind, schedule_decay_ratio = _normalize_schedule_identity(
        schedule_kind,
        schedule_decay_ratio,
    )
    expected = {
        "schedule_kind": schedule_kind,
        "warmup_steps": int(warmup_steps),
        "total_steps": int(total_steps),
        "schedule_decay_ratio": schedule_decay_ratio,
    }
    actual = _scheduler_identity_from_instance(scheduler)
    if actual != expected:
        raise RuntimeError(
            f"declared scheduler recipe {expected!r} disagrees with live "
            f"scheduler recipe {actual!r}"
        )
    return actual


def _schedule_multiplier(
    schedule_kind,
    step,
    warmup_steps,
    total_steps,
    schedule_decay_ratio,
):
    schedule_kind, schedule_decay_ratio = _normalize_schedule_identity(
        schedule_kind,
        schedule_decay_ratio,
    )
    if schedule_kind == "cosine":
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if warmup_steps == 0:
            progress = step / max(1, total_steps - 1)
        else:
            progress = (step - warmup_steps + 1) / max(
                1,
                total_steps - warmup_steps,
            )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    decay_steps = max(1, int(total_steps * schedule_decay_ratio))
    decay_start = max(warmup_steps + 1, total_steps - decay_steps)
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    if step < decay_start:
        return 1.0
    progress = (step - decay_start + 1) / max(1, total_steps - decay_start)
    progress = min(max(progress, 0.0), 1.0)
    return 1.0 - progress


def _validate_optimizer_scheduler_semantics(
    optimizer_payload,
    scheduler_payload,
    *,
    optimizer_kind,
    schedule_kind,
    schedule_decay_ratio,
    warmup_steps,
    total_steps,
    expected_step,
):
    optimizer_kind = _normalize_optimizer_kind(optimizer_kind)
    schedule_kind, schedule_decay_ratio = _normalize_schedule_identity(
        schedule_kind,
        schedule_decay_ratio,
    )
    if set(optimizer_payload) != {"state", "param_groups"}:
        raise RuntimeError(
            "optimizer state must contain exactly 'state' and 'param_groups'"
        )
    expected_scheduler_fields = {
        "base_lrs",
        "last_epoch",
        "_step_count",
        "_is_initial",
        "_get_lr_called_within_step",
        "_last_lr",
        "lr_lambdas",
    }
    if set(scheduler_payload) != expected_scheduler_fields:
        missing = sorted(expected_scheduler_fields - set(scheduler_payload))
        unexpected = sorted(set(scheduler_payload) - expected_scheduler_fields)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise RuntimeError(
            "scheduler state fields disagree with the strict LambdaLR schema; "
            + "; ".join(details)
        )
    param_groups = optimizer_payload["param_groups"]
    optimizer_state = optimizer_payload["state"]
    if len(param_groups) != 2:
        raise RuntimeError(
            "strict AdamW state must contain exactly decay and no-decay groups"
        )
    known_parameter_ids = set()
    group_lrs = []
    initial_lrs = []
    group_betas = []
    group_epsilons = []
    group_weight_decays = []
    expected_group_fields = {
        "params",
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
        "decoupled_weight_decay",
        "initial_lr",
    }
    for group_index, group in enumerate(param_groups):
        if not isinstance(group, dict):
            raise RuntimeError(f"optimizer param_group {group_index} must be a dictionary")
        if set(group) != expected_group_fields:
            missing = sorted(expected_group_fields - set(group))
            unexpected = sorted(set(group) - expected_group_fields)
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unexpected:
                details.append("unexpected " + ", ".join(unexpected))
            raise RuntimeError(
                f"optimizer param_group {group_index} fields disagree with strict "
                "AdamW; " + "; ".join(details)
            )
        parameters = group.get("params")
        if not isinstance(parameters, list):
            raise RuntimeError(
                f"optimizer param_group {group_index} params must be a list"
            )
        for parameter_id in parameters:
            if isinstance(parameter_id, bool) or not isinstance(parameter_id, int):
                raise RuntimeError("optimizer parameter IDs must be integers")
            if parameter_id < 0:
                raise RuntimeError("optimizer parameter IDs must be non-negative")
            if parameter_id in known_parameter_ids:
                raise RuntimeError(
                    f"optimizer contains duplicate parameter ID {parameter_id}"
                )
            known_parameter_ids.add(parameter_id)

        lr = _finite_real(group.get("lr"), f"optimizer param_group {group_index} lr")
        if lr < 0:
            raise RuntimeError(
                f"optimizer param_group {group_index} lr must be non-negative"
            )
        group_lrs.append(lr)
        weight_decay = _finite_real(
            group.get("weight_decay"),
            f"optimizer param_group {group_index} weight_decay",
        )
        if weight_decay < 0:
            raise RuntimeError(
                f"optimizer param_group {group_index} weight_decay must be non-negative"
            )
        group_weight_decays.append(weight_decay)

        fixed_options = {
            "amsgrad": False,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
            "decoupled_weight_decay": True,
        }
        for option, expected in fixed_options.items():
            if group[option] is not expected:
                raise RuntimeError(
                    f"optimizer param_group {group_index} {option} must be "
                    f"{expected!r} for strict AdamW"
                )
        for option in ("foreach", "fused"):
            if group[option] is not None:
                raise RuntimeError(
                    f"optimizer param_group {group_index} {option} must remain None"
                )

        if "betas" not in group:
            raise RuntimeError(
                f"optimizer AdamW param_group {group_index} is missing betas"
            )
        betas = group["betas"]
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise RuntimeError(
                f"optimizer param_group {group_index} betas must contain two values"
            )
        normalized_betas = [
            _finite_real(beta, f"optimizer param_group {group_index} betas")
            for beta in betas
        ]
        if any(beta < 0 or beta >= 1 for beta in normalized_betas):
            raise RuntimeError(
                f"optimizer param_group {group_index} betas must satisfy 0 <= beta < 1"
            )
        group_betas.append(tuple(normalized_betas))
        if "eps" not in group:
            raise RuntimeError(
                f"optimizer AdamW param_group {group_index} is missing eps"
            )
        epsilon = _finite_real(
            group["eps"],
            f"optimizer param_group {group_index} eps",
        )
        if epsilon <= 0:
            raise RuntimeError(
                f"optimizer param_group {group_index} eps must be positive"
            )
        group_epsilons.append(epsilon)
        if "initial_lr" not in group:
            raise RuntimeError(
                f"optimizer param_group {group_index} is missing initial_lr"
            )
        initial_lr = _finite_real(
            group["initial_lr"],
            f"optimizer param_group {group_index} initial_lr",
        )
        if initial_lr < 0:
            raise RuntimeError(
                f"optimizer param_group {group_index} initial_lr must be non-negative"
            )
        initial_lrs.append(initial_lr)

    if known_parameter_ids != set(range(len(known_parameter_ids))):
        raise RuntimeError("optimizer parameter IDs must form one contiguous range")
    if group_weight_decays[1] != 0.0:
        raise RuntimeError("optimizer no-decay group must have weight_decay=0")
    if len(set(group_lrs)) != 1 or len(set(initial_lrs)) != 1:
        raise RuntimeError("optimizer groups must share one learning-rate schedule")
    if len(set(group_betas)) != 1 or len(set(group_epsilons)) != 1:
        raise RuntimeError("optimizer groups must share AdamW betas and eps")

    for parameter_id, parameter_state in optimizer_state.items():
        if isinstance(parameter_id, bool) or not isinstance(parameter_id, int):
            raise RuntimeError("optimizer state keys must be integer parameter IDs")
        if parameter_id not in known_parameter_ids:
            raise RuntimeError(
                f"optimizer state references unknown parameter ID {parameter_id}"
            )
        if not isinstance(parameter_state, dict):
            raise RuntimeError(
                f"optimizer state for parameter ID {parameter_id} must be a dictionary"
            )
        expected_state_fields = {"step", "exp_avg", "exp_avg_sq"}
        if set(parameter_state) != expected_state_fields:
            raise RuntimeError(
                f"optimizer state for parameter ID {parameter_id} must contain "
                "exactly step, exp_avg and exp_avg_sq"
            )
        step_tensor = parameter_state["step"]
        exp_avg = parameter_state["exp_avg"]
        exp_avg_sq = parameter_state["exp_avg_sq"]
        if (
            not isinstance(step_tensor, torch.Tensor)
            or step_tensor.is_meta
            or step_tensor.shape != ()
            or step_tensor.dtype != torch.float32
        ):
            raise RuntimeError(
                f"optimizer state step for parameter ID {parameter_id} must be a "
                "materialized FP32 scalar tensor"
            )
        step_value = float(step_tensor.detach().cpu().item())
        if (
            not math.isfinite(step_value)
            or step_value < 1
            or not step_value.is_integer()
        ):
            raise RuntimeError(
                f"optimizer state step for parameter ID {parameter_id} must be a "
                "positive integer-valued scalar"
            )
        for state_name, tensor in (("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.is_meta
                or not tensor.is_floating_point()
                or tensor.layout != torch.strided
            ):
                raise RuntimeError(
                    f"optimizer {state_name} for parameter ID {parameter_id} must "
                    "be a materialized strided floating tensor"
                )
            if not bool(torch.isfinite(tensor.detach()).all().item()):
                raise RuntimeError(
                    f"optimizer {state_name} for parameter ID {parameter_id} is "
                    "non-finite"
                )
        if exp_avg.shape != exp_avg_sq.shape or exp_avg.dtype != exp_avg_sq.dtype:
            raise RuntimeError(
                f"optimizer moments for parameter ID {parameter_id} disagree in "
                "shape or dtype"
            )
        if bool((exp_avg_sq.detach() < 0).any().item()):
            raise RuntimeError(
                f"optimizer exp_avg_sq for parameter ID {parameter_id} must be "
                "non-negative"
            )

    base_lrs = scheduler_payload.get("base_lrs")
    last_lrs = scheduler_payload.get("_last_lr")
    if not isinstance(base_lrs, list) or len(base_lrs) != len(param_groups):
        raise RuntimeError(
            "scheduler base_lrs must be a list matching optimizer param_groups"
        )
    if not isinstance(last_lrs, list) or len(last_lrs) != len(param_groups):
        raise RuntimeError(
            "scheduler _last_lr must be a list matching optimizer param_groups"
        )
    normalized_base_lrs = [
        _finite_real(value, f"scheduler base_lrs[{index}]")
        for index, value in enumerate(base_lrs)
    ]
    normalized_last_lrs = [
        _finite_real(value, f"scheduler _last_lr[{index}]")
        for index, value in enumerate(last_lrs)
    ]
    if any(value < 0 for value in normalized_base_lrs):
        raise RuntimeError("scheduler base_lrs must be non-negative")
    if any(value < 0 for value in normalized_last_lrs):
        raise RuntimeError("scheduler _last_lr must be non-negative")
    if normalized_base_lrs != initial_lrs:
        raise RuntimeError(
            "scheduler base_lrs disagree with optimizer initial_lr values"
        )
    if normalized_last_lrs != group_lrs:
        raise RuntimeError(
            "scheduler _last_lr disagrees with optimizer current lr values"
        )
    expected_multiplier = _schedule_multiplier(
        schedule_kind,
        expected_step,
        warmup_steps,
        total_steps,
        schedule_decay_ratio,
    )
    expected_last_lrs = [
        base_lr * expected_multiplier for base_lr in normalized_base_lrs
    ]
    if any(
        not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15)
        for actual, expected in zip(normalized_last_lrs, expected_last_lrs)
    ):
        raise RuntimeError(
            f"scheduler LR state disagrees with declared {schedule_kind} recipe"
        )
    if scheduler_payload["_is_initial"] is not False:
        raise RuntimeError("scheduler _is_initial must be False in a saved checkpoint")
    if scheduler_payload["_get_lr_called_within_step"] is not False:
        raise RuntimeError(
            "scheduler _get_lr_called_within_step must be False in a saved checkpoint"
        )
    scheduler_epoch = scheduler_payload.get("last_epoch")
    if isinstance(scheduler_epoch, bool) or not isinstance(scheduler_epoch, int):
        raise RuntimeError("scheduler last_epoch must be an integer")
    if scheduler_epoch != expected_step:
        raise RuntimeError(
            f"Scheduler step mismatch: last_epoch={scheduler_epoch}, "
            f"training step={expected_step}"
        )
    scheduler_step_count = scheduler_payload.get("_step_count")
    if isinstance(scheduler_step_count, bool) or not isinstance(
        scheduler_step_count,
        int,
    ):
        raise RuntimeError("scheduler _step_count must be an integer")
    if scheduler_step_count != expected_step + 1:
        raise RuntimeError(
            f"Scheduler _step_count={scheduler_step_count}, "
            f"expected {expected_step + 1}"
        )
    lr_lambdas = scheduler_payload["lr_lambdas"]
    if (
        not isinstance(lr_lambdas, list)
        or len(lr_lambdas) != len(param_groups)
        or any(value is not None for value in lr_lambdas)
    ):
        raise RuntimeError(
            "scheduler lr_lambdas must contain one None entry per optimizer group"
        )


def _adamw_parameter_groups_for_model(model):
    """Return the canonical decay/no-decay trainable-parameter grouping."""
    cpt_model = _unwrap_cpt_model(model)
    decay = []
    no_decay = []
    for parameter in cpt_model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return decay, no_decay


def _assert_optimizer_bound_to_model(optimizer, model):
    """Reject a live optimizer constructed from another, merely isomorphic model."""
    expected_groups = _adamw_parameter_groups_for_model(model)
    if len(optimizer.param_groups) != len(expected_groups):
        raise RuntimeError(
            "live optimizer parameter groups do not match the live model"
        )
    for group_index, (group, expected_parameters) in enumerate(
        zip(optimizer.param_groups, expected_groups)
    ):
        actual_parameters = group.get("params")
        if (
            not isinstance(actual_parameters, list)
            or len(actual_parameters) != len(expected_parameters)
            or any(
                actual is not expected
                for actual, expected in zip(actual_parameters, expected_parameters)
            )
        ):
            raise RuntimeError(
                f"live optimizer param_group {group_index} is not bound to the "
                "live model parameters"
            )


def _validate_adamw_state_against_model(
    optimizer_payload,
    model,
    *,
    expected_step,
    optimizer_kind,
):
    """Bind serialized AdamW IDs and moments to the live trainable parameters."""
    optimizer_kind = _normalize_optimizer_kind(optimizer_kind)
    expected_groups = _adamw_parameter_groups_for_model(model)
    id_to_parameter = {}
    next_parameter_id = 0
    for group_index, parameters in enumerate(expected_groups):
        expected_ids = list(
            range(next_parameter_id, next_parameter_id + len(parameters))
        )
        actual_ids = optimizer_payload["param_groups"][group_index]["params"]
        if actual_ids != expected_ids:
            raise RuntimeError(
                f"optimizer param_group {group_index} IDs do not match the live "
                "model parameter order"
            )
        for parameter_id, parameter in zip(expected_ids, parameters):
            id_to_parameter[parameter_id] = parameter
        next_parameter_id += len(parameters)

    for parameter_id, parameter_state in optimizer_payload["state"].items():
        parameter = id_to_parameter[parameter_id]
        step_value = float(parameter_state["step"].detach().cpu().item())
        if step_value > expected_step:
            raise RuntimeError(
                f"optimizer parameter ID {parameter_id} step={step_value:g} exceeds "
                f"training step={expected_step}"
            )
        for state_name in ("exp_avg", "exp_avg_sq"):
            tensor = parameter_state[state_name]
            if tensor.shape != parameter.shape:
                raise RuntimeError(
                    f"optimizer {state_name} shape for parameter ID {parameter_id} "
                    f"is {tuple(tensor.shape)}, expected {tuple(parameter.shape)}"
                )
            expected_dtype = parameter.dtype
            if tensor.dtype != expected_dtype:
                raise RuntimeError(
                    f"optimizer {state_name} dtype for parameter ID {parameter_id} "
                    f"is {tensor.dtype}, expected {expected_dtype} for "
                    f"optimizer_kind={optimizer_kind}"
                )


def make_adamw(
    model,
    lr,
    weight_decay,
    betas=(0.9, 0.95),
    bf16_states=False,
):
    """构建 AdamW：矩阵权重衰减，RMSNorm 等 1D 参数不衰减。"""
    if not isinstance(bf16_states, bool):
        raise ValueError("bf16_states must be a boolean")
    normalized_lr = _finite_real(lr, "AdamW lr")
    if normalized_lr < 0:
        raise ValueError("AdamW lr must be non-negative")
    normalized_weight_decay = _finite_real(weight_decay, "AdamW weight_decay")
    if normalized_weight_decay < 0:
        raise ValueError("AdamW weight_decay must be non-negative")
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError("AdamW betas must contain two values")
    normalized_betas = tuple(_finite_real(beta, "AdamW beta") for beta in betas)
    if any(beta < 0 or beta >= 1 for beta in normalized_betas):
        raise ValueError("AdamW betas must satisfy 0 <= beta < 1")
    decay, no_decay = _adamw_parameter_groups_for_model(model)
    optimizer_class = BF16AdamW if bf16_states else torch.optim.AdamW
    return optimizer_class(
        [
            {"params": decay, "weight_decay": normalized_weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=normalized_lr,
        betas=normalized_betas,
    )

def validate_training_state(
    state,
    checkpoint_step=None,
    model=None,
    expected_run_id=None,
    expected_code_manifest=None,
    expected_data_manifest=None,
    expected_tokenizer_manifest=None,
    expected_world_size=None,
    expected_model_state_sha256=None,
    expected_config_file_sha256=None,
):
    """Strictly validate a CPT training-state payload before resume."""
    if not isinstance(state, dict):
        raise RuntimeError("training_state.pt must contain a dictionary")
    required = {
        "schema_name", "schema_version", "cpt_state_version",
        "optimizer_kind", "schedule_kind", "schedule_decay_ratio",
        "opt", "sched", "step", "total_tok", "warmup_steps", "total_steps",
        "fi", "ptr", "batch_size", "seq_len", "rng_world_size",
        "rng_states", "run_id",
        "config_sha256", "code_manifest", "data_manifest",
        "tokenizer_manifest", "model_state_sha256", "config_file_sha256",
    }
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError(
            "Legacy or incomplete training checkpoint is incompatible with CPT "
            f"resume; missing fields: {', '.join(missing)}"
        )
    unexpected = sorted(set(state) - required)
    if unexpected:
        raise RuntimeError(
            "Training checkpoint contains unsupported fields: "
            + ", ".join(unexpected)
        )
    if state["schema_name"] != TRAINING_STATE_SCHEMA_NAME:
        raise RuntimeError(
            f"Unsupported training-state schema: {state['schema_name']!r}"
        )
    if state["schema_version"] != TRAINING_STATE_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported training-state schema version: {state['schema_version']!r}; "
            f"expected {TRAINING_STATE_SCHEMA_VERSION}"
        )
    optimizer_kind = _normalize_optimizer_kind(state["optimizer_kind"])
    schedule_kind, schedule_decay_ratio = _normalize_schedule_identity(
        state["schedule_kind"],
        state["schedule_decay_ratio"],
    )

    integer_fields = (
        "cpt_state_version", "step", "total_tok", "warmup_steps", "total_steps",
        "fi", "ptr", "batch_size", "seq_len", "rng_world_size",
    )
    for name in integer_fields:
        value = state[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(f"training-state field {name!r} must be an integer")
    if state["step"] < 0 or state["total_tok"] < 0:
        raise RuntimeError("step and total_tok must be non-negative")
    if state["warmup_steps"] < 0 or state["total_steps"] <= 0:
        raise RuntimeError("warmup_steps must be non-negative and total_steps positive")
    if state["fi"] < 0 or state["ptr"] < 0:
        raise RuntimeError("fi and ptr must be non-negative")
    if state["batch_size"] <= 0 or state["seq_len"] <= 0:
        raise RuntimeError("batch_size and seq_len must be positive")
    if not isinstance(state["opt"], dict) or not isinstance(state["sched"], dict):
        raise RuntimeError("optimizer and scheduler states must be dictionaries")
    if state["step"] > state["total_steps"]:
        raise RuntimeError("training step cannot exceed total_steps")
    warmup_cap = max(
        state["total_steps"] - (2 if schedule_kind == "wsd" else 1),
        0,
    )
    if state["warmup_steps"] > warmup_cap:
        raise RuntimeError("warmup_steps exceeds the strict schedule range")
    if state["cpt_state_version"] != state["step"]:
        raise RuntimeError(
            f"Checkpoint transaction mismatch: step={state['step']}, "
            f"CPT state version={state['cpt_state_version']}"
        )
    if checkpoint_step is not None and state["step"] != int(checkpoint_step):
        raise RuntimeError(
            f"Checkpoint directory step={checkpoint_step} disagrees with "
            f"training_state.pt step={state['step']}"
        )

    expected_total_tok = state["step"] * state["batch_size"] * state["seq_len"]
    if state["total_tok"] != expected_total_tok:
        raise RuntimeError(
            f"Checkpoint token counter mismatch: total_tok={state['total_tok']}, "
            f"expected {expected_total_tok} from step/batch_size/seq_len"
        )
    chunk = (state["seq_len"] + 1) * state["batch_size"]
    if state["ptr"] % chunk != 0:
        raise RuntimeError(
            f"Checkpoint data pointer ptr={state['ptr']} is not aligned to chunk={chunk}"
        )

    param_groups = state["opt"].get("param_groups")
    optimizer_state = state["opt"].get("state")
    if not isinstance(param_groups, list) or not param_groups:
        raise RuntimeError("optimizer state must contain non-empty param_groups")
    if not isinstance(optimizer_state, dict):
        raise RuntimeError("optimizer state must contain a state dictionary")
    _validate_optimizer_scheduler_semantics(
        state["opt"],
        state["sched"],
        optimizer_kind=optimizer_kind,
        schedule_kind=schedule_kind,
        schedule_decay_ratio=schedule_decay_ratio,
        warmup_steps=state["warmup_steps"],
        total_steps=state["total_steps"],
        expected_step=state["step"],
    )
    scheduler_epoch = state["sched"].get("last_epoch")
    if isinstance(scheduler_epoch, bool) or not isinstance(scheduler_epoch, int):
        raise RuntimeError("scheduler last_epoch must be an integer")
    if scheduler_epoch != state["step"]:
        raise RuntimeError(
            f"Scheduler step mismatch: last_epoch={scheduler_epoch}, "
            f"training step={state['step']}"
        )
    scheduler_step_count = state["sched"].get("_step_count")
    if isinstance(scheduler_step_count, bool) or not isinstance(
        scheduler_step_count,
        int,
    ):
        raise RuntimeError("scheduler _step_count must be an integer")
    if scheduler_step_count != state["step"] + 1:
        raise RuntimeError(
            f"Scheduler _step_count={scheduler_step_count}, "
            f"expected {state['step'] + 1}"
        )
    _assert_nested_finite(state["opt"], "optimizer")
    _assert_nested_finite(state["sched"], "scheduler")

    run_id = state["run_id"]
    if not isinstance(run_id, str) or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise RuntimeError("run_id must be 32 lowercase hexadecimal characters")
    config_hash = state["config_sha256"]
    if not isinstance(config_hash, str) or _SHA256_PATTERN.fullmatch(config_hash) is None:
        raise RuntimeError("config_sha256 must be a lowercase SHA-256 digest")
    model_state_hash = state["model_state_sha256"]
    if (
        not isinstance(model_state_hash, str)
        or _SHA256_PATTERN.fullmatch(model_state_hash) is None
    ):
        raise RuntimeError("model_state_sha256 must be a lowercase SHA-256 digest")
    config_file_hash = state["config_file_sha256"]
    if (
        not isinstance(config_file_hash, str)
        or _SHA256_PATTERN.fullmatch(config_file_hash) is None
    ):
        raise RuntimeError("config_file_sha256 must be a lowercase SHA-256 digest")
    _validate_manifest(state["code_manifest"], "code")
    _validate_manifest(state["data_manifest"], "data")
    _validate_manifest(state["tokenizer_manifest"], "tokenizer")
    _validate_rank_rng_states(state["rng_states"], state["rng_world_size"])
    if (
        expected_world_size is not None
        and state["rng_world_size"] != expected_world_size
    ):
        raise RuntimeError(
            f"Checkpoint rng_world_size={state['rng_world_size']} disagrees with "
            f"runtime world_size={expected_world_size}"
        )

    if expected_run_id is not None and run_id != expected_run_id:
        raise RuntimeError(
            f"Checkpoint run_id={run_id} disagrees with expected run_id={expected_run_id}"
        )
    if expected_code_manifest is not None and state["code_manifest"] != expected_code_manifest:
        raise RuntimeError("Current training code manifest disagrees with checkpoint")
    if expected_data_manifest is not None and state["data_manifest"] != expected_data_manifest:
        raise RuntimeError("Current data-shard manifest disagrees with checkpoint")
    if (
        expected_tokenizer_manifest is not None
        and state["tokenizer_manifest"] != expected_tokenizer_manifest
    ):
        raise RuntimeError("Current tokenizer manifest disagrees with checkpoint")
    if (
        expected_model_state_sha256 is not None
        and model_state_hash != expected_model_state_sha256
    ):
        raise RuntimeError(
            "pytorch_model.bin SHA-256 disagrees with training_state.pt"
        )
    if (
        expected_config_file_sha256 is not None
        and config_file_hash != expected_config_file_sha256
    ):
        raise RuntimeError("config.json SHA-256 disagrees with training_state.pt")
    if model is not None:
        model_version = get_cpt_state_version(model)
        if model_version != state["cpt_state_version"]:
            raise RuntimeError(
                f"Model CPT state version={model_version} disagrees with "
                f"training_state.pt version={state['cpt_state_version']}"
            )
        live_config_hash = canonical_config_hash(_unwrap_cpt_model(model).config)
        if live_config_hash != config_hash:
            raise RuntimeError(
                f"Model config hash={live_config_hash} disagrees with "
                f"training_state.pt config hash={config_hash}"
            )
        _validate_adamw_state_against_model(
            state["opt"],
            model,
            expected_step=state["step"],
            optimizer_kind=optimizer_kind,
        )
    return state


def save_training_state(
    path,
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
    *,
    optimizer_kind,
    schedule_kind,
    schedule_decay_ratio,
    cpt_state_version,
    run_id,
    config_sha256,
    code_manifest,
    data_manifest,
    tokenizer_manifest,
    model_state_sha256,
    config_file_sha256,
    rng_states=None,
    rng_world_size=None,
):
    """Validate and save a self-contained strict CPT training state."""
    optimizer_kind = _normalize_optimizer_kind(optimizer_kind)
    live_optimizer_kind = _optimizer_kind_from_instance(opt)
    if live_optimizer_kind != optimizer_kind:
        raise RuntimeError(
            f"optimizer_kind={optimizer_kind!r} disagrees with live optimizer "
            f"kind={live_optimizer_kind!r}"
        )
    schedule_kind, schedule_decay_ratio = _normalize_schedule_identity(
        schedule_kind,
        schedule_decay_ratio,
    )
    _assert_scheduler_identity(
        sched,
        optimizer=opt,
        schedule_kind=schedule_kind,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        schedule_decay_ratio=schedule_decay_ratio,
    )
    if rng_states is None:
        if rng_world_size not in (None, 1):
            raise RuntimeError(
                "Explicit rng_states are required when rng_world_size is not 1"
            )
        rng_world_size = 1
        rng_states = [{"rank": 0, "state": capture_rng_state()}]
    elif rng_world_size is None:
        raise RuntimeError("rng_world_size is required with explicit rng_states")
    state = {
        "schema_name": TRAINING_STATE_SCHEMA_NAME,
        "schema_version": TRAINING_STATE_SCHEMA_VERSION,
        "cpt_state_version": cpt_state_version,
        "optimizer_kind": optimizer_kind,
        "schedule_kind": schedule_kind,
        "schedule_decay_ratio": schedule_decay_ratio,
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "step": step,
        "total_tok": total_tok,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "fi": fi,
        "ptr": ptr,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "rng_world_size": rng_world_size,
        "rng_states": rng_states,
        "run_id": run_id,
        "config_sha256": config_sha256,
        "code_manifest": code_manifest,
        "data_manifest": data_manifest,
        "tokenizer_manifest": tokenizer_manifest,
        "model_state_sha256": model_state_sha256,
        "config_file_sha256": config_file_sha256,
    }
    validate_training_state(
        state,
        checkpoint_step=step,
        expected_run_id=run_id,
        expected_code_manifest=code_manifest,
        expected_data_manifest=data_manifest,
        expected_tokenizer_manifest=tokenizer_manifest,
        expected_world_size=rng_world_size,
        expected_model_state_sha256=model_state_sha256,
        expected_config_file_sha256=config_file_sha256,
    )
    _torch_save_fsync(state, path)
    return state


def save_checkpoint(model, opt, sched, output_dir, step, total_tok,
                    warmup_steps, total_steps, fi, ptr,
                    batch_size, seq_len, *, run_id, code_manifest,
                    data_manifest, tokenizer_manifest, data_files,
                    tokenizer_dir, optimizer_kind, schedule_kind,
                    schedule_decay_ratio, final=False):
    """完整写入临时目录后原子发布 checkpoint。"""
    optimizer_kind = _normalize_optimizer_kind(optimizer_kind)
    schedule_kind, schedule_decay_ratio = _normalize_schedule_identity(
        schedule_kind,
        schedule_decay_ratio,
    )
    _assert_training_state_usable(model, operation="save a checkpoint")
    suffix = "_final" if final else ""
    target = Path(output_dir) / f"step_{step:07d}{suffix}"
    if isinstance(data_files, (str, bytes, os.PathLike)):
        raise TypeError("data_files must be an ordered collection of shard paths")
    try:
        runtime_data_files = tuple(data_files)
    except TypeError as exc:
        raise TypeError(
            "data_files must be an ordered collection of shard paths"
        ) from exc
    if not runtime_data_files:
        raise RuntimeError("data_files must contain at least one shard path")
    if tokenizer_dir is None:
        raise RuntimeError("tokenizer_dir is required for strict checkpoint save")

    def preflight():
        live_optimizer_kind = _optimizer_kind_from_instance(opt)
        if live_optimizer_kind != optimizer_kind:
            raise RuntimeError(
                f"optimizer_kind={optimizer_kind!r} disagrees with live "
                f"optimizer kind={live_optimizer_kind!r}"
            )
        _assert_optimizer_bound_to_model(opt, model)
        _assert_scheduler_identity(
            sched,
            optimizer=opt,
            schedule_kind=schedule_kind,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            schedule_decay_ratio=schedule_decay_ratio,
        )
        live_code_manifest = build_code_manifest()
        if live_code_manifest != code_manifest:
            raise RuntimeError("Training code changed after the run manifest was frozen")
        live_data_manifest = build_data_manifest(runtime_data_files)
        if live_data_manifest != data_manifest:
            raise RuntimeError(
                "Current data-shard manifest changed after the run manifest was frozen"
            )
        live_tokenizer_manifest = build_tokenizer_manifest(tokenizer_dir)
        if live_tokenizer_manifest != tokenizer_manifest:
            raise RuntimeError(
                "Current tokenizer manifest changed after the run manifest was frozen"
            )
        cpt_model = _unwrap_cpt_model(model)
        cpt_state_version = get_cpt_state_version(cpt_model)
        if cpt_state_version != step:
            raise RuntimeError(
                f"Cannot save mixed checkpoint: training step={step}, "
                f"CPT state version={cpt_state_version}"
            )
        _validate_training_numerics(model, opt, sched)
        _validate_adamw_state_against_model(
            opt.state_dict(),
            cpt_model,
            expected_step=step,
            optimizer_kind=optimizer_kind,
        )
        config_hash = canonical_config_hash(cpt_model.config)
        return cpt_model, cpt_state_version, config_hash

    cpt_model, cpt_state_version, config_hash = _run_consensus_phase(
        model,
        "checkpoint_preflight",
        preflight,
    )
    identity = {
        "run_id": run_id,
        "code_sha256": code_manifest.get("sha256"),
        "data_sha256": data_manifest.get("sha256"),
        "tokenizer_sha256": tokenizer_manifest.get("sha256"),
        "config_sha256": config_hash,
        "step": step,
        "total_tok": total_tok,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "fi": fi,
        "ptr": ptr,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "cpt_state_version": cpt_state_version,
        "optimizer_kind": optimizer_kind,
        "schedule_kind": schedule_kind,
        "schedule_decay_ratio": schedule_decay_ratio,
        "final": bool(final),
    }
    rng_world_size = distributed_world_size()
    rng_states = _gather_checkpoint_rng_states(model, identity)

    def publish_from_rank0():
        output_path = Path(output_dir)
        if _is_link_or_reparse_point(output_path):
            raise RuntimeError(
                f"Checkpoint output directory must not be a link or reparse point: "
                f"{output_path}"
            )
        cleanup_stale_checkpoint_temps(output_dir)
        temp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        if _is_link_or_reparse_point(target):
            raise RuntimeError(
                f"Checkpoint target must not be a link or reparse point: {target}"
            )
        if target.exists():
            raise FileExistsError(f"Checkpoint already exists: {target}")
        if _is_link_or_reparse_point(temp):
            raise RuntimeError(
                f"Refusing to remove checkpoint temp reparse point: {temp}"
            )
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir(parents=True)
        try:
            model_state_path = temp / "pytorch_model.bin"
            config_path = temp / "config.json"
            _torch_save_fsync(cpt_model.state_dict(), model_state_path)
            cpt_model.config.save_pretrained(str(temp))
            _fsync_file(config_path)
            model_state_sha256 = _sha256_file(model_state_path)
            config_file_sha256 = _sha256_file(config_path)
            save_training_state(
                temp / "training_state.pt", opt, sched, step, total_tok,
                warmup_steps, total_steps, fi, ptr, batch_size, seq_len,
                optimizer_kind=optimizer_kind,
                schedule_kind=schedule_kind,
                schedule_decay_ratio=schedule_decay_ratio,
                cpt_state_version=cpt_state_version,
                run_id=run_id,
                config_sha256=config_hash,
                code_manifest=code_manifest,
                data_manifest=data_manifest,
                tokenizer_manifest=tokenizer_manifest,
                model_state_sha256=model_state_sha256,
                config_file_sha256=config_file_sha256,
                rng_states=rng_states,
                rng_world_size=rng_world_size,
            )
            _atomic_publish_directory(temp, target)
        except BaseException as save_error:
            try:
                if _is_link_or_reparse_point(temp):
                    raise RuntimeError(
                        f"Refusing to remove checkpoint temp reparse point: {temp}"
                    )
                if temp.exists():
                    shutil.rmtree(temp)
            except BaseException as cleanup_error:
                raise RuntimeError(
                    "Checkpoint publication failed and its temporary directory "
                    f"could not be safely removed: {cleanup_error}"
                ) from save_error
            raise
        return target

    _run_rank0_coordinated(model, "checkpoint_publish", publish_from_rank0)
    return target


def _pid_is_running(pid):
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def cleanup_stale_checkpoint_temps(output_dir):
    """Remove abandoned atomic-save directories whose writer PID is gone."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return []
    removed = []
    for path in output_path.iterdir():
        match = _TEMP_CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        if _is_link_or_reparse_point(path):
            raise RuntimeError(
                f"Refusing to remove checkpoint temp reparse point: {path}"
            )
        if not path.is_dir():
            continue
        if _pid_is_running(int(match.group(1))):
            continue
        shutil.rmtree(path)
        removed.append(path)
    return removed


def prune_periodic_checkpoints(output_dir, keep_last):
    """仅保留最近的周期 checkpoint；final checkpoint 永不删除。"""
    if isinstance(keep_last, bool) or not isinstance(keep_last, int) or keep_last <= 0:
        raise ValueError("keep_last must be a positive integer")
    output_path = Path(output_dir)
    if not output_path.exists():
        return []
    output_root = output_path.resolve()
    checkpoints = []
    for path in output_path.iterdir():
        match = _PERIODIC_CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        if _is_link_or_reparse_point(path):
            raise RuntimeError(
                f"Checkpoint pruning refuses a link or reparse point: {path}"
            )
        if not path.is_dir():
            continue
        required_paths = [path / filename for filename in _CHECKPOINT_REQUIRED_FILES]
        if not all(
            item.is_file() and not _is_link_or_reparse_point(item)
            for item in required_paths
        ):
            continue
        if path.resolve().parent != output_root:
            continue
        checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda item: item[0])
    removed = []
    for _, path in checkpoints[:-keep_last]:
        match = _PERIODIC_CHECKPOINT_PATTERN.fullmatch(path.name)
        required_paths = [path / filename for filename in _CHECKPOINT_REQUIRED_FILES]
        if (
            match is None
            or _is_link_or_reparse_point(path)
            or not path.is_dir()
            or path.parent.resolve() != output_root
            or path.resolve().parent != output_root
            or not all(
                item.is_file() and not _is_link_or_reparse_point(item)
                for item in required_paths
            )
        ):
            raise RuntimeError(
                f"Checkpoint changed before pruning; refusing to delete {path}"
            )
        shutil.rmtree(path)
        removed.append(path)
    return removed


def check_checkpoint_disk_space(model, output_dir, keep_last):
    """确认磁盘可容纳保留的周期 checkpoint、final 和一次原子临时写入。"""
    output_path = Path(output_dir)
    probe_path = output_path
    while not probe_path.exists():
        probe_path = probe_path.parent

    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters() if parameter.requires_grad
    )
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    checkpoint_bytes = int((3 * parameter_bytes + buffer_bytes) * 1.15)
    checkpoint_bytes = max(checkpoint_bytes, 64 * 1024**2)
    target_bytes = checkpoint_bytes * (keep_last + 2)
    existing_bytes = sum(
        file.stat().st_size
        for checkpoint in output_path.glob("step_*") if checkpoint.is_dir()
        for file in checkpoint.rglob("*") if file.is_file()
    ) if output_path.exists() else 0
    additional_bytes = max(checkpoint_bytes, target_bytes - existing_bytes)
    free_bytes = shutil.disk_usage(probe_path).free
    if free_bytes < additional_bytes:
        raise RuntimeError(
            f"Insufficient checkpoint disk space: need about "
            f"{additional_bytes / 1024**3:.1f} GiB more, "
            f"only {free_bytes / 1024**3:.1f} GiB free at {probe_path}"
        )
    print(
        f"Checkpoint disk preflight: ~{checkpoint_bytes / 1024**3:.1f} GiB each, "
        f"{free_bytes / 1024**3:.1f} GiB free",
        flush=True,
    )


def make_cosine_schedule(opt, warmup_steps, total_steps):
    """创建 cosine LR schedule with linear warmup，返回 LambdaLR。"""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    warmup_steps = min(warmup_steps, max(total_steps - 1, 0))

    def lr_lambda(s):
        return _schedule_multiplier(
            "cosine",
            s,
            warmup_steps,
            total_steps,
            None,
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return _register_schedule_identity(
        scheduler,
        schedule_kind="cosine",
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        schedule_decay_ratio=None,
    )


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
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    decay_ratio = _finite_real(decay_ratio, "WSD decay_ratio")
    if decay_ratio <= 0 or decay_ratio > 1:
        raise ValueError("WSD decay_ratio must satisfy 0 < value <= 1")
    warmup_steps = min(warmup_steps, max(total_steps - 2, 0))
    def lr_lambda(s):
        return _schedule_multiplier(
            "wsd",
            s,
            warmup_steps,
            total_steps,
            decay_ratio,
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return _register_schedule_identity(
        scheduler,
        schedule_kind="wsd",
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        schedule_decay_ratio=decay_ratio,
    )


# ============================================================
# CPU eval 子进程
# ============================================================

def run_cpu_eval(checkpoint_path, eval_dir, tokenizer_dir):
    """子进程 CPU GLUE eval。"""
    script = Path(__file__).parent / "eval_glue.py"
    output = Path(eval_dir) / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(tokenizer_dir)
    # 删除旧结果，防止子进程失败时误读
    if output.exists():
        output.unlink()
    cmd = [sys.executable, str(script),
           "--checkpoint", checkpoint_path, "--tokenizer", str(tokenizer_path),
           "--tasks", "sst2,mrpc,qnli,rte,cola", "--limit", "200",
           "--batch-size", "2", "--max-length", "256",
           "--device", "cpu", "--precision", "fp32",
           "--output", str(output), "--seed", "1234"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            stderr_tail = r.stderr.strip()[-500:] if r.stderr else "(empty)"
            print(f"  [eval err] exit={r.returncode} stderr={stderr_tail}", flush=True)
            return None, None  # 失败后不读旧结果
        if output.exists():
            with open(output) as f: d = json.load(f)
            return d.get("aggregate", {}).get("mean_score"), d.get("results", {})
    except Exception as e:
        print(f"  [eval err] {e}", flush=True)
    return None, None


def training_loop(model, opt, sched, files, fi, ptr, total_tok, bs, seq, chunk,
                  output_dir, max_steps, save_every_min, log_every, step_start=0,
                  schedule_args=None, eval_on_save=False,
                  keep_last_checkpoints=5, checkpoint_metadata=None,
                  tokenizer_dir=None, failure_policy="exact-rollback",
                  optimizer_kind=None):
    """按绝对 step 目标训练。

    schedule_args: 可选 dict with warmup_steps, total_steps，用于 checkpoint 恢复。
    """
    failure_policy = _normalize_failure_policy(failure_policy)
    optimizer_kind = _normalize_optimizer_kind(optimizer_kind)
    live_optimizer_kind = _optimizer_kind_from_instance(opt)
    if live_optimizer_kind != optimizer_kind:
        raise RuntimeError(
            f"optimizer_kind={optimizer_kind!r} disagrees with live optimizer "
            f"kind={live_optimizer_kind!r}"
        )
    _assert_optimizer_bound_to_model(opt, model)
    _assert_training_state_usable(model, operation="start the training loop")
    if not isinstance(checkpoint_metadata, dict) or set(checkpoint_metadata) != {
        "run_id", "code_manifest", "data_manifest", "tokenizer_manifest",
    }:
        raise RuntimeError("strict checkpoint_metadata is required for training")
    if tokenizer_dir is None:
        raise RuntimeError("tokenizer_dir is required for strict checkpoint identity")
    if not isinstance(schedule_args, dict) or set(schedule_args) != {
        "warmup_steps", "total_steps", "schedule_kind",
        "schedule_decay_ratio",
    }:
        raise RuntimeError("strict schedule_args is required for training")
    _normalize_schedule_identity(
        schedule_args["schedule_kind"],
        schedule_args["schedule_decay_ratio"],
    )
    _assert_scheduler_identity(
        sched,
        optimizer=opt,
        schedule_kind=schedule_args["schedule_kind"],
        warmup_steps=schedule_args["warmup_steps"],
        total_steps=schedule_args["total_steps"],
        schedule_decay_ratio=schedule_args["schedule_decay_ratio"],
    )
    _validate_optimizer_scheduler_semantics(
        opt.state_dict(),
        sched.state_dict(),
        optimizer_kind=optimizer_kind,
        schedule_kind=schedule_args["schedule_kind"],
        schedule_decay_ratio=schedule_args["schedule_decay_ratio"],
        warmup_steps=schedule_args["warmup_steps"],
        total_steps=schedule_args["total_steps"],
        expected_step=step_start,
    )
    _validate_adamw_state_against_model(
        opt.state_dict(),
        model,
        expected_step=step_start,
        optimizer_kind=optimizer_kind,
    )
    os.makedirs(output_dir, exist_ok=True)

    def cleanup_startup_temps():
        removed_temps = cleanup_stale_checkpoint_temps(output_dir)
        for path in removed_temps:
            print(f"  -> Removed stale checkpoint temp: {path}", flush=True)

    _run_rank0_coordinated(model, "checkpoint_temp_cleanup", cleanup_startup_temps)
    shard = torch.load(files[fi], weights_only=True)
    training_device = next(model.parameters()).device
    tok_base = total_tok
    sa = schedule_args
    metadata = checkpoint_metadata
    t0 = time.time()
    last_save = t0
    last_saved_step = None
    step = step_start
    initial_cpt_version = get_cpt_state_version(model)
    if initial_cpt_version != step_start:
        raise RuntimeError(
            f"Training starts at step={step_start}, but model CPT state version="
            f"{initial_cpt_version}"
        )
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

            # ---- 分片循环：这里只准备候选游标，成功提交前不推进正式状态 ----
            candidate_fi = fi
            candidate_ptr = ptr
            candidate_shard = shard
            if candidate_ptr + chunk > len(candidate_shard):
                candidate_ptr = 0
                candidate_shard = None
                shard = None
                for _ in range(len(files)):
                    candidate_fi = (candidate_fi + 1) % len(files)
                    candidate_shard = torch.load(
                        files[candidate_fi], weights_only=True
                    )
                    if len(candidate_shard) >= chunk:
                        break
                else:
                    raise RuntimeError(f"No shard contains at least {chunk} tokens")

            batch = candidate_shard[candidate_ptr:candidate_ptr + chunk]
            if batch.numel() != chunk:
                print(
                    f"  WARN: short read {batch.numel()}/{chunk} "
                    f"shard={candidate_fi}",
                    flush=True,
                )
                candidate_ptr = 0
                batch = None
                if candidate_shard is shard:
                    shard = None
                candidate_shard = None
                for _ in range(len(files)):
                    candidate_fi = (candidate_fi + 1) % len(files)
                    candidate_shard = torch.load(
                        files[candidate_fi], weights_only=True
                    )
                    if len(candidate_shard) >= chunk:
                        break
                else:
                    raise RuntimeError(f"No shard contains at least {chunk} tokens")
                batch = candidate_shard[:chunk]
                if batch.numel() != chunk:
                    raise RuntimeError(
                        f"Failed to prepare a full batch of {chunk} tokens"
                    )

            batch = batch.view(bs, seq + 1).to(
                training_device,
                non_blocking=training_device.type == "cuda",
            )
            next_ptr = candidate_ptr + chunk

            def forward_fn():
                if training_device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        return model(batch[:, :-1], labels=batch[:, 1:])
                return model(batch[:, :-1], labels=batch[:, 1:])

            out, _ = execute_training_iteration(
                model,
                opt,
                sched,
                forward_fn,
                max_grad_norm=1.0,
                failure_policy=failure_policy,
            )

            committed_cpt_version = get_cpt_state_version(model)
            if committed_cpt_version != step + 1:
                raise RuntimeError(
                    f"CPT commit returned version={committed_cpt_version}; "
                    f"expected {step + 1}"
                )

            # optimizer/scheduler/CPT 全部成功后，正式发布训练计数与数据位置。
            shard = candidate_shard
            fi = candidate_fi
            ptr = next_ptr
            total_tok += bs * seq
            step += 1

            stop_signal = _synchronize_stop_signal(model, stop_signal)

            if step % log_every == 0 and is_main_process():
                elapsed = time.time() - t0
                hours_elapsed = elapsed / 3600
                print(f"  step {step:7d}: loss={out['loss'].item():.4f} "
                      f"tok/s={(total_tok - tok_base) / elapsed:.0f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"shard={fi}/{len(files)} "
                      f"[{hours_elapsed:.1f}h]", flush=True)

            # ---- 保存 + eval ----
            periodic_save_due = _broadcast_rank0_flag(
                model,
                time.time() - last_save > save_every_min * 60,
            )
            if periodic_save_due:
                d = save_checkpoint(
                    model, opt, sched, output_dir, step, total_tok,
                    sa.get("warmup_steps", 0), sa.get("total_steps", 0), fi, ptr,
                    bs, seq,
                    run_id=metadata["run_id"],
                    code_manifest=metadata["code_manifest"],
                    data_manifest=metadata["data_manifest"],
                    tokenizer_manifest=metadata["tokenizer_manifest"],
                    data_files=files,
                    tokenizer_dir=tokenizer_dir,
                    optimizer_kind=optimizer_kind,
                    schedule_kind=sa["schedule_kind"],
                    schedule_decay_ratio=sa["schedule_decay_ratio"],
                )

                def periodic_postprocess():
                    print(f"  -> Saved {d}", flush=True)
                    prune_periodic_checkpoints(output_dir, keep_last_checkpoints)
                    if eval_on_save:
                        eval_dir = f"evals/{Path(output_dir).name}/step_{step:07d}"
                        mean, res = run_cpu_eval(d, eval_dir, tokenizer_dir)
                        if mean is not None and res:
                            parts = [
                                f'{task}={result.get("accuracy", result.get("matthews_correlation", result.get("f1", float("nan")))):.3f}'
                                for task, result in sorted(res.items())
                            ]
                            print(
                                f"  [eval] mean={mean:.4f} | {' '.join(parts)}",
                                flush=True,
                            )

                _run_rank0_coordinated(
                    model,
                    "checkpoint_postprocess",
                    periodic_postprocess,
                )
                last_save = time.time()
                last_saved_step = step

            if stop_signal is not None:
                if last_saved_step == step:
                    _run_rank0_coordinated(
                        model,
                        "emergency_checkpoint_notice",
                        lambda: print(
                            "  -> Current step was already checkpointed",
                            flush=True,
                        ),
                    )
                else:
                    d = save_checkpoint(
                        model, opt, sched, output_dir, step, total_tok,
                        sa.get("warmup_steps", 0), sa.get("total_steps", 0), fi, ptr,
                        bs, seq,
                        run_id=metadata["run_id"],
                        code_manifest=metadata["code_manifest"],
                        data_manifest=metadata["data_manifest"],
                        tokenizer_manifest=metadata["tokenizer_manifest"],
                        data_files=files,
                        tokenizer_dir=tokenizer_dir,
                        optimizer_kind=optimizer_kind,
                        schedule_kind=sa["schedule_kind"],
                        schedule_decay_ratio=sa["schedule_decay_ratio"],
                    )

                    def emergency_postprocess():
                        print(f"  -> Emergency checkpoint saved: {d}", flush=True)
                        prune_periodic_checkpoints(
                            output_dir,
                            keep_last_checkpoints,
                        )

                    _run_rank0_coordinated(
                        model,
                        "emergency_checkpoint_postprocess",
                        emergency_postprocess,
                    )
                raise SystemExit(128 + stop_signal)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    elapsed = time.time() - t0
    return step, total_tok, fi, ptr, elapsed


def final_save(model, opt, sched, output_dir, step, total_tok, elapsed,
               fi, ptr, batch_size, seq_len, schedule_args=None,
               checkpoint_metadata=None, data_files=None, tokenizer_dir=None,
               optimizer_kind=None):
    """保存最终 checkpoint。"""
    if not isinstance(checkpoint_metadata, dict):
        raise RuntimeError("strict checkpoint_metadata is required for final save")
    if not isinstance(schedule_args, dict) or set(schedule_args) != {
        "warmup_steps", "total_steps", "schedule_kind",
        "schedule_decay_ratio",
    }:
        raise RuntimeError("strict schedule_args is required for final save")
    optimizer_kind = _normalize_optimizer_kind(optimizer_kind)
    sa = schedule_args
    d = save_checkpoint(
        model, opt, sched, output_dir, step, total_tok,
        sa.get("warmup_steps", 0), sa.get("total_steps", 0), fi, ptr,
        batch_size, seq_len,
        run_id=checkpoint_metadata["run_id"],
        code_manifest=checkpoint_metadata["code_manifest"],
        data_manifest=checkpoint_metadata["data_manifest"],
        tokenizer_manifest=checkpoint_metadata["tokenizer_manifest"],
        data_files=data_files,
        tokenizer_dir=tokenizer_dir,
        optimizer_kind=optimizer_kind,
        schedule_kind=sa["schedule_kind"],
        schedule_decay_ratio=sa["schedule_decay_ratio"],
        final=True,
    )
    if is_main_process():
        print(f"\nDone: {step} steps {total_tok / 1e9:.3f}B tokens "
          f"session={elapsed / 3600:.1f}h → {d}", flush=True)
