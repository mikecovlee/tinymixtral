"""Micro-benchmark the legacy and aggregated training finite-state gates."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_utils


DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex64,
)


def make_checks(device, tensor_count, tensor_numel):
    generator = torch.Generator(device=device).manual_seed(20260731)
    checks = []
    for index in range(tensor_count):
        dtype = DTYPES[index % len(DTYPES)]
        if index % 41 == 0:
            tensor = torch.empty(0, dtype=dtype, device=device)
        elif dtype.is_complex:
            tensor = torch.complex(
                torch.randn(tensor_numel, device=device, generator=generator),
                torch.randn(tensor_numel, device=device, generator=generator),
            )
        else:
            tensor = torch.randn(
                tensor_numel,
                dtype=dtype,
                device=device,
                generator=generator,
            )
        checks.append((tensor, f"synthetic[{index}]"))
    return checks


def legacy_gate(checks):
    for value, message in checks:
        if not train_utils._numeric_leaf_is_finite(value):
            raise FloatingPointError(message)


def aggregate_gate(checks):
    train_utils._assert_numeric_checks_finite(checks)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def item_count(function, checks):
    original_item = torch.Tensor.item
    calls = []

    def counted_item(tensor, *args, **kwargs):
        calls.append(str(tensor.device))
        return original_item(tensor, *args, **kwargs)

    torch.Tensor.item = counted_item
    try:
        function(checks)
    finally:
        torch.Tensor.item = original_item
    return calls


def benchmark(function, checks, device, warmups, repeats):
    for _ in range(warmups):
        function(checks)
    synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        function(checks)
    synchronize(device)
    return (time.perf_counter() - started) * 1000.0 / repeats


def run_device(device, tensor_count, tensor_numel, warmups, repeats):
    checks = make_checks(device, tensor_count, tensor_numel)
    legacy_ms = benchmark(legacy_gate, checks, device, warmups, repeats)
    aggregate_ms = benchmark(aggregate_gate, checks, device, warmups, repeats)
    legacy_items = item_count(legacy_gate, checks)
    aggregate_items = item_count(aggregate_gate, checks)

    failure_checks = list(checks)
    failure_index = tensor_count // 2
    corrupted = failure_checks[failure_index][0].clone()
    if corrupted.numel() == 0:
        corrupted = torch.ones(1, dtype=torch.float32, device=device)
    corrupted.view(-1)[0] = complex(float("nan"), 0.0) if corrupted.is_complex() else float("nan")
    expected_message = f"synthetic[{failure_index}]"
    failure_checks[failure_index] = (corrupted, expected_message)
    observed_message = None
    try:
        aggregate_gate(failure_checks)
    except FloatingPointError as exc:
        observed_message = str(exc)
    if observed_message != expected_message:
        raise RuntimeError(
            f"slow diagnostic mismatch: expected {expected_message!r}, "
            f"got {observed_message!r}"
        )

    tensor_bytes = sum(value.numel() * value.element_size() for value, _ in checks)
    record = {
        "device": str(device),
        "tensor_count": tensor_count,
        "nonempty_tensor_count": sum(value.numel() > 0 for value, _ in checks),
        "tensor_bytes": tensor_bytes,
        "legacy_ms": legacy_ms,
        "aggregate_ms": aggregate_ms,
        "speedup": legacy_ms / aggregate_ms,
        "legacy_item_calls": len(legacy_items),
        "aggregate_item_calls": len(aggregate_items),
        "legacy_item_devices": sorted(set(legacy_items)),
        "aggregate_item_devices": sorted(set(aggregate_items)),
        "failure_diagnostic_exact": True,
    }
    del checks, failure_checks, corrupted
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-count", type=int, default=512)
    parser.add_argument("--tensor-numel", type=int, default=32768)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.tensor_count, args.tensor_numel, args.warmups, args.repeats) <= 0:
        raise ValueError("all benchmark dimensions must be positive")
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda", torch.cuda.current_device()))
    report = {
        "status": "passed",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "records": [
            run_device(
                device,
                args.tensor_count,
                args.tensor_numel,
                args.warmups,
                args.repeats,
            )
            for device in devices
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
