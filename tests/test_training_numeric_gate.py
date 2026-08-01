import math

import numpy as np
import pytest
import torch

import scripts.train_utils as train_utils


class _NumericModel(torch.nn.Module):
    def __init__(self, *, device="cpu"):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([1.0, -2.0], dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "numeric_buffer",
            torch.tensor([3.0], dtype=torch.float64, device=device),
        )

    def validate_cpt_transaction(self, transaction):
        return None

    def commit_cpt_transaction(self, transaction):
        return None

    def abort_cpt_transaction(self, transaction):
        return None

    def get_cpt_state_version(self):
        return 0


class _TensorStateScheduler:
    def __init__(self, optimizer, *, device="cpu"):
        self.optimizer = optimizer
        self.tensor_state = torch.tensor([1.0], dtype=torch.float64, device=device)
        self.last_lr = [float(optimizer.param_groups[0]["lr"])]

    def step(self):
        self.optimizer.param_groups[0]["lr"] *= 0.9
        self.last_lr = [float(self.optimizer.param_groups[0]["lr"])]

    def state_dict(self):
        return {"tensor_state": self.tensor_state, "last_lr": self.last_lr}

    def load_state_dict(self, state):
        self.tensor_state.copy_(state["tensor_state"])
        self.last_lr = list(state["last_lr"])

    def get_last_lr(self):
        return list(self.last_lr)


def _optimizer_with_nested_state(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    optimizer.state[parameter]["nested"] = {
        "empty": torch.empty(0, device=parameter.device, dtype=torch.float32),
        "mixed": [
            torch.tensor([1.0], device=parameter.device, dtype=torch.float16),
            torch.tensor([2.0], device=parameter.device, dtype=torch.bfloat16),
            torch.tensor(
                [complex(torch.finfo(torch.float32).max, torch.finfo(torch.float32).max)],
                device=parameter.device,
                dtype=torch.complex64,
            ),
        ],
    }
    return optimizer


def test_aggregate_gate_accepts_empty_mixed_dtype_and_extreme_complex_cpu():
    values = {
        "empty": torch.empty(0, dtype=torch.float32),
        "float16": torch.tensor([1.0], dtype=torch.float16),
        "bfloat16": torch.tensor([-2.0], dtype=torch.bfloat16),
        "float64": torch.tensor([3.0], dtype=torch.float64),
        "complex": torch.tensor(
            [complex(torch.finfo(torch.float32).max, torch.finfo(torch.float32).max)],
            dtype=torch.complex64,
        ),
        "numpy": np.asarray([1.0, -4.0], dtype=np.float64),
        "scalar": 2.5,
        "ignored_integer": torch.tensor([torch.iinfo(torch.int64).max]),
    }
    checks = []
    train_utils._append_nested_numeric_checks(checks, values, "root")

    assert train_utils._fast_numeric_checks_are_finite(checks)
    train_utils._assert_numeric_checks_finite(checks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_aggregate_gate_accepts_empty_mixed_dtype_and_extreme_complex_cuda():
    values = [
        torch.empty(0, device="cuda", dtype=torch.float32),
        torch.tensor([1.0], device="cuda", dtype=torch.float16),
        torch.tensor([2.0], device="cuda", dtype=torch.bfloat16),
        torch.tensor([3.0], device="cuda", dtype=torch.float64),
        torch.tensor(
            [complex(torch.finfo(torch.float32).max, torch.finfo(torch.float32).max)],
            device="cuda",
            dtype=torch.complex64,
        ),
    ]
    checks = [(value, f"bad value {index}") for index, value in enumerate(values)]

    assert train_utils._fast_numeric_checks_are_finite(checks)
    train_utils._assert_numeric_checks_finite(checks)


def test_aggregate_gate_synchronizes_once_per_represented_cpu_device(monkeypatch):
    checks = [
        (torch.ones(128, dtype=dtype), f"bad {dtype}")
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    ]
    checks.extend(
        [
            (torch.ones(64, dtype=torch.complex64), "bad complex"),
            (torch.empty(0), "bad empty"),
            (np.ones(16, dtype=np.float32), "bad numpy"),
        ]
    )
    original_item = torch.Tensor.item
    item_devices = []

    def counted_item(tensor, *args, **kwargs):
        item_devices.append(str(tensor.device))
        return original_item(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", counted_item)

    assert train_utils._fast_numeric_checks_are_finite(checks)
    assert item_devices == ["cpu"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_aggregate_gate_synchronizes_once_per_cpu_and_cuda_device(monkeypatch):
    checks = [
        (torch.ones(128, dtype=torch.float32), "bad cpu"),
        (torch.ones(128, device="cuda", dtype=torch.float16), "bad cuda fp16"),
        (torch.ones(128, device="cuda", dtype=torch.complex64), "bad cuda complex"),
    ]
    original_item = torch.Tensor.item
    item_devices = []

    def counted_item(tensor, *args, **kwargs):
        item_devices.append(str(tensor.device))
        return original_item(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", counted_item)

    assert train_utils._fast_numeric_checks_are_finite(checks)
    assert sorted(item_devices) == ["cpu", "cuda:0"]


def test_aggregate_failure_uses_original_first_error_path():
    checks = [
        (torch.tensor([1.0]), "first finite"),
        (torch.tensor([float("nan")]), "expected first failure"),
        (torch.tensor([float("inf")]), "later failure"),
    ]

    with pytest.raises(FloatingPointError, match="expected first failure"):
        train_utils._assert_numeric_checks_finite(checks)


def test_optimizer_nested_state_covers_empty_mixed_dtype_and_complex_failure():
    model = _NumericModel()
    optimizer = _optimizer_with_nested_state(model)

    train_utils._validate_training_numerics(model, optimizer, None)

    parameter = next(model.parameters())
    optimizer.state[parameter]["nested"]["mixed"][2][0] = complex(float("nan"), 0.0)
    with pytest.raises(
        FloatingPointError,
        match=r"Non-finite numeric state at optimizer\.state\.0\.nested\.mixed\[2\]",
    ):
        train_utils._validate_training_numerics(model, optimizer, None)


def test_full_training_gate_uses_one_item_for_one_device(monkeypatch):
    model = _NumericModel()
    optimizer = _optimizer_with_nested_state(model)
    scheduler = _TensorStateScheduler(optimizer)
    original_item = torch.Tensor.item
    item_devices = []

    def counted_item(tensor, *args, **kwargs):
        item_devices.append(str(tensor.device))
        return original_item(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", counted_item)

    train_utils._validate_training_numerics(model, optimizer, scheduler)
    assert item_devices == ["cpu"]


def test_post_scheduler_gate_skips_unchanged_model_and_optimizer_tensors(monkeypatch):
    model = _NumericModel()
    optimizer = _optimizer_with_nested_state(model)
    scheduler = _TensorStateScheduler(optimizer)
    guarded_tensor_ids = {
        id(value)
        for value in list(model.parameters()) + list(model.buffers())
    }
    for state in optimizer.state.values():
        for _, tensor in train_utils._iter_nested_numeric_tensors(state, "state"):
            guarded_tensor_ids.add(id(tensor))
    guard = train_utils._capture_non_scheduler_tensor_guard(model, optimizer)
    observed_ids = set()
    original_fast_gate = train_utils._fast_numeric_checks_are_finite

    def recording_fast_gate(checks):
        observed_ids.update(
            id(value) for value, _ in checks if isinstance(value, torch.Tensor)
        )
        return original_fast_gate(checks)

    monkeypatch.setattr(
        train_utils,
        "_fast_numeric_checks_are_finite",
        recording_fast_gate,
    )
    scheduler.step()
    train_utils._validate_post_scheduler_numerics(
        model,
        optimizer,
        scheduler,
        guard,
    )

    assert observed_ids.isdisjoint(guarded_tensor_ids)
    assert id(scheduler.tensor_state) in observed_ids


def test_scheduler_out_of_scope_optimizer_tensor_corruption_falls_back_to_full_gate():
    model = _NumericModel()
    optimizer = _optimizer_with_nested_state(model)
    scheduler = _TensorStateScheduler(optimizer)
    guard = train_utils._capture_non_scheduler_tensor_guard(model, optimizer)
    parameter = next(model.parameters())

    optimizer.state[parameter]["momentum_buffer"][0] = float("nan")

    with pytest.raises(
        FloatingPointError,
        match=r"Non-finite numeric state at optimizer\.state\.0\.momentum_buffer",
    ):
        train_utils._validate_post_scheduler_numerics(
            model,
            optimizer,
            scheduler,
            guard,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_full_training_gate_uses_one_cuda_item_for_cuda_state(monkeypatch):
    model = _NumericModel(device="cuda")
    optimizer = _optimizer_with_nested_state(model)
    scheduler = _TensorStateScheduler(optimizer, device="cuda")
    original_item = torch.Tensor.item
    item_devices = []

    def counted_item(tensor, *args, **kwargs):
        item_devices.append(str(tensor.device))
        return original_item(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", counted_item)

    train_utils._validate_training_numerics(model, optimizer, scheduler)
    assert item_devices == ["cuda:0"]


def test_numeric_leaf_scalar_semantics_match_math_isfinite():
    for value in (0.0, -1.5, np.float32(2.0)):
        assert train_utils._numeric_leaf_is_finite(value)
    for value in (float("nan"), float("inf"), np.float64(-math.inf)):
        assert not train_utils._numeric_leaf_is_finite(value)
