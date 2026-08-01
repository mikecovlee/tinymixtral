from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import sys
import time

import pytest
import torch

import scripts.verify_cpt_moe_gpu as gpu_verify
import scripts.verify_cpt_moe_deep_gpu as deep_gpu_verify


def test_gpu_reports_bind_their_own_verification_scripts():
    assert "scripts/verify_cpt_moe_gpu.py" in gpu_verify.CORE_SOURCE_FILES
    assert "scripts/verify_cpt_moe_deep_gpu.py" in deep_gpu_verify.CORE_FILES


def test_process_memory_probe_is_structured_and_nonnegative():
    stats = gpu_verify.process_memory_stats()
    assert isinstance(stats["supported"], bool)
    if stats["supported"]:
        for name in (
            "working_set_bytes",
            "peak_working_set_bytes",
            "private_bytes",
            "pagefile_bytes",
            "peak_pagefile_bytes",
            "page_fault_count",
        ):
            assert isinstance(stats[name], int)
            assert stats[name] >= 0


def test_atomic_gpu_report_publish_refuses_overwrite_and_cleans_temporary_files(
    tmp_path,
):
    output = tmp_path / "report.json"
    gpu_verify.atomic_write_json(output, {"run": 1})

    assert output.read_text(encoding="utf-8").strip() == '{\n  "run": 1\n}'
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        gpu_verify.atomic_write_json(output, {"run": 2})
    assert list(tmp_path.glob(".*.tmp-*")) == []


def test_atomic_gpu_report_concurrent_publish_has_exactly_one_winner(tmp_path):
    output = tmp_path / "report.json"

    def publish(run_id: int):
        try:
            gpu_verify.atomic_write_json(output, {"run": run_id})
            return "published"
        except FileExistsError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, (1, 2)))

    assert sorted(outcomes) == ["published", "rejected"]
    assert list(tmp_path.glob(".*.tmp-*")) == []


def test_deep_gpu_report_publish_refuses_overwrite_and_cleans_temporary_files(
    tmp_path,
):
    output = tmp_path / "deep-report.json"
    deep_gpu_verify.atomic_write_json(output, {"run": 1})

    assert output.read_text(encoding="utf-8").strip() == '{\n  "run": 1\n}'
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        deep_gpu_verify.atomic_write_json(output, {"run": 2})
    assert list(tmp_path.glob(".*.tmp-*")) == []


def test_deep_gpu_report_concurrent_publish_has_exactly_one_winner(tmp_path):
    output = tmp_path / "deep-report.json"

    def publish(run_id: int):
        try:
            deep_gpu_verify.atomic_write_json(output, {"run": run_id})
            return "published"
        except FileExistsError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, (1, 2)))

    assert sorted(outcomes) == ["published", "rejected"]
    assert list(tmp_path.glob(".*.tmp-*")) == []


def test_deep_gpu_cli_refuses_existing_output_before_running(
    monkeypatch,
    tmp_path,
    capsys,
):
    output = tmp_path / "existing.json"
    output.write_text("old evidence", encoding="utf-8")
    monkeypatch.setattr(deep_gpu_verify, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_cpt_moe_deep_gpu.py", "--output", str(output)],
    )

    with pytest.raises(SystemExit) as error:
        deep_gpu_verify.parse_args()

    assert error.value.code == 2
    assert "overwrite is refused" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == "old evidence"


@pytest.mark.parametrize(
    (
        "successful_steps",
        "requested_steps",
        "attempt_statuses",
        "resource_stop",
        "budget_stop",
        "expected",
    ),
    [
        (2, 2, ["pass", "pass"], False, False, "pass"),
        (
            1,
            2,
            ["pass", "error"],
            False,
            False,
            "training_failed_after_success",
        ),
        (0, 2, ["oom"], False, False, "training_failed"),
        (
            1,
            1,
            ["recovered_oom", "pass"],
            False,
            False,
            "pass_with_shorter_sequence_fallback",
        ),
        (1, 2, ["pass"], True, False, "partial_resource_budget"),
        (1, 2, ["pass"], False, True, "partial_resource_budget"),
        (0, 2, [], False, True, "forward_only"),
    ],
)
def test_default_training_status_never_hides_failed_requested_step(
    successful_steps,
    requested_steps,
    attempt_statuses,
    resource_stop,
    budget_stop,
    expected,
):
    attempts = [
        {
            "status": "oom" if status == "recovered_oom" else status,
            "recovered_by_shorter_sequence_retry": status == "recovered_oom",
        }
        for status in attempt_statuses
    ]
    assert (
        gpu_verify.classify_default_training_status(
            successful_steps=successful_steps,
            requested_steps=requested_steps,
            attempts=attempts,
            stopped_for_resource_margin=resource_stop,
            stopped_for_soft_budget=budget_stop,
        )
        == expected
    )


class _FakeDefaultModel:
    def __init__(self, identity: int):
        self.identity = identity
        self.num_parameters = 1
        self._use_activation_checkpointing = False
        price = SimpleNamespace(grad=None)
        self.layers = [SimpleNamespace(moe=SimpleNamespace(cpt_router=SimpleNamespace(
            congestion_price=price
        )))]

    def to(self, *args, **kwargs):
        return self

    def gradient_checkpointing_enable(self):
        self._use_activation_checkpointing = True

    def parameters(self):
        return iter(())


class _FakeOptimizer:
    def zero_grad(self, set_to_none=True):
        return None


class _FakeRecorder:
    def __init__(self, *args, **kwargs):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    ("failure_policy", "expected_constructions"),
    [("fail-stop", 2), ("exact-rollback", 1)],
)
def test_default_oom_fallback_rebuilds_only_poisoned_fail_stop_runtime(
    monkeypatch,
    failure_policy,
    expected_constructions,
):
    constructions: list[_FakeDefaultModel] = []
    attempted_models: list[_FakeDefaultModel] = []

    def construct(_config):
        model = _FakeDefaultModel(len(constructions))
        constructions.append(model)
        return model

    def training_attempt(model, *_args, **_kwargs):
        attempted_models.append(model)
        if len(attempted_models) == 1:
            return {
                "status": "oom",
                "duration_seconds": 0.01,
                "training_failure_policy": failure_policy,
            }
        return {
            "status": "pass",
            "duration_seconds": 0.01,
            "training_failure_policy": failure_policy,
        }

    batch16 = (
        torch.zeros(1, 16, dtype=torch.int64),
        torch.zeros(1, 16, dtype=torch.int64),
        torch.ones(1, 16, dtype=torch.bool),
    )
    batch8 = tuple(value[:, :8].clone() for value in batch16)

    monkeypatch.setattr(gpu_verify, "clean_cuda", lambda: None)
    monkeypatch.setattr(gpu_verify, "reset_cuda_peak", lambda: None)
    monkeypatch.setattr(gpu_verify, "seed_everything", lambda _seed: None)
    monkeypatch.setattr(gpu_verify, "TinyMixtralConfig", lambda: object())
    monkeypatch.setattr(gpu_verify, "TinyMixtralForCausalLM", construct)
    monkeypatch.setattr(
        gpu_verify,
        "run_default_forward",
        lambda *_args, **_kwargs: ({"status": "pass"}, batch16),
    )
    monkeypatch.setattr(gpu_verify, "make_default_batch", lambda *_args, **_kwargs: batch8)
    monkeypatch.setattr(gpu_verify, "low_memory_adamw", lambda *_args, **_kwargs: _FakeOptimizer())
    monkeypatch.setattr(gpu_verify, "GradientRecorder", _FakeRecorder)
    monkeypatch.setattr(gpu_verify, "default_gradient_names", lambda: ())
    monkeypatch.setattr(gpu_verify, "run_default_training_attempt", training_attempt)
    monkeypatch.setattr(gpu_verify, "model_versions", lambda _model: [0])
    monkeypatch.setattr(gpu_verify, "model_prices", lambda _model: [[0.0]])
    monkeypatch.setattr(gpu_verify, "price_summary", lambda _model: {"max": 0.0})
    monkeypatch.setattr(gpu_verify, "parameter_and_buffer_dtypes", lambda _model: {})
    monkeypatch.setattr(gpu_verify, "tensor_bytes", lambda _values: 0)
    monkeypatch.setattr(gpu_verify, "optimizer_state_tensor_bytes", lambda _optimizer: 0)
    monkeypatch.setattr(gpu_verify, "memory_stats", lambda: {})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    result = gpu_verify.run_default_experiment(
        max_steps=1,
        script_started=time.perf_counter(),
        soft_budget_seconds=1000.0,
        failure_policy=failure_policy,
    )

    assert result["status"] == "pass_with_shorter_sequence_fallback"
    assert result["successful_strict_steps"] == 1
    assert len(constructions) == expected_constructions
    assert len(attempted_models) == 2
    if failure_policy == "fail-stop":
        assert attempted_models[1] is not attempted_models[0]
        assert len(result["fail_stop_rebuilds"]) == 1
    else:
        assert attempted_models[1] is attempted_models[0]
        assert result["fail_stop_rebuilds"] == []
