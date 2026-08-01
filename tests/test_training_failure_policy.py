import copy
from datetime import timedelta
from pathlib import Path
import random
import types

import numpy as np
import pytest
import torch
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp

import scripts.train_utils as train_utils
from scripts.train_utils import (
    TrainingFailStop,
    build_code_manifest,
    build_data_manifest,
    build_tokenizer_manifest,
    capture_rng_state,
    execute_training_iteration,
    make_adamw,
    make_cosine_schedule,
    make_wsd_schedule,
    restore_rank_rng_state,
    save_checkpoint,
    validate_training_state,
)


class _RouterView:
    def __init__(self, model):
        self.model = model

    @property
    def anchors(self):
        return self.model.anchor


class _TransactionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.anchor = torch.nn.Parameter(torch.tensor([[1.0]]))
        self.register_buffer("version", torch.zeros((), dtype=torch.int64))
        self.aborts = 0
        self.commits = 0

    def output(self):
        return {
            "loss": self.weight.square() + 0.0 * self.anchor.sum(),
            "cpt_transaction": {"base_version": int(self.version.item())},
        }

    def validate_cpt_transaction(self, transaction):
        if transaction["base_version"] != int(self.version.item()):
            raise RuntimeError("stale transaction")

    def commit_cpt_transaction(self, transaction):
        self.validate_cpt_transaction(transaction)
        self.version.add_(1)
        self.commits += 1

    def abort_cpt_transaction(self, transaction):
        self.aborts += 1

    def get_cpt_state_version(self):
        return int(self.version.item())

    def _cpt_routers(self):
        return (_RouterView(self),)


class _Scheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, state):
        self.steps = state["steps"]

    def get_last_lr(self):
        return [0.1]


class _UpdateThenFailSGD(torch.optim.SGD):
    def step(self, closure=None):
        result = super().step(closure)
        raise RuntimeError("injected partial optimizer failure")


class _DistributedRouter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 1
        self.projection = torch.nn.Parameter(
            torch.tensor([[0.5]], dtype=torch.float32)
        )
        self.register_buffer(
            "congestion_price",
            torch.zeros(1, dtype=torch.float32),
        )
        self.register_buffer("state_version", torch.zeros((), dtype=torch.int64))


class _DistributedTransactionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.router = _DistributedRouter()
        self.aborts = 0
        self.commits = 0

    def forward(self):
        return {
            "loss": self.weight.square() + self.router.projection.square().sum(),
            "cpt_transaction": {
                "base_version": int(self.router.state_version.item())
            },
        }

    def validate_cpt_transaction(self, transaction):
        if transaction["base_version"] != int(self.router.state_version.item()):
            raise RuntimeError("stale distributed transaction")

    def commit_cpt_transaction(self, transaction):
        self.validate_cpt_transaction(transaction)
        self.router.state_version.add_(1)
        self.commits += 1

    def abort_cpt_transaction(self, transaction):
        self.aborts += 1

    def get_cpt_state_version(self):
        return int(self.router.state_version.item())

    def _cpt_routers(self):
        return (self.router,)


class _RankSelectiveUpdateThenFailSGD(torch.optim.SGD):
    def __init__(self, parameters, *, rank):
        super().__init__(parameters, lr=0.1, momentum=0.9)
        self.rank = rank

    def step(self, closure=None):
        result = super().step(closure)
        if self.rank == 1:
            raise RuntimeError("rank 1 injected failure after optimizer write")
        return result


def _two_rank_fail_stop_worker(rank, world_size, store_path, result_dir):
    torch_dist.init_process_group(
        backend="gloo",
        init_method=Path(store_path).resolve().as_uri(),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        torch.manual_seed(991)
        local_model = _DistributedTransactionModel()
        model = torch.nn.parallel.DistributedDataParallel(local_model)
        optimizer = _RankSelectiveUpdateThenFailSGD(model.parameters(), rank=rank)
        scheduler = _Scheduler()
        initial_weight = local_model.weight.detach().clone()
        initial_projection = local_model.router.projection.detach().clone()
        error = None
        retry_error = None

        try:
            execute_training_iteration(
                model,
                optimizer,
                scheduler,
                model,
                failure_policy="fail-stop",
            )
        except BaseException as exc:
            error = exc

        try:
            execute_training_iteration(
                model,
                optimizer,
                scheduler,
                model,
                failure_policy="fail-stop",
            )
        except BaseException as exc:
            retry_error = exc

        result = {
            "rank": rank,
            "error_type": type(error).__name__ if error is not None else None,
            "error": str(error) if error is not None else None,
            "retry_error_type": (
                type(retry_error).__name__ if retry_error is not None else None
            ),
            "retry_error": str(retry_error) if retry_error is not None else None,
            "model_poisoned": hasattr(
                local_model, "_cpt_training_poison_reason"
            ),
            "optimizer_poisoned": hasattr(
                optimizer, "_cpt_training_poison_reason"
            ),
            "scheduler_poisoned": hasattr(
                scheduler, "_cpt_training_poison_reason"
            ),
            "version": local_model.get_cpt_state_version(),
            "commits": local_model.commits,
            "aborts": local_model.aborts,
            "scheduler_steps": scheduler.steps,
            "gradients_cleared": all(
                parameter.grad is None for parameter in model.parameters()
            ),
            "weight_was_written": not torch.equal(
                local_model.weight.detach(), initial_weight
            ),
            "projection_was_written": not torch.equal(
                local_model.router.projection.detach(), initial_projection
            ),
        }
        torch.save(result, Path(result_dir) / f"rank_{rank}.pt")
    finally:
        torch_dist.destroy_process_group()


def _tiny_recovery_config():
    from model.config import TinyMixtralConfig

    return TinyMixtralConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        expert_intermediate_size=24,
        cpt_num_prototypes=4,
        cpt_projection_dim=8,
        cpt_init_seed=37,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        attention_dropout=0.2,
        tie_word_embeddings=True,
        initializer_range=0.02,
    )


def _clone_nested(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    return copy.deepcopy(value)


def _assert_nested_exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested_exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_exact(actual_item, expected_item)
    else:
        assert actual == expected


def _prepare_recovery_artifacts(root):
    data_dir = root / "data"
    tokenizer_dir = root / "tokenizer"
    checkpoint_dir = root / "checkpoints"
    data_dir.mkdir(parents=True)
    tokenizer_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)

    shard_paths = [
        data_dir / "train_00000.pt",
        data_dir / "train_00001.pt",
    ]
    torch.save(torch.arange(15, dtype=torch.int64) % 32, shard_paths[0])
    torch.save(torch.arange(15, 35, dtype=torch.int64) % 32, shard_paths[1])
    (tokenizer_dir / "tokenizer.json").write_text(
        '{"version":"1.0"}',
        encoding="utf-8",
    )
    (tokenizer_dir / "tokenizer_config.json").write_text(
        '{"model_max_length":16}',
        encoding="utf-8",
    )
    metadata = {
        "run_id": "f" * 32,
        "code_manifest": build_code_manifest(),
        "data_manifest": build_data_manifest(shard_paths),
        "tokenizer_manifest": build_tokenizer_manifest(tokenizer_dir),
    }
    return checkpoint_dir, shard_paths, tokenizer_dir, metadata


def _candidate_batch(shards, cursor, *, batch_size=1, seq_len=4):
    chunk = batch_size * (seq_len + 1)
    candidate_fi = cursor["fi"]
    candidate_ptr = cursor["ptr"]
    candidate_shard = shards[candidate_fi]
    if candidate_ptr + chunk > len(candidate_shard):
        for _ in range(len(shards)):
            candidate_fi = (candidate_fi + 1) % len(shards)
            candidate_shard = shards[candidate_fi]
            candidate_ptr = 0
            if len(candidate_shard) >= chunk:
                break
        else:
            raise RuntimeError("No recovery-test shard contains one full batch")
    batch = candidate_shard[candidate_ptr:candidate_ptr + chunk]
    if batch.numel() != chunk:
        raise RuntimeError("Recovery-test shard produced a short batch")
    return batch.view(batch_size, seq_len + 1), candidate_fi, candidate_ptr + chunk


def _run_replay_step(model, optimizer, scheduler, shards, cursor):
    batch, candidate_fi, next_ptr = _candidate_batch(shards, cursor)
    probe = {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "torch": torch.rand(5),
    }

    def forward_fn():
        return model(batch[:, :-1], labels=batch[:, 1:])

    output, grad_norm = execute_training_iteration(
        model,
        optimizer,
        scheduler,
        forward_fn,
        failure_policy="fail-stop",
    )
    cursor["fi"] = candidate_fi
    cursor["ptr"] = next_ptr
    cursor["step"] += 1
    cursor["total_tok"] += 4
    return {
        "batch": batch.detach().clone(),
        "probe": probe,
        "loss": output["loss"].detach().clone(),
        "grad_norm": grad_norm.detach().clone(),
        "cursor": dict(cursor),
    }


def _load_recovery_checkpoint(checkpoint, metadata):
    from model.modeling import TinyMixtralForCausalLM

    state = torch.load(
        checkpoint / "training_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = TinyMixtralForCausalLM.from_pretrained(str(checkpoint)).train()
    optimizer_kind = state["optimizer_kind"]
    saved_groups = state["opt"]["param_groups"]
    saved_lr = float(saved_groups[0]["lr"])
    saved_weight_decay = max(
        float(group["weight_decay"]) for group in saved_groups
    )
    optimizer = make_adamw(
        model,
        lr=saved_lr,
        weight_decay=saved_weight_decay,
        bf16_states=optimizer_kind == "bf16_adamw",
    )
    optimizer.load_state_dict(state["opt"])
    saved_lrs = [group["lr"] for group in optimizer.param_groups]
    if state["schedule_kind"] == "wsd":
        scheduler = make_wsd_schedule(
            optimizer,
            warmup_steps=state["warmup_steps"],
            total_steps=state["total_steps"],
            decay_ratio=state["schedule_decay_ratio"],
        )
    else:
        scheduler = make_cosine_schedule(
            optimizer,
            warmup_steps=state["warmup_steps"],
            total_steps=state["total_steps"],
        )
    for group, saved_group_lr in zip(optimizer.param_groups, saved_lrs):
        group["lr"] = saved_group_lr
    scheduler.load_state_dict(state["sched"])
    validate_training_state(
        state,
        checkpoint_step=state["step"],
        model=model,
        expected_run_id=metadata["run_id"],
        expected_code_manifest=metadata["code_manifest"],
        expected_data_manifest=metadata["data_manifest"],
        expected_tokenizer_manifest=metadata["tokenizer_manifest"],
        expected_world_size=1,
    )
    restore_rank_rng_state(state, rank=0, world_size=1)
    cursor = {
        "step": state["step"],
        "total_tok": state["total_tok"],
        "fi": state["fi"],
        "ptr": state["ptr"],
    }
    return model, optimizer, scheduler, cursor, state


def test_fail_stop_success_path_never_captures_full_cpu_snapshot(monkeypatch):
    model = _TransactionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _Scheduler()

    def forbidden_snapshot(*args, **kwargs):
        raise AssertionError("fail-stop must not capture a full training snapshot")

    monkeypatch.setattr(train_utils, "capture_training_step_entry", forbidden_snapshot)
    monkeypatch.setattr(train_utils, "_capture_training_snapshot", forbidden_snapshot)

    output, grad_norm = execute_training_iteration(
        model,
        optimizer,
        scheduler,
        model.output,
        failure_policy="fail-stop",
    )

    assert torch.isfinite(output["loss"])
    assert torch.isfinite(grad_norm)
    assert model.get_cpt_state_version() == 1
    assert model.commits == 1
    assert model.aborts == 0
    assert scheduler.steps == 1
    assert not hasattr(model, "_cpt_training_poison_reason")


def test_fail_stop_partial_optimizer_write_poisoned_and_cannot_continue_or_save(
    tmp_path,
    monkeypatch,
):
    model = _TransactionModel()
    optimizer = _UpdateThenFailSGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = _Scheduler()
    model_before = copy.deepcopy(model.state_dict())

    def forbidden_snapshot(*args, **kwargs):
        raise AssertionError("fail-stop must not capture a full training snapshot")

    monkeypatch.setattr(train_utils, "capture_training_step_entry", forbidden_snapshot)
    monkeypatch.setattr(train_utils, "_capture_training_snapshot", forbidden_snapshot)

    with pytest.raises(TrainingFailStop, match="must not be reused or checkpointed"):
        execute_training_iteration(
            model,
            optimizer,
            scheduler,
            model.output,
            failure_policy="fail-stop",
        )

    assert model.get_cpt_state_version() == 0
    assert model.commits == 0
    assert model.aborts == 1
    assert scheduler.steps == 0
    assert model.weight.grad is None
    assert not torch.equal(model.state_dict()["weight"], model_before["weight"])
    assert "injected partial optimizer failure" in model._cpt_training_poison_reason

    with pytest.raises(TrainingFailStop, match="prior fail-stop event"):
        execute_training_iteration(
            model,
            optimizer,
            scheduler,
            model.output,
            failure_policy="fail-stop",
        )

    with pytest.raises(TrainingFailStop, match="prior fail-stop event"):
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            tmp_path,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            optimizer_kind="adamw",
            schedule_kind="cosine",
            schedule_decay_ratio=None,
            run_id="0" * 32,
            code_manifest={},
            data_manifest={},
            tokenizer_manifest={},
            data_files=(),
            tokenizer_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_fail_stop_forward_failure_poisoned_without_advancing_transaction(monkeypatch):
    model = _TransactionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _Scheduler()

    def forbidden_snapshot(*args, **kwargs):
        raise AssertionError("fail-stop must not capture a full training snapshot")

    monkeypatch.setattr(train_utils, "capture_training_step_entry", forbidden_snapshot)

    def failed_forward():
        torch.rand(3)
        raise RuntimeError("injected forward failure")

    with pytest.raises(TrainingFailStop, match="Forward failed under fail-stop policy"):
        execute_training_iteration(
            model,
            optimizer,
            scheduler,
            failed_forward,
            failure_policy="fail-stop",
        )

    assert model.get_cpt_state_version() == 0
    assert model.commits == 0
    assert scheduler.steps == 0
    assert "injected forward failure" in model._cpt_training_poison_reason


def test_exact_rollback_remains_exact_after_partial_optimizer_write():
    model = _TransactionModel()
    optimizer = _UpdateThenFailSGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = _Scheduler()
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    rng_before = torch.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="injected partial optimizer failure"):
        execute_training_iteration(
            model,
            optimizer,
            scheduler,
            lambda: (torch.rand(5), model.output())[1],
            failure_policy="exact-rollback",
        )

    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, model_before[key], rtol=0, atol=0)
    assert optimizer.state_dict() == optimizer_before
    assert scheduler.state_dict() == scheduler_before
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    assert model.get_cpt_state_version() == 0
    assert not hasattr(model, "_cpt_training_poison_reason")


def test_exact_rollback_failure_after_optimizer_error_poisoned_live_state(monkeypatch):
    model = _TransactionModel()
    optimizer = _UpdateThenFailSGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = _Scheduler()

    def incomplete_restore(active_model, active_optimizer, active_scheduler, snapshot):
        with torch.no_grad():
            active_model.weight.add_(17.0)
        raise RuntimeError("injected model rollback failure")

    monkeypatch.setattr(
        train_utils,
        "_restore_training_snapshot",
        incomplete_restore,
    )

    with pytest.raises(
        TrainingFailStop,
        match="exact rollback was incomplete",
    ):
        execute_training_iteration(
            model,
            optimizer,
            scheduler,
            model.output,
            failure_policy="exact-rollback",
        )

    assert "injected model rollback failure" in model._cpt_training_poison_reason
    assert hasattr(optimizer, "_cpt_training_poison_reason")
    assert hasattr(scheduler, "_cpt_training_poison_reason")
    assert model.get_cpt_state_version() == 0
    assert model.commits == 0

    with pytest.raises(TrainingFailStop, match="prior fail-stop event"):
        execute_training_iteration(
            model,
            optimizer,
            scheduler,
            model.output,
            failure_policy="exact-rollback",
        )


def test_exact_rollback_failure_after_forward_error_poisoned_live_state(monkeypatch):
    model = _TransactionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _Scheduler()

    def incomplete_restore(active_model, active_optimizer, active_scheduler, snapshot):
        raise RuntimeError("injected forward rollback failure")

    monkeypatch.setattr(
        train_utils,
        "_restore_training_snapshot",
        incomplete_restore,
    )

    def failed_forward():
        raise RuntimeError("injected forward failure before output")

    with pytest.raises(
        TrainingFailStop,
        match="Forward failed and exact rollback was incomplete",
    ):
        execute_training_iteration(
            model,
            optimizer,
            scheduler,
            failed_forward,
            failure_policy="exact-rollback",
        )

    assert "injected forward rollback failure" in model._cpt_training_poison_reason
    assert hasattr(optimizer, "_cpt_training_poison_reason")
    assert hasattr(scheduler, "_cpt_training_poison_reason")
    assert model.get_cpt_state_version() == 0
    assert model.commits == 0


@pytest.mark.parametrize("raw_world_size", [None, "2"])
def test_training_entry_rejects_preinitialized_multi_rank_process_group(
    monkeypatch,
    raw_world_size,
):
    if raw_world_size is None:
        monkeypatch.delenv("WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("WORLD_SIZE", raw_world_size)
    monkeypatch.setattr(train_utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(train_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(train_utils.dist, "get_world_size", lambda: 2)

    with pytest.raises(RuntimeError, match="End-to-end DDP training entrypoint"):
        train_utils.reject_uninitialized_torchrun_environment()


@pytest.mark.parametrize("policy", [None, True, "rollback", "FAIL-STOP", ""])
def test_failure_policy_rejects_noncanonical_values(policy):
    model = _TransactionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="failure_policy"):
        execute_training_iteration(
            model,
            optimizer,
            None,
            model.output,
            failure_policy=policy,
        )


@pytest.mark.skipif(
    not torch_dist.is_available() or not torch_dist.is_gloo_available(),
    reason="PyTorch Gloo distributed backend is unavailable",
)
def test_two_rank_gloo_partial_optimizer_failure_poisoned_everywhere_without_commit(
    tmp_path,
):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    torch_mp.spawn(
        _two_rank_fail_stop_worker,
        args=(
            2,
            str(tmp_path / "gloo_fail_stop"),
            str(result_dir),
        ),
        nprocs=2,
        join=True,
    )
    results = [
        torch.load(
            result_dir / f"rank_{rank}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for rank in range(2)
    ]

    assert [result["rank"] for result in results] == [0, 1]
    assert all(result["error_type"] == "TrainingFailStop" for result in results)
    assert all(
        "must not be reused or checkpointed" in result["error"]
        for result in results
    )
    assert all(
        result["retry_error_type"] == "TrainingFailStop" for result in results
    )
    assert all(
        "prior fail-stop event" in result["retry_error"] for result in results
    )
    assert all(result["model_poisoned"] for result in results)
    assert all(result["optimizer_poisoned"] for result in results)
    assert all(result["scheduler_poisoned"] for result in results)
    assert all(result["version"] == 0 for result in results)
    assert all(result["commits"] == 0 for result in results)
    assert all(result["aborts"] == 1 for result in results)
    assert all(result["scheduler_steps"] == 0 for result in results)
    assert all(result["gradients_cleared"] for result in results)
    assert all(result["weight_was_written"] for result in results)
    assert all(result["projection_was_written"] for result in results)
    assert "rank 1 injected failure after optimizer write" in results[1]["error"]
    assert "failed on another distributed rank" in results[0]["error"]


@pytest.mark.parametrize(
    ("optimizer_kind", "schedule_kind", "schedule_decay_ratio"),
    [
        ("adamw", "cosine", None),
        ("bf16_adamw", "wsd", 0.5),
    ],
)
def test_fail_stop_checkpoint_recovery_replays_cursor_rng_and_next_batch_exactly(
    tmp_path,
    optimizer_kind,
    schedule_kind,
    schedule_decay_ratio,
):
    from model.modeling import TinyMixtralForCausalLM

    root = tmp_path / "fail_stop_recovery"
    checkpoint_dir, shard_paths, tokenizer_dir, metadata = (
        _prepare_recovery_artifacts(root)
    )
    shards = [torch.load(path, weights_only=True) for path in shard_paths]
    random.seed(7401)
    np.random.seed(7401)
    torch.manual_seed(7401)
    model = TinyMixtralForCausalLM(_tiny_recovery_config()).train()
    optimizer = make_adamw(
        model,
        lr=1e-3,
        weight_decay=0.0,
        bf16_states=optimizer_kind == "bf16_adamw",
    )
    if schedule_kind == "wsd":
        scheduler = make_wsd_schedule(
            optimizer,
            warmup_steps=1,
            total_steps=8,
            decay_ratio=schedule_decay_ratio,
        )
    else:
        scheduler = make_cosine_schedule(
            optimizer,
            warmup_steps=1,
            total_steps=8,
        )
    cursor = {"step": 0, "total_tok": 0, "fi": 0, "ptr": 0}
    for _ in range(2):
        _run_replay_step(model, optimizer, scheduler, shards, cursor)
    assert cursor == {"step": 2, "total_tok": 8, "fi": 0, "ptr": 10}

    checkpoint = save_checkpoint(
        model,
        optimizer,
        scheduler,
        checkpoint_dir,
        step=cursor["step"],
        total_tok=cursor["total_tok"],
        warmup_steps=1,
        total_steps=8,
        fi=cursor["fi"],
        ptr=cursor["ptr"],
        batch_size=1,
        seq_len=4,
        optimizer_kind=optimizer_kind,
        schedule_kind=schedule_kind,
        schedule_decay_ratio=schedule_decay_ratio,
        run_id=metadata["run_id"],
        code_manifest=metadata["code_manifest"],
        data_manifest=metadata["data_manifest"],
        tokenizer_manifest=metadata["tokenizer_manifest"],
        data_files=shard_paths,
        tokenizer_dir=tokenizer_dir,
    )
    assert checkpoint.name == "step_0000002"

    reference_model, reference_optimizer, reference_scheduler, reference_cursor, _ = (
        _load_recovery_checkpoint(checkpoint, metadata)
    )
    reference_records = [
        _run_replay_step(
            reference_model,
            reference_optimizer,
            reference_scheduler,
            shards,
            reference_cursor,
        )
        for _ in range(3)
    ]
    reference_model_state = _clone_nested(reference_model.state_dict())
    reference_optimizer_state = _clone_nested(reference_optimizer.state_dict())
    reference_scheduler_state = _clone_nested(reference_scheduler.state_dict())
    reference_rng_state = _clone_nested(capture_rng_state())
    reference_next_batch = _candidate_batch(shards, reference_cursor)[0]

    failed_model, failed_optimizer, failed_scheduler, failed_cursor, _ = (
        _load_recovery_checkpoint(checkpoint, metadata)
    )
    failed_records = [
        _run_replay_step(
            failed_model,
            failed_optimizer,
            failed_scheduler,
            shards,
            failed_cursor,
        )
        for _ in range(2)
    ]
    for actual, expected in zip(failed_records, reference_records[:2]):
        _assert_nested_exact(actual, expected)
    assert failed_cursor == {"step": 4, "total_tok": 16, "fi": 1, "ptr": 5}
    failed_cursor_before = dict(failed_cursor)
    failed_version_before = failed_model.get_cpt_state_version()
    original_step = failed_optimizer.step

    def update_then_fail(self, closure=None):
        result = original_step(closure)
        raise RuntimeError("injected recovery partial AdamW write")

    failed_optimizer.step = types.MethodType(update_then_fail, failed_optimizer)
    with pytest.raises(TrainingFailStop, match="must not be reused or checkpointed"):
        _run_replay_step(
            failed_model,
            failed_optimizer,
            failed_scheduler,
            shards,
            failed_cursor,
        )
    assert failed_cursor == failed_cursor_before
    assert failed_model.get_cpt_state_version() == failed_version_before == 4
    assert hasattr(failed_model, "_cpt_training_poison_reason")
    assert "injected recovery partial AdamW write" in (
        failed_model._cpt_training_poison_reason
    )

    with pytest.raises(TrainingFailStop, match="prior fail-stop event"):
        save_checkpoint(
            failed_model,
            failed_optimizer,
            failed_scheduler,
            checkpoint_dir,
            step=failed_cursor["step"],
            total_tok=failed_cursor["total_tok"],
            warmup_steps=1,
            total_steps=8,
            fi=failed_cursor["fi"],
            ptr=failed_cursor["ptr"],
            batch_size=1,
            seq_len=4,
            optimizer_kind=optimizer_kind,
            schedule_kind=schedule_kind,
            schedule_decay_ratio=schedule_decay_ratio,
            run_id=metadata["run_id"],
            code_manifest=metadata["code_manifest"],
            data_manifest=metadata["data_manifest"],
            tokenizer_manifest=metadata["tokenizer_manifest"],
            data_files=shard_paths,
            tokenizer_dir=tokenizer_dir,
        )
    assert sorted(path.name for path in checkpoint_dir.glob("step_*")) == [
        "step_0000002"
    ]
    assert not list(checkpoint_dir.glob(".step_*.tmp-*"))

    del failed_model, failed_optimizer, failed_scheduler
    recovered_model, recovered_optimizer, recovered_scheduler, recovered_cursor, state = (
        _load_recovery_checkpoint(checkpoint, metadata)
    )
    assert state["step"] == 2
    assert state["total_tok"] == 8
    assert state["fi"] == 0
    assert state["ptr"] == 10
    assert recovered_cursor == {"step": 2, "total_tok": 8, "fi": 0, "ptr": 10}
    restored_next_batch = _candidate_batch(shards, recovered_cursor)[0]
    torch.testing.assert_close(
        restored_next_batch,
        reference_records[0]["batch"],
        rtol=0,
        atol=0,
    )

    recovered_records = [
        _run_replay_step(
            recovered_model,
            recovered_optimizer,
            recovered_scheduler,
            shards,
            recovered_cursor,
        )
        for _ in range(3)
    ]
    for actual, expected in zip(recovered_records, reference_records):
        _assert_nested_exact(actual, expected)
    assert recovered_cursor == reference_cursor
    assert recovered_model.get_cpt_state_version() == recovered_cursor["step"] == 5
    _assert_nested_exact(recovered_model.state_dict(), reference_model_state)
    _assert_nested_exact(recovered_optimizer.state_dict(), reference_optimizer_state)
    _assert_nested_exact(recovered_scheduler.state_dict(), reference_scheduler_state)
    _assert_nested_exact(capture_rng_state(), reference_rng_state)
    torch.testing.assert_close(
        _candidate_batch(shards, recovered_cursor)[0],
        reference_next_batch,
        rtol=0,
        atol=0,
    )
