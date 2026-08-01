import ast
import copy
from datetime import timedelta
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp

from scripts.train_utils import (
    BF16AdamW,
    TRAINING_STATE_SCHEMA_NAME,
    TRAINING_STATE_SCHEMA_VERSION,
    build_code_manifest,
    build_data_manifest,
    build_tokenizer_manifest,
    canonical_config_hash,
    capture_rng_state,
    cleanup_stale_checkpoint_temps,
    execute_training_iteration,
    execute_training_step,
    make_adamw,
    make_cosine_schedule,
    make_wsd_schedule,
    prune_periodic_checkpoints,
    restore_rank_rng_state,
    restore_rng_state,
    save_checkpoint,
    save_training_state,
    training_loop,
    validate_training_state,
    verify_checkpoint_file_hashes,
)


class _DummyConfig:
    def save_pretrained(self, path) -> None:
        Path(path, "config.json").write_text("{}", encoding="utf-8")

    def to_dict(self):
        return {"model_type": "dummy-cpt"}


class _DummyRouterView:
    def __init__(self, model):
        self._model = model
        self.num_experts = 1

    @property
    def congestion_price(self):
        return self._model.dummy_congestion_price

    @property
    def state_version(self):
        return self._model.cpt_version

    def named_parameters(self, recurse=True):
        del recurse
        return (("projection", self._model.weight),)


class _DummyCPTModel(torch.nn.Module):
    def __init__(
        self,
        events,
        *,
        validate_error=None,
        commit_error=None,
        commit_error_after_write=False,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.register_buffer("cpt_version", torch.tensor(0, dtype=torch.int64))
        self.register_buffer(
            "dummy_congestion_price",
            torch.zeros(1, dtype=torch.float32),
        )
        self.events = events
        self.validate_error = validate_error
        self.commit_error = commit_error
        self.commit_error_after_write = commit_error_after_write
        self.config = _DummyConfig()

    @property
    def version(self):
        return int(self.cpt_version.item())

    @version.setter
    def version(self, value):
        self.cpt_version.fill_(int(value))

    def forward(self, *args, **kwargs):
        return self.output()

    def transaction(self):
        return {"base_version": self.version, "id": object()}

    def output(self):
        return {
            "loss": self.weight.square(),
            "cpt_transaction": self.transaction(),
        }

    def validate_cpt_transaction(self, transaction) -> None:
        self.events.append("validate")
        if self.validate_error is not None:
            raise self.validate_error
        if transaction["base_version"] != self.version:
            raise RuntimeError("stale transaction")

    def commit_cpt_transaction(self, transaction) -> None:
        self.events.append("commit")
        if self.commit_error is not None and not self.commit_error_after_write:
            raise self.commit_error
        if transaction["base_version"] != self.version:
            raise RuntimeError("stale transaction")
        self.cpt_version.add_(1)
        if self.commit_error is not None:
            raise self.commit_error

    def abort_cpt_transaction(self, transaction) -> None:
        self.events.append("abort")

    def get_cpt_state_version(self):
        return self.version

    def _cpt_routers(self):
        return (_DummyRouterView(self),)


class _RecordingSGD(torch.optim.SGD):
    def __init__(self, parameters, events, *, fail=False):
        super().__init__(parameters, lr=0.1)
        self.events = events
        self.fail = fail

    def step(self, closure=None):
        self.events.append("optimizer")
        if self.fail:
            raise RuntimeError("optimizer failed")
        return super().step(closure)

    def zero_grad(self, set_to_none=True):
        self.events.append("zero_grad")
        return super().zero_grad(set_to_none=set_to_none)


class _RecordingScheduler:
    def __init__(self, events, *, fail=False, fail_after_step=False):
        self.events = events
        self.fail = fail
        self.fail_after_step = fail_after_step
        self.last_epoch = 0
        self._step_count = 1
        self.last_lr = [0.1]

    def step(self):
        self.events.append("scheduler")
        if self.fail and not self.fail_after_step:
            raise RuntimeError("scheduler failed")
        self.last_epoch += 1
        self._step_count += 1
        if self.fail:
            raise RuntimeError("scheduler failed")

    def state_dict(self):
        return {
            "last_epoch": self.last_epoch,
            "_step_count": self._step_count,
            "last_lr": list(self.last_lr),
        }

    def load_state_dict(self, state):
        self.last_epoch = state["last_epoch"]
        self._step_count = state["_step_count"]
        self.last_lr = list(state["last_lr"])

    def get_last_lr(self):
        return list(self.last_lr)


class _CorruptingSGD(torch.optim.SGD):
    def __init__(self, parameters, target):
        super().__init__(parameters, lr=0.1)
        self.target = target

    def step(self, closure=None):
        result = super().step(closure)
        with torch.no_grad():
            self.target.view(-1)[0] = float("nan")
        return result


class _UpdateThenFailSGD(torch.optim.SGD):
    def __init__(self, parameters, *, fail=True):
        super().__init__(parameters, lr=0.1, momentum=0.9)
        self.fail = fail

    def step(self, closure=None):
        result = super().step(closure)
        if self.fail:
            raise RuntimeError("optimizer failed after partial write")
        return result


class _CorruptingOptimizerStateSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.1, momentum=0.9)

    def step(self, closure=None):
        result = super().step(closure)
        for state in self.state.values():
            state["momentum_buffer"].view(-1)[0] = float("nan")
            break
        return result


class _CleanupFailsOnceSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.1)
        self.fail_next_cleanup = True

    def zero_grad(self, set_to_none=True):
        if self.fail_next_cleanup:
            self.fail_next_cleanup = False
            raise RuntimeError("gradient cleanup failed")
        return super().zero_grad(set_to_none=set_to_none)


class _CorruptingScheduler(_RecordingScheduler):
    def __init__(self, events, optimizer):
        super().__init__(events)
        self.optimizer = optimizer

    def step(self):
        super().step()
        self.optimizer.param_groups[0]["lr"] = float("nan")
        self.last_lr[0] = float("nan")


class _FailingVersionTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, source):
        return torch.Tensor._make_subclass(
            cls,
            source.detach().clone(),
            False,
        )

    def add_(self, *args, **kwargs):
        raise RuntimeError("injected CPT version write failure")


class _InfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return torch.full_like(grad_output, float("inf"))


def _tiny_cpt_config():
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
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        initializer_range=0.02,
    )


def _clone_nested(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    return copy.deepcopy(value)


def _assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _make_checkpoint_metadata(tmp_path):
    data_dir = tmp_path / "manifest_data"
    data_dir.mkdir(exist_ok=True)
    shard = data_dir / "train_00000.pt"
    torch.save(torch.arange(32, dtype=torch.int64), shard)

    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    (tokenizer_dir / "tokenizer.json").write_text(
        '{"version":"1.0"}',
        encoding="utf-8",
    )
    (tokenizer_dir / "tokenizer_config.json").write_text(
        '{"model_max_length":16}',
        encoding="utf-8",
    )
    return {
        "run_id": "a" * 32,
        "code_manifest": build_code_manifest(),
        "data_manifest": build_data_manifest([shard]),
        "tokenizer_manifest": build_tokenizer_manifest(tokenizer_dir),
    }


def _advance_schedule(optimizer, scheduler, steps):
    for _ in range(steps):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()


def _make_valid_training_state(
    tmp_path,
    *,
    step=2,
    batch_size=1,
    seq_len=8,
    optimizer_kind="adamw",
    schedule_kind="cosine",
    schedule_decay_ratio=None,
):
    model = _DummyCPTModel([])
    model.version = step
    optimizer = make_adamw(
        model,
        lr=0.1,
        weight_decay=0.0,
        bf16_states=optimizer_kind == "bf16_adamw",
    )
    total_steps = max(step + 2, 4)
    warmup_cap = max(total_steps - (2 if schedule_kind == "wsd" else 1), 0)
    warmup_steps = min(1, max(step - 1, 0), warmup_cap)
    if schedule_kind == "wsd":
        if schedule_decay_ratio is None:
            schedule_decay_ratio = 0.25
        scheduler = make_wsd_schedule(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            decay_ratio=schedule_decay_ratio,
        )
    else:
        scheduler = make_cosine_schedule(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
    _advance_schedule(optimizer, scheduler, step)
    metadata = _make_checkpoint_metadata(tmp_path)
    path = tmp_path / f"training_state_{step}.pt"
    save_training_state(
        path,
        optimizer,
        scheduler,
        step=step,
        total_tok=step * batch_size * seq_len,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        fi=0,
        ptr=step * batch_size * (seq_len + 1),
        batch_size=batch_size,
        seq_len=seq_len,
        optimizer_kind=optimizer_kind,
        schedule_kind=schedule_kind,
        schedule_decay_ratio=schedule_decay_ratio,
        cpt_state_version=step,
        run_id=metadata["run_id"],
        config_sha256=canonical_config_hash(model.config),
        code_manifest=metadata["code_manifest"],
        data_manifest=metadata["data_manifest"],
        tokenizer_manifest=metadata["tokenizer_manifest"],
        model_state_sha256="a" * 64,
        config_file_sha256="b" * 64,
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    return state, model, metadata


def _duplicate_first_optimizer_parameter(state):
    for group in state["opt"]["param_groups"]:
        if group["params"]:
            group["params"].append(group["params"][0])
            return
    raise AssertionError("valid optimizer state unexpectedly contains no parameters")


def _ddp_checkpoint_worker(
    rank,
    world_size,
    store_path,
    output_dir,
    shard_path,
    tokenizer_dir,
    result_dir,
    expect_failure,
    rank1_runtime_shard,
):
    torch_dist.init_process_group(
        backend="gloo",
        init_method=Path(store_path).resolve().as_uri(),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        events = []
        model = _DummyCPTModel(events)
        model.version = 1
        optimizer = make_adamw(model, lr=0.1, weight_decay=0.0)
        scheduler = make_cosine_schedule(
            optimizer,
            warmup_steps=0,
            total_steps=4,
        )
        _advance_schedule(optimizer, scheduler, 1)
        code_manifest = build_code_manifest()
        data_manifest = build_data_manifest([shard_path])
        tokenizer_manifest = build_tokenizer_manifest(tokenizer_dir)

        seed = 4100 + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng_before_save = capture_rng_state()
        expected = {
            "python": random.random(),
            "numpy": float(np.random.random()),
            "torch": torch.rand(5),
        }
        restore_rng_state(rng_before_save)

        checkpoint = None
        error = None
        try:
            runtime_shard = (
                rank1_runtime_shard
                if rank == 1 and rank1_runtime_shard is not None
                else shard_path
            )
            checkpoint = save_checkpoint(
                model,
                optimizer,
                scheduler,
                output_dir,
                step=1,
                total_tok=8,
                warmup_steps=0,
                total_steps=4,
                fi=0,
                ptr=9,
                batch_size=1,
                seq_len=8,
                optimizer_kind="adamw",
                schedule_kind="cosine",
                schedule_decay_ratio=None,
                run_id="d" * 32,
                code_manifest=code_manifest,
                data_manifest=data_manifest,
                tokenizer_manifest=tokenizer_manifest,
                data_files=[runtime_shard],
                tokenizer_dir=tokenizer_dir,
            )
        except BaseException as exc:
            import traceback

            error = (
                f"{type(exc).__name__}: {exc}\n"
                + traceback.format_exc()
            )

        result = {
            "rank": rank,
            "error": error,
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "restored": False,
        }
        if not expect_failure and error is None:
            state = torch.load(
                Path(checkpoint) / "training_state.pt",
                map_location="cpu",
                weights_only=True,
            )
            validate_training_state(
                state,
                checkpoint_step=1,
                model=model,
                expected_run_id="d" * 32,
                expected_code_manifest=code_manifest,
                expected_data_manifest=data_manifest,
                expected_tokenizer_manifest=tokenizer_manifest,
                expected_world_size=world_size,
            )
            random.seed(99999)
            np.random.seed(99999)
            torch.manual_seed(99999)
            restore_rank_rng_state(state)
            actual = {
                "python": random.random(),
                "numpy": float(np.random.random()),
                "torch": torch.rand(5),
            }
            result["restored"] = (
                actual["python"] == expected["python"]
                and actual["numpy"] == expected["numpy"]
                and torch.equal(actual["torch"], expected["torch"])
            )
            result["rng_world_size"] = state["rng_world_size"]
            result["rng_ranks"] = [entry["rank"] for entry in state["rng_states"]]

        torch.save(result, Path(result_dir) / f"rank_{rank}.pt")
    finally:
        torch_dist.destroy_process_group()


def _ddp_cpt_state_gate_worker(
    rank,
    world_size,
    store_path,
    result_dir,
    mismatch_kind,
):
    torch_dist.init_process_group(
        backend="gloo",
        init_method=Path(store_path).resolve().as_uri(),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        events = []
        model = _DummyCPTModel(events)
        if mismatch_kind == "price" and rank == 1:
            model.dummy_congestion_price.fill_(0.5)
        elif mismatch_kind == "version" and rank == 1:
            model.version = 1
        elif mismatch_kind == "learnable" and rank == 1:
            with torch.no_grad():
                model.weight.fill_(2.0)

        learning_rate = 0.2 if mismatch_kind == "post_optimizer" and rank == 1 else 0.1
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = _RecordingScheduler(events)
        model_before = _clone_nested(model.state_dict())
        optimizer_before = _clone_nested(optimizer.state_dict())
        scheduler_before = _clone_nested(scheduler.state_dict())
        error = None
        try:
            execute_training_step(model, model.output(), optimizer, scheduler)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"

        restored = True
        try:
            _assert_nested_equal(model.state_dict(), model_before)
            _assert_nested_equal(optimizer.state_dict(), optimizer_before)
            _assert_nested_equal(scheduler.state_dict(), scheduler_before)
        except AssertionError:
            restored = False
        torch.save(
            {
                "rank": rank,
                "error": error,
                "events": events,
                "restored": restored,
                "version": model.version,
                "price": model.dummy_congestion_price.detach().clone(),
                "weight": model.weight.detach().clone(),
            },
            Path(result_dir) / f"rank_{rank}.pt",
        )
    finally:
        torch_dist.destroy_process_group()


def _manual_cpt_transaction(model, load=(3.0, 1.0, 0.0, 0.0)):
    from model.cpt_router import CPTLayerProposal

    count = int(sum(load))
    proposals = []
    for layer_index, layer in enumerate(model.layers):
        router = layer.moe.cpt_router
        proposals.append(
            CPTLayerProposal(
                layer_index=layer_index,
                load_sum=torch.tensor(
                    load,
                    device=router.congestion_price.device,
                    dtype=torch.float32,
                ),
                token_count=torch.tensor(
                    count,
                    device=router.congestion_price.device,
                    dtype=torch.int64,
                ),
                state_version=router.state_version.detach().clone(),
                valid=torch.tensor(
                    True,
                    device=router.congestion_price.device,
                ),
            )
        )
    return model.prepare_cpt_transaction(proposals)


def test_execute_training_step_orders_validation_optimizer_scheduler_and_commit():
    events = []
    model = _DummyCPTModel(events)
    optimizer = _RecordingSGD(model.parameters(), events)
    scheduler = _RecordingScheduler(events)

    grad_norm = execute_training_step(
        model,
        model.output(),
        optimizer,
        scheduler,
        max_grad_norm=10.0,
    )

    assert torch.isfinite(grad_norm)
    assert events == ["validate", "optimizer", "scheduler", "zero_grad", "commit"]
    assert model.version == 1
    assert model.weight.grad is None


def test_execute_training_step_integrates_with_real_cpt_model(
    fixed_batch,
):
    from model.modeling import TinyMixtralForCausalLM

    tiny_config = _tiny_cpt_config()
    torch.manual_seed(301)
    model = TinyMixtralForCausalLM(tiny_config).train()
    model.gradient_checkpointing_enable()
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    input_ids, labels, attention_mask = fixed_batch

    output = model(
        input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    transaction = output["cpt_transaction"]
    assert model.get_cpt_state_version() == 0

    execute_training_step(model, output, optimizer, max_grad_norm=1.0)

    assert model.get_cpt_state_version() == 1
    assert transaction.closed
    assert not transaction.aborted
    assert all(
        layer.moe.cpt_router.congestion_price.dtype == torch.float32
        for layer in model.layers
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_real_cpt_price_formula_and_stale_transaction_rejection():
    from model.modeling import TinyMixtralForCausalLM

    model = TinyMixtralForCausalLM(_tiny_cpt_config()).train()
    stale = _manual_cpt_transaction(model)
    current = _manual_cpt_transaction(model)
    expected = []
    for layer in model.layers:
        router = layer.moe.cpt_router
        with torch.no_grad():
            router.congestion_price.copy_(
                torch.tensor([0.0, 0.002, 0.01, 0.0])
            )
        mean_probability = torch.tensor([0.75, 0.25, 0.0, 0.0])
        expected.append(
            torch.clamp_min(
                router.congestion_price
                + router.price_learning_rate
                * (
                    mean_probability
                    - router.capacity_factor / router.num_experts
                ),
                0.0,
            )
        )

    model.commit_cpt_transaction(current)

    assert model.get_cpt_state_version() == 1
    for layer, expected_price in zip(model.layers, expected):
        torch.testing.assert_close(
            layer.moe.cpt_router.congestion_price,
            expected_price,
        )
    with pytest.raises(RuntimeError, match="does not match active version"):
        model.commit_cpt_transaction(stale)
    model.abort_cpt_transaction(stale)
    assert stale.closed and stale.aborted


def test_commit_rejects_tampered_prepared_price_without_state_advance():
    from model.modeling import TinyMixtralForCausalLM

    model = TinyMixtralForCausalLM(_tiny_cpt_config()).train()
    transaction = _manual_cpt_transaction(model)
    model.validate_cpt_transaction(transaction)
    tampered = list(transaction._prepared_prices)
    tampered[0] = tampered[0].clone()
    tampered[0][0] = float("nan")
    transaction._prepared_prices = tuple(tampered)

    with pytest.raises(RuntimeError, match="prepared CPT price is invalid"):
        model.commit_cpt_transaction(transaction)

    assert model.get_cpt_state_version() == 0
    assert all(
        torch.equal(
            layer.moe.cpt_router.congestion_price,
            torch.zeros_like(layer.moe.cpt_router.congestion_price),
        )
        for layer in model.layers
    )
    model.abort_cpt_transaction(transaction)
    assert transaction.closed and transaction.aborted


@pytest.mark.parametrize("parameter_name", ["projection", "energy"])
def test_optimizer_corrupted_cpt_parameter_fails_stop_before_cpt_commit(
    fixed_batch,
    parameter_name,
):
    from model.modeling import TinyMixtralForCausalLM

    model = TinyMixtralForCausalLM(_tiny_cpt_config()).train()
    target = getattr(model.layers[0].moe.cpt_router, parameter_name)
    optimizer = _CorruptingSGD(model.parameters(), target)
    model_before = _clone_nested(model.state_dict())
    optimizer_before = _clone_nested(optimizer.state_dict())
    input_ids, labels, attention_mask = fixed_batch
    output = model(
        input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    transaction = output["cpt_transaction"]

    with pytest.raises(
        FloatingPointError,
        match=rf"Non-finite model parameter after optimizer: .*{parameter_name}",
    ):
        execute_training_step(model, output, optimizer)

    _assert_nested_equal(model.state_dict(), model_before)
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)
    assert model.get_cpt_state_version() == 0
    assert transaction.closed and transaction.aborted
    assert all(parameter.grad is None for parameter in model.parameters())


def test_optimizer_corrupted_ordinary_parameter_rolls_back_without_commit(
    fixed_batch,
):
    from model.modeling import TinyMixtralForCausalLM

    model = TinyMixtralForCausalLM(_tiny_cpt_config()).train()
    target = model.embed_tokens.weight
    optimizer = _CorruptingSGD(model.parameters(), target)
    input_ids, labels, attention_mask = fixed_batch
    output = model(input_ids, attention_mask=attention_mask, labels=labels)
    transaction = output["cpt_transaction"]
    model_before = _clone_nested(model.state_dict())
    optimizer_before = _clone_nested(optimizer.state_dict())

    with pytest.raises(
        FloatingPointError,
        match="Non-finite model parameter after optimizer: embed_tokens.weight",
    ):
        execute_training_step(model, output, optimizer)

    _assert_nested_equal(model.state_dict(), model_before)
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)
    assert model.get_cpt_state_version() == 0
    assert transaction.closed and transaction.aborted
    assert all(parameter.grad is None for parameter in model.parameters())


def test_optimizer_partial_write_restores_model_optimizer_and_scheduler():
    events = []
    model = _DummyCPTModel(events)
    optimizer = _UpdateThenFailSGD(model.parameters(), fail=False)
    scheduler = _RecordingScheduler(events)

    model.weight.grad = torch.tensor(0.25)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    optimizer.fail = True
    model_before = _clone_nested(model.state_dict())
    optimizer_before = _clone_nested(optimizer.state_dict())
    scheduler_before = _clone_nested(scheduler.state_dict())

    with pytest.raises(RuntimeError, match="optimizer failed after partial write"):
        execute_training_step(model, model.output(), optimizer, scheduler)

    _assert_nested_equal(model.state_dict(), model_before)
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)
    _assert_nested_equal(scheduler.state_dict(), scheduler_before)
    assert model.version == 0
    assert "commit" not in events
    assert "abort" in events
    assert model.weight.grad is None


def test_nonfinite_optimizer_state_rolls_back_without_commit():
    events = []
    model = _DummyCPTModel(events)
    optimizer = _CorruptingOptimizerStateSGD(model.parameters())
    scheduler = _RecordingScheduler(events)
    model_before = _clone_nested(model.state_dict())
    optimizer_before = _clone_nested(optimizer.state_dict())
    scheduler_before = _clone_nested(scheduler.state_dict())

    with pytest.raises(
        FloatingPointError,
        match=r"Non-finite numeric state at optimizer\.state\..*momentum_buffer",
    ):
        execute_training_step(model, model.output(), optimizer, scheduler)

    _assert_nested_equal(model.state_dict(), model_before)
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)
    _assert_nested_equal(scheduler.state_dict(), scheduler_before)
    assert model.version == 0
    assert "commit" not in events


def test_nonfinite_scheduler_lr_rolls_back_every_applied_state():
    events = []
    model = _DummyCPTModel(events)
    optimizer = _RecordingSGD(model.parameters(), events)
    scheduler = _CorruptingScheduler(events, optimizer)
    model_before = _clone_nested(model.state_dict())
    optimizer_before = _clone_nested(optimizer.state_dict())
    scheduler_before = _clone_nested(scheduler.state_dict())

    with pytest.raises(
        FloatingPointError,
        match=r"Non-finite numeric state at optimizer\.param_groups\[0\]\.lr",
    ):
        execute_training_step(model, model.output(), optimizer, scheduler)

    _assert_nested_equal(model.state_dict(), model_before)
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)
    _assert_nested_equal(scheduler.state_dict(), scheduler_before)
    assert model.version == 0
    assert "commit" not in events


def test_gradient_cleanup_failure_aborts_before_final_cpt_commit():
    events = []
    model = _DummyCPTModel(events)
    optimizer = _CleanupFailsOnceSGD(model.parameters())
    scheduler = _RecordingScheduler(events)
    model_before = _clone_nested(model.state_dict())
    optimizer_before = _clone_nested(optimizer.state_dict())
    scheduler_before = _clone_nested(scheduler.state_dict())

    with pytest.raises(RuntimeError, match="gradient cleanup failed"):
        execute_training_step(model, model.output(), optimizer, scheduler)

    _assert_nested_equal(model.state_dict(), model_before)
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)
    _assert_nested_equal(scheduler.state_dict(), scheduler_before)
    assert "commit" not in events
    assert "abort" in events
    assert model.version == 0
    assert model.weight.grad is None


def test_pre_forward_rng_state_is_restored_after_failed_iteration():
    random.seed(90210)
    np.random.seed(90210)
    torch.manual_seed(90210)
    rng_entry = capture_rng_state()
    expected = (random.random(), float(np.random.random()), torch.rand(4))
    restore_rng_state(rng_entry)

    events = []
    model = _DummyCPTModel(events)
    optimizer = _UpdateThenFailSGD(model.parameters())
    scheduler = _RecordingScheduler(events)

    def forward_fn():
        random.random()
        np.random.random()
        torch.rand(7)
        return model.output()

    with pytest.raises(RuntimeError, match="optimizer failed after partial write"):
        execute_training_iteration(model, optimizer, scheduler, forward_fn)

    actual = (random.random(), float(np.random.random()), torch.rand(4))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    assert model.version == 0
    assert model.weight.item() == 1.0
    assert model.weight.grad is None


def test_real_commit_rolls_back_every_layer_after_late_write_failure():
    from model.modeling import TinyMixtralForCausalLM

    model = TinyMixtralForCausalLM(_tiny_cpt_config()).train()
    with torch.no_grad():
        for layer_index, layer in enumerate(model.layers):
            router = layer.moe.cpt_router
            router.anchors.mul_(2.0 + layer_index)
            router.congestion_price.fill_(0.001 * (layer_index + 1))
    transaction = _manual_cpt_transaction(model)
    second_router = model.layers[1].moe.cpt_router
    second_router.state_version = _FailingVersionTensor(
        second_router.state_version
    )
    anchor_backups = [
        layer.moe.cpt_router.anchors.detach().clone()
        for layer in model.layers
    ]
    price_backups = [
        layer.moe.cpt_router.congestion_price.detach().clone()
        for layer in model.layers
    ]
    version_backups = [
        layer.moe.cpt_router.state_version.detach().clone()
        for layer in model.layers
    ]

    with pytest.raises(RuntimeError, match="injected CPT version write failure"):
        model.commit_cpt_transaction(transaction)

    for layer, anchor, price, version in zip(
        model.layers,
        anchor_backups,
        price_backups,
        version_backups,
    ):
        router = layer.moe.cpt_router
        torch.testing.assert_close(router.anchors, anchor, rtol=0, atol=0)
        torch.testing.assert_close(
            router.congestion_price,
            price,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(router.state_version, version, rtol=0, atol=0)
    model.abort_cpt_transaction(transaction)
    assert transaction.closed and transaction.aborted


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 is unavailable",
)
def test_execute_training_step_cuda_bfloat16_checkpointed(fixed_batch):
    from model.modeling import TinyMixtralForCausalLM

    torch.manual_seed(303)
    torch.cuda.manual_seed_all(303)
    model = TinyMixtralForCausalLM(_tiny_cpt_config()).to(
        device="cuda",
        dtype=torch.bfloat16,
    ).train()
    model.gradient_checkpointing_enable()
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(
        optimizer,
        warmup_steps=1,
        total_steps=4,
    )
    input_ids, labels, attention_mask = (
        tensor.cuda() for tensor in fixed_batch
    )

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    transaction = output["cpt_transaction"]

    execute_training_step(model, output, optimizer, scheduler)

    assert model.get_cpt_state_version() == 1
    assert scheduler.last_epoch == 1
    assert transaction.closed
    assert not transaction.aborted
    assert all(
        layer.moe.cpt_router.congestion_price.dtype == torch.float32
        for layer in model.layers
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_multirank_consensus_rolls_back_when_another_rank_fails_optimizer(
    monkeypatch,
):
    import scripts.train_utils as train_utils

    events = []
    model = _DummyCPTModel(events)
    optimizer = _RecordingSGD(model.parameters(), events)
    scheduler = _RecordingScheduler(events)
    model_before = _clone_nested(model.state_dict())
    optimizer_before = _clone_nested(optimizer.state_dict())
    scheduler_before = _clone_nested(scheduler.state_dict())
    collective_flags = []

    monkeypatch.setattr(train_utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(train_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(train_utils.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(train_utils.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(
        train_utils,
        "_validate_distributed_cpt_consistency",
        lambda model, context: None,
    )

    def all_reduce(success, op=None):
        collective_flags.append(int(success.item()))
        if len(collective_flags) == 5:
            success.zero_()

    monkeypatch.setattr(train_utils.dist, "all_reduce", all_reduce)

    with pytest.raises(
        RuntimeError,
        match="Training phase 'optimizer' failed on another distributed rank",
    ):
        execute_training_step(model, model.output(), optimizer, scheduler)

    assert collective_flags == [1, 1, 1, 1, 1, 1]
    _assert_nested_equal(model.state_dict(), model_before)
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)
    _assert_nested_equal(scheduler.state_dict(), scheduler_before)
    assert "commit" not in events
    assert "abort" in events
    assert model.version == 0
    assert model.weight.grad is None


def test_real_cpt_checkpoint_roundtrip_keeps_transaction_version(
    tmp_path,
    fixed_batch,
):
    from model.modeling import TinyMixtralForCausalLM

    torch.manual_seed(302)
    model = TinyMixtralForCausalLM(_tiny_cpt_config()).train()
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, warmup_steps=1, total_steps=4)
    input_ids, labels, attention_mask = fixed_batch
    output = model(
        input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    execute_training_step(model, output, optimizer, scheduler)
    metadata = _make_checkpoint_metadata(tmp_path)

    checkpoint = save_checkpoint(
        model,
        optimizer,
        scheduler,
        tmp_path,
        step=1,
        total_tok=16,
        warmup_steps=1,
        total_steps=4,
        fi=0,
        ptr=18,
        batch_size=2,
        seq_len=8,
        optimizer_kind="adamw",
        schedule_kind="cosine",
        schedule_decay_ratio=None,
        run_id=metadata["run_id"],
        code_manifest=metadata["code_manifest"],
        data_manifest=metadata["data_manifest"],
        tokenizer_manifest=metadata["tokenizer_manifest"],
        data_files=[tmp_path / "manifest_data" / "train_00000.pt"],
        tokenizer_dir=tmp_path / "tokenizer",
    )
    state = torch.load(
        checkpoint / "training_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    verified_hashes = verify_checkpoint_file_hashes(checkpoint, state)
    assert verified_hashes["model_state_sha256"] == state["model_state_sha256"]
    assert verified_hashes["config_file_sha256"] == state["config_file_sha256"]
    restored = TinyMixtralForCausalLM.from_pretrained(str(checkpoint))

    validate_training_state(
        state,
        checkpoint_step=1,
        model=restored,
        expected_run_id=metadata["run_id"],
        expected_code_manifest=metadata["code_manifest"],
        expected_data_manifest=metadata["data_manifest"],
        expected_tokenizer_manifest=metadata["tokenizer_manifest"],
    )
    assert restored.get_cpt_state_version() == 1
    assert all(
        layer.moe.cpt_router.congestion_price.dtype == torch.float32
        for layer in restored.layers
    )

    model_path = checkpoint / "pytorch_model.bin"
    original_model_bytes = model_path.read_bytes()
    torch.manual_seed(303)
    mismatched_model = TinyMixtralForCausalLM(_tiny_cpt_config())
    torch.save(mismatched_model.state_dict(), model_path)
    with pytest.raises(RuntimeError, match="pytorch_model.bin SHA-256"):
        verify_checkpoint_file_hashes(checkpoint, state)

    model_path.write_bytes(original_model_bytes)
    config_path = checkpoint / "config.json"
    config_path.write_bytes(config_path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="config.json SHA-256"):
        verify_checkpoint_file_hashes(checkpoint, state)


def test_real_cpt_learnable_state_digest_is_stable_and_sensitive():
    import scripts.train_utils as train_utils
    from model.modeling import TinyMixtralForCausalLM

    torch.manual_seed(911)
    model = TinyMixtralForCausalLM(_tiny_cpt_config())
    router = model.layers[0].moe.cpt_router
    first = train_utils._digest_cpt_learnable_state(router, 0)
    second = train_utils._digest_cpt_learnable_state(router, 0)
    assert first == second
    with torch.no_grad():
        router.projection.view(-1)[0].add_(0.125)
    changed = train_utils._digest_cpt_learnable_state(router, 0)
    assert changed != first


def test_nonfinite_loss_aborts_before_validation_or_optimizer():
    events = []
    model = _DummyCPTModel(events)
    optimizer = _RecordingSGD(model.parameters(), events)
    output = model.output()
    output["loss"] = model.weight * torch.tensor(float("nan"))

    with pytest.raises(FloatingPointError, match="Non-finite training loss"):
        execute_training_step(model, output, optimizer)

    assert events == ["abort", "zero_grad"]
    assert model.version == 0
    assert model.weight.item() == 1.0


def test_invalid_transaction_aborts_before_backward_or_optimizer():
    events = []
    model = _DummyCPTModel(events, validate_error=RuntimeError("invalid proposal"))
    optimizer = _RecordingSGD(model.parameters(), events)

    with pytest.raises(RuntimeError, match="invalid proposal"):
        execute_training_step(model, model.output(), optimizer)

    assert events == ["validate", "abort", "zero_grad"]
    assert model.weight.grad is None
    assert model.version == 0


def test_nonfinite_gradient_aborts_before_optimizer():
    events = []
    model = _DummyCPTModel(events)
    optimizer = _RecordingSGD(model.parameters(), events)
    output = model.output()
    output["loss"] = _InfiniteGradient.apply(model.weight)

    with pytest.raises(FloatingPointError, match="Non-finite gradient norm"):
        execute_training_step(model, output, optimizer)

    assert events == ["validate", "abort", "zero_grad"]
    assert model.version == 0
    assert model.weight.grad is None


@pytest.mark.parametrize("failure_site", ["optimizer", "scheduler", "commit"])
def test_post_backward_failures_abort_transaction(failure_site):
    events = []
    model = _DummyCPTModel(
        events,
        commit_error=RuntimeError("commit failed") if failure_site == "commit" else None,
        commit_error_after_write=failure_site == "commit",
    )
    optimizer = _RecordingSGD(
        model.parameters(),
        events,
        fail=failure_site == "optimizer",
    )
    scheduler = _RecordingScheduler(
        events,
        fail=failure_site == "scheduler",
        fail_after_step=failure_site == "scheduler",
    )
    model_before = _clone_nested(model.state_dict())
    optimizer_before = _clone_nested(optimizer.state_dict())
    scheduler_before = _clone_nested(scheduler.state_dict())

    with pytest.raises(RuntimeError, match=f"{failure_site} failed"):
        execute_training_step(model, model.output(), optimizer, scheduler)

    assert "abort" in events
    assert events[-1] == "zero_grad"
    _assert_nested_equal(model.state_dict(), model_before)
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)
    _assert_nested_equal(scheduler.state_dict(), scheduler_before)
    assert model.version == 0


def test_training_state_schema_and_cpt_version_roundtrip(tmp_path):
    events = []
    model = _DummyCPTModel(events)
    model.version = 3
    optimizer = make_adamw(model, lr=0.1, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, warmup_steps=1, total_steps=8)
    _advance_schedule(optimizer, scheduler, 3)
    metadata = _make_checkpoint_metadata(tmp_path)
    path = tmp_path / "training_state.pt"

    save_training_state(
        path,
        optimizer,
        scheduler,
        step=3,
        total_tok=48,
        warmup_steps=1,
        total_steps=8,
        fi=1,
        ptr=54,
        batch_size=2,
        seq_len=8,
        optimizer_kind="adamw",
        schedule_kind="cosine",
        schedule_decay_ratio=None,
        cpt_state_version=3,
        run_id=metadata["run_id"],
        config_sha256=canonical_config_hash(model.config),
        code_manifest=metadata["code_manifest"],
        data_manifest=metadata["data_manifest"],
        tokenizer_manifest=metadata["tokenizer_manifest"],
        model_state_sha256="a" * 64,
        config_file_sha256="b" * 64,
    )
    state = torch.load(path, map_location="cpu", weights_only=True)

    assert state["schema_name"] == TRAINING_STATE_SCHEMA_NAME
    assert state["schema_version"] == TRAINING_STATE_SCHEMA_VERSION
    assert state["cpt_state_version"] == 3
    assert state["optimizer_kind"] == "adamw"
    assert state["schedule_kind"] == "cosine"
    assert state["schedule_decay_ratio"] is None
    assert state["rng_world_size"] == 1
    assert [entry["rank"] for entry in state["rng_states"]] == [0]
    assert validate_training_state(
        state,
        checkpoint_step=3,
        model=model,
        expected_run_id=metadata["run_id"],
        expected_code_manifest=metadata["code_manifest"],
        expected_data_manifest=metadata["data_manifest"],
        expected_tokenizer_manifest=metadata["tokenizer_manifest"],
        expected_world_size=1,
        expected_model_state_sha256="a" * 64,
        expected_config_file_sha256="b" * 64,
    ) is state


@pytest.mark.parametrize(
    ("optimizer_kind", "schedule_kind", "schedule_decay_ratio"),
    [
        ("adamw", "cosine", None),
        ("adamw", "wsd", 0.5),
        ("bf16_adamw", "cosine", None),
        ("bf16_adamw", "wsd", 0.5),
    ],
)
def test_training_recipe_2x2_roundtrip_continues_exactly(
    tmp_path,
    optimizer_kind,
    schedule_kind,
    schedule_decay_ratio,
):
    step = 2
    warmup_steps = 1
    total_steps = 6
    source = _DummyCPTModel([])
    source.version = step
    source_optimizer = make_adamw(
        source,
        lr=0.1,
        weight_decay=0.0,
        bf16_states=optimizer_kind == "bf16_adamw",
    )
    if schedule_kind == "wsd":
        source_scheduler = make_wsd_schedule(
            source_optimizer,
            warmup_steps,
            total_steps,
            decay_ratio=schedule_decay_ratio,
        )
    else:
        source_scheduler = make_cosine_schedule(
            source_optimizer,
            warmup_steps,
            total_steps,
        )
    _advance_schedule(source_optimizer, source_scheduler, step)
    metadata = _make_checkpoint_metadata(tmp_path)
    state_path = tmp_path / f"{optimizer_kind}_{schedule_kind}.pt"
    save_training_state(
        state_path,
        source_optimizer,
        source_scheduler,
        step=step,
        total_tok=16,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        fi=0,
        ptr=18,
        batch_size=1,
        seq_len=8,
        optimizer_kind=optimizer_kind,
        schedule_kind=schedule_kind,
        schedule_decay_ratio=schedule_decay_ratio,
        cpt_state_version=step,
        run_id=metadata["run_id"],
        config_sha256=canonical_config_hash(source.config),
        code_manifest=metadata["code_manifest"],
        data_manifest=metadata["data_manifest"],
        tokenizer_manifest=metadata["tokenizer_manifest"],
        model_state_sha256="a" * 64,
        config_file_sha256="b" * 64,
    )
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    assert state["optimizer_kind"] == optimizer_kind
    assert state["schedule_kind"] == schedule_kind
    assert state["schedule_decay_ratio"] == schedule_decay_ratio

    restored = _DummyCPTModel([])
    restored.load_state_dict(source.state_dict())
    restored_optimizer = make_adamw(
        restored,
        lr=0.1,
        weight_decay=0.0,
        bf16_states=optimizer_kind == "bf16_adamw",
    )
    restored_optimizer.load_state_dict(state["opt"])
    saved_lrs = [group["lr"] for group in restored_optimizer.param_groups]
    if schedule_kind == "wsd":
        restored_scheduler = make_wsd_schedule(
            restored_optimizer,
            warmup_steps,
            total_steps,
            decay_ratio=schedule_decay_ratio,
        )
    else:
        restored_scheduler = make_cosine_schedule(
            restored_optimizer,
            warmup_steps,
            total_steps,
        )
    for group, saved_lr in zip(restored_optimizer.param_groups, saved_lrs):
        group["lr"] = saved_lr
    restored_scheduler.load_state_dict(state["sched"])

    assert isinstance(restored_optimizer, BF16AdamW) == (
        optimizer_kind == "bf16_adamw"
    )
    assert validate_training_state(
        state,
        checkpoint_step=step,
        model=restored,
    ) is state

    # For WSD this step crosses from the stable phase into decay.
    _advance_schedule(source_optimizer, source_scheduler, 1)
    _advance_schedule(restored_optimizer, restored_scheduler, 1)
    _assert_nested_equal(restored.state_dict(), source.state_dict())
    _assert_nested_equal(
        restored_optimizer.state_dict(),
        source_optimizer.state_dict(),
    )
    _assert_nested_equal(
        restored_scheduler.state_dict(),
        source_scheduler.state_dict(),
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("optimizer_kind", "sgd", "optimizer_kind must be one of"),
        ("schedule_kind", "linear", "schedule_kind must be one of"),
        (
            "schedule_decay_ratio",
            0.1,
            "schedule_decay_ratio must be None for the cosine schedule",
        ),
    ],
)
def test_training_state_rejects_invalid_recipe_identity(
    tmp_path,
    field,
    value,
    message,
):
    state, _, _ = _make_valid_training_state(tmp_path, step=2)
    state[field] = value

    with pytest.raises(RuntimeError, match=message):
        validate_training_state(state, checkpoint_step=2)


@pytest.mark.parametrize(
    ("decay_ratio", "message"),
    [
        (float("nan"), "schedule_decay_ratio must be finite"),
        (0.0, "schedule_decay_ratio must satisfy"),
        (-0.1, "schedule_decay_ratio must satisfy"),
        (1.1, "schedule_decay_ratio must satisfy"),
    ],
)
def test_training_state_rejects_invalid_wsd_decay_ratio(
    tmp_path,
    decay_ratio,
    message,
):
    state, _, _ = _make_valid_training_state(tmp_path, step=2)
    state["schedule_kind"] = "wsd"
    state["schedule_decay_ratio"] = decay_ratio

    with pytest.raises(RuntimeError, match=message):
        validate_training_state(state, checkpoint_step=2)


def test_training_state_rejects_scheduler_recipe_mismatch(tmp_path):
    state, _, _ = _make_valid_training_state(tmp_path, step=2)
    state["schedule_kind"] = "wsd"
    state["schedule_decay_ratio"] = 0.25

    with pytest.raises(RuntimeError, match="declared wsd recipe"):
        validate_training_state(state, checkpoint_step=2)


def test_training_state_rejects_legacy_and_cross_file_version_mismatch(tmp_path):
    with pytest.raises(RuntimeError, match="Legacy or incomplete"):
        validate_training_state({"step": 2}, checkpoint_step=2)

    state, _, _ = _make_valid_training_state(tmp_path, step=2)
    with pytest.raises(RuntimeError, match="directory step=3"):
        validate_training_state(state, checkpoint_step=3)

    state["cpt_state_version"] = 1
    with pytest.raises(RuntimeError, match="transaction mismatch"):
        validate_training_state(state, checkpoint_step=2)


def test_training_state_rejects_unexpected_schema_fields(tmp_path):
    state, _, _ = _make_valid_training_state(tmp_path, step=2)
    state["silently_ignored"] = True

    with pytest.raises(RuntimeError, match="unsupported fields: silently_ignored"):
        validate_training_state(state, checkpoint_step=2)


def test_training_state_rejects_missing_distributed_rng_fields(tmp_path):
    state, _, _ = _make_valid_training_state(tmp_path, step=2)
    state.pop("rng_states")

    with pytest.raises(RuntimeError, match="missing fields: rng_states"):
        validate_training_state(state, checkpoint_step=2)


def test_training_state_rejects_duplicate_rng_ranks(tmp_path):
    state, _, _ = _make_valid_training_state(tmp_path, step=2)
    rank_zero = copy.deepcopy(state["rng_states"][0])
    state["rng_world_size"] = 2
    state["rng_states"] = [rank_zero, copy.deepcopy(rank_zero)]

    with pytest.raises(RuntimeError, match="duplicate ranks"):
        validate_training_state(state, checkpoint_step=2)


@pytest.mark.parametrize(
    ("rng_world_size", "entry_count", "message"),
    [
        (2, 1, "contains 1 entries, expected rng_world_size=2"),
        (1, 2, "contains 2 entries, expected rng_world_size=1"),
    ],
)
def test_training_state_rejects_missing_or_extra_rank_rng_payloads(
    tmp_path,
    rng_world_size,
    entry_count,
    message,
):
    state, _, _ = _make_valid_training_state(tmp_path, step=2)
    rank_zero = copy.deepcopy(state["rng_states"][0])
    state["rng_world_size"] = rng_world_size
    state["rng_states"] = [copy.deepcopy(rank_zero) for _ in range(entry_count)]

    with pytest.raises(RuntimeError, match=message):
        validate_training_state(state, checkpoint_step=2)


def test_training_state_rejects_runtime_world_size_mismatch(tmp_path):
    state, _, _ = _make_valid_training_state(tmp_path, step=2)

    with pytest.raises(RuntimeError, match="runtime world_size=2"):
        validate_training_state(
            state,
            checkpoint_step=2,
            expected_world_size=2,
        )


def test_restore_rank_rng_state_selects_requested_rank():
    random.seed(81)
    np.random.seed(81)
    torch.manual_seed(81)
    rank_zero = capture_rng_state()

    random.seed(82)
    np.random.seed(82)
    torch.manual_seed(82)
    rank_one = capture_rng_state()
    expected = (random.random(), float(np.random.random()), torch.rand(4))

    training_state = {
        "rng_world_size": 2,
        "rng_states": [
            {"rank": 0, "state": rank_zero},
            {"rank": 1, "state": rank_one},
        ],
    }
    restore_rank_rng_state(training_state, rank=1, world_size=2)
    actual = (random.random(), float(np.random.random()), torch.rand(4))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_save_checkpoint_rejects_mixed_training_and_cpt_versions(tmp_path):
    events = []
    model = _DummyCPTModel(events)
    model.version = 2
    optimizer = make_adamw(model, lr=0.1, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, warmup_steps=1, total_steps=8)
    _advance_schedule(optimizer, scheduler, 1)
    metadata = _make_checkpoint_metadata(tmp_path)

    with pytest.raises(RuntimeError, match="mixed checkpoint"):
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            tmp_path,
            step=1,
            total_tok=8,
            warmup_steps=1,
            total_steps=8,
            fi=0,
            ptr=9,
            batch_size=1,
            seq_len=8,
            optimizer_kind="adamw",
            schedule_kind="cosine",
            schedule_decay_ratio=None,
            run_id=metadata["run_id"],
            code_manifest=metadata["code_manifest"],
            data_manifest=metadata["data_manifest"],
            tokenizer_manifest=metadata["tokenizer_manifest"],
            data_files=[tmp_path / "manifest_data" / "train_00000.pt"],
            tokenizer_dir=tmp_path / "tokenizer",
        )

    assert not list(tmp_path.glob("step_*"))
    assert not list(tmp_path.glob(".step_*.tmp-*"))


@pytest.mark.parametrize("drift_kind", ["data", "tokenizer"])
def test_save_checkpoint_rejects_live_manifest_drift(tmp_path, drift_kind):
    model = _DummyCPTModel([])
    model.version = 1
    optimizer = make_adamw(model, lr=0.1, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, warmup_steps=0, total_steps=4)
    _advance_schedule(optimizer, scheduler, 1)
    metadata = _make_checkpoint_metadata(tmp_path)
    shard = tmp_path / "manifest_data" / "train_00000.pt"
    tokenizer_file = tmp_path / "tokenizer" / "tokenizer.json"
    if drift_kind == "data":
        torch.save(torch.arange(64, dtype=torch.int64), shard)
    else:
        tokenizer_file.write_text('{"version":"2.0"}', encoding="utf-8")

    with pytest.raises(RuntimeError, match=drift_kind):
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            tmp_path,
            step=1,
            total_tok=8,
            warmup_steps=0,
            total_steps=4,
            fi=0,
            ptr=9,
            batch_size=1,
            seq_len=8,
            optimizer_kind="adamw",
            schedule_kind="cosine",
            schedule_decay_ratio=None,
            run_id=metadata["run_id"],
            code_manifest=metadata["code_manifest"],
            data_manifest=metadata["data_manifest"],
            tokenizer_manifest=metadata["tokenizer_manifest"],
            data_files=[shard],
            tokenizer_dir=tmp_path / "tokenizer",
        )

    assert not list(tmp_path.glob("step_*"))
    assert not list(tmp_path.glob(".step_*.tmp-*"))


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("total_tok", "Checkpoint token counter mismatch"),
        ("ptr", "is not aligned to chunk"),
    ],
)
def test_training_state_rejects_counter_and_pointer_mismatch(
    tmp_path,
    field,
    message,
):
    state, _, _ = _make_valid_training_state(tmp_path, step=2)
    state[field] += 1

    with pytest.raises(RuntimeError, match=message):
        validate_training_state(state, checkpoint_step=2)


def test_training_state_rejects_scheduler_step_mismatch(tmp_path):
    state, _, _ = _make_valid_training_state(tmp_path, step=2)
    state["sched"]["last_epoch"] += 1

    with pytest.raises(RuntimeError, match="Scheduler step mismatch"):
        validate_training_state(state, checkpoint_step=2)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda state: state["opt"]["param_groups"][0].__setitem__("weight_decay", -0.5), "weight_decay"),
        (lambda state: state["opt"]["param_groups"][0].__setitem__("betas", (-0.1, 0.95)), "betas"),
        (lambda state: state["opt"]["param_groups"][0].__setitem__("eps", 0.0), "eps"),
        (lambda state: state["opt"]["param_groups"][0].__setitem__("maximize", True), "maximize"),
        (lambda state: state["opt"]["param_groups"][0].__setitem__("amsgrad", True), "amsgrad"),
        (
            lambda state: state["opt"]["param_groups"][0].__setitem__(
                "decoupled_weight_decay",
                False,
            ),
            "decoupled_weight_decay",
        ),
        (lambda state: state["opt"]["param_groups"][0].__setitem__("foreach", False), "foreach"),
        (lambda state: state["opt"]["param_groups"][1].__setitem__("weight_decay", 0.5), "no-decay"),
        (lambda state: state["opt"]["param_groups"][0].__setitem__("unexpected", 1), "fields disagree"),
        (lambda state: state["opt"]["param_groups"][0].pop("betas"), "missing betas"),
        (lambda state: state["opt"]["param_groups"][0].pop("eps"), "missing eps"),
        (_duplicate_first_optimizer_parameter, "duplicate parameter"),
        (
            lambda state: state["sched"].__setitem__(
                "base_lrs",
                [9.0] * len(state["opt"]["param_groups"]),
            ),
            "base_lrs",
        ),
        (
            lambda state: state["sched"].__setitem__(
                "_last_lr",
                [9.0] * len(state["opt"]["param_groups"]),
            ),
            "_last_lr",
        ),
        (
            lambda state: state["sched"].__setitem__(
                "lr_lambdas",
                [object()] * len(state["opt"]["param_groups"]),
            ),
            "lr_lambdas",
        ),
        (lambda state: state["sched"].__setitem__("unexpected", 1), "scheduler state fields"),
    ],
)
def test_training_state_rejects_semantically_invalid_optimizer_scheduler_state(
    tmp_path,
    mutation,
    message,
):
    state, _, _ = _make_valid_training_state(tmp_path, step=2)
    mutation(state)

    with pytest.raises(RuntimeError, match=message):
        validate_training_state(state, checkpoint_step=2)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda parameter_state: (
                parameter_state.__setitem__(
                    "exp_avg",
                    parameter_state["exp_avg"].reshape(1),
                ),
                parameter_state.__setitem__(
                    "exp_avg_sq",
                    parameter_state["exp_avg_sq"].reshape(1),
                ),
            ),
            "shape.*expected",
        ),
        (
            lambda parameter_state: (
                parameter_state.__setitem__(
                    "exp_avg",
                    parameter_state["exp_avg"].double(),
                ),
                parameter_state.__setitem__(
                    "exp_avg_sq",
                    parameter_state["exp_avg_sq"].double(),
                ),
            ),
            "dtype.*expected",
        ),
        (
            lambda parameter_state: parameter_state["step"].fill_(3),
            "step=3 exceeds training step=2",
        ),
    ],
)
def test_training_state_binds_optimizer_moments_to_live_model(
    tmp_path,
    mutation,
    message,
):
    state, model, _ = _make_valid_training_state(tmp_path, step=2)
    parameter_state = next(iter(state["opt"]["state"].values()))
    mutation(parameter_state)

    with pytest.raises(RuntimeError, match=message):
        validate_training_state(
            state,
            checkpoint_step=2,
            model=model,
        )


def test_training_state_rejects_missing_scheduler_step_count(tmp_path):
    state, _, _ = _make_valid_training_state(tmp_path, step=2)
    state["sched"].pop("_step_count")

    with pytest.raises(RuntimeError, match="missing _step_count"):
        validate_training_state(state, checkpoint_step=2)


def test_training_state_rejects_model_config_hash_mismatch(tmp_path):
    state, model, _ = _make_valid_training_state(tmp_path, step=2)
    state["config_sha256"] = "b" * 64

    with pytest.raises(RuntimeError, match="Model config hash=.*disagrees"):
        validate_training_state(state, checkpoint_step=2, model=model)


def test_data_manifest_binds_explicit_shard_order(tmp_path):
    shard_a = tmp_path / "train_a.pt"
    shard_b = tmp_path / "train_b.pt"
    torch.save(torch.tensor([1, 2, 3]), shard_a)
    torch.save(torch.tensor([4, 5, 6]), shard_b)

    forward_order = build_data_manifest([shard_a, shard_b])
    reverse_order = build_data_manifest([shard_b, shard_a])

    assert [entry["name"] for entry in forward_order["entries"]] == [
        "train_a.pt",
        "train_b.pt",
    ]
    assert [entry["order"] for entry in forward_order["entries"]] == [0, 1]
    assert [entry["name"] for entry in reverse_order["entries"]] == [
        "train_b.pt",
        "train_a.pt",
    ]
    assert forward_order["sha256"] != reverse_order["sha256"]

    state, _, _ = _make_valid_training_state(tmp_path, step=2)
    state["data_manifest"] = forward_order
    with pytest.raises(RuntimeError, match="data-shard manifest disagrees"):
        validate_training_state(
            state,
            checkpoint_step=2,
            expected_data_manifest=reverse_order,
        )


def test_tokenizer_manifest_binds_recursive_artifact_identity(tmp_path):
    first_dir = tmp_path / "tokenizer_first"
    second_dir = tmp_path / "tokenizer_second"
    (first_dir / "nested").mkdir(parents=True)
    (second_dir / "nested").mkdir(parents=True)
    (first_dir / "tokenizer.json").write_text("same", encoding="utf-8")
    (second_dir / "tokenizer.json").write_text("same", encoding="utf-8")
    (first_dir / "nested" / "added_tokens.json").write_text("v1", encoding="utf-8")
    (second_dir / "nested" / "added_tokens.json").write_text("v2", encoding="utf-8")

    first = build_tokenizer_manifest(first_dir)
    second = build_tokenizer_manifest(second_dir)
    assert first["sha256"] != second["sha256"]
    assert [entry["name"] for entry in first["entries"]] == [
        "nested/added_tokens.json",
        "tokenizer.json",
    ]

    state, _, _ = _make_valid_training_state(tmp_path, step=2)
    state["tokenizer_manifest"] = first
    with pytest.raises(RuntimeError, match="tokenizer manifest disagrees"):
        validate_training_state(
            state,
            checkpoint_step=2,
            expected_tokenizer_manifest=second,
        )


def test_training_state_rejects_code_manifest_mismatch(tmp_path):
    state, _, metadata = _make_valid_training_state(tmp_path, step=2)
    different = copy.deepcopy(metadata["code_manifest"])
    different["sha256"] = "b" * 64

    with pytest.raises(RuntimeError, match="training code manifest disagrees"):
        validate_training_state(
            state,
            checkpoint_step=2,
            expected_code_manifest=different,
        )


def test_rng_state_weights_only_roundtrip(tmp_path):
    random.seed(717)
    np.random.seed(717)
    torch.manual_seed(717)
    state = capture_rng_state()
    path = tmp_path / "rng.pt"
    torch.save(state, path)
    restored = torch.load(path, map_location="cpu", weights_only=True)

    restore_rng_state(restored)
    first = (random.random(), float(np.random.random()), torch.rand(5))
    restore_rng_state(state)
    second = (random.random(), float(np.random.random()), torch.rand(5))

    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)


def test_cleanup_stale_checkpoint_temps_only_removes_dead_writer(
    tmp_path,
    monkeypatch,
):
    import scripts.train_utils as train_utils

    dead = tmp_path / ".step_0000001.tmp-999999"
    live = tmp_path / f".step_0000002.tmp-{os.getpid()}"
    unrelated = tmp_path / ".step_invalid.tmp-999999"
    dead.mkdir()
    live.mkdir()
    unrelated.mkdir()
    monkeypatch.setattr(
        train_utils,
        "_pid_is_running",
        lambda pid: pid == os.getpid(),
    )

    removed = cleanup_stale_checkpoint_temps(tmp_path)

    assert removed == [dead]
    assert not dead.exists()
    assert live.is_dir()
    assert unrelated.is_dir()


def test_cleanup_stale_checkpoint_temps_refuses_reparse_points(
    tmp_path,
    monkeypatch,
):
    import scripts.train_utils as train_utils

    redirected = tmp_path / ".step_0000001.tmp-999999"
    redirected.mkdir()
    original_check = train_utils._is_link_or_reparse_point
    monkeypatch.setattr(
        train_utils,
        "_is_link_or_reparse_point",
        lambda path: Path(path) == redirected or original_check(path),
    )

    with pytest.raises(RuntimeError, match="temp reparse point"):
        cleanup_stale_checkpoint_temps(tmp_path)

    assert redirected.is_dir()


def test_prune_periodic_checkpoints_only_deletes_old_complete_schema_dirs(tmp_path):
    required_files = ("config.json", "pytorch_model.bin", "training_state.pt")

    def complete_checkpoint(name):
        directory = tmp_path / name
        directory.mkdir()
        for filename in required_files:
            (directory / filename).write_bytes(b"checkpoint")
        return directory

    oldest = complete_checkpoint("step_0000001")
    retained = [
        complete_checkpoint("step_0000002"),
        complete_checkpoint("step_0000003"),
    ]
    notes = tmp_path / "step_0000000_notes"
    notes.mkdir()
    (notes / "user.txt").write_text("keep", encoding="utf-8")
    incomplete = tmp_path / "step_0000000"
    incomplete.mkdir()
    (incomplete / "config.json").write_text("{}", encoding="utf-8")
    final = complete_checkpoint("step_0000004_final")
    noncanonical = complete_checkpoint("step_4")

    removed = prune_periodic_checkpoints(tmp_path, keep_last=2)

    assert removed == [oldest]
    assert not oldest.exists()
    assert all(path.is_dir() for path in retained)
    assert notes.is_dir()
    assert (notes / "user.txt").read_text(encoding="utf-8") == "keep"
    assert incomplete.is_dir()
    assert final.is_dir()
    assert noncanonical.is_dir()


def _run_two_rank_checkpoint_case(
    tmp_path,
    *,
    expect_failure,
    rank1_manifest_drift=False,
):
    output_dir = tmp_path / "checkpoints"
    output_dir.mkdir()
    shard = tmp_path / "train_00000.pt"
    shard.write_bytes(b"shared distributed token shard")
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    rank1_runtime_shard = None
    if rank1_manifest_drift:
        rank1_dir = tmp_path / "rank1_data"
        rank1_dir.mkdir()
        rank1_runtime_shard = rank1_dir / shard.name
        rank1_runtime_shard.write_bytes(b"different rank-local token shard")
    if expect_failure and not rank1_manifest_drift:
        target = output_dir / "step_0000001"
        target.mkdir()
        (target / "sentinel.txt").write_text("do not replace", encoding="utf-8")

    torch_mp.spawn(
        _ddp_checkpoint_worker,
        args=(
            2,
            str(tmp_path / "gloo_filestore"),
            str(output_dir),
            str(shard),
            str(tokenizer_dir),
            str(result_dir),
            expect_failure,
            str(rank1_runtime_shard) if rank1_runtime_shard is not None else None,
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
    return output_dir, results


@pytest.mark.skipif(
    not torch_dist.is_available() or not torch_dist.is_gloo_available(),
    reason="PyTorch Gloo distributed backend is unavailable",
)
def test_two_rank_gloo_checkpoint_is_single_writer_and_restores_rank_rng(tmp_path):
    output_dir, results = _run_two_rank_checkpoint_case(
        tmp_path,
        expect_failure=False,
    )

    assert [result["rank"] for result in results] == [0, 1]
    assert all(result["error"] is None for result in results)
    assert all(result["restored"] for result in results)
    assert all(result["rng_world_size"] == 2 for result in results)
    assert all(result["rng_ranks"] == [0, 1] for result in results)
    assert len(list(output_dir.glob("step_*"))) == 1
    assert not list(output_dir.glob(".step_*.tmp-*"))
    state = torch.load(
        output_dir / "step_0000001" / "training_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert state["rng_world_size"] == 2
    assert [entry["rank"] for entry in state["rng_states"]] == [0, 1]


@pytest.mark.skipif(
    not torch_dist.is_available() or not torch_dist.is_gloo_available(),
    reason="PyTorch Gloo distributed backend is unavailable",
)
def test_two_rank_gloo_rank0_save_failure_reaches_every_rank(tmp_path):
    output_dir, results = _run_two_rank_checkpoint_case(
        tmp_path,
        expect_failure=True,
    )

    assert all(result["checkpoint"] is None for result in results)
    assert all(result["error"] is not None for result in results)
    assert all("checkpoint_publish" in result["error"] for result in results)
    assert all("Checkpoint already exists" in result["error"] for result in results)
    assert (output_dir / "step_0000001" / "sentinel.txt").read_text(
        encoding="utf-8"
    ) == "do not replace"
    assert not list(output_dir.glob(".step_*.tmp-*"))


@pytest.mark.skipif(
    not torch_dist.is_available() or not torch_dist.is_gloo_available(),
    reason="PyTorch Gloo distributed backend is unavailable",
)
def test_two_rank_gloo_manifest_drift_reaches_every_rank_before_publish(tmp_path):
    output_dir, results = _run_two_rank_checkpoint_case(
        tmp_path,
        expect_failure=True,
        rank1_manifest_drift=True,
    )

    assert all(result["checkpoint"] is None for result in results)
    assert all(result["error"] is not None for result in results)
    assert any("data-shard manifest changed" in result["error"] for result in results)
    assert not list(output_dir.glob("step_*"))
    assert not list(output_dir.glob(".step_*.tmp-*"))


@pytest.mark.skipif(
    not torch_dist.is_available() or not torch_dist.is_gloo_available(),
    reason="PyTorch Gloo distributed backend is unavailable",
)
def test_two_rank_gloo_accepts_identical_cpt_state_and_commits(tmp_path):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    torch_mp.spawn(
        _ddp_cpt_state_gate_worker,
        args=(
            2,
            str(tmp_path / "gloo_state_gate"),
            str(result_dir),
            "none",
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

    assert all(result["error"] is None for result in results)
    assert all(result["version"] == 1 for result in results)
    assert all("commit" in result["events"] for result in results)


@pytest.mark.skipif(
    not torch_dist.is_available() or not torch_dist.is_gloo_available(),
    reason="PyTorch Gloo distributed backend is unavailable",
)
@pytest.mark.parametrize(
    "mismatch_kind",
    ["price", "version", "learnable", "post_optimizer"],
)
def test_two_rank_gloo_rejects_divergent_cpt_state_without_advancing(
    tmp_path,
    mismatch_kind,
):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    torch_mp.spawn(
        _ddp_cpt_state_gate_worker,
        args=(
            2,
            str(tmp_path / "gloo_state_gate"),
            str(result_dir),
            mismatch_kind,
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
    assert all(result["error"] is not None for result in results)
    assert all(result["restored"] for result in results)
    assert all("commit" not in result["events"] for result in results)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--max-steps", "0"], "max-steps must be positive"),
        (["--warmup-steps", "-1"], "warmup-steps must be non-negative"),
        (["--lr", "0"], "lr must be finite and positive"),
        (["--lr", "nan"], "lr must be finite and positive"),
        (["--wd", "-0.1"], "wd must be finite and non-negative"),
        (["--wd", "inf"], "wd must be finite and non-negative"),
    ],
)
def test_train_cli_rejects_invalid_step_controls(arguments, message):
    script = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    result = subprocess.run(
        [sys.executable, "-B", str(script), *arguments],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 2
    assert message in result.stderr


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--trials", "0"], "trials must be positive"),
        (["--steps-per-trial", "-1"], "steps-per-trial must be positive"),
        (["--batch-size", "0"], "batch-size must be positive"),
        (["--seq-len", "0"], "seq-len must be positive"),
        (["--eval-limit", "0"], "eval-limit must be positive"),
        (["--eval-batch", "0"], "eval-batch must be positive"),
        (["--max-data-tokens", "0"], "max-data-tokens must be positive"),
        (["--tasks", ", ,"], "tasks must contain at least one task"),
    ],
)
def test_search_cli_rejects_invalid_controls_before_writing_output(
    tmp_path,
    arguments,
    message,
):
    script = Path(__file__).resolve().parents[1] / "scripts" / "search.py"
    output_dir = tmp_path / "search_output"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "--output-dir",
            str(output_dir),
            *arguments,
        ],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert not output_dir.exists()


def test_generate_trials_rejects_nonpositive_count():
    from scripts.search import generate_trials

    with pytest.raises(ValueError, match="positive"):
        generate_trials(-1)


@pytest.mark.parametrize("script_name", ["train.py", "resume.py"])
def test_training_entrypoints_fail_fast_for_uninitialized_torchrun(
    tmp_path,
    script_name,
):
    script = Path(__file__).resolve().parents[1] / "scripts" / script_name
    output_dir = tmp_path / "must_not_be_created"
    env = os.environ.copy()
    env["WORLD_SIZE"] = "2"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "--output-dir",
            str(output_dir),
        ],
        cwd=script.parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 2
    assert "DDP training entrypoint is not implemented" in result.stderr
    assert not output_dir.exists()


def test_cosine_schedule_rejects_negative_warmup():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    with pytest.raises(ValueError, match="warmup_steps must be non-negative"):
        make_cosine_schedule(optimizer, warmup_steps=-1, total_steps=8)


def test_training_loop_rejects_scheduler_bound_to_another_optimizer_before_io(
    tmp_path,
):
    model = _DummyCPTModel([])
    optimizer = make_adamw(model, lr=0.1, weight_decay=0.0)
    other_optimizer = make_adamw(model, lr=0.1, weight_decay=0.0)
    scheduler = make_cosine_schedule(
        other_optimizer,
        warmup_steps=0,
        total_steps=4,
    )
    output_dir = tmp_path / "must_not_be_created"

    with pytest.raises(RuntimeError, match="not bound to the live optimizer"):
        training_loop(
            model,
            opt=optimizer,
            sched=scheduler,
            files=[tmp_path / "must_not_be_read.pt"],
            fi=0,
            ptr=0,
            total_tok=0,
            bs=1,
            seq=8,
            chunk=9,
            output_dir=output_dir,
            max_steps=4,
            save_every_min=10**9,
            log_every=100,
            step_start=0,
            schedule_args={
                "warmup_steps": 0,
                "total_steps": 4,
                "schedule_kind": "cosine",
                "schedule_decay_ratio": None,
            },
            checkpoint_metadata={
                "run_id": "0" * 32,
                "code_manifest": {},
                "data_manifest": {},
                "tokenizer_manifest": {},
            },
            tokenizer_dir=tmp_path / "tokenizer",
            optimizer_kind="adamw",
        )

    assert not output_dir.exists()


def test_training_loop_rejects_optimizer_bound_to_another_model_before_io(
    tmp_path,
):
    model = _DummyCPTModel([])
    other_model = _DummyCPTModel([])
    optimizer = make_adamw(other_model, lr=0.1, weight_decay=0.0)
    scheduler = make_cosine_schedule(
        optimizer,
        warmup_steps=0,
        total_steps=4,
    )
    output_dir = tmp_path / "must_not_be_created"

    with pytest.raises(RuntimeError, match="not bound to the live model"):
        training_loop(
            model,
            opt=optimizer,
            sched=scheduler,
            files=[tmp_path / "must_not_be_read.pt"],
            fi=0,
            ptr=0,
            total_tok=0,
            bs=1,
            seq=8,
            chunk=9,
            output_dir=output_dir,
            max_steps=4,
            save_every_min=10**9,
            log_every=100,
            step_start=0,
            schedule_args={
                "warmup_steps": 0,
                "total_steps": 4,
                "schedule_kind": "cosine",
                "schedule_decay_ratio": None,
            },
            checkpoint_metadata={
                "run_id": "0" * 32,
                "code_manifest": {},
                "data_manifest": {},
                "tokenizer_manifest": {},
            },
            tokenizer_dir=tmp_path / "tokenizer",
            optimizer_kind="adamw",
        )

    assert not output_dir.exists()


def test_training_loop_rejects_stale_wsd_progress_before_io(tmp_path):
    model = _DummyCPTModel([])
    optimizer = make_adamw(model, lr=0.1, weight_decay=0.0)
    scheduler = make_wsd_schedule(
        optimizer,
        warmup_steps=1,
        total_steps=10,
        decay_ratio=0.5,
    )
    output_dir = tmp_path / "must_not_be_created"

    # Fresh step 0 and resumed step 4 are both at peak LR, so a current-LR-only
    # check cannot distinguish the stale scheduler progress.
    with pytest.raises(RuntimeError, match="Scheduler step mismatch"):
        training_loop(
            model,
            opt=optimizer,
            sched=scheduler,
            files=[tmp_path / "must_not_be_read.pt"],
            fi=0,
            ptr=36,
            total_tok=32,
            bs=1,
            seq=8,
            chunk=9,
            output_dir=output_dir,
            max_steps=10,
            save_every_min=10**9,
            log_every=100,
            step_start=4,
            schedule_args={
                "warmup_steps": 1,
                "total_steps": 10,
                "schedule_kind": "wsd",
                "schedule_decay_ratio": 0.5,
            },
            checkpoint_metadata={
                "run_id": "0" * 32,
                "code_manifest": {},
                "data_manifest": {},
                "tokenizer_manifest": {},
            },
            tokenizer_dir=tmp_path / "tokenizer",
            optimizer_kind="adamw",
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lr": float("inf"), "weight_decay": 0.1}, "lr must be finite"),
        ({"lr": -0.1, "weight_decay": 0.1}, "lr must be non-negative"),
        ({"lr": 0.1, "weight_decay": float("nan")}, "weight_decay must be finite"),
        ({"lr": 0.1, "weight_decay": -0.1}, "weight_decay must be non-negative"),
        (
            {"lr": 0.1, "weight_decay": 0.1, "betas": (1.0, 0.95)},
            "betas must satisfy",
        ),
    ],
)
def test_make_adamw_rejects_invalid_hyperparameters(kwargs, message):
    model = torch.nn.Linear(2, 2)

    with pytest.raises((RuntimeError, ValueError), match=message):
        make_adamw(model, **kwargs)


@pytest.mark.parametrize("raw_world_size", ["true", "0", "-1", "2.0", " 2"])
def test_torchrun_world_size_parser_rejects_noncanonical_values(
    monkeypatch,
    raw_world_size,
):
    import scripts.train_utils as train_utils

    monkeypatch.setenv("WORLD_SIZE", raw_world_size)
    with pytest.raises(RuntimeError, match="canonical positive integer"):
        train_utils.reject_uninitialized_torchrun_environment()


def test_training_loop_shard_rollover_publishes_committed_shard_cursor(
    tmp_path,
    monkeypatch,
):
    import scripts.train_utils as train_utils

    chunk = 6
    first = tmp_path / "train_00000.pt"
    second = tmp_path / "train_00001.pt"
    torch.save(torch.arange(chunk, dtype=torch.int64), first)
    torch.save(torch.arange(100, 100 + 2 * chunk, dtype=torch.int64), second)
    model = _DummyCPTModel([])
    optimizer = make_adamw(model, lr=0.1, weight_decay=0.0)
    scheduler = make_cosine_schedule(
        optimizer,
        warmup_steps=0,
        total_steps=3,
    )
    metadata = _make_checkpoint_metadata(tmp_path)

    def committed_iteration(
        model,
        optimizer,
        scheduler,
        forward_fn,
        max_grad_norm,
        failure_policy,
    ):
        assert failure_policy == "exact-rollback"
        output = forward_fn()
        model.validate_cpt_transaction(output["cpt_transaction"])
        model.commit_cpt_transaction(output["cpt_transaction"])
        return output, torch.tensor(0.0)

    monkeypatch.setattr(
        train_utils,
        "execute_training_iteration",
        committed_iteration,
    )

    step, total_tok, fi, ptr, _ = training_loop(
        model,
        opt=optimizer,
        sched=scheduler,
        files=[first, second],
        fi=0,
        ptr=0,
        total_tok=0,
        bs=1,
        seq=5,
        chunk=chunk,
        output_dir=tmp_path / "loop_output",
        max_steps=3,
        save_every_min=10**9,
        log_every=100,
        schedule_args={
            "warmup_steps": 0,
            "total_steps": 3,
            "schedule_kind": "cosine",
            "schedule_decay_ratio": None,
        },
        checkpoint_metadata=metadata,
        tokenizer_dir=tmp_path / "tokenizer",
        optimizer_kind="adamw",
    )

    assert step == 3
    assert total_tok == 15
    assert fi == 1
    assert ptr == 2 * chunk
    assert model.version == 3


def test_optimizer_and_scheduler_steps_are_centralized_in_shared_helper():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    violations = []

    for path in scripts_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        function_stack = []

        class StepVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                function_stack.append(node.name)
                self.generic_visit(node)
                function_stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "step":
                    owner = function_stack[-1] if function_stack else "<module>"
                    if not (
                        path.name == "train_utils.py"
                        and "execute_training_step" in function_stack
                    ):
                        violations.append((path.name, node.lineno, owner))
                self.generic_visit(node)

        StepVisitor().visit(tree)

    assert violations == []
