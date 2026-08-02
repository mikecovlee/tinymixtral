import copy

import pytest
import torch

import scripts.train_utils as train_utils
from model.config import TinyMixtralConfig
from model.cpt_router import CPTLayerProposal, CPTTransaction
from model.modeling import TinyMixtralForCausalLM
from scripts.train_utils import (
    _execute_cpt_optimizer_step,
    make_adamw,
    make_cosine_schedule,
    save_checkpoint,
    training_loop,
)


def tiny_model():
    config = TinyMixtralConfig(
        vocab_size=37,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=24,
        num_local_experts=3,
        num_experts_per_tok=2,
        expert_intermediate_size=16,
        cpt_projection_dim=4,
        cpt_init_seed=101,
    )
    return TinyMixtralForCausalLM(config)


def persistent_snapshot(model):
    return tuple(
        (
            router.anchors.detach().clone(),
            router.congestion_price.detach().clone(),
            router.optimizer_step.detach().clone(),
            router.state_version.detach().clone(),
        )
        for router in model._cpt_routers()
    )


def assert_persistent_equal(model, snapshot):
    for router, (anchors, price, optimizer_step, version) in zip(
        model._cpt_routers(),
        snapshot,
    ):
        assert torch.equal(router.anchors, anchors)
        assert torch.equal(router.congestion_price, price)
        assert torch.equal(router.optimizer_step, optimizer_step)
        assert torch.equal(router.state_version, version)


def forward_transaction(model, mask=None):
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    return model(input_ids, attention_mask=mask, labels=input_ids)["cpt_transaction"]


def partition_trainable_parameters(model):
    decay = []
    no_decay = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return decay, no_decay


def set_zero_cpt_gradients(model):
    for parameter in model.cpt_trainable_parameters():
        parameter.grad = torch.zeros_like(parameter)


def skewed_transaction(model, token_count=10):
    proposals = []
    for index, router in enumerate(model._cpt_routers()):
        load = torch.zeros(router.num_experts, dtype=torch.float32)
        load[0] = float(token_count)
        proposals.append(
            CPTLayerProposal(
                layer_index=index,
                load_sum=load,
                token_count=torch.tensor(token_count, dtype=torch.int64),
                state_version=router.state_version.detach().clone(),
            )
        )
    return CPTTransaction(tuple(proposals))


def test_forward_is_pure_for_all_persistent_cpt_state():
    torch.manual_seed(1)
    model = tiny_model()
    before = persistent_snapshot(model)
    transaction = forward_transaction(model)
    assert len(transaction.proposals) == len(model.layers)
    assert_persistent_equal(model, before)
    model.abort_cpt_transaction(transaction)
    assert_persistent_equal(model, before)


def test_config_drift_is_rejected_before_optimizer_step():
    model = tiny_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    output = model(input_ids, labels=input_ids)
    output["loss"].backward()
    transaction = output["cpt_transaction"]
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, 0, 2)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    model.config.num_experts_per_tok = 1

    with pytest.raises(RuntimeError, match="num_experts_per_tok"):
        _execute_cpt_optimizer_step(model, optimizer, scheduler, transaction)
    for parameter, saved in zip(model.parameters(), before):
        assert torch.equal(parameter, saved)
    assert not optimizer.state
    assert transaction.consumed
    assert model.get_cpt_state_version() == 0


def test_successful_step_commits_all_layers_retracts_anchors_and_lags_price():
    model = tiny_model()
    transaction = skewed_transaction(model, token_count=12)
    kernels_before = [
        router.expert_kernel().detach().clone()
        for router in model._cpt_routers()
    ]
    optimizer = make_adamw(model, lr=0.0, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, warmup_steps=0, total_steps=2)
    set_zero_cpt_gradients(model)

    version = _execute_cpt_optimizer_step(model, optimizer, scheduler, transaction)
    assert version == 1
    assert model.get_cpt_optimizer_step() == 1
    for router, kernel_before in zip(model._cpt_routers(), kernels_before):
        expected = torch.zeros(router.num_experts)
        expected[0] = router.price_learning_rate * (
            1.0 - router.capacity_factor / router.num_experts
        )
        torch.testing.assert_close(router.congestion_price, expected, atol=1e-7, rtol=0)
        torch.testing.assert_close(
            torch.linalg.vector_norm(router.anchors, dim=0),
            torch.ones(router.num_prototypes),
            atol=2e-6,
            rtol=2e-6,
        )
        # The completed forward used lambda_n; only the next forward sees lambda_{n+1}.
        assert not torch.equal(router.expert_kernel(), kernel_before)
    assert transaction.consumed


def test_all_padding_step_keeps_price_but_advances_success_version():
    model = tiny_model()
    mask = torch.zeros(2, 5, dtype=torch.bool)
    transaction = forward_transaction(model, mask=mask)
    before_prices = [router.congestion_price.clone() for router in model._cpt_routers()]
    optimizer = make_adamw(model, lr=0.0, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, 0, 2)
    _execute_cpt_optimizer_step(model, optimizer, scheduler, transaction)
    for router, price in zip(model._cpt_routers(), before_prices):
        assert int(router.optimizer_step) == 0
        assert int(router.state_version) == 1
        assert torch.equal(router.congestion_price, price)


def test_partial_cpt_gradients_are_rejected_before_optimizer_step(monkeypatch):
    model = tiny_model()
    transaction = skewed_transaction(model)
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, 0, 2)
    parameters = model.cpt_trainable_parameters()
    parameters[0].grad = torch.zeros_like(parameters[0])
    calls = []
    original_step = optimizer.step

    def tracked_step(*args, **kwargs):
        calls.append((args, kwargs))
        return original_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", tracked_step)
    before = persistent_snapshot(model)
    with pytest.raises(RuntimeError, match="gradients must be present"):
        _execute_cpt_optimizer_step(
            model,
            optimizer,
            scheduler,
            transaction,
        )

    assert calls == []
    assert not optimizer.state
    assert transaction.consumed
    assert_persistent_equal(model, before)


def test_valid_transaction_without_backward_is_rejected_before_step(monkeypatch):
    model = tiny_model()
    transaction = skewed_transaction(model)
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, 0, 2)
    calls = []
    original_step = optimizer.step

    def tracked_step(*args, **kwargs):
        calls.append((args, kwargs))
        return original_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", tracked_step)
    before = persistent_snapshot(model)
    with pytest.raises(RuntimeError, match="requires gradients"):
        _execute_cpt_optimizer_step(
            model,
            optimizer,
            scheduler,
            transaction,
        )

    assert calls == []
    assert not optimizer.state
    assert transaction.consumed
    assert_persistent_equal(model, before)


def test_all_padding_transaction_with_gradients_is_rejected_before_step(
    monkeypatch,
):
    model = tiny_model()
    mask = torch.zeros(2, 5, dtype=torch.bool)
    transaction = forward_transaction(model, mask=mask)
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, 0, 2)
    set_zero_cpt_gradients(model)
    calls = []
    original_step = optimizer.step

    def tracked_step(*args, **kwargs):
        calls.append((args, kwargs))
        return original_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", tracked_step)
    before = persistent_snapshot(model)
    with pytest.raises(RuntimeError, match="all-padding"):
        _execute_cpt_optimizer_step(
            model,
            optimizer,
            scheduler,
            transaction,
        )

    assert calls == []
    assert not optimizer.state
    assert transaction.consumed
    assert_persistent_equal(model, before)


def test_nonunit_anchor_tampering_is_rejected_before_optimizer_step(monkeypatch):
    model = tiny_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    output = model(input_ids, labels=input_ids)
    output["loss"].backward()
    transaction = output["cpt_transaction"]
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, 0, 2)

    with torch.no_grad():
        model._cpt_routers()[0].anchors[0].add_(0.375)
    anchor_norms = torch.linalg.vector_norm(
        model._cpt_routers()[0].anchors,
        dim=0,
    )
    assert not torch.allclose(
        anchor_norms,
        torch.ones_like(anchor_norms),
        atol=5e-5,
        rtol=5e-5,
    )

    calls = {"optimizer": 0, "scheduler": 0, "commit": 0}
    original_optimizer_step = optimizer.step
    original_scheduler_step = scheduler.step
    original_commit = model.commit_cpt_transaction

    def tracked_optimizer_step(*args, **kwargs):
        calls["optimizer"] += 1
        return original_optimizer_step(*args, **kwargs)

    def tracked_scheduler_step(*args, **kwargs):
        calls["scheduler"] += 1
        return original_scheduler_step(*args, **kwargs)

    def tracked_commit(*args, **kwargs):
        calls["commit"] += 1
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", tracked_optimizer_step)
    monkeypatch.setattr(scheduler, "step", tracked_scheduler_step)
    monkeypatch.setattr(model, "commit_cpt_transaction", tracked_commit)
    parameter_snapshot = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )
    persistent_state = persistent_snapshot(model)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scheduler_state = copy.deepcopy(scheduler.state_dict())

    with pytest.raises(RuntimeError, match="unit L2 norm"):
        _execute_cpt_optimizer_step(
            model,
            optimizer,
            scheduler,
            transaction,
        )

    assert calls == {"optimizer": 0, "scheduler": 0, "commit": 0}
    assert transaction.consumed
    assert_persistent_equal(model, persistent_state)
    for parameter, expected in zip(model.parameters(), parameter_snapshot):
        assert torch.equal(parameter, expected)
    assert_nested_state_equal(optimizer.state_dict(), optimizer_state)
    assert_nested_state_equal(scheduler.state_dict(), scheduler_state)
    assert getattr(model, "_tinymixtral_fail_stop", False) is False


@pytest.mark.parametrize("failure", ["nan", "version", "count"])
def test_invalid_single_layer_proposal_causes_zero_layer_commits(failure):
    model = tiny_model()
    transaction = skewed_transaction(model)
    proposals = list(transaction.proposals)
    original = proposals[1]
    if failure == "nan":
        bad_load = original.load_sum.clone()
        bad_load[0] = float("nan")
        proposals[1] = CPTLayerProposal(
            original.layer_index, bad_load, original.token_count, original.state_version
        )
    elif failure == "version":
        proposals[1] = CPTLayerProposal(
            original.layer_index,
            original.load_sum,
            original.token_count,
            original.state_version + 1,
        )
    else:
        bad_count = original.token_count + 1
        bad_load = original.load_sum.clone()
        bad_load[0] += 1
        proposals[1] = CPTLayerProposal(
            original.layer_index, bad_load, bad_count, original.state_version
        )
    transaction = CPTTransaction(tuple(proposals))
    before = persistent_snapshot(model)
    with pytest.raises(RuntimeError):
        model.commit_cpt_transaction(transaction)
    assert_persistent_equal(model, before)


@pytest.mark.parametrize("invalid_layer_index", [True, 1.0, "1"])
def test_non_integer_proposal_layer_index_is_rejected_without_writes(
    invalid_layer_index,
):
    model = tiny_model()
    proposals = list(skewed_transaction(model).proposals)
    original = proposals[1]
    proposals[1] = CPTLayerProposal(
        invalid_layer_index,
        original.load_sum,
        original.token_count,
        original.state_version,
    )
    transaction = CPTTransaction(tuple(proposals))
    before = persistent_snapshot(model)

    with pytest.raises(TypeError, match="layer_index"):
        model.commit_cpt_transaction(transaction)

    assert transaction.consumed
    assert_persistent_equal(model, before)


def test_wrong_integer_proposal_layer_index_is_rejected_without_writes():
    model = tiny_model()
    proposals = list(skewed_transaction(model).proposals)
    original = proposals[1]
    proposals[1] = CPTLayerProposal(
        0,
        original.load_sum,
        original.token_count,
        original.state_version,
    )
    transaction = CPTTransaction(tuple(proposals))
    before = persistent_snapshot(model)

    with pytest.raises(RuntimeError, match="layer index mismatch"):
        model.commit_cpt_transaction(transaction)

    assert transaction.consumed
    assert_persistent_equal(model, before)


def test_duplicate_commit_is_rejected_before_any_write():
    model = tiny_model()
    transaction = skewed_transaction(model)
    model.commit_cpt_transaction(transaction)
    after = persistent_snapshot(model)
    with pytest.raises(RuntimeError, match="already consumed"):
        model.commit_cpt_transaction(transaction)
    assert_persistent_equal(model, after)


def test_apply_failure_rolls_back_every_layer(monkeypatch):
    model = tiny_model()
    transaction = skewed_transaction(model)
    before = persistent_snapshot(model)

    def fail_apply(*_args, **_kwargs):
        raise RuntimeError("injected commit write failure")

    monkeypatch.setattr(model.layers[1].moe.cpt_router, "apply_commit", fail_apply)
    with pytest.raises(RuntimeError, match="injected"):
        model.commit_cpt_transaction(transaction)
    assert_persistent_equal(model, before)
    assert transaction.consumed
    with pytest.raises(RuntimeError, match="already consumed"):
        model.commit_cpt_transaction(transaction)


def test_direct_commit_rollback_failure_consumes_and_poisons(monkeypatch, tmp_path):
    model = tiny_model()
    transaction = skewed_transaction(model)

    def fail_apply(*_args, **_kwargs):
        raise RuntimeError("injected commit write failure")

    def fail_restore(*_args, **_kwargs):
        raise RuntimeError("injected direct rollback failure")

    monkeypatch.setattr(model.layers[1].moe.cpt_router, "apply_commit", fail_apply)
    monkeypatch.setattr(
        model.layers[0].moe.cpt_router,
        "restore_commit_snapshot",
        fail_restore,
    )
    with pytest.raises(RuntimeError, match="recovery was incomplete") as caught:
        model.commit_cpt_transaction(transaction)
    assert "injected direct rollback failure" in str(caught.value.__cause__)
    assert transaction.consumed
    assert model._tinymixtral_fail_stop is True

    with pytest.raises(RuntimeError, match="fail-stop poisoned"):
        model.commit_cpt_transaction(skewed_transaction(model))
    with pytest.raises(RuntimeError, match="fail-stop poisoned"):
        model.save_pretrained(tmp_path / "direct_model")


class RecordingScheduler:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.calls = 0
        self.version_seen = None

    def step(self):
        self.calls += 1
        self.version_seen = self.model.get_cpt_state_version()


class CountingScheduler:
    def __init__(self, scheduler, calls):
        self.scheduler = scheduler
        self.calls = calls

    def step(self):
        self.calls["scheduler"] += 1
        return self.scheduler.step()


class PartialFailureOptimizer(torch.optim.Optimizer):
    def __init__(self, model, target):
        decay, no_decay = partition_trainable_parameters(model)
        super().__init__([{"params": decay}, {"params": no_decay}], defaults={})
        self.target = target

    @torch.no_grad()
    def step(self, closure=None):
        parameter = self.target
        parameter.add_(7.0)
        self.state[parameter]["marker"] = torch.tensor(9.0)
        raise RuntimeError("partial optimizer failure")


def test_partial_optimizer_exception_restores_cpt_parameters_and_moments():
    model = tiny_model()
    transaction = skewed_transaction(model)
    parameters = model.cpt_trainable_parameters()
    before_parameters = [parameter.detach().clone() for parameter in parameters]
    before_persistent = persistent_snapshot(model)
    optimizer = PartialFailureOptimizer(model, parameters[0])
    scheduler = RecordingScheduler(model, optimizer)
    set_zero_cpt_gradients(model)

    with pytest.raises(RuntimeError, match="partial optimizer failure"):
        _execute_cpt_optimizer_step(model, optimizer, scheduler, transaction)
    for parameter, before in zip(parameters, before_parameters):
        assert torch.equal(parameter, before)
        assert parameter not in optimizer.state
    assert_persistent_equal(model, before_persistent)
    assert scheduler.calls == 0
    assert transaction.consumed


def test_optimizer_failure_with_failed_rollback_poison_blocks_all_reuse(
    monkeypatch, tmp_path
):
    model = tiny_model()
    transaction = skewed_transaction(model)
    target = model.cpt_trainable_parameters()[0]
    optimizer = PartialFailureOptimizer(model, target)
    scheduler = RecordingScheduler(model, optimizer)
    set_zero_cpt_gradients(model)

    def fail_restore(*_args, **_kwargs):
        raise RuntimeError("injected rollback failure")

    monkeypatch.setattr(
        train_utils, "_restore_cpt_optimizer_state", fail_restore
    )
    with pytest.raises(RuntimeError, match="recovery was incomplete") as caught:
        _execute_cpt_optimizer_step(model, optimizer, scheduler, transaction)
    assert "injected rollback failure" in str(caught.value.__cause__)
    assert transaction.consumed
    for training_object in (model, optimizer, scheduler):
        assert training_object._tinymixtral_fail_stop is True

    with pytest.raises(RuntimeError, match="fail-stop poisoned"):
        _execute_cpt_optimizer_step(
            model, optimizer, scheduler, skewed_transaction(model)
        )
    with pytest.raises(RuntimeError, match="fail-stop poisoned"):
        save_checkpoint(
            model, optimizer, scheduler, tmp_path / "checkpoint",
            step=0, total_tok=0, warmup_steps=0, total_steps=2,
            fi=0, ptr=0, batch_size=2, seq_len=5,
        )
    with pytest.raises(RuntimeError, match="fail-stop poisoned"):
        model.save_pretrained(tmp_path / "direct_model")


def test_post_step_invalid_anchor_restores_cpt_and_does_not_step_scheduler(
    monkeypatch,
):
    model = tiny_model()
    transaction = skewed_transaction(model)
    parameters = model.cpt_trainable_parameters()
    before_parameters = [parameter.detach().clone() for parameter in parameters]
    before_persistent = persistent_snapshot(model)
    optimizer = make_adamw(model, lr=0.0, weight_decay=0.0)
    scheduler = RecordingScheduler(model, optimizer)
    set_zero_cpt_gradients(model)
    original_step = optimizer.step

    @torch.no_grad()
    def zero_anchor_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        model.layers[0].moe.cpt_router.anchors.zero_()
        return result

    monkeypatch.setattr(optimizer, "step", zero_anchor_step)

    with pytest.raises(RuntimeError, match="anchor columns"):
        _execute_cpt_optimizer_step(model, optimizer, scheduler, transaction)
    for parameter, before in zip(parameters, before_parameters):
        assert torch.equal(parameter, before)
    assert_persistent_equal(model, before_persistent)
    assert scheduler.calls == 0


def test_scheduler_runs_only_after_successful_cpt_commit():
    model = tiny_model()
    transaction = skewed_transaction(model)
    optimizer = make_adamw(model, lr=0.0, weight_decay=0.0)
    scheduler = RecordingScheduler(model, optimizer)
    set_zero_cpt_gradients(model)
    version = _execute_cpt_optimizer_step(model, optimizer, scheduler, transaction)
    assert version == 1
    assert scheduler.calls == 1
    assert scheduler.version_seen == 1


def test_standard_adamw_step_precision_limit_is_rejected_before_write(
    monkeypatch,
):
    model = tiny_model()
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, 0, 3)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    output = model(input_ids, labels=input_ids)
    output["loss"].backward()
    _execute_cpt_optimizer_step(
        model,
        optimizer,
        scheduler,
        output["cpt_transaction"],
    )
    optimizer.zero_grad(set_to_none=True)

    precision_limit = 1 << 24
    for router in model._cpt_routers():
        router.optimizer_step.fill_(precision_limit)
        router.state_version.fill_(precision_limit)
    for parameter in model.cpt_trainable_parameters():
        optimizer.state[parameter]["step"].fill_(precision_limit)

    output = model(input_ids, labels=input_ids)
    output["loss"].backward()
    calls = []
    original_step = optimizer.step

    def tracked_step(*args, **kwargs):
        calls.append((args, kwargs))
        return original_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", tracked_step)
    before = persistent_snapshot(model)
    with pytest.raises(RuntimeError, match="cannot represent the next"):
        _execute_cpt_optimizer_step(
            model,
            optimizer,
            scheduler,
            output["cpt_transaction"],
        )

    assert calls == []
    assert output["cpt_transaction"].consumed
    assert_persistent_equal(model, before)


class FailingScheduler(RecordingScheduler):
    def step(self):
        raise RuntimeError("injected scheduler failure")


def test_scheduler_failure_rolls_back_cpt_and_poison_blocks_step_and_save(tmp_path):
    model = tiny_model()
    transaction = skewed_transaction(model)
    optimizer = make_adamw(model, lr=0.0, weight_decay=0.0)
    scheduler = FailingScheduler(model, optimizer)
    set_zero_cpt_gradients(model)
    before_parameters = [
        parameter.detach().clone() for parameter in model.cpt_trainable_parameters()
    ]
    before_persistent = persistent_snapshot(model)

    with pytest.raises(RuntimeError, match="scheduler failure"):
        _execute_cpt_optimizer_step(model, optimizer, scheduler, transaction)
    for parameter, before in zip(model.cpt_trainable_parameters(), before_parameters):
        assert torch.equal(parameter, before)
    assert_persistent_equal(model, before_persistent)
    assert transaction.consumed
    with pytest.raises(RuntimeError, match="fail-stop poisoned"):
        _execute_cpt_optimizer_step(
            model, optimizer, scheduler, skewed_transaction(model)
        )
    with pytest.raises(RuntimeError, match="fail-stop poisoned"):
        save_checkpoint(
            model, optimizer, scheduler, tmp_path,
            step=0, total_tok=0, warmup_steps=0, total_steps=2,
            fi=0, ptr=0, batch_size=2, seq_len=5,
        )
    with pytest.raises(RuntimeError, match="fail-stop poisoned"):
        model.save_pretrained(tmp_path / "direct_model")


class BackwardFailure(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss):
        del ctx
        return loss.clone()

    @staticmethod
    def backward(ctx, grad_output):
        del ctx, grad_output
        raise RuntimeError("injected backward failure")


class InfiniteLossGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss):
        del ctx
        return loss.clone()

    @staticmethod
    def backward(ctx, grad_output):
        del ctx
        return torch.full_like(grad_output, float("inf"))


def assert_nested_state_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
        return
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_nested_state_equal(actual[key], expected[key])
        return
    if isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            assert_nested_state_equal(actual_item, expected_item)
        return
    assert actual == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        ("nonfinite_loss", "Non-finite loss"),
        ("backward_exception", "injected backward failure"),
        ("nonfinite_gradient", "Non-finite gradient norm"),
    ],
)
def test_training_loop_pre_step_failure_is_transactional(
    failure,
    expected_error,
    monkeypatch,
    tmp_path,
):
    model = tiny_model().to("cuda").to(torch.bfloat16)
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, 0, 3)

    initial_ids = torch.randint(
        0,
        model.config.vocab_size,
        (2, 5),
        device="cuda",
    )
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        initial_output = model(initial_ids, labels=initial_ids)
    initial_output["loss"].backward()
    _execute_cpt_optimizer_step(
        model,
        optimizer,
        scheduler,
        initial_output["cpt_transaction"],
    )
    optimizer.zero_grad(set_to_none=True)

    parameter_snapshot = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )
    persistent_state = persistent_snapshot(model)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scheduler_state = copy.deepcopy(scheduler.state_dict())
    captured = {}
    calls = {"optimizer": 0, "scheduler": 0, "save": 0}

    original_forward = model.forward
    original_optimizer_step = optimizer.step
    counted_scheduler = CountingScheduler(scheduler, calls)

    def injected_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        captured["transaction"] = output["cpt_transaction"]
        if failure == "nonfinite_loss":
            output["loss"] = output["loss"] * float("nan")
        elif failure == "backward_exception":
            output["loss"] = BackwardFailure.apply(output["loss"])
        else:
            output["loss"] = InfiniteLossGradient.apply(output["loss"])
        return output

    def counted_optimizer_step(*args, **kwargs):
        calls["optimizer"] += 1
        return original_optimizer_step(*args, **kwargs)

    def counted_save(*args, **kwargs):
        del args, kwargs
        calls["save"] += 1
        raise AssertionError("failed training step must not save")

    monkeypatch.setattr(model, "forward", injected_forward)
    monkeypatch.setattr(optimizer, "step", counted_optimizer_step)
    monkeypatch.setattr(train_utils, "save_checkpoint", counted_save)

    shard_path = tmp_path / "train_00000.pt"
    torch.save(
        torch.randint(0, model.config.vocab_size, (12,), dtype=torch.long),
        shard_path,
    )
    with pytest.raises((FloatingPointError, RuntimeError), match=expected_error):
        training_loop(
            model,
            optimizer,
            counted_scheduler,
            [str(shard_path)],
            fi=0,
            ptr=0,
            total_tok=10,
            bs=2,
            seq=5,
            chunk=12,
            output_dir=tmp_path / "output",
            max_steps=2,
            save_every_min=10_000,
            log_every=10_000,
            step_start=1,
            schedule_args={"warmup_steps": 0, "total_steps": 3},
        )

    assert calls == {"optimizer": 0, "scheduler": 0, "save": 0}
    assert captured["transaction"].consumed
    assert model.get_cpt_state_version() == 1
    assert_persistent_equal(model, persistent_state)
    for parameter, expected in zip(model.parameters(), parameter_snapshot):
        assert torch.equal(parameter, expected)
        assert parameter.grad is None
    assert_nested_state_equal(optimizer.state_dict(), optimizer_state)
    assert_nested_state_equal(scheduler.state_dict(), scheduler_state)
