import copy
import types

import pytest
import torch

import hf.modeling_tinymixtral as hf_modeling
import model.modeling as native_modeling
from hf.configuration_tinymixtral import TinyMixtralConfig as HFConfig
from hf.modeling_tinymixtral import TinyMixtralForCausalLM as HFModel
from model.config import TinyMixtralConfig as NativeConfig
from model.modeling import TinyMixtralForCausalLM as NativeModel
from scripts.publish_hf import _normalize_source_config
from scripts.train_utils import (
    canonical_config_hash,
    execute_training_iteration,
    make_adamw,
)


CONFIG_KWARGS = {
    "vocab_size": 32,
    "hidden_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "max_position_embeddings": 16,
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
    "expert_intermediate_size": 24,
    "cpt_num_prototypes": 4,
    "cpt_projection_dim": 8,
    "cpt_init_seed": 17,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10_000.0,
    "attention_dropout": 0.0,
    "tie_word_embeddings": True,
    "initializer_range": 0.02,
}

INPUT_IDS = torch.tensor(
    [
        [1, 3, 5, 7, 9, 11],
        [2, 4, 6, 8, 10, 12],
    ],
    dtype=torch.long,
)


def _classes(kind):
    if kind == "native":
        return NativeConfig, NativeModel
    if kind == "hf":
        return HFConfig, HFModel
    raise AssertionError(kind)


def _output_parts(kind, output):
    if kind == "native":
        return (
            output["loss"],
            output["logits"],
            output["cpt_transaction"],
            output["cpt_sequence_states"],
        )
    return (
        output.loss,
        output.logits,
        output.cpt_transaction,
        output.cpt_sequence_states,
    )


def _run_model(kind, state_dict, *, global_recompute, mode, dropout=0.0):
    config_class, model_class = _classes(kind)
    kwargs = dict(CONFIG_KWARGS)
    kwargs["attention_dropout"] = dropout
    kwargs["cpt_router_recompute"] = mode
    model = model_class(config_class(**kwargs)).train()
    model.load_state_dict(copy.deepcopy(state_dict), strict=True)
    if global_recompute:
        model.gradient_checkpointing_enable()
    output = model(INPUT_IDS, labels=INPUT_IDS)
    loss, logits, transaction, sequence_states = _output_parts(kind, output)
    loss.backward()
    return model, loss, logits, transaction, sequence_states


def _assert_model_results_equal(left, right):
    (
        left_model,
        left_loss,
        left_logits,
        left_transaction,
        left_states,
    ) = left
    (
        right_model,
        right_loss,
        right_logits,
        right_transaction,
        right_states,
    ) = right

    torch.testing.assert_close(left_loss, right_loss)
    torch.testing.assert_close(left_logits, right_logits)
    assert left_transaction is not None and right_transaction is not None
    assert len(left_transaction.proposals) == len(right_transaction.proposals)
    for left_proposal, right_proposal in zip(
        left_transaction.proposals,
        right_transaction.proposals,
    ):
        assert left_proposal.layer_index == right_proposal.layer_index
        torch.testing.assert_close(left_proposal.load_sum, right_proposal.load_sum)
        assert torch.equal(left_proposal.token_count, right_proposal.token_count)
        assert torch.equal(left_proposal.state_version, right_proposal.state_version)
        assert torch.equal(left_proposal.valid, right_proposal.valid)

    assert len(left_states) == len(right_states)
    for left_state, right_state in zip(left_states, right_states):
        torch.testing.assert_close(left_state.state_s, right_state.state_s)
        torch.testing.assert_close(left_state.state_nu, right_state.state_nu)
        assert torch.equal(left_state.initialized, right_state.initialized)
        assert torch.equal(left_state.state_version, right_state.state_version)

    left_parameters = dict(left_model.named_parameters())
    right_parameters = dict(right_model.named_parameters())
    assert left_parameters.keys() == right_parameters.keys()
    for name, left_parameter in left_parameters.items():
        right_gradient = right_parameters[name].grad
        if left_parameter.grad is None or right_gradient is None:
            assert left_parameter.grad is None and right_gradient is None, name
            continue
        torch.testing.assert_close(
            left_parameter.grad,
            right_gradient,
            msg=lambda message: f"{name}: {message}",
        )
    assert left_model.get_cpt_state_version() == 0
    assert right_model.get_cpt_state_version() == 0


def _snapshot_sequence_states(states):
    return tuple(
        (
            state.state_s.clone(),
            state.state_nu.clone(),
            state.initialized.clone(),
            state.state_version.clone(),
        )
        for state in states
    )


def _assert_sequence_states_match_snapshot(states, snapshots):
    for state, snapshot in zip(states, snapshots):
        torch.testing.assert_close(state.state_s, snapshot[0])
        torch.testing.assert_close(state.state_nu, snapshot[1])
        assert torch.equal(state.initialized, snapshot[2])
        assert torch.equal(state.state_version, snapshot[3])


@pytest.mark.parametrize("config_class", [NativeConfig, HFConfig])
@pytest.mark.parametrize("mode", ["global", "on", "off"])
def test_router_recompute_config_accepts_and_roundtrips(config_class, mode):
    config = config_class(**CONFIG_KWARGS, cpt_router_recompute=mode)
    assert config.cpt_router_recompute == mode
    assert config.to_dict()["cpt_router_recompute"] == mode
    restored = config_class.from_dict(config.to_dict())
    assert restored.cpt_router_recompute == mode


@pytest.mark.parametrize("config_class", [NativeConfig, HFConfig])
@pytest.mark.parametrize(
    "mode",
    [None, True, False, 0, "", "auto", "GLOBAL", " on "],
)
def test_router_recompute_config_rejects_invalid_values(config_class, mode):
    with pytest.raises(ValueError, match="cpt_router_recompute"):
        config_class(**CONFIG_KWARGS, cpt_router_recompute=mode)


def test_old_config_defaults_only_missing_recompute_to_global():
    native_payload = NativeConfig(**CONFIG_KWARGS).to_dict()
    native_payload.pop("cpt_router_recompute")
    restored_native = NativeConfig.from_dict(native_payload)
    assert restored_native.cpt_router_recompute == "global"

    missing_architecture = dict(native_payload)
    missing_architecture.pop("hidden_size")
    with pytest.raises(ValueError, match="hidden_size"):
        NativeConfig.from_dict(missing_architecture)

    hf_payload = HFConfig(**CONFIG_KWARGS).to_dict()
    hf_payload.pop("cpt_router_recompute")
    restored_hf = HFConfig.from_dict(hf_payload)
    assert restored_hf.cpt_router_recompute == "global"

    normalized = _normalize_source_config(native_payload)
    assert "cpt_router_recompute" not in native_payload
    assert normalized["cpt_router_recompute"] == "global"
    explicit = dict(normalized, cpt_router_recompute="off")
    assert _normalize_source_config(explicit)["cpt_router_recompute"] == "off"


def test_router_recompute_does_not_change_strict_model_config_hash():
    hashes = {
        canonical_config_hash(
            NativeConfig(**CONFIG_KWARGS, cpt_router_recompute=mode)
        )
        for mode in ("global", "on", "off")
    }
    assert len(hashes) == 1
    changed = dict(CONFIG_KWARGS)
    changed["attention_dropout"] = 0.125
    assert canonical_config_hash(NativeConfig(**changed)) not in hashes


@pytest.mark.parametrize("kind", ["native", "hf"])
def test_set_router_recompute_updates_serialized_config(kind):
    config_class, model_class = _classes(kind)
    model = model_class(config_class(**CONFIG_KWARGS))
    for mode in ("on", "off", "global"):
        model.set_router_recompute(mode)
        assert model.config.cpt_router_recompute == mode
        assert model.config.to_dict()["cpt_router_recompute"] == mode
    with pytest.raises(ValueError, match="cpt_router_recompute"):
        model.set_router_recompute("auto")


@pytest.mark.parametrize("kind", ["native", "hf"])
@pytest.mark.parametrize(
    ("global_recompute", "mode", "expected"),
    [
        (False, "global", {"attention": 1, "router": 1, "experts": 1}),
        (False, "on", {"attention": 1, "router": 2, "experts": 1}),
        (False, "off", {"attention": 1, "router": 1, "experts": 1}),
        (True, "global", {"attention": 2, "router": 2, "experts": 2}),
        (True, "on", {"attention": 2, "router": 2, "experts": 2}),
        (True, "off", {"attention": 2, "router": 1, "experts": 2}),
    ],
)
def test_router_recompute_decision_matrix(
    kind,
    global_recompute,
    mode,
    expected,
):
    config_class, model_class = _classes(kind)
    torch.manual_seed(20260731)
    model = model_class(
        config_class(**CONFIG_KWARGS, cpt_router_recompute=mode)
    ).train()
    if global_recompute:
        model.gradient_checkpointing_enable()

    counts = {"attention": 0, "router": 0, "experts": 0}
    layer = model.layers[0]
    original_attention = layer.self_attn.forward
    original_router = layer.moe.cpt_router.forward
    original_experts = layer.moe._expert_forward

    def counted_attention(self, *args, **kwargs):
        counts["attention"] += 1
        return original_attention(*args, **kwargs)

    def counted_router(self, *args, **kwargs):
        counts["router"] += 1
        return original_router(*args, **kwargs)

    def counted_experts(self, *args, **kwargs):
        counts["experts"] += 1
        return original_experts(*args, **kwargs)

    layer.self_attn.forward = types.MethodType(
        counted_attention,
        layer.self_attn,
    )
    layer.moe.cpt_router.forward = types.MethodType(
        counted_router,
        layer.moe.cpt_router,
    )
    layer.moe._expert_forward = types.MethodType(
        counted_experts,
        layer.moe,
    )

    output = model(INPUT_IDS, labels=INPUT_IDS)
    loss = output["loss"] if kind == "native" else output.loss
    loss.backward()
    assert counts == expected


@pytest.mark.parametrize("kind", ["native", "hf"])
@pytest.mark.parametrize(
    ("global_recompute", "mode"),
    [
        (False, "global"),
        (False, "on"),
        (False, "off"),
        (True, "global"),
        (True, "on"),
        (True, "off"),
    ],
)
def test_all_router_recompute_paths_match_outputs_state_and_gradients(
    kind,
    global_recompute,
    mode,
):
    config_class, model_class = _classes(kind)
    torch.manual_seed(20260801)
    base = model_class(config_class(**CONFIG_KWARGS))
    state_dict = copy.deepcopy(base.state_dict())
    reference = _run_model(
        kind,
        state_dict,
        global_recompute=False,
        mode="off",
    )
    candidate = _run_model(
        kind,
        state_dict,
        global_recompute=global_recompute,
        mode=mode,
    )
    _assert_model_results_equal(reference, candidate)


@pytest.mark.parametrize("kind", ["native", "hf"])
@pytest.mark.parametrize(
    ("global_recompute", "mode"),
    [(False, "on"), (True, "off")],
)
def test_selective_router_recompute_preserves_continuation_state(
    kind,
    global_recompute,
    mode,
):
    config_class, model_class = _classes(kind)
    torch.manual_seed(20260806)
    base = model_class(config_class(**CONFIG_KWARGS))
    state_dict = copy.deepcopy(base.state_dict())
    reference_model = model_class(
        config_class(**CONFIG_KWARGS, cpt_router_recompute="off")
    )
    candidate_model = model_class(
        config_class(**CONFIG_KWARGS, cpt_router_recompute=mode)
    )
    reference_model.load_state_dict(copy.deepcopy(state_dict), strict=True)
    candidate_model.load_state_dict(copy.deepcopy(state_dict), strict=True)
    if global_recompute:
        candidate_model.gradient_checkpointing_enable()

    prefix = INPUT_IDS[:, :3]
    reference_model.eval()
    candidate_model.eval()
    with torch.no_grad():
        reference_prefix = reference_model(prefix)
        candidate_prefix = candidate_model(prefix)
    reference_prefix_states = _output_parts(
        kind,
        reference_prefix,
    )[3]
    candidate_prefix_states = _output_parts(
        kind,
        candidate_prefix,
    )[3]
    reference_snapshot = _snapshot_sequence_states(reference_prefix_states)
    candidate_snapshot = _snapshot_sequence_states(candidate_prefix_states)
    for left, right in zip(reference_snapshot, candidate_snapshot):
        for left_tensor, right_tensor in zip(left, right):
            torch.testing.assert_close(left_tensor, right_tensor)

    suffix = INPUT_IDS[:, 3:]
    reference_model.train()
    candidate_model.train()
    reference_output = reference_model(
        suffix,
        labels=suffix,
        cpt_sequence_states=reference_prefix_states,
    )
    candidate_output = candidate_model(
        suffix,
        labels=suffix,
        cpt_sequence_states=candidate_prefix_states,
    )
    reference_parts = _output_parts(kind, reference_output)
    candidate_parts = _output_parts(kind, candidate_output)
    reference_parts[0].backward()
    candidate_parts[0].backward()
    _assert_model_results_equal(
        (reference_model, *reference_parts),
        (candidate_model, *candidate_parts),
    )
    _assert_sequence_states_match_snapshot(
        reference_prefix_states,
        reference_snapshot,
    )
    _assert_sequence_states_match_snapshot(
        candidate_prefix_states,
        candidate_snapshot,
    )


@pytest.mark.parametrize("kind", ["native", "hf"])
@pytest.mark.parametrize(
    ("global_recompute", "mode"),
    [(False, "on"), (True, "global")],
)
def test_router_checkpoint_uses_private_continuation_snapshot(
    kind,
    global_recompute,
    mode,
):
    config_class, model_class = _classes(kind)
    torch.manual_seed(20260807)
    base = model_class(config_class(**CONFIG_KWARGS))
    state_dict = copy.deepcopy(base.state_dict())

    def prepare_model():
        model = model_class(
            config_class(**CONFIG_KWARGS, cpt_router_recompute=mode)
        )
        model.load_state_dict(copy.deepcopy(state_dict), strict=True)
        if global_recompute:
            model.gradient_checkpointing_enable()
        model.eval()
        with torch.no_grad():
            prefix_output = model(INPUT_IDS[:, :3])
        prefix_states = _output_parts(kind, prefix_output)[3]
        model.train()
        suffix = INPUT_IDS[:, 3:]
        suffix_output = model(
            suffix,
            labels=suffix,
            cpt_sequence_states=prefix_states,
        )
        return model, prefix_states, _output_parts(kind, suffix_output)

    reference_model, reference_states, reference_parts = prepare_model()
    candidate_model, candidate_states, candidate_parts = prepare_model()
    reference_snapshot = _snapshot_sequence_states(reference_states)
    candidate_before_mutation = _snapshot_sequence_states(candidate_states)

    # Frozen dataclass fields still contain mutable tensors. Simulate a caller
    # mutating those tensors after forward but before checkpoint recomputation.
    with torch.no_grad():
        for state in candidate_states:
            state.state_s.zero_()
            state.state_nu.zero_()
    assert any(
        not torch.equal(state.state_s, snapshot[0])
        or not torch.equal(state.state_nu, snapshot[1])
        for state, snapshot in zip(candidate_states, candidate_before_mutation)
    )

    reference_parts[0].backward()
    candidate_parts[0].backward()
    _assert_model_results_equal(
        (reference_model, *reference_parts),
        (candidate_model, *candidate_parts),
    )
    _assert_sequence_states_match_snapshot(
        reference_states,
        reference_snapshot,
    )


@pytest.mark.parametrize(
    ("global_recompute", "mode"),
    [(False, "on"), (True, "global"), (True, "off")],
)
def test_router_recompute_strict_step_commits_exactly_once(
    global_recompute,
    mode,
):
    torch.manual_seed(20260804)
    model = NativeModel(
        NativeConfig(**CONFIG_KWARGS, cpt_router_recompute=mode)
    ).train()
    if global_recompute:
        model.gradient_checkpointing_enable()
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)

    def forward_fn():
        return model(INPUT_IDS, labels=INPUT_IDS)

    output, grad_norm = execute_training_iteration(
        model,
        optimizer,
        scheduler=None,
        forward_fn=forward_fn,
        max_grad_norm=1.0,
    )
    transaction = output["cpt_transaction"]
    assert torch.isfinite(grad_norm)
    assert transaction.closed
    assert not transaction.aborted
    assert len(transaction.microbatch_ids) == 1
    assert model.get_cpt_state_version() == 1


@pytest.mark.parametrize("kind", ["native", "hf"])
@pytest.mark.parametrize(
    ("global_recompute", "mode"),
    [
        (False, "on"),
        (True, "global"),
        (True, "off"),
    ],
)
def test_router_recompute_preserves_attention_dropout_rng(
    kind,
    global_recompute,
    mode,
):
    config_class, model_class = _classes(kind)
    kwargs = dict(CONFIG_KWARGS)
    kwargs["attention_dropout"] = 0.25
    torch.manual_seed(20260802)
    base = model_class(config_class(**kwargs))
    state_dict = copy.deepcopy(base.state_dict())

    torch.manual_seed(20260803)
    reference = _run_model(
        kind,
        state_dict,
        global_recompute=False,
        mode="off",
        dropout=0.25,
    )
    reference_next_random = torch.rand(16)

    torch.manual_seed(20260803)
    candidate = _run_model(
        kind,
        state_dict,
        global_recompute=global_recompute,
        mode=mode,
        dropout=0.25,
    )
    candidate_next_random = torch.rand(16)

    _assert_model_results_equal(reference, candidate)
    assert torch.equal(reference_next_random, candidate_next_random)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("kind", ["native", "hf"])
@pytest.mark.parametrize(
    ("global_recompute", "mode"),
    [(False, "on"), (True, "global"), (True, "off")],
)
def test_router_recompute_cuda_bf16_paths_are_finite_and_equivalent(
    kind,
    global_recompute,
    mode,
):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")
    config_class, model_class = _classes(kind)
    cuda_kwargs = dict(CONFIG_KWARGS)
    cuda_kwargs["attention_dropout"] = 0.30
    torch.manual_seed(20260805)
    base = model_class(config_class(**cuda_kwargs))
    state_dict = copy.deepcopy(base.state_dict())

    def run(global_flag, policy):
        model = model_class(
            config_class(
                **cuda_kwargs,
                cpt_router_recompute=policy,
            )
        ).to(device="cuda", dtype=torch.bfloat16).train()
        model.load_state_dict(copy.deepcopy(state_dict), strict=True)
        if global_flag:
            model.gradient_checkpointing_enable()
        input_ids = INPUT_IDS.cuda()
        torch.manual_seed(20260808)
        torch.cuda.manual_seed_all(20260808)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids, labels=input_ids)
        loss, logits, transaction, states = _output_parts(kind, output)
        loss.backward()
        gradients = {
            name: parameter.grad.detach().float().cpu()
            if parameter.grad is not None
            else None
            for name, parameter in model.named_parameters()
        }
        return (
            loss.detach().float().cpu(),
            logits.detach().float().cpu(),
            transaction,
            tuple(
                (
                    state.state_s.cpu(),
                    state.state_nu.cpu(),
                    state.initialized.cpu(),
                    state.state_version.cpu(),
                )
                for state in states
            ),
            gradients,
            model.get_cpt_state_version(),
            torch.cuda.get_rng_state().cpu(),
        )

    reference = run(False, "off")
    candidate = run(global_recompute, mode)
    torch.testing.assert_close(reference[0], candidate[0])
    torch.testing.assert_close(reference[1], candidate[1])
    for left, right in zip(
        reference[2].proposals,
        candidate[2].proposals,
    ):
        torch.testing.assert_close(left.load_sum.cpu(), right.load_sum.cpu())
        assert torch.equal(left.token_count.cpu(), right.token_count.cpu())
        assert torch.equal(left.state_version.cpu(), right.state_version.cpu())
        assert torch.equal(left.valid.cpu(), right.valid.cpu())
    for left_state, right_state in zip(reference[3], candidate[3]):
        for left_tensor, right_tensor in zip(left_state, right_state):
            torch.testing.assert_close(left_tensor, right_tensor)
    assert reference[4].keys() == candidate[4].keys()
    for name, left_gradient in reference[4].items():
        right_gradient = candidate[4][name]
        if left_gradient is None or right_gradient is None:
            assert left_gradient is None and right_gradient is None, name
            continue
        assert torch.isfinite(right_gradient).all(), name
        torch.testing.assert_close(
            left_gradient,
            right_gradient,
            msg=lambda message: f"{name}: {message}",
        )
    assert reference[5] == candidate[5] == 0
    assert torch.equal(reference[6], candidate[6])


@pytest.mark.parametrize(
    ("kind", "modeling_module"),
    [("native", native_modeling), ("hf", hf_modeling)],
)
def test_router_force_on_is_inactive_without_training_autograd(
    kind,
    modeling_module,
    monkeypatch,
):
    config_class, model_class = _classes(kind)
    model = model_class(
        config_class(**CONFIG_KWARGS, cpt_router_recompute="on")
    )
    model.gradient_checkpointing_enable()

    checkpoint_calls = 0
    original_checkpoint = modeling_module.checkpoint

    def counted_checkpoint(*args, **kwargs):
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        return original_checkpoint(*args, **kwargs)

    monkeypatch.setattr(modeling_module, "checkpoint", counted_checkpoint)
    model.eval()
    with torch.no_grad():
        model(INPUT_IDS)
    model.train()
    with torch.no_grad():
        model(INPUT_IDS)
    assert checkpoint_calls == 0
