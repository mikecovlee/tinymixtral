import copy
import math
from dataclasses import replace

import pytest
import torch

from hf.configuration_tinymixtral import TinyMixtralConfig as HFConfig
from hf.modeling_tinymixtral import TinyMixtralForCausalLM as HFModel
from hf.modeling_tinymixtral import (
    _segment_transition_mask as hf_segment_transition_mask,
)
from model.config import TinyMixtralConfig as NativeConfig
from model.cpt_router import CPTRouter, CPTSequenceState
from model.modeling import TinyMixtralForCausalLM as NativeModel
from model.modeling import (
    _segment_transition_mask as native_segment_transition_mask,
)
from scripts.publish_hf import (
    CONFIG_FIELDS,
    _assert_canonical_config_equal,
    _canonical_config,
    _require_complete_config,
    _require_consistent_cpt_state_version,
)


def _config_kwargs() -> dict:
    return {
        "vocab_size": 64,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 32,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "expert_intermediate_size": 24,
        "cpt_router_version": 1,
        "cpt_num_prototypes": 4,
        "cpt_projection_dim": 8,
        "cpt_rho_beta": 0.95,
        "cpt_beta_max": 0.45,
        "cpt_expert_temperature": 1.0,
        "cpt_state_radius": 1.0,
        "cpt_eps_z": 1e-6,
        "cpt_eps_m": 1e-6,
        "cpt_eps_init": 1e-8,
        "cpt_capacity_factor": 1.25,
        "cpt_init_seed": 19,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "attention_dropout": 0.0,
        "tie_word_embeddings": True,
        "initializer_range": 0.02,
    }


@pytest.mark.parametrize(
    ("config_class", "model_class"),
    ((NativeConfig, NativeModel), (HFConfig, HFModel)),
)
def test_model_only_uses_trusted_dense_router_controls_for_implicit_inputs(
    config_class,
    model_class,
    monkeypatch,
) -> None:
    """Only the fully implicit public layout may reach Router as ``None``."""
    model = model_class(config_class(**_config_kwargs())).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    observed_controls = []
    original_forward = CPTRouter.forward

    def observed_forward(
        self,
        hidden_states,
        route_valid_mask=None,
        reset_mask=None,
        sequence_state=None,
        cpt_sequence_ids=None,
    ):
        observed_controls.append((route_valid_mask, reset_mask))
        return original_forward(
            self,
            hidden_states,
            route_valid_mask=route_valid_mask,
            reset_mask=reset_mask,
            sequence_state=sequence_state,
            cpt_sequence_ids=cpt_sequence_ids,
        )

    monkeypatch.setattr(CPTRouter, "forward", observed_forward)

    def run_and_take_controls(**kwargs):
        observed_controls.clear()
        with torch.inference_mode():
            model(input_ids, **kwargs)
        assert len(observed_controls) == model.config.num_hidden_layers
        return tuple(observed_controls)

    implicit = run_and_take_controls()
    assert all(route is None and reset is None for route, reset in implicit)

    explicit_attention = run_and_take_controls(
        attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
    )
    assert all(
        route is not None and reset is not None
        for route, reset in explicit_attention
    )

    explicit_reset = run_and_take_controls(
        cpt_reset_mask=torch.zeros_like(input_ids, dtype=torch.bool),
    )
    assert all(
        route is not None and reset is not None
        for route, reset in explicit_reset
    )

    explicit_packing = run_and_take_controls(
        cpt_segment_ids=torch.zeros_like(input_ids),
    )
    assert all(
        route is not None and reset is not None
        for route, reset in explicit_packing
    )


@pytest.mark.parametrize(
    ("config_class", "model_class"),
    ((NativeConfig, NativeModel), (HFConfig, HFModel)),
)
def test_poisoned_fail_stop_model_rejects_direct_forward(
    config_class,
    model_class,
) -> None:
    model = model_class(config_class(**_config_kwargs())).eval()
    model._cpt_training_poison_reason = "RuntimeError: injected partial write"

    with pytest.raises(
        RuntimeError,
        match="live model instance is poisoned.*successfully published checkpoint",
    ):
        model(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))


@pytest.mark.parametrize(
    ("config_class", "model_class"),
    ((NativeConfig, NativeModel), (HFConfig, HFModel)),
)
def test_poisoned_fail_stop_model_rejects_save_pretrained_before_writing(
    tmp_path,
    config_class,
    model_class,
) -> None:
    model = model_class(config_class(**_config_kwargs())).eval()
    model._cpt_training_poison_reason = "RuntimeError: injected partial write"
    output = tmp_path / model_class.__module__.replace(".", "_")

    with pytest.raises(
        RuntimeError,
        match="live model instance is poisoned.*successfully published checkpoint",
    ):
        model.save_pretrained(output)

    assert not output.exists()


FLOAT_FIELDS = (
    "cpt_rho_beta",
    "cpt_beta_max",
    "cpt_expert_temperature",
    "cpt_state_radius",
    "cpt_eps_z",
    "cpt_eps_m",
    "cpt_eps_init",
    "cpt_capacity_factor",
    "rms_norm_eps",
    "rope_theta",
    "attention_dropout",
    "initializer_range",
)


@pytest.mark.parametrize("field", FLOAT_FIELDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_native_and_hf_reject_every_nonfinite_float(field, value) -> None:
    kwargs = _config_kwargs()
    kwargs[field] = value
    for config_class in (NativeConfig, HFConfig):
        with pytest.raises(ValueError, match=f"{field} must be finite"):
            config_class(**kwargs)


@pytest.mark.parametrize("field", FLOAT_FIELDS)
@pytest.mark.parametrize("value", [True, "0.1", object()])
def test_native_and_hf_reject_nonreal_float_types(field, value) -> None:
    kwargs = _config_kwargs()
    kwargs[field] = value
    for config_class in (NativeConfig, HFConfig):
        with pytest.raises(ValueError, match=f"{field} must be a real number"):
            config_class(**kwargs)


def test_native_and_hf_canonicalize_integer_float_fields() -> None:
    kwargs = _config_kwargs()
    kwargs["cpt_expert_temperature"] = 2
    kwargs["cpt_capacity_factor"] = 2
    native = NativeConfig(**kwargs)
    hf = HFConfig(**kwargs)
    for name in FLOAT_FIELDS:
        assert isinstance(getattr(native, name), float)
        assert isinstance(getattr(hf, name), float)
        assert getattr(native, name) == getattr(hf, name)


def test_native_and_hf_reject_unrepresentable_real_as_nonfinite() -> None:
    kwargs = _config_kwargs()
    kwargs["rope_theta"] = 10**10_000
    for config_class in (NativeConfig, HFConfig):
        with pytest.raises(ValueError, match="rope_theta must be finite"):
            config_class(**kwargs)


def test_native_and_hf_reject_rho_that_rounds_to_one_in_fp32() -> None:
    kwargs = _config_kwargs()
    kwargs["cpt_rho_beta"] = math.nextafter(1.0, 0.0)
    for config_class in (NativeConfig, HFConfig):
        with pytest.raises(ValueError, match="represented in FP32"):
            config_class(**kwargs)


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_native_and_hf_require_boolean_embedding_tying(value) -> None:
    kwargs = _config_kwargs()
    kwargs["tie_word_embeddings"] = value
    for config_class in (NativeConfig, HFConfig):
        with pytest.raises(ValueError, match="tie_word_embeddings must be"):
            config_class(**kwargs)


def test_native_config_from_dict_rejects_missing_but_filters_unknown_fields() -> None:
    payload = NativeConfig(**_config_kwargs()).to_dict()
    assert NativeConfig.from_dict(payload).to_dict() == payload

    missing = dict(payload)
    missing.pop("rope_theta")
    with pytest.raises(ValueError, match="missing.*rope_theta"):
        NativeConfig.from_dict(missing)

    unknown = dict(payload)
    unknown["rope_tetha"] = unknown["rope_theta"]
    assert NativeConfig.from_dict(unknown).to_dict() == payload


def test_retired_auxiliary_loss_coefficient_is_discarded() -> None:
    native_payload = NativeConfig(**_config_kwargs()).to_dict()
    native_payload["router_aux_loss_coef"] = 123.0
    native = NativeConfig.from_dict(native_payload)
    assert "router_aux_loss_coef" not in native.to_dict()
    assert not hasattr(native, "router_aux_loss_coef")

    first = HFConfig(**_config_kwargs(), router_aux_loss_coef=0.0)
    second = HFConfig(**_config_kwargs(), router_aux_loss_coef=123.0)
    assert "router_aux_loss_coef" not in first.to_dict()
    assert "router_aux_loss_coef" not in second.to_dict()
    assert not hasattr(first, "router_aux_loss_coef")
    assert not hasattr(second, "router_aux_loss_coef")
    assert first.to_dict() == second.to_dict()


def _paired_models():
    kwargs = _config_kwargs()
    torch.manual_seed(20260801)
    native = NativeModel(NativeConfig(**kwargs)).eval()
    hf = HFModel(HFConfig(**kwargs)).eval()
    hf.load_state_dict(copy.deepcopy(native.state_dict()), strict=True)
    return native, hf


def _state_from_moe_output(output, moe) -> CPTSequenceState:
    return moe.cpt_router._make_sequence_state(
        state_s=output[5],
        state_nu=output[6],
        initialized=output[7],
        state_version=output[8],
    )


def _assert_sequence_state_equal(left, right) -> None:
    torch.testing.assert_close(left.state_s, right.state_s, rtol=0, atol=0)
    torch.testing.assert_close(left.state_nu, right.state_nu, rtol=0, atol=0)
    assert torch.equal(left.initialized, right.initialized)
    assert torch.equal(left.state_version, right.state_version)
    assert left.layer_index == right.layer_index


def _output_parts(output, output_style):
    if output_style == "dict":
        return (
            output["loss"],
            output["logits"],
            output["cpt_sequence_states"],
        )
    return output.loss, output.logits, output.cpt_sequence_states


def test_segment_ids_isolate_attention_and_reset_router_with_native_hf_parity():
    native, hf = _paired_models()
    first_prefix = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed_prefix = torch.tensor([[31, 32, 33, 4, 5, 6]])
    segment_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.int64)
    router_only_reset = torch.tensor(
        [[True, False, False, True, False, False]],
        dtype=torch.bool,
    )

    with torch.inference_mode():
        native_first = native(
            first_prefix,
            cpt_segment_ids=segment_ids,
        )["logits"][:, 3:]
        native_changed = native(
            changed_prefix,
            cpt_segment_ids=segment_ids,
        )["logits"][:, 3:]
        hf_first = hf(
            first_prefix,
            cpt_segment_ids=segment_ids,
        ).logits[:, 3:]
        router_only_first = native(
            first_prefix,
            cpt_reset_mask=router_only_reset,
        )["logits"][:, 3:]
        router_only_changed = native(
            changed_prefix,
            cpt_reset_mask=router_only_reset,
        )["logits"][:, 3:]

    torch.testing.assert_close(
        native_first,
        native_changed,
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(hf_first, native_first)
    assert not torch.allclose(router_only_first, router_only_changed)


@pytest.mark.parametrize(
    ("segment_ids", "error_type", "message"),
    [
        (torch.zeros(1, 6, dtype=torch.float32), TypeError, "integer dtype"),
        (torch.tensor([[0, 0, 2, 2, 1, 1]]), ValueError, "nondecreasing"),
        (torch.zeros(1, 5, dtype=torch.int64), ValueError, "must have shape"),
    ],
)
def test_segment_id_validation_matches_native_and_hf(
    segment_ids,
    error_type,
    message,
) -> None:
    native, hf = _paired_models()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    for model in (native, hf):
        with pytest.raises(error_type, match=message):
            model(input_ids, cpt_segment_ids=segment_ids)


def test_segment_transitions_skip_invalid_gaps_without_losing_order() -> None:
    route_valid_mask = torch.tensor(
        [[True, False, True, False, True]],
        dtype=torch.bool,
    )
    segment_ids = torch.tensor([[0, -100, 0, 999, 1]], dtype=torch.int64)
    expected = torch.tensor(
        [[False, False, False, False, True]],
        dtype=torch.bool,
    )
    for transition_mask in (
        native_segment_transition_mask,
        hf_segment_transition_mask,
    ):
        assert torch.equal(
            transition_mask(route_valid_mask, segment_ids),
            expected,
        )
        with pytest.raises(ValueError, match="nondecreasing"):
            transition_mask(
                route_valid_mask,
                torch.tensor([[1, -100, 0, 999, 2]], dtype=torch.int64),
            )


def test_hf_generate_matches_manual_full_prefix_greedy_decode() -> None:
    _, model = _paired_models()
    input_ids = torch.tensor([[1, 2, 3]])
    observed_lengths = []

    def capture_length(_module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None:
            ids = args[0]
        observed_lengths.append(ids.shape[1])

    handle = model.register_forward_pre_hook(capture_length, with_kwargs=True)
    try:
        with torch.inference_mode():
            generated = model.generate(
                input_ids,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=0,
                eos_token_id=None,
            )
    finally:
        handle.remove()

    manual = input_ids.clone()
    with torch.inference_mode():
        for _ in range(2):
            next_token = model(manual).logits[:, -1].argmax(dim=-1, keepdim=True)
            manual = torch.cat((manual, next_token), dim=1)

    assert observed_lengths == [3, 4]
    assert torch.equal(generated, manual)


@pytest.mark.parametrize("framework", ["native", "hf"])
def test_sparse_moe_continuation_matches_one_pass_and_explicit_reset(
    framework,
) -> None:
    native, hf = _paired_models()
    moe = native.layers[0].moe if framework == "native" else hf.layers[0].moe
    generator = torch.Generator(device="cpu").manual_seed(20260803)
    hidden_states = torch.randn(2, 7, 16, generator=generator)

    with torch.inference_mode():
        full = moe(hidden_states, return_sequence_state=True)
        first = moe(hidden_states[:, :3], return_sequence_state=True)
        first_state = _state_from_moe_output(first, moe)
        first_state_snapshot = replace(
            first_state,
            state_s=first_state.state_s.clone(),
            state_nu=first_state.state_nu.clone(),
            initialized=first_state.initialized.clone(),
            state_version=first_state.state_version.clone(),
        )
        continued = moe(
            hidden_states[:, 3:],
            sequence_state=first_state,
            return_sequence_state=True,
        )
        fresh = moe(hidden_states[:, 3:], return_sequence_state=True)
        forced_reset = moe(
            hidden_states[:, 3:],
            reset_mask=torch.tensor(
                [[True, False, False, False]] * 2,
                dtype=torch.bool,
            ),
            sequence_state=first_state,
            return_sequence_state=True,
        )

    torch.testing.assert_close(
        full[0],
        torch.cat((first[0], continued[0]), dim=1),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_sequence_state_equal(
        _state_from_moe_output(full, moe),
        _state_from_moe_output(continued, moe),
    )
    _assert_sequence_state_equal(first_state, first_state_snapshot)
    torch.testing.assert_close(forced_reset[0], fresh[0], rtol=0, atol=0)
    _assert_sequence_state_equal(
        _state_from_moe_output(forced_reset, moe),
        _state_from_moe_output(fresh, moe),
    )
    assert not torch.equal(continued[6], fresh[6])


def test_model_continuation_native_hf_parity_schema_and_all_padding() -> None:
    native, hf = _paired_models()
    prefix = torch.tensor([[1, 2, 3], [4, 5, 6]])
    suffix = torch.tensor([[7, 8, 9], [10, 11, 12]])

    with torch.inference_mode():
        native_prefix = native(prefix)
        hf_prefix = hf(prefix)
        native_continued = native(
            suffix,
            cpt_sequence_states=native_prefix["cpt_sequence_states"],
        )
        hf_continued = hf(
            suffix,
            cpt_sequence_states=hf_prefix.cpt_sequence_states,
        )

    torch.testing.assert_close(
        native_continued["logits"],
        hf_continued.logits,
        rtol=0,
        atol=0,
    )
    assert len(native_continued["cpt_sequence_states"]) == len(native.layers)
    for layer_index, (native_state, hf_state) in enumerate(
        zip(
            native_continued["cpt_sequence_states"],
            hf_continued.cpt_sequence_states,
        )
    ):
        _assert_sequence_state_equal(native_state, hf_state)
        router = native.layers[layer_index].moe.cpt_router
        assert native_state.state_s.shape == (
            2,
            router.projection_dim,
            router.num_prototypes,
        )
        assert native_state.state_nu.shape == (2, router.num_prototypes)
        assert native_state.initialized.shape == (2,)
        assert native_state.state_version.shape == ()
        assert native_state.state_s.dtype == torch.float32
        assert native_state.state_nu.dtype == torch.float32
        assert native_state.initialized.dtype == torch.bool
        assert native_state.state_version.dtype == torch.int64
        assert not native_state.state_s.requires_grad
        assert not native_state.state_nu.requires_grad

    all_padding = torch.zeros_like(suffix, dtype=torch.bool)
    with torch.inference_mode():
        native_padding = native(
            suffix,
            attention_mask=all_padding,
            cpt_sequence_states=native_prefix["cpt_sequence_states"],
        )
        hf_padding = hf(
            suffix,
            attention_mask=all_padding,
            cpt_sequence_states=hf_prefix.cpt_sequence_states,
        )
    for state_in, state_out in zip(
        native_prefix["cpt_sequence_states"],
        native_padding["cpt_sequence_states"],
    ):
        _assert_sequence_state_equal(state_in, state_out)
    for state_in, state_out in zip(
        hf_prefix.cpt_sequence_states,
        hf_padding.cpt_sequence_states,
    ):
        _assert_sequence_state_equal(state_in, state_out)


@pytest.mark.parametrize(
    ("config_class", "model_class", "output_style"),
    [
        (NativeConfig, NativeModel, "dict"),
        (HFConfig, HFModel, "model_output"),
    ],
)
def test_cross_microbatch_new_segment_requires_explicit_boundary_reset(
    config_class,
    model_class,
    output_style,
) -> None:
    torch.manual_seed(20260804)
    model = model_class(config_class(**_config_kwargs())).eval()
    prefix = torch.tensor([[1, 2, 3]])
    suffix = torch.tensor([[4, 5, 6]])
    segment_ids = torch.ones_like(suffix)
    boundary_reset = torch.tensor([[True, False, False]])

    with torch.inference_mode():
        prefix_output = model(
            prefix,
            cpt_segment_ids=torch.zeros_like(prefix),
        )
        if output_style == "dict":
            prefix_states = prefix_output["cpt_sequence_states"]
        else:
            prefix_states = prefix_output.cpt_sequence_states
        continued = model(
            suffix,
            cpt_segment_ids=segment_ids,
            cpt_sequence_states=prefix_states,
        )
        reset = model(
            suffix,
            cpt_segment_ids=segment_ids,
            cpt_reset_mask=boundary_reset,
            cpt_sequence_states=prefix_states,
        )
        fresh = model(suffix, cpt_segment_ids=segment_ids)

    _, reset_logits, reset_states = _output_parts(reset, output_style)
    _, fresh_logits, fresh_states = _output_parts(fresh, output_style)
    _, _, continued_states = _output_parts(continued, output_style)
    torch.testing.assert_close(reset_logits, fresh_logits, rtol=0, atol=0)
    for reset_state, fresh_state in zip(reset_states, fresh_states):
        _assert_sequence_state_equal(reset_state, fresh_state)
    assert any(
        not torch.equal(continued_state.state_nu, fresh_state.state_nu)
        for continued_state, fresh_state in zip(continued_states, fresh_states)
    )


def test_model_rejects_wrong_continuation_layer_count_and_keeps_it_transient():
    native, hf = _paired_models()
    input_ids = torch.tensor([[1, 2, 3]])
    native_keys = tuple(native.state_dict())
    hf_keys = tuple(hf.state_dict())
    with torch.inference_mode():
        native_output = native(input_ids)
        hf_output = hf(input_ids)
    assert tuple(native.state_dict()) == native_keys
    assert tuple(hf.state_dict()) == hf_keys
    assert all("sequence_state" not in key for key in native_keys)
    assert all("sequence_state" not in key for key in hf_keys)

    for model, states in (
        (native, native_output["cpt_sequence_states"]),
        (hf, hf_output.cpt_sequence_states),
    ):
        with pytest.raises(ValueError, match="cpt_sequence_states"):
            model(input_ids, cpt_sequence_states=states[:-1])


def test_model_rejects_swapped_layers_and_cross_model_continuation_states():
    native, hf = _paired_models()
    input_ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        native_states = native(input_ids)["cpt_sequence_states"]
        hf_states = hf(input_ids).cpt_sequence_states

    for model, states in ((native, native_states), (hf, hf_states)):
        with pytest.raises(ValueError, match="layer identity"):
            model(
                input_ids,
                cpt_sequence_states=tuple(reversed(states)),
            )

    with pytest.raises(ValueError, match="different CPT router instance"):
        native(input_ids, cpt_sequence_states=hf_states)
    with pytest.raises(ValueError, match="different CPT router instance"):
        hf(input_ids, cpt_sequence_states=native_states)


@pytest.mark.parametrize(
    ("config_class", "model_class"),
    [
        (NativeConfig, NativeModel),
        (HFConfig, HFModel),
    ],
)
def test_native_hf_and_publish_guard_reject_mixed_layer_versions(
    config_class,
    model_class,
) -> None:
    source = model_class(config_class(**_config_kwargs()))
    state = copy.deepcopy(source.state_dict())
    state["layers.1.moe.cpt_router.state_version"].fill_(1)
    target = model_class(config_class(**_config_kwargs()))
    with pytest.raises(RuntimeError, match="CPT layer versions disagree"):
        target.load_state_dict(state, strict=True)

    # The publish helper is a second explicit gate at the source/reloaded
    # round-trip boundary, even if a model is mutated after loading.
    source.layers[1].moe.cpt_router.state_version.fill_(1)
    with pytest.raises(
        RuntimeError,
        match="publish test has inconsistent CPT layer state versions",
    ):
        _require_consistent_cpt_state_version(source, "publish test")


@pytest.mark.parametrize(
    ("config_class", "model_class", "output_style"),
    [
        (NativeConfig, NativeModel, "dict"),
        (HFConfig, HFModel, "model_output"),
    ],
)
def test_segment_attention_matches_with_activation_checkpointing(
    config_class,
    model_class,
    output_style,
) -> None:
    torch.manual_seed(20260802)
    regular = model_class(config_class(**_config_kwargs())).train()
    checkpointed = model_class(config_class(**_config_kwargs())).train()
    checkpointed.load_state_dict(copy.deepcopy(regular.state_dict()), strict=True)
    checkpointed.gradient_checkpointing_enable()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = torch.tensor([[2, 3, 4, 5, 6, 7]])
    segment_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])

    regular_output = regular(
        input_ids,
        labels=labels,
        cpt_segment_ids=segment_ids,
    )
    checkpointed_output = checkpointed(
        input_ids,
        labels=labels,
        cpt_segment_ids=segment_ids,
    )
    if output_style == "dict":
        regular_loss = regular_output["loss"]
        checkpointed_loss = checkpointed_output["loss"]
        regular_logits = regular_output["logits"]
        checkpointed_logits = checkpointed_output["logits"]
    else:
        regular_loss = regular_output.loss
        checkpointed_loss = checkpointed_output.loss
        regular_logits = regular_output.logits
        checkpointed_logits = checkpointed_output.logits

    regular_loss.backward()
    checkpointed_loss.backward()
    torch.testing.assert_close(regular_logits, checkpointed_logits)
    torch.testing.assert_close(regular_loss, checkpointed_loss)
    regular_parameters = dict(regular.named_parameters())
    checkpointed_parameters = dict(checkpointed.named_parameters())
    assert regular_parameters.keys() == checkpointed_parameters.keys()
    for name, parameter in regular_parameters.items():
        torch.testing.assert_close(
            parameter.grad,
            checkpointed_parameters[name].grad,
            msg=lambda message: f"{name}: {message}",
        )


@pytest.mark.parametrize(
    ("config_class", "model_class", "output_style"),
    [
        (NativeConfig, NativeModel, "dict"),
        (HFConfig, HFModel, "model_output"),
    ],
)
def test_continuation_state_matches_with_activation_checkpointing(
    config_class,
    model_class,
    output_style,
) -> None:
    torch.manual_seed(20260805)
    regular = model_class(config_class(**_config_kwargs())).eval()
    checkpointed = model_class(config_class(**_config_kwargs())).eval()
    checkpointed.load_state_dict(copy.deepcopy(regular.state_dict()), strict=True)
    prefix = torch.tensor([[1, 2, 3], [4, 5, 6]])
    suffix = torch.tensor([[7, 8, 9], [10, 11, 12]])
    labels = torch.tensor([[8, 9, 10], [11, 12, 13]])

    with torch.inference_mode():
        prefix_output = regular(prefix)
        checkpointed_prefix_output = checkpointed(prefix)
    if output_style == "dict":
        continuation_states = prefix_output["cpt_sequence_states"]
        checkpointed_continuation_states = checkpointed_prefix_output[
            "cpt_sequence_states"
        ]
    else:
        continuation_states = prefix_output.cpt_sequence_states
        checkpointed_continuation_states = (
            checkpointed_prefix_output.cpt_sequence_states
        )
    for regular_state, checkpointed_state in zip(
        continuation_states,
        checkpointed_continuation_states,
    ):
        _assert_sequence_state_equal(regular_state, checkpointed_state)
    state_snapshots = tuple(
        replace(
            state,
            state_s=state.state_s.clone(),
            state_nu=state.state_nu.clone(),
            initialized=state.initialized.clone(),
            state_version=state.state_version.clone(),
        )
        for state in continuation_states
    )
    checkpointed_state_snapshots = tuple(
        replace(
            state,
            state_s=state.state_s.clone(),
            state_nu=state.state_nu.clone(),
            initialized=state.initialized.clone(),
            state_version=state.state_version.clone(),
        )
        for state in checkpointed_continuation_states
    )

    regular.train()
    checkpointed.train()
    checkpointed.gradient_checkpointing_enable()
    regular_output = regular(
        suffix,
        labels=labels,
        cpt_sequence_states=continuation_states,
    )
    checkpointed_output = checkpointed(
        suffix,
        labels=labels,
        cpt_sequence_states=checkpointed_continuation_states,
    )
    regular_loss, regular_logits, regular_states = _output_parts(
        regular_output,
        output_style,
    )
    checkpointed_loss, checkpointed_logits, checkpointed_states = (
        _output_parts(checkpointed_output, output_style)
    )
    torch.testing.assert_close(regular_logits, checkpointed_logits)
    torch.testing.assert_close(regular_loss, checkpointed_loss)
    for regular_state, checkpointed_state in zip(
        regular_states,
        checkpointed_states,
    ):
        _assert_sequence_state_equal(regular_state, checkpointed_state)
    for state, snapshot in zip(continuation_states, state_snapshots):
        _assert_sequence_state_equal(state, snapshot)
    for state, snapshot in zip(
        checkpointed_continuation_states,
        checkpointed_state_snapshots,
    ):
        _assert_sequence_state_equal(state, snapshot)

    regular_loss.backward()
    checkpointed_loss.backward()
    regular_parameters = dict(regular.named_parameters())
    checkpointed_parameters = dict(checkpointed.named_parameters())
    assert regular_parameters.keys() == checkpointed_parameters.keys()
    for name, parameter in regular_parameters.items():
        checkpointed_gradient = checkpointed_parameters[name].grad
        if parameter.grad is None or checkpointed_gradient is None:
            assert parameter.grad is None and checkpointed_gradient is None
            continue
        torch.testing.assert_close(
            parameter.grad,
            checkpointed_gradient,
            msg=lambda message: f"{name}: {message}",
        )


def test_publish_requires_and_preserves_all_behavior_critical_config() -> None:
    config = HFConfig(**_config_kwargs())
    payload = _canonical_config(config)
    assert set(payload) == set(CONFIG_FIELDS)
    _require_complete_config(payload)
    _assert_canonical_config_equal(config, HFConfig(**payload), "round-trip")

    incomplete = dict(payload)
    incomplete.pop("rope_theta")
    with pytest.raises(RuntimeError, match="rope_theta"):
        _require_complete_config(incomplete)

    changed = dict(payload)
    changed["rope_theta"] = math.nextafter(
        changed["rope_theta"],
        float("inf"),
    )
    with pytest.raises(RuntimeError, match="rope_theta"):
        _assert_canonical_config_equal(
            config,
            HFConfig(**changed),
            "changed config",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_native_moves_cpu_attention_mask_to_cuda_input_device() -> None:
    model, _ = _paired_models()
    model = model.cuda().eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    attention_mask = torch.ones(1, 4, dtype=torch.bool, device="cpu")
    with torch.inference_mode():
        output = model(input_ids, attention_mask=attention_mask)
    assert output["logits"].device.type == "cuda"
    assert torch.isfinite(output["logits"]).all()
