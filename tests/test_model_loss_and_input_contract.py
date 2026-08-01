import copy
import math
import re

import pytest
import torch
import torch.nn.functional as F

from hf.configuration_tinymixtral import TinyMixtralConfig as HFConfig
from hf.modeling_tinymixtral import SparseMoE as HFSparseMoE
from hf.modeling_tinymixtral import TinyMixtralForCausalLM as HFModel
from hf.modeling_tinymixtral import (
    _prepare_causal_lm_loss_tensors as hf_prepare_loss,
)
from model.config import TinyMixtralConfig as NativeConfig
from model.modeling import SparseMoE as NativeSparseMoE
from model.modeling import TinyMixtralForCausalLM as NativeModel
from model.modeling import (
    _prepare_causal_lm_loss_tensors as native_prepare_loss,
)


def _config_kwargs(**overrides) -> dict:
    values = {
        "vocab_size": 32,
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 8,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "expert_intermediate_size": 24,
        "cpt_router_version": 1,
        "cpt_router_recompute": "off",
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
        "cpt_init_seed": 41,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "attention_dropout": 0.0,
        "tie_word_embeddings": True,
        "initializer_range": 0.02,
    }
    values.update(overrides)
    return values


def _classes(kind):
    if kind == "native":
        return NativeConfig, NativeModel
    if kind == "hf":
        return HFConfig, HFModel
    raise AssertionError(kind)


def _loss(output, kind):
    return output["loss"] if kind == "native" else output.loss


def _logits(output, kind):
    return output["logits"] if kind == "native" else output.logits


def _paired_models(*, train=False):
    torch.manual_seed(20260731)
    native = NativeModel(NativeConfig(**_config_kwargs()))
    hf = HFModel(HFConfig(**_config_kwargs()))
    hf.load_state_dict(copy.deepcopy(native.state_dict()), strict=True)
    if train:
        native.train()
        hf.train()
    else:
        native.eval()
        hf.eval()
    return native, hf


def _assert_parameter_gradients_equal(left, right):
    left_parameters = dict(left.named_parameters())
    right_parameters = dict(right.named_parameters())
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


def test_hf_default_labels_follow_standard_causal_lm_shift_and_padding_mask():
    _, model = _paired_models()
    input_ids = torch.tensor(
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        dtype=torch.int64,
    )
    attention_mask = torch.tensor(
        [[0, 1, 1, 1, 1], [1, 1, 1, 0, 0]],
        dtype=torch.bool,
    )

    with torch.inference_mode():
        output = model(
            input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

    effective_labels = input_ids[:, 1:].clone()
    valid_targets = attention_mask[:, :-1] & attention_mask[:, 1:]
    effective_labels.masked_fill_(~valid_targets, -100)
    expected = F.cross_entropy(
        output.logits[:, :-1].reshape(-1, model.config.vocab_size),
        effective_labels.reshape(-1),
        ignore_index=-100,
    )
    torch.testing.assert_close(output.loss, expected)


def test_hf_explicit_pre_shift_matches_native_legacy_default_and_direct_ce():
    native, hf = _paired_models()
    full_tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    input_ids = full_tokens[:, :-1]
    labels = full_tokens[:, 1:]

    with torch.inference_mode():
        native_output = native(input_ids, labels=labels)
        hf_output = hf(
            input_ids,
            labels=labels,
            labels_are_pre_shifted=True,
        )

    torch.testing.assert_close(hf_output.logits, native_output["logits"])
    torch.testing.assert_close(hf_output.loss, native_output["loss"])
    expected = F.cross_entropy(
        hf_output.logits.reshape(-1, hf.config.vocab_size),
        labels.reshape(-1),
    )
    torch.testing.assert_close(hf_output.loss, expected)


@pytest.mark.parametrize("prepare_loss", [native_prepare_loss, hf_prepare_loss])
def test_standard_packed_loss_masks_boundaries_single_token_segments_and_padding(
    prepare_loss,
):
    logits = torch.randn(1, 6, 32)
    labels = torch.tensor([[10, 11, 12, 13, 14, 15]])
    route_valid_mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    segment_ids = torch.tensor([[0, 0, 1, 2, 2, 3]])

    loss_logits, effective_labels = prepare_loss(
        logits,
        labels,
        route_valid_mask,
        segment_ids,
        None,
        labels_are_pre_shifted=False,
    )

    torch.testing.assert_close(loss_logits, logits[:, :-1])
    assert torch.equal(
        effective_labels,
        torch.tensor([[11, -100, -100, 14, -100]]),
    )


@pytest.mark.parametrize("prepare_loss", [native_prepare_loss, hf_prepare_loss])
def test_pre_shifted_packed_loss_masks_unknown_final_target_without_label_ids(
    prepare_loss,
):
    logits = torch.randn(1, 4, 32)
    labels = torch.tensor([[10, 11, 12, 13]])
    route_valid_mask = torch.ones(1, 4, dtype=torch.bool)
    segment_ids = torch.tensor([[0, 0, 1, 1]])

    loss_logits, effective_labels = prepare_loss(
        logits,
        labels,
        route_valid_mask,
        segment_ids,
        None,
        labels_are_pre_shifted=True,
    )

    torch.testing.assert_close(loss_logits, logits)
    assert torch.equal(
        effective_labels,
        torch.tensor([[10, -100, 12, -100]]),
    )


@pytest.mark.parametrize("prepare_loss", [native_prepare_loss, hf_prepare_loss])
def test_pre_shifted_label_segment_ids_preserve_observable_final_target(
    prepare_loss,
):
    logits = torch.randn(1, 4, 32)
    labels = torch.tensor([[10, 11, 12, 13]])
    route_valid_mask = torch.ones(1, 4, dtype=torch.bool)
    segment_ids = torch.tensor([[0, 0, 1, 1]])
    label_segment_ids = torch.tensor([[0, 1, 1, 1]])

    _, effective_labels = prepare_loss(
        logits,
        labels,
        route_valid_mask,
        segment_ids,
        label_segment_ids,
        labels_are_pre_shifted=True,
    )

    assert torch.equal(
        effective_labels,
        torch.tensor([[10, -100, 12, 13]]),
    )


@pytest.mark.parametrize("kind", ["native", "hf"])
def test_packed_boundary_token_cannot_change_loss_or_gradients(kind):
    config_class, model_class = _classes(kind)
    torch.manual_seed(20260801)
    reference = model_class(config_class(**_config_kwargs())).train()
    candidate = model_class(config_class(**_config_kwargs())).train()
    candidate.load_state_dict(copy.deepcopy(reference.state_dict()), strict=True)
    reference_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed_ids = reference_ids.clone()
    changed_ids[0, 2] = 17
    labels = torch.tensor([[20, 7, 21, 22, 8, 23]])
    segment_ids = torch.tensor([[0, 0, 1, 2, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    common = {
        "attention_mask": attention_mask,
        "labels": labels,
        "cpt_segment_ids": segment_ids,
        "labels_are_pre_shifted": False,
    }

    reference_output = reference(reference_ids, **common)
    candidate_output = candidate(changed_ids, **common)
    _loss(reference_output, kind).backward()
    _loss(candidate_output, kind).backward()

    torch.testing.assert_close(
        _loss(reference_output, kind),
        _loss(candidate_output, kind),
    )
    _assert_parameter_gradients_equal(reference, candidate)


def test_native_hf_packed_standard_loss_logits_and_gradients_match():
    native, hf = _paired_models(train=True)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = torch.tensor([[20, 7, 21, 22, 8, 23]])
    segment_ids = torch.tensor([[0, 0, 1, 2, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    kwargs = {
        "attention_mask": attention_mask,
        "labels": labels,
        "cpt_segment_ids": segment_ids,
        "labels_are_pre_shifted": False,
    }

    native_output = native(input_ids, **kwargs)
    hf_output = hf(input_ids, **kwargs)
    native_output["loss"].backward()
    hf_output.loss.backward()

    torch.testing.assert_close(hf_output.logits, native_output["logits"])
    torch.testing.assert_close(hf_output.loss, native_output["loss"])
    _assert_parameter_gradients_equal(native, hf)


@pytest.mark.parametrize("kind", ["native", "hf"])
@pytest.mark.parametrize(
    ("global_recompute", "router_mode"),
    [(False, "on"), (True, "off"), (True, "global")],
)
def test_packed_standard_loss_is_recompute_invariant(
    kind,
    global_recompute,
    router_mode,
):
    config_class, model_class = _classes(kind)
    torch.manual_seed(20260802)
    base = model_class(config_class(**_config_kwargs())).train()
    state_dict = copy.deepcopy(base.state_dict())
    reference = model_class(
        config_class(**_config_kwargs(cpt_router_recompute="off"))
    ).train()
    candidate = model_class(
        config_class(**_config_kwargs(cpt_router_recompute=router_mode))
    ).train()
    reference.load_state_dict(copy.deepcopy(state_dict), strict=True)
    candidate.load_state_dict(copy.deepcopy(state_dict), strict=True)
    if global_recompute:
        candidate.gradient_checkpointing_enable()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = torch.tensor([[20, 7, 21, 22, 8, 23]])
    segment_ids = torch.tensor([[0, 0, 1, 2, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    kwargs = {
        "attention_mask": attention_mask,
        "labels": labels,
        "cpt_segment_ids": segment_ids,
        "labels_are_pre_shifted": False,
    }

    reference_output = reference(input_ids, **kwargs)
    candidate_output = candidate(input_ids, **kwargs)
    _loss(reference_output, kind).backward()
    _loss(candidate_output, kind).backward()

    torch.testing.assert_close(
        _logits(reference_output, kind),
        _logits(candidate_output, kind),
    )
    torch.testing.assert_close(
        _loss(reference_output, kind),
        _loss(candidate_output, kind),
    )
    _assert_parameter_gradients_equal(reference, candidate)


def _invalid_masks():
    return [
        pytest.param(torch.tensor([[0, 1, 2, 0]]), id="two"),
        pytest.param(torch.tensor([[0, 1, -1, 0]]), id="negative"),
        pytest.param(torch.tensor([[0.0, 1.0, float("nan"), 0.0]]), id="nan"),
        pytest.param(torch.tensor([[0.0, 1.0, float("inf"), 0.0]]), id="inf"),
        pytest.param(torch.tensor([[0, 1, 1j, 0]]), id="complex"),
        pytest.param([[0, 1, 0, 0]], id="non_tensor"),
        pytest.param(torch.ones(1, 3), id="wrong_shape"),
    ]


@pytest.mark.parametrize("kind", ["native", "hf"])
@pytest.mark.parametrize("field", ["attention_mask", "cpt_reset_mask"])
@pytest.mark.parametrize("invalid_mask", _invalid_masks())
def test_public_binary_masks_reject_nonbinary_or_ambiguous_values(
    kind,
    field,
    invalid_mask,
):
    config_class, model_class = _classes(kind)
    model = model_class(config_class(**_config_kwargs())).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with pytest.raises((TypeError, ValueError), match=field):
        model(input_ids, **{field: invalid_mask})


@pytest.mark.parametrize("kind", ["native", "hf"])
@pytest.mark.parametrize("field", ["attention_mask", "cpt_reset_mask"])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float32])
def test_public_binary_masks_accept_exact_zero_one_values(kind, field, dtype):
    config_class, model_class = _classes(kind)
    model = model_class(config_class(**_config_kwargs())).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    values = [1, 1, 0, 0] if field == "attention_mask" else [1, 0, 0, 0]
    mask = torch.tensor([values], dtype=dtype)
    with torch.inference_mode():
        output = model(input_ids, **{field: mask})
    assert torch.isfinite(_logits(output, kind)).all()


@pytest.mark.parametrize("kind", ["native", "hf"])
def test_sequence_length_guard_precedes_rope_indexing(kind):
    config_class, model_class = _classes(kind)
    model = model_class(config_class(**_config_kwargs())).eval()
    input_ids = torch.ones(1, 9, dtype=torch.int64)
    with pytest.raises(ValueError, match="exceeds max_position_embeddings"):
        model(input_ids)


@pytest.mark.parametrize("kind", ["native", "hf"])
def test_input_ids_require_materialized_integer_matrix_with_nonempty_batch(kind):
    config_class, model_class = _classes(kind)
    model = model_class(config_class(**_config_kwargs())).eval()
    with pytest.raises(TypeError, match="torch.int32 or torch.int64"):
        model(torch.ones(1, 4, dtype=torch.float32))
    with pytest.raises(ValueError, match="batch size must be positive"):
        model(torch.empty(0, 4, dtype=torch.int64))
    with pytest.raises(TypeError, match="materialized"):
        model(torch.empty(1, 4, dtype=torch.int64, device="meta"))


@pytest.mark.parametrize("config_class", [NativeConfig, HFConfig])
def test_config_rejects_odd_rotary_head_dimension(config_class):
    kwargs = _config_kwargs(
        hidden_size=10,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=5,
    )
    with pytest.raises(ValueError, match="head_dim must be even"):
        config_class(**kwargs)


@pytest.mark.parametrize("config_class", [NativeConfig, HFConfig])
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"cpt_init_seed": 2**100}, "unsigned 64-bit"),
        (
            {"cpt_init_seed": (1 << 64) - 1, "num_hidden_layers": 2},
            "unsigned 64-bit",
        ),
        ({"cpt_eps_z": 1e-50}, "cpt_eps_z must remain positive in FP32"),
        ({"rms_norm_eps": 1e-50}, "rms_norm_eps must remain positive in FP32"),
        (
            {"initializer_range": 1e-50},
            "initializer_range must remain positive in FP32",
        ),
        ({"rope_theta": 1e300}, "rope_theta must remain finite in FP32"),
        (
            {"attention_dropout": math.nextafter(1.0, 0.0)},
            "attention_dropout must remain in [0, 1) in FP32",
        ),
        (
            {"cpt_rho_beta": math.nextafter(1.0, 0.0)},
            "represented in FP32",
        ),
        (
            {"cpt_expert_temperature": 7e-44},
            "cpt_price_learning_rate must remain positive in FP32",
        ),
        (
            {"cpt_projection_dim": 10**100},
            "cpt_projection_temperature must remain positive in FP32",
        ),
        (
            {"cpt_num_prototypes": 10**100},
            "cpt_kappa_beta must remain positive in FP32",
        ),
    ],
)
def test_config_rejects_values_unsafe_for_actual_fp32_execution(
    config_class,
    overrides,
    message,
):
    with pytest.raises(ValueError, match=re.escape(message)):
        config_class(**_config_kwargs(**overrides))


@pytest.mark.parametrize(
    ("config_class", "model_class", "sparse_class"),
    [
        (NativeConfig, NativeModel, NativeSparseMoE),
        (HFConfig, HFModel, HFSparseMoE),
    ],
)
def test_expert_raw_parameters_honor_initializer_range_without_overwrite(
    config_class,
    model_class,
    sparse_class,
):
    small_std = 0.01
    large_std = 0.04
    torch.manual_seed(20260803)
    small = model_class(
        config_class(**_config_kwargs(initializer_range=small_std))
    )
    torch.manual_seed(20260803)
    large = model_class(
        config_class(**_config_kwargs(initializer_range=large_std))
    )

    assert isinstance(small.layers[0].moe, sparse_class)
    for name in ("gate_proj", "up_proj", "down_proj"):
        small_parameter = getattr(small.layers[0].moe, name)
        large_parameter = getattr(large.layers[0].moe, name)
        torch.testing.assert_close(
            large_parameter,
            small_parameter * (large_std / small_std),
        )
