import copy
import math

import pytest
import torch
import torch.nn.functional as F
from transformers.utils import ModelOutput

from evals.prompt_scoring import score_answers
from hf.configuration_tinymixtral import TinyMixtralConfig as HFTinyMixtralConfig
from hf.modeling_tinymixtral import TinyMixtralForCausalLM as HFTinyMixtralForCausalLM
from model.config import TinyMixtralConfig as NativeTinyMixtralConfig
from model.modeling import TinyMixtralForCausalLM as NativeTinyMixtralForCausalLM
from scripts.publish_hf import _reject_incompatible_checkpoint


def _config_kwargs() -> dict:
    return {
        "vocab_size": 64,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 16,
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
        "cpt_init_seed": 17,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "attention_dropout": 0.0,
        "tie_word_embeddings": True,
        "initializer_range": 0.02,
    }


def _models(*, train: bool = False):
    kwargs = _config_kwargs()
    torch.manual_seed(20260730)
    native = NativeTinyMixtralForCausalLM(
        NativeTinyMixtralConfig(**kwargs)
    )
    hf = HFTinyMixtralForCausalLM(HFTinyMixtralConfig(**kwargs))
    result = hf.load_state_dict(copy.deepcopy(native.state_dict()), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    if train:
        native.train()
        hf.train()
    else:
        native.eval()
        hf.eval()
    return native, hf


def _left_padded_batch():
    input_ids = torch.tensor(
        [
            [0, 0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8, 9],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    return input_ids, attention_mask


def test_hf_config_matches_native_cpt_contract_and_roundtrips() -> None:
    kwargs = _config_kwargs()
    native = NativeTinyMixtralConfig(**kwargs)
    hf = HFTinyMixtralConfig(**kwargs)

    serialized_fields = (
        "cpt_router_version",
        "cpt_num_prototypes",
        "cpt_projection_dim",
        "cpt_rho_beta",
        "cpt_beta_max",
        "cpt_expert_temperature",
        "cpt_state_radius",
        "cpt_eps_z",
        "cpt_eps_m",
        "cpt_eps_init",
        "cpt_capacity_factor",
        "cpt_init_seed",
    )
    derived_fields = (
        "cpt_rho_beta_effective",
        "cpt_kappa_beta",
        "cpt_lambda_sa",
        "cpt_projection_temperature",
        "cpt_state_step_size",
        "cpt_energy_init_scale",
        "cpt_price_learning_rate",
    )
    for name in serialized_fields + derived_fields:
        assert getattr(hf, name) == pytest.approx(getattr(native, name))

    config_dict = hf.to_dict()
    assert all(name in config_dict for name in serialized_fields)
    assert all(name not in config_dict for name in derived_fields)
    restored = HFTinyMixtralConfig.from_dict(config_dict)
    for name in serialized_fields + derived_fields:
        assert getattr(restored, name) == pytest.approx(getattr(hf, name))
    assert hf.cpt_projection_temperature == pytest.approx(1 / math.sqrt(8))
    assert not hasattr(hf, "router_jitter_noise")


def test_hf_config_discards_retired_linear_router_metadata() -> None:
    low = HFTinyMixtralConfig(
        **_config_kwargs(),
        router_aux_loss_coef=0.0,
        router_jitter_noise=0.0,
    )
    high = HFTinyMixtralConfig(
        **_config_kwargs(),
        router_aux_loss_coef=123.0,
        router_jitter_noise=456.0,
    )
    for config in (low, high):
        assert not hasattr(config, "router_aux_loss_coef")
        assert not hasattr(config, "router_jitter_noise")
        payload = config.to_dict()
        assert "router_aux_loss_coef" not in payload
        assert "router_jitter_noise" not in payload
    assert low.to_dict() == high.to_dict()


@pytest.mark.parametrize(
    "overrides",
    [
        {"cpt_projection_dim": 8.5},
        {"cpt_num_prototypes": True},
        {"cpt_router_version": True},
        {"cpt_router_version": 0},
        {"cpt_router_version": 2},
        {"cpt_init_seed": 1.5},
        {"num_local_experts": 1, "num_experts_per_tok": 1},
        {"cpt_num_prototypes": 1},
        {"cpt_projection_dim": 1, "cpt_num_prototypes": 3},
    ],
)
def test_native_and_hf_configs_reject_the_same_invalid_dimensions(overrides):
    kwargs = _config_kwargs()
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        NativeTinyMixtralConfig(**kwargs)
    with pytest.raises(ValueError):
        HFTinyMixtralConfig(**kwargs)


def test_native_hf_strict_state_and_left_padding_router_parity() -> None:
    native, hf = _models()
    input_ids, attention_mask = _left_padded_batch()

    assert set(native.state_dict()) == set(hf.state_dict())
    assert not any(key.endswith(".moe.router.weight") for key in hf.state_dict())

    with torch.inference_mode():
        native_output = native(input_ids, attention_mask=attention_mask)
        hf_output = hf(input_ids, attention_mask=attention_mask)

    assert isinstance(hf_output, ModelOutput)
    torch.testing.assert_close(hf_output.logits, native_output["logits"])
    torch.testing.assert_close(hf_output["logits"], hf_output.logits)
    assert "aux_loss" not in hf_output
    assert "aux_loss" not in native_output
    assert hf_output.cpt_transaction is None

    native_hidden = native.embed_tokens(input_ids)
    hf_hidden = hf.embed_tokens(input_ids)
    native_router = native.layers[0].moe.cpt_router(
        native_hidden,
        route_valid_mask=attention_mask,
    )
    hf_router = hf.layers[0].moe.cpt_router(
        hf_hidden,
        route_valid_mask=attention_mask,
    )
    torch.testing.assert_close(
        hf_router.q_probabilities,
        native_router.q_probabilities,
    )
    torch.testing.assert_close(
        hf_router.expert_kernel,
        native_router.expert_kernel,
    )
    torch.testing.assert_close(
        hf_router.probabilities,
        native_router.probabilities,
    )
    assert torch.equal(
        hf_router.flat_valid_indices,
        native_router.flat_valid_indices,
    )
    assert hf_router.probabilities.shape == (9, 4)
    torch.testing.assert_close(
        hf_router.probabilities.sum(dim=-1),
        torch.ones(9),
    )


def test_hf_all_padding_is_finite_and_has_zero_router_mass() -> None:
    _, hf = _models()
    input_ids = torch.zeros((2, 5), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    with torch.inference_mode():
        output = hf(input_ids, attention_mask=attention_mask)
        router_output = hf.layers[0].moe.cpt_router(
            hf.embed_tokens(input_ids),
            route_valid_mask=attention_mask,
        )

    assert torch.isfinite(output.logits).all()
    assert "aux_loss" not in output
    assert router_output.probabilities.shape == (0, 4)
    assert router_output.q_probabilities.shape == (0, 4)
    assert router_output.proposal.token_count.item() == 0
    torch.testing.assert_close(
        router_output.proposal.load_sum,
        torch.zeros(4),
    )
    assert bool(router_output.proposal.valid.item())


def test_native_hf_explicit_reset_mask_parity_and_validation() -> None:
    native, hf = _models()
    input_ids, attention_mask = _left_padded_batch()
    reset_mask = torch.zeros_like(attention_mask)
    reset_mask[0, 4] = True
    reset_mask[1, 3] = True

    with torch.inference_mode():
        native_output = native(
            input_ids,
            attention_mask=attention_mask,
            cpt_reset_mask=reset_mask,
        )
        hf_output = hf(
            input_ids,
            attention_mask=attention_mask,
            cpt_reset_mask=reset_mask,
        )
    torch.testing.assert_close(hf_output.logits, native_output["logits"])

    invalid_reset = reset_mask.clone()
    invalid_reset[0, 0] = True
    with pytest.raises(
        ValueError,
        match="cannot mark an invalid route position",
    ):
        hf(
            input_ids,
            attention_mask=attention_mask,
            cpt_reset_mask=invalid_reset,
        )


def test_native_hf_training_transaction_commit_parity() -> None:
    native, hf = _models(train=True)
    input_ids, attention_mask = _left_padded_batch()

    native_output = native(
        input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
        labels_are_pre_shifted=True,
    )
    hf_output = hf(
        input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
        labels_are_pre_shifted=True,
    )
    torch.testing.assert_close(hf_output.loss, native_output["loss"])
    expected_labels = input_ids.clone()
    expected_labels.masked_fill_(~attention_mask, -100)
    expected_task_loss = F.cross_entropy(
        hf_output.logits.reshape(-1, hf.config.vocab_size),
        expected_labels.reshape(-1),
        ignore_index=-100,
    )
    torch.testing.assert_close(hf_output.loss, expected_task_loss)
    assert "aux_loss" not in hf_output
    assert "aux_loss" not in native_output
    assert len(hf_output.cpt_transaction.proposals) == len(native.layers)

    native.commit_cpt_transaction(native_output["cpt_transaction"])
    hf.commit_cpt_transaction(hf_output.cpt_transaction)
    assert native.get_cpt_state_version() == 1
    assert hf.get_cpt_state_version() == 1
    for native_layer, hf_layer in zip(native.layers, hf.layers):
        native_router = native_layer.moe.cpt_router
        hf_router = hf_layer.moe.cpt_router
        torch.testing.assert_close(
            hf_router.anchors,
            native_router.anchors,
        )
        torch.testing.assert_close(
            hf_router.congestion_price,
            native_router.congestion_price,
        )
        assert hf_router.congestion_price.dtype == torch.float32
        assert torch.equal(hf_router.state_version, native_router.state_version)

    with pytest.raises(RuntimeError, match="already committed"):
        hf.commit_cpt_transaction(hf_output.cpt_transaction)


def test_hf_activation_checkpoint_and_fp32_price_buffer() -> None:
    _, regular = _models(train=True)
    checkpointed = HFTinyMixtralForCausalLM(regular.config).train()
    checkpointed.load_state_dict(copy.deepcopy(regular.state_dict()), strict=True)
    checkpointed.gradient_checkpointing_enable()
    input_ids, attention_mask = _left_padded_batch()

    regular_output = regular(
        input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )
    checkpointed_output = checkpointed(
        input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )
    regular_output.loss.backward()
    checkpointed_output.loss.backward()
    torch.testing.assert_close(regular_output.loss, checkpointed_output.loss)
    for (name, parameter), (other_name, other_parameter) in zip(
        regular.named_parameters(),
        checkpointed.named_parameters(),
    ):
        assert name == other_name
        assert parameter.grad is not None
        assert other_parameter.grad is not None
        torch.testing.assert_close(parameter.grad, other_parameter.grad)

    checkpointed.to(torch.bfloat16)
    for layer in checkpointed.layers:
        assert layer.moe.cpt_router.congestion_price.dtype == torch.float32
        assert layer.moe.cpt_router.projection.dtype == torch.float32
        assert layer.moe.cpt_router.anchors.dtype == torch.float32
        assert layer.moe.cpt_router.energy.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_hf_cuda_bf16_forward_backward_keeps_probability_state_fp32() -> None:
    assert torch.cuda.is_bf16_supported()
    kwargs = _config_kwargs()
    kwargs["num_hidden_layers"] = 1
    torch.manual_seed(20260731)
    torch.cuda.manual_seed_all(20260731)
    model = HFTinyMixtralForCausalLM(
        HFTinyMixtralConfig(**kwargs)
    ).to(device="cuda", dtype=torch.bfloat16).train()
    model.gradient_checkpointing_enable()
    input_ids, attention_mask = _left_padded_batch()
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
    assert torch.isfinite(output.loss)
    output.loss.backward()

    router = model.layers[0].moe.cpt_router
    assert router.congestion_price.dtype == torch.float32
    assert output.cpt_transaction.proposals[0].load_sum.dtype == torch.float32
    for name in ("projection", "anchors", "energy"):
        gradient = getattr(router, name).grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()


class _MinimalTokenizer:
    def __init__(self):
        self.padding_side = "right"
        self.truncation_side = "right"
        self.pad_token = None
        self.eos_token = "<eos>"
        self.pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [1 + (ord(character) % 31) for character in text]


def test_prompt_scoring_accepts_hf_model_output() -> None:
    _, hf = _models()
    tokenizer = _MinimalTokenizer()
    results = score_answers(
        hf,
        tokenizer,
        prompts=["question"],
        answer_choices=[[" yes", " no"]],
        label_ids=[[1, 0]],
        batch_size=2,
        max_length=12,
        device=torch.device("cpu"),
    )
    assert len(results) == 1
    assert set(results[0].label_scores) == {0, 1}
    assert results[0].predicted_label in {0, 1}


def test_publish_rejects_legacy_or_missing_cpt_state() -> None:
    with pytest.raises(RuntimeError, match="CPT router v1"):
        _reject_incompatible_checkpoint(
            {"layers.0.moe.router.weight": torch.zeros(4, 16)}
        )
    with pytest.raises(RuntimeError, match="no CPT router state"):
        _reject_incompatible_checkpoint({"embed_tokens.weight": torch.zeros(2, 2)})
    _reject_incompatible_checkpoint(
        {"layers.0.moe.cpt_router.energy": torch.zeros(4, 4)}
    )
