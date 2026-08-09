import inspect

import pytest
import torch
import torch.nn.functional as F

from model.config import TinyMixtralConfig
from model.cpt_router import CPTRouter
from model.modeling import GQAAttention, SparseMoE, TinyMixtralForCausalLM


def tiny_config(**overrides):
    values = {
        "vocab_size": 41,
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "max_position_embeddings": 32,
        "num_local_experts": 3,
        "num_experts_per_tok": 2,
        "expert_intermediate_size": 16,
        "cpt_projection_dim": 4,
        "cpt_state_chunk_size": 1,
        "cpt_init_seed": 17,
    }
    values.update(overrides)
    return TinyMixtralConfig(**values)


@torch.no_grad()
def column_vector_oracle(router, hidden_states, valid_mask=None):
    """Literal column-vector implementation of the authoritative equations."""
    x = hidden_states.float()
    batch, sequence, _ = x.shape
    if valid_mask is None:
        valid_mask = torch.ones(batch, sequence, dtype=torch.bool, device=x.device)
    else:
        valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)

    energy = router.energy - router.energy.mean(dim=-1, keepdim=True)
    kernel = torch.softmax(
        (energy - router.congestion_price.unsqueeze(0)) / router.expert_temperature,
        dim=-1,
    )
    result = torch.zeros(
        batch,
        sequence,
        router.num_experts,
        device=x.device,
        dtype=torch.float32,
    )
    for batch_index in range(batch):
        short_state = torch.zeros(
            router.projection_dim,
            router.num_prototypes,
            device=x.device,
        )
        responsibility = torch.zeros(router.num_prototypes, device=x.device)
        for position in range(sequence):
            if not bool(valid_mask[batch_index, position]):
                continue
            x_column = x[batch_index, position].unsqueeze(1)
            projected = router.projection @ x_column
            projected_norm = float(torch.linalg.vector_norm(projected))
            z = projected / max(projected_norm, router.eps_z)
            beta = (
                router.beta_max
                * responsibility
                / (responsibility + router.kappa_beta)
            )
            mixed = (
                router.anchors * (1.0 - beta.unsqueeze(0))
                + short_state * beta.unsqueeze(0)
            )
            prototype_norms = torch.linalg.vector_norm(mixed, dim=0, keepdim=True)
            prototypes = mixed / prototype_norms.clamp_min(router.eps_m)
            q = torch.softmax(
                (prototypes.T @ z).squeeze(1) / router.prototype_temperature,
                dim=0,
            )
            result[batch_index, position] = kernel.T @ q

            gradient = (
                (short_state - z.expand(-1, router.num_prototypes))
                * q.unsqueeze(0)
                + router.lambda_sa * (short_state - router.anchors)
            )
            candidate = short_state - router.state_step_size * gradient
            norms = torch.linalg.vector_norm(candidate, dim=0, keepdim=True)
            short_state = candidate / torch.maximum(
                torch.ones_like(norms), norms / router.state_radius
            )
            responsibility = router.rho_beta * responsibility + q
    return result


def test_config_derives_k_and_authoritative_defaults():
    config = tiny_config()
    assert config.cpt_num_prototypes == 2 * config.num_local_experts == 6
    assert config.cpt_kappa_beta == pytest.approx(
        1.0 / (6 * (1.0 - config.cpt_rho_beta)), rel=2e-6
    )
    assert config.cpt_lambda_sa == pytest.approx(1.0 / 6, rel=2e-6)
    assert config.cpt_prototype_temperature == pytest.approx(0.5)
    assert config.cpt_state_step_size == pytest.approx(
        0.1 / (1.0 + 1.0 / 6), rel=2e-6
    )
    assert config.cpt_energy_init_scale == pytest.approx(
        0.05 * config.cpt_expert_temperature
    )
    assert config.cpt_price_learning_rate == pytest.approx(
        1e-2 * config.cpt_expert_temperature
    )


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"cpt_num_prototypes": 5}, r"2 \* num_local_experts"),
        ({"cpt_router_version": 2}, "version=1"),
        ({"cpt_projection_dim": 1}, "more than two"),
        ({"cpt_rho_beta": float("nan")}, "finite"),
        ({"cpt_beta_max": 0.0}, "cpt_beta_max"),
        ({"cpt_beta_max": 0.5}, "cpt_beta_max"),
        ({"cpt_capacity_factor": 0.9}, "at least 1"),
        ({"num_experts_per_tok": 2.0}, "positive integer"),
        ({"num_experts_per_tok": True}, "positive integer"),
        ({"num_hidden_layers": 2.0}, "positive integer"),
        ({"num_hidden_layers": True}, "positive integer"),
        ({"hidden_size": 8.0}, "positive integer"),
        ({"hidden_size": True}, "positive integer"),
        ({"num_local_experts": 3.0}, "positive integer"),
        ({"num_local_experts": True}, "positive integer"),
        ({"num_experts_per_tok": 1}, "must remain 2"),
        ({"cpt_state_chunk_size": 0}, "positive integer"),
        ({"cpt_state_chunk_size": True}, "positive integer"),
        ({"cpt_state_chunk_size": 2.0}, "positive integer"),
        ({"cpt_state_corrector": 1}, "boolean"),
        ({"cpt_state_corrector": 0.0}, "boolean"),
        ({"cpt_state_corrector": "true"}, "boolean"),
    ],
)
def test_config_rejects_invalid_cpt_contract(updates, message):
    with pytest.raises(ValueError, match=message):
        tiny_config(**updates)


def test_default_state_step_size_uses_effective_lambda_sa():
    config = tiny_config(cpt_lambda_sa=0.25)
    assert config.cpt_state_step_size == pytest.approx(0.1 / 1.25, rel=2e-6)


def test_energy_initialization_accepts_valid_scale_below_eps_init():
    router = CPTRouter(
        tiny_config(cpt_energy_init_scale=1e-9),
        layer_index=0,
    )
    centered_energy = router.energy - router.energy.mean(dim=-1, keepdim=True)
    assert torch.isfinite(centered_energy).all()
    assert not torch.equal(
        centered_energy,
        centered_energy[:1].expand_as(centered_energy),
    )


def test_initialization_uses_eps_init_only_in_authoritative_energy_denominator():
    router = CPTRouter(
        tiny_config(cpt_projection_dim=2, cpt_eps_init=10.0),
        layer_index=0,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(router.anchors, dim=0),
        torch.ones(router.num_prototypes),
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.unique(router.anchors.T, dim=0).shape[0] == router.num_prototypes
    centered_energy = router.energy - router.energy.mean(dim=-1, keepdim=True)
    assert torch.isfinite(centered_energy).all()
    assert not torch.equal(
        centered_energy,
        centered_energy[:1].expand_as(centered_energy),
    )


@pytest.mark.parametrize("seed", [3, 5, 7, 11, 13])
def test_token_major_forward_matches_column_vector_oracle(seed):
    torch.manual_seed(seed)
    router = CPTRouter(tiny_config(), layer_index=0)
    hidden = torch.randn(2, 5, router.hidden_size)
    mask = torch.tensor([[1, 1, 0, 1, 0], [0, 1, 1, 1, 1]], dtype=torch.bool)
    actual = router(hidden, mask).probabilities
    expected = column_vector_oracle(router, hidden, mask)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        actual[mask].sum(dim=-1), torch.ones(int(mask.sum())), atol=2e-6, rtol=0
    )
    assert torch.equal(actual[~mask], torch.zeros_like(actual[~mask]))


def test_one_chunk_routes_tokens_in_parallel_from_the_entry_state():
    torch.manual_seed(19)
    router = CPTRouter(tiny_config(cpt_state_chunk_size=8), layer_index=0)
    hidden = torch.randn(2, 6, router.hidden_size)
    mask = torch.tensor([[1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 1]], dtype=torch.bool)

    actual = router(hidden, mask).probabilities
    expected = torch.zeros_like(actual)
    for batch_index, position in mask.nonzero().tolist():
        expected[batch_index, position] = router(
            hidden[batch_index : batch_index + 1, position : position + 1]
        ).probabilities[0, 0]

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_chunk_state_changes_only_later_chunks():
    torch.manual_seed(23)
    router = CPTRouter(tiny_config(cpt_state_chunk_size=2), layer_index=0)
    hidden = torch.randn(1, 4, router.hidden_size)
    changed = hidden.clone()
    changed[:, 0] = changed[:, 0] * -7.0 + 3.0

    original_probabilities = router(hidden).probabilities
    changed_probabilities = router(changed).probabilities

    torch.testing.assert_close(
        original_probabilities[:, 1],
        changed_probabilities[:, 1],
        atol=2e-6,
        rtol=2e-6,
    )
    assert not torch.allclose(
        original_probabilities[:, 2:],
        changed_probabilities[:, 2:],
        atol=1e-7,
        rtol=1e-7,
    )


def test_blockwise_router_preserves_probability_mass_padding_and_gradients():
    torch.manual_seed(29)
    router = CPTRouter(tiny_config(cpt_state_chunk_size=3), layer_index=0)
    hidden = torch.randn(2, 7, router.hidden_size, requires_grad=True)
    mask = torch.tensor(
        [[1, 1, 0, 1, 1, 1, 0], [0, 1, 1, 0, 1, 1, 1]],
        dtype=torch.bool,
    )

    output = router(hidden, mask)
    torch.testing.assert_close(
        output.probabilities[mask].sum(dim=-1),
        torch.ones(int(mask.sum())),
        atol=2e-6,
        rtol=0,
    )
    assert torch.equal(
        output.probabilities[~mask],
        torch.zeros_like(output.probabilities[~mask]),
    )

    output.probabilities.square().sum().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    assert torch.equal(hidden.grad[~mask], torch.zeros_like(hidden.grad[~mask]))
    for parameter in router.trainable_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_corrector_defaults_to_off():
    assert tiny_config().cpt_state_corrector is False


def test_corrector_is_exact_for_single_token_chunks():
    torch.manual_seed(41)
    standard = CPTRouter(tiny_config(cpt_state_chunk_size=1), layer_index=0)
    corrected = CPTRouter(
        tiny_config(cpt_state_chunk_size=1, cpt_state_corrector=True),
        layer_index=0,
    )
    hidden = torch.randn(2, 6, standard.hidden_size)
    mask = torch.tensor([[1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 1]], dtype=torch.bool)
    torch.testing.assert_close(
        corrected(hidden, mask).probabilities,
        standard(hidden, mask).probabilities,
        atol=0,
        rtol=0,
    )


def test_corrector_improves_fidelity_to_strict_v1():
    torch.manual_seed(43)
    reference = CPTRouter(tiny_config(cpt_state_chunk_size=1), layer_index=0)
    hidden = torch.randn(2, 32, reference.hidden_size)
    mask = torch.randint(0, 2, (2, 32), dtype=torch.bool)
    mask[:, 0] = True
    reference_probabilities = reference(hidden, mask).probabilities.detach()

    def mean_error(**overrides):
        router = CPTRouter(
            tiny_config(cpt_state_chunk_size=8, **overrides),
            layer_index=0,
        )
        probabilities = router(hidden, mask).probabilities.detach()
        valid = mask.unsqueeze(-1).expand_as(probabilities)
        return (
            (probabilities[valid] - reference_probabilities[valid])
            .abs()
            .mean()
            .item()
        )

    plain_error = mean_error()
    corrected_error = mean_error(cpt_state_corrector=True)
    assert plain_error > 0
    assert corrected_error < 0.5 * plain_error


def test_corrector_preserves_probability_mass_padding_and_gradients():
    torch.manual_seed(47)
    router = CPTRouter(
        tiny_config(cpt_state_chunk_size=3, cpt_state_corrector=True),
        layer_index=0,
    )
    hidden = torch.randn(2, 7, router.hidden_size, requires_grad=True)
    mask = torch.tensor(
        [[1, 1, 0, 1, 1, 1, 0], [0, 1, 1, 0, 1, 1, 1]],
        dtype=torch.bool,
    )

    output = router(hidden, mask)
    torch.testing.assert_close(
        output.probabilities[mask].sum(dim=-1),
        torch.ones(int(mask.sum())),
        atol=2e-6,
        rtol=0,
    )
    assert torch.equal(
        output.probabilities[~mask],
        torch.zeros_like(output.probabilities[~mask]),
    )

    output.probabilities.square().sum().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    assert torch.equal(hidden.grad[~mask], torch.zeros_like(hidden.grad[~mask]))
    for parameter in router.trainable_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("scale", [0.0, 1e-12, 1.0, 1e6])
def test_router_is_finite_for_zero_near_zero_and_extreme_inputs(scale):
    torch.manual_seed(9)
    router = CPTRouter(tiny_config(), layer_index=0)
    hidden = torch.randn(2, 4, router.hidden_size) * scale
    output = router(hidden)
    assert torch.isfinite(output.probabilities).all()
    assert torch.isfinite(output.proposal.load_sum).all()
    torch.testing.assert_close(
        output.probabilities.sum(dim=-1),
        torch.ones(2, 4),
        atol=2e-6,
        rtol=0,
    )


def test_strict_v1_projection_has_no_projection_softmax():
    router = CPTRouter(tiny_config(), layer_index=0)
    hidden = torch.tensor([[[2.0, -1.0, 0.5, 3.0, -2.0, 1.5, 0.25, -0.75]]])
    actual = router(hidden).probabilities[0, 0]

    projected = router.projection @ hidden[0, 0].float().unsqueeze(1)
    wrong_z = torch.softmax(projected.squeeze(1), dim=0)
    wrong_z = wrong_z / torch.linalg.vector_norm(wrong_z).clamp_min(router.eps_z)
    prototypes = router.anchors / torch.linalg.vector_norm(
        router.anchors, dim=0, keepdim=True
    ).clamp_min(router.eps_m)
    wrong_q = torch.softmax(
        prototypes.T @ wrong_z / router.prototype_temperature, dim=0
    )
    wrong_pi = router.expert_kernel().T @ wrong_q
    assert not torch.allclose(actual, wrong_pi, atol=1e-5, rtol=1e-5)


def test_router_emits_final_probabilities_with_one_micro_batch_gemm():
    source = inspect.getsource(CPTRouter._forward_impl)
    assert "flat_prototype_probabilities.index_select" in source
    assert "valid_probabilities = prototype_probabilities @ kernel" in source
    assert ".view(batch_size, sequence_length, self.num_experts)" in source
    assert source.count("@ kernel") == 1


def test_padding_is_not_routed_and_does_not_advance_sequence_state():
    torch.manual_seed(11)
    router = CPTRouter(tiny_config(), layer_index=0)
    valid = torch.randn(1, 3, router.hidden_size)
    pads_left = torch.randn(1, 2, router.hidden_size) * 100
    pads_right = torch.randn(1, 2, router.hidden_size) * 100
    padded = torch.cat((pads_left, valid, pads_right), dim=1)
    mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0]], dtype=torch.bool)

    plain_output = router(valid)
    padded_output = router(padded, mask)
    torch.testing.assert_close(
        padded_output.probabilities[:, 2:5],
        plain_output.probabilities,
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.equal(
        padded_output.probabilities[~mask],
        torch.zeros_like(padded_output.probabilities[~mask]),
    )
    torch.testing.assert_close(
        padded_output.proposal.load_sum,
        plain_output.proposal.load_sum,
        atol=2e-6,
        rtol=2e-6,
    )
    assert int(padded_output.proposal.token_count) == 3


def test_padding_nan_hidden_has_zero_gradient_and_detached_proposal():
    torch.manual_seed(13)
    router = CPTRouter(tiny_config(), layer_index=0)
    mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    hidden = torch.randn(1, 4, router.hidden_size)
    hidden[~mask] = float("nan")
    hidden.requires_grad_()

    output = router(hidden, mask)
    assert torch.isfinite(output.probabilities).all()
    assert torch.equal(
        output.probabilities[~mask],
        torch.zeros_like(output.probabilities[~mask]),
    )
    for value in (
        output.proposal.load_sum,
        output.proposal.token_count,
        output.proposal.state_version,
    ):
        assert not value.requires_grad

    expert_weights = torch.tensor([0.3, -0.8, 1.1])
    (output.probabilities[mask] * expert_weights).sum().backward()
    assert torch.equal(hidden.grad[~mask], torch.zeros_like(hidden.grad[~mask]))
    assert torch.isfinite(hidden.grad[mask]).all()
    assert float(hidden.grad[mask].abs().sum()) > 0


def test_later_token_router_loss_has_no_cross_token_bptt():
    torch.manual_seed(17)
    router = CPTRouter(tiny_config(), layer_index=0)
    hidden = torch.randn(1, 4, router.hidden_size, requires_grad=True)
    output = router(hidden)
    expert_weights = torch.tensor([0.25, -0.75, 1.25])
    (output.probabilities[:, -1] * expert_weights).sum().backward()

    assert torch.equal(hidden.grad[:, :-1], torch.zeros_like(hidden.grad[:, :-1]))
    assert torch.isfinite(hidden.grad[:, -1]).all()
    assert float(hidden.grad[:, -1].abs().sum()) > 0


def test_batch_rows_are_independent_and_router_is_causal():
    torch.manual_seed(19)
    router = CPTRouter(tiny_config(), layer_index=1)
    hidden = torch.randn(2, 5, router.hidden_size)
    batched = router(hidden).probabilities
    separate = torch.cat(
        [
            router(hidden[index : index + 1]).probabilities
            for index in range(hidden.shape[0])
        ]
    )
    torch.testing.assert_close(batched, separate, atol=2e-6, rtol=2e-6)

    changed = hidden.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 50
    changed_output = router(changed).probabilities
    torch.testing.assert_close(batched[:, :3], changed_output[:, :3], atol=0, rtol=0)


def test_main_task_gradients_reach_hidden_projection_anchors_and_energy():
    torch.manual_seed(23)
    router = CPTRouter(tiny_config(), layer_index=0)
    hidden = torch.randn(2, 4, router.hidden_size, requires_grad=True)
    probabilities = router(hidden).probabilities
    expert_weights = torch.tensor([0.2, -0.7, 1.3])
    loss = (probabilities * expert_weights).sum()
    loss.backward()

    for tensor in (hidden, router.projection, router.anchors, router.energy):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert float(tensor.grad.abs().sum()) > 0
    assert router.congestion_price.grad is None
    assert "congestion_price" not in dict(router.named_parameters())


def test_sparse_moe_top2_consumes_cpt_probabilities_directly(monkeypatch):
    torch.manual_seed(29)
    moe = SparseMoE(tiny_config(), layer_index=0)
    hidden = torch.randn(2, 3, moe.hidden_size)
    expected = moe.cpt_router(hidden).probabilities.reshape(-1, moe.num_experts)
    captured = []
    original_topk = torch.topk

    def recording_topk(input_tensor, *args, **kwargs):
        captured.append(input_tensor.detach().float())
        return original_topk(input_tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "topk", recording_topk)
    moe(hidden)
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], expected, atol=0, rtol=0)


def test_native_model_keeps_upstream_interfaces_and_zero_aux_loss():
    torch.manual_seed(31)
    model = TinyMixtralForCausalLM(tiny_config())
    input_ids = torch.randint(0, model.config.vocab_size, (2, 6))
    labels = torch.randint(0, model.config.vocab_size, (2, 6))
    output = model(input_ids, labels=labels)
    ce = F.cross_entropy(
        output["logits"].reshape(-1, model.config.vocab_size), labels.reshape(-1)
    )
    assert output["aux_loss"].dtype == torch.float32
    assert not output["aux_loss"].requires_grad
    assert output["aux_loss"].item() == 0.0
    torch.testing.assert_close(output["loss"], ce, atol=0, rtol=0)

    for coefficient in (12345.0, float("inf"), float("nan")):
        model.config.router_aux_loss_coef = coefficient
        output_with_arbitrary_coef = model(input_ids, labels=labels)
        torch.testing.assert_close(
            output_with_arbitrary_coef["loss"], ce, atol=0, rtol=0
        )
    assert all(not hasattr(layer.moe, "router") for layer in model.layers)
    assert all(hasattr(layer.moe, "cpt_router") for layer in model.layers)

    assert tuple(inspect.signature(GQAAttention.forward).parameters) == (
        "self", "hidden_states", "attention_mask", "position_ids"
    )
    assert tuple(inspect.signature(TinyMixtralForCausalLM.forward).parameters) == (
        "self", "input_ids", "attention_mask", "labels", "return_dict"
    )


def test_activation_checkpointing_matches_plain_forward_and_gradients():
    torch.manual_seed(37)
    plain = TinyMixtralForCausalLM(tiny_config())
    checkpointed = TinyMixtralForCausalLM(tiny_config())
    checkpointed.load_state_dict(plain.state_dict(), strict=True)
    checkpointed.gradient_checkpointing_enable()
    plain.train()
    checkpointed.train()
    input_ids = torch.randint(0, plain.config.vocab_size, (2, 5))
    labels = torch.randint(0, plain.config.vocab_size, (2, 5))

    plain_output = plain(input_ids, labels=labels)
    checkpoint_output = checkpointed(input_ids, labels=labels)
    torch.testing.assert_close(
        checkpoint_output["logits"], plain_output["logits"], atol=0, rtol=0
    )
    plain_output["loss"].backward()
    checkpoint_output["loss"].backward()
    for plain_parameter, checkpoint_parameter in zip(
        plain.cpt_trainable_parameters(), checkpointed.cpt_trainable_parameters()
    ):
        torch.testing.assert_close(
            checkpoint_parameter.grad, plain_parameter.grad, atol=2e-6, rtol=2e-5
        )
    assert plain.get_cpt_state_version() == checkpointed.get_cpt_state_version() == 0


def test_cpt_precision_island_survives_model_bfloat16_conversion():
    model = TinyMixtralForCausalLM(tiny_config())
    snapshots = []
    for layer_index, layer in enumerate(model.layers):
        router = layer.moe.cpt_router
        with torch.no_grad():
            router.congestion_price.copy_(
                torch.linspace(
                    0.000123 + layer_index,
                    0.004321 + layer_index,
                    router.num_experts,
                )
            )
        for parameter in router.trainable_parameters():
            parameter.grad = torch.randn_like(parameter)
        snapshots.append(
            {
                name: value.detach().clone()
                for name, value in (
                    ("projection", router.projection),
                    ("projection_grad", router.projection.grad),
                    ("anchors", router.anchors),
                    ("anchors_grad", router.anchors.grad),
                    ("energy", router.energy),
                    ("energy_grad", router.energy.grad),
                    ("congestion_price", router.congestion_price),
                )
            }
        )

    model.to(torch.bfloat16)

    assert model.embed_tokens.weight.dtype == torch.bfloat16
    for layer, snapshot in zip(model.layers, snapshots):
        router = layer.moe.cpt_router
        assert router.projection.dtype == torch.float32
        assert router.anchors.dtype == torch.float32
        assert router.energy.dtype == torch.float32
        assert router.congestion_price.dtype == torch.float32
        assert router.state_version.dtype == torch.int64
        for name, value in (
            ("projection", router.projection),
            ("projection_grad", router.projection.grad),
            ("anchors", router.anchors),
            ("anchors_grad", router.anchors.grad),
            ("energy", router.energy),
            ("energy_grad", router.energy.grad),
            ("congestion_price", router.congestion_price),
        ):
            assert torch.equal(value, snapshot[name])
    model.validate_persistent_cpt_state()
