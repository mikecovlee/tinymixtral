import copy

import pytest
import torch
import torch.nn.functional as F

from model.cpt_router import CPTRouter
from model.modeling import TinyMixtralForCausalLM


class _GeneralRowPathRouter(CPTRouter):
    """Test-only Router that exercises gather/scatter with an all-valid mask."""

    @staticmethod
    def _all_routes_valid(route_valid_mask: torch.Tensor) -> bool:
        return False


def _reference_router(
    router: CPTRouter,
    hidden_states: torch.Tensor,
    route_valid_mask: torch.Tensor | None = None,
    reset_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent token-by-token implementation of the column-vector math."""
    batch_size, seq_len, _ = hidden_states.shape
    if route_valid_mask is None:
        route_valid_mask = torch.ones(
            batch_size,
            seq_len,
            dtype=torch.bool,
            device=hidden_states.device,
        )
    else:
        route_valid_mask = route_valid_mask.to(
            device=hidden_states.device,
            dtype=torch.bool,
        )
    first_valid = route_valid_mask & (
        route_valid_mask.to(torch.int64).cumsum(dim=1) == 1
    )
    if reset_mask is None:
        reset_mask = first_valid
    else:
        reset_mask = reset_mask.to(
            device=hidden_states.device,
            dtype=torch.bool,
        ) | first_valid

    routing_hidden_states = torch.where(
        route_valid_mask[:, :, None],
        hidden_states,
        torch.zeros_like(hidden_states),
    )
    projected_rows = F.linear(
        routing_hidden_states,
        router.projection,
    ).float()
    z_rows = projected_rows / projected_rows.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(router.eps_z)

    state_s = torch.zeros(
        batch_size,
        router.projection_dim,
        router.num_prototypes,
        dtype=torch.float32,
        device=hidden_states.device,
    )
    state_nu = torch.zeros(
        batch_size,
        router.num_prototypes,
        dtype=torch.float32,
        device=hidden_states.device,
    )
    anchors = router.anchors.float()
    q_by_position = []

    for position in range(seq_len):
        valid = route_valid_mask[:, position]
        reset = reset_mask[:, position]
        state_s = torch.where(
            reset[:, None, None],
            torch.zeros_like(state_s),
            state_s,
        )
        state_nu = torch.where(
            reset[:, None],
            torch.zeros_like(state_nu),
            state_nu,
        )

        beta = router.beta_max * state_nu / (
            state_nu + router.kappa_beta
        )
        mixed = (
            anchors[None, :, :] * (1.0 - beta[:, None, :])
            + state_s * beta[:, None, :]
        )
        prototypes = mixed / mixed.norm(
            dim=1,
            keepdim=True,
        ).clamp_min(router.eps_m)
        prototype_logits = torch.einsum(
            "bd,bdk->bk",
            z_rows[:, position, :],
            prototypes,
        ) / router.projection_temperature
        q_current = torch.softmax(prototype_logits, dim=-1)
        q_by_position.append(
            torch.where(
                valid[:, None],
                q_current,
                torch.zeros_like(q_current),
            )
        )

        gradient_s = (
            (state_s - z_rows[:, position, :, None])
            * q_current[:, None, :]
            + router.lambda_sa * (state_s - anchors[None, :, :])
        )
        candidate_s = state_s - router.state_step_size * gradient_s
        candidate_s = candidate_s / torch.maximum(
            torch.ones((), dtype=candidate_s.dtype, device=candidate_s.device),
            candidate_s.norm(dim=1, keepdim=True) / router.state_radius,
        )
        candidate_nu = router.rho_beta * state_nu + q_current
        state_s = torch.where(
            valid[:, None, None],
            candidate_s,
            state_s,
        )
        state_nu = torch.where(
            valid[:, None],
            candidate_nu,
            state_nu,
        )

    if q_by_position:
        q_all_rows = torch.stack(q_by_position, dim=1)
    else:
        q_all_rows = torch.zeros(
            batch_size,
            0,
            router.num_prototypes,
            dtype=torch.float32,
            device=hidden_states.device,
        )
    flat_valid_indices = route_valid_mask.reshape(-1).nonzero(
        as_tuple=False
    ).flatten()
    q_valid_rows = q_all_rows.reshape(
        -1,
        router.num_prototypes,
    ).index_select(0, flat_valid_indices)

    centered_energy = router.energy.float()
    centered_energy = centered_energy - centered_energy.mean(
        dim=-1,
        keepdim=True,
    )
    expert_kernel = torch.softmax(
        (
            centered_energy
            - router.congestion_price.detach().float()[None, :]
        ) / router.expert_temperature,
        dim=-1,
    )
    probabilities = q_valid_rows @ expert_kernel
    return q_valid_rows, expert_kernel, probabilities, flat_valid_indices


def test_cpt_initialization_invariants(tiny_config) -> None:
    router = CPTRouter(tiny_config, layer_index=0)

    projection_gram = router.projection.float() @ router.projection.float().T
    torch.testing.assert_close(
        projection_gram,
        torch.eye(tiny_config.cpt_projection_dim),
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        router.anchors.float().norm(dim=0),
        torch.ones(tiny_config.cpt_num_prototypes),
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        router.energy.float().mean(dim=0),
        torch.zeros(tiny_config.num_local_experts),
        atol=2e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        router.energy.float().mean(dim=1),
        torch.zeros(tiny_config.cpt_num_prototypes),
        atol=2e-7,
        rtol=0.0,
    )
    assert torch.isfinite(router.energy).all()
    assert router.energy.abs().amax() <= tiny_config.cpt_energy_init_scale
    assert router.energy.abs().amax() > 0

    expert_kernel = router._expert_kernel()
    torch.testing.assert_close(
        expert_kernel.sum(dim=-1),
        torch.ones(tiny_config.cpt_num_prototypes),
    )
    assert torch.pdist(expert_kernel).max() > 0
    assert router.congestion_price.dtype == torch.float32
    assert torch.equal(
        router.congestion_price,
        torch.zeros(tiny_config.num_local_experts),
    )
    assert router.state_version.dtype == torch.int64
    assert router.state_version.item() == 0
    assert router.router_algorithm_version.dtype == torch.int64
    assert router.router_algorithm_version.item() == tiny_config.cpt_router_version


def test_strict_tex_projection_is_directly_normalized_per_token(
    tiny_config,
) -> None:
    """Verify x -> P x -> per-token L2 normalization with column vectors."""
    router = CPTRouter(tiny_config, layer_index=0).eval()
    controlled_anchors = torch.zeros_like(router.anchors)
    controlled_anchors[0, 0] = 1.0
    controlled_anchors[0, 1] = -1.0
    controlled_anchors[1, 2] = 1.0
    controlled_anchors[1, 3] = -1.0
    with torch.no_grad():
        router.anchors.copy_(controlled_anchors)

    desired_projected_column = torch.tensor(
        [[3.0], [-2.0], [1.0], [-0.5], [0.25], [-0.75], [0.5], [-1.25]],
        dtype=torch.float32,
    )
    hidden_column = router.projection.float().T @ desired_projected_column
    projected_column = router.projection.float() @ hidden_column
    z_column = projected_column / projected_column.norm(p=2).clamp_min(router.eps_z)
    prototype_columns = controlled_anchors / controlled_anchors.norm(
        dim=0,
        keepdim=True,
    ).clamp_min(router.eps_m)
    expected_q_column = torch.softmax(
        prototype_columns.T @ z_column / router.projection_temperature,
        dim=0,
    )

    with torch.no_grad():
        actual = router(hidden_column.T.unsqueeze(0))

    torch.testing.assert_close(
        projected_column,
        desired_projected_column,
        rtol=0.0,
        atol=2e-6,
    )
    torch.testing.assert_close(z_column.norm(p=2), torch.tensor(1.0))
    torch.testing.assert_close(actual.q_probabilities.T, expected_q_column)

    # A projection-coordinate softmax would change the strict TeX geometry.
    softmax_column = torch.softmax(projected_column, dim=0)
    softmax_z_column = softmax_column / softmax_column.norm(p=2).clamp_min(
        router.eps_z
    )
    softmax_q_column = torch.softmax(
        prototype_columns.T @ softmax_z_column / router.projection_temperature,
        dim=0,
    )
    assert (expected_q_column - softmax_q_column).abs().max() > 1e-4

    zero_hidden_column = torch.zeros(tiny_config.hidden_size, 1)
    with torch.no_grad():
        zero_actual = router(zero_hidden_column.T.unsqueeze(0))
    torch.testing.assert_close(
        zero_actual.q_probabilities,
        torch.full_like(
            zero_actual.q_probabilities,
            1.0 / router.num_prototypes,
        ),
    )


def test_production_router_matches_independent_reference_and_column_formula(
    tiny_config,
) -> None:
    torch.manual_seed(401)
    router = CPTRouter(tiny_config, layer_index=1).eval()
    hidden_states = torch.randn(2, 5, tiny_config.hidden_size)
    route_valid_mask = torch.tensor(
        [
            [True, True, False, True, True],
            [False, True, True, True, False],
        ]
    )
    reset_mask = torch.tensor(
        [
            [True, False, False, True, False],
            [False, True, False, False, False],
        ]
    )

    with torch.no_grad():
        actual = router(
            hidden_states,
            route_valid_mask=route_valid_mask,
            reset_mask=reset_mask,
        )
        expected_q, expected_b, expected_pi_rows, expected_indices = (
            _reference_router(
                router,
                hidden_states,
                route_valid_mask,
                reset_mask,
            )
        )

    torch.testing.assert_close(actual.q_probabilities, expected_q)
    torch.testing.assert_close(actual.expert_kernel, expected_b)
    torch.testing.assert_close(actual.probabilities, expected_pi_rows)
    assert torch.equal(actual.flat_valid_indices, expected_indices)

    # Column-vector theory: Pi=B^T Q. Token-major code stores Pi^T=Q^T B.
    q_columns = actual.q_probabilities.T
    probability_columns = actual.expert_kernel.T @ q_columns
    torch.testing.assert_close(actual.probabilities, probability_columns.T)
    torch.testing.assert_close(
        actual.probabilities.sum(dim=-1),
        torch.ones(actual.probabilities.size(0)),
    )
    assert not torch.allclose(
        actual.probabilities,
        torch.softmax(actual.probabilities, dim=-1),
    )
    assert actual.proposal.valid.item()
    assert actual.proposal.token_count.item() == int(route_valid_mask.sum())
    torch.testing.assert_close(
        actual.proposal.load_sum,
        actual.probabilities.sum(dim=0),
    )


def test_dense_training_fast_path_matches_independent_column_reference(
    tiny_config,
) -> None:
    """The all-valid optimization must preserve the exact causal equations."""
    torch.manual_seed(4011)
    router = CPTRouter(tiny_config, layer_index=0).eval()
    hidden_states = torch.randn(3, 9, tiny_config.hidden_size)
    route_valid_mask = torch.ones(3, 9, dtype=torch.bool)
    reset_mask = torch.zeros_like(route_valid_mask)
    reset_mask[:, 0] = True
    reset_mask[1, 5] = True

    with torch.no_grad():
        actual = router(
            hidden_states,
            route_valid_mask=route_valid_mask,
            reset_mask=reset_mask,
        )
        expected_q, expected_b, expected_pi_rows, expected_indices = (
            _reference_router(
                router,
                hidden_states,
                route_valid_mask,
                reset_mask,
            )
        )

    torch.testing.assert_close(actual.q_probabilities, expected_q)
    torch.testing.assert_close(actual.expert_kernel, expected_b)
    torch.testing.assert_close(actual.probabilities, expected_pi_rows)
    assert torch.equal(actual.flat_valid_indices, expected_indices)
    assert actual.proposal.valid.item()


def test_trusted_dense_defaults_match_explicit_controls_and_gradients(
    tiny_config,
    monkeypatch,
) -> None:
    """The ``None`` provenance must be exact, differentiable, and sync-free."""
    torch.manual_seed(40115)
    trusted = CPTRouter(tiny_config, layer_index=0).train()
    explicit = CPTRouter(tiny_config, layer_index=0).train()
    explicit.load_state_dict(copy.deepcopy(trusted.state_dict()), strict=True)

    trusted_hidden = torch.randn(
        2,
        7,
        tiny_config.hidden_size,
        requires_grad=True,
    )
    explicit_hidden = trusted_hidden.detach().clone().requires_grad_(True)
    route_valid_mask = torch.ones(2, 7, dtype=torch.bool)
    reset_mask = torch.zeros_like(route_valid_mask)
    reset_mask[:, 0] = True

    def reject_dense_tensor_probe(_mask):
        raise AssertionError("trusted dense controls must not inspect the mask")

    def reject_dynamic_dense_output(*_args, **_kwargs):
        raise AssertionError(
            "trusted dense controls must not use nonzero/tolist synchronization"
        )

    monkeypatch.setattr(trusted, "_all_routes_valid", reject_dense_tensor_probe)
    original_nonzero = torch.Tensor.nonzero
    original_tolist = torch.Tensor.tolist
    monkeypatch.setattr(torch.Tensor, "nonzero", reject_dynamic_dense_output)
    monkeypatch.setattr(torch.Tensor, "tolist", reject_dynamic_dense_output)
    trusted_output = trusted(trusted_hidden)
    monkeypatch.setattr(torch.Tensor, "nonzero", original_nonzero)
    monkeypatch.setattr(torch.Tensor, "tolist", original_tolist)
    explicit_output = explicit(
        explicit_hidden,
        route_valid_mask=route_valid_mask,
        reset_mask=reset_mask,
    )

    torch.testing.assert_close(
        trusted_output.q_probabilities,
        explicit_output.q_probabilities,
    )
    torch.testing.assert_close(
        trusted_output.probabilities,
        explicit_output.probabilities,
    )
    torch.testing.assert_close(
        trusted_output.sequence_state.state_s,
        explicit_output.sequence_state.state_s,
    )
    torch.testing.assert_close(
        trusted_output.sequence_state.state_nu,
        explicit_output.sequence_state.state_nu,
    )
    expected_indices = torch.arange(trusted_hidden.shape[0] * trusted_hidden.shape[1])
    assert torch.equal(trusted_output.flat_valid_indices, expected_indices)
    assert torch.equal(explicit_output.flat_valid_indices, expected_indices)

    trusted_loss = (
        trusted_output.probabilities.square().mean()
        + trusted_output.q_probabilities.square().mean()
    )
    explicit_loss = (
        explicit_output.probabilities.square().mean()
        + explicit_output.q_probabilities.square().mean()
    )
    trusted_loss.backward()
    explicit_loss.backward()
    torch.testing.assert_close(trusted_hidden.grad, explicit_hidden.grad)
    for (trusted_name, trusted_parameter), (
        explicit_name,
        explicit_parameter,
    ) in zip(trusted.named_parameters(), explicit.named_parameters()):
        assert trusted_name == explicit_name
        assert trusted_parameter.grad is not None
        assert explicit_parameter.grad is not None
        torch.testing.assert_close(
            trusted_parameter.grad,
            explicit_parameter.grad,
        )


def test_trusted_dense_default_reset_matches_explicit_continuation(
    tiny_config,
) -> None:
    router = CPTRouter(tiny_config, layer_index=0).eval()
    hidden_states = torch.randn(2, 8, tiny_config.hidden_size)

    with torch.inference_mode():
        prefix = router(hidden_states[:, :3])
        default_continuation = router(
            hidden_states[:, 3:],
            sequence_state=prefix.sequence_state,
        )
        explicit_continuation = router(
            hidden_states[:, 3:],
            route_valid_mask=torch.ones(2, 5, dtype=torch.bool),
            reset_mask=torch.zeros(2, 5, dtype=torch.bool),
            sequence_state=prefix.sequence_state,
        )

    torch.testing.assert_close(
        default_continuation.q_probabilities,
        explicit_continuation.q_probabilities,
    )
    torch.testing.assert_close(
        default_continuation.probabilities,
        explicit_continuation.probabilities,
    )
    torch.testing.assert_close(
        default_continuation.sequence_state.state_s,
        explicit_continuation.sequence_state.state_s,
    )
    torch.testing.assert_close(
        default_continuation.sequence_state.state_nu,
        explicit_continuation.sequence_state.state_nu,
    )


def test_dense_and_general_row_paths_match_outputs_states_and_gradients(
    tiny_config,
) -> None:
    torch.manual_seed(4012)
    dense = CPTRouter(tiny_config, layer_index=0).train()
    general = _GeneralRowPathRouter(tiny_config, layer_index=0).train()
    general.load_state_dict(copy.deepcopy(dense.state_dict()), strict=True)
    hidden_dense = torch.randn(
        3,
        11,
        tiny_config.hidden_size,
        requires_grad=True,
    )
    hidden_general = hidden_dense.detach().clone().requires_grad_(True)
    route_valid_mask = torch.ones(3, 11, dtype=torch.bool)
    reset_mask = torch.zeros_like(route_valid_mask)
    reset_mask[:, 0] = True
    reset_mask[2, 7] = True

    dense_output = dense(hidden_dense, route_valid_mask, reset_mask)
    general_output = general(hidden_general, route_valid_mask, reset_mask)
    torch.testing.assert_close(
        dense_output.q_probabilities,
        general_output.q_probabilities,
    )
    torch.testing.assert_close(
        dense_output.probabilities,
        general_output.probabilities,
    )
    torch.testing.assert_close(
        dense_output.sequence_state.state_s,
        general_output.sequence_state.state_s,
    )
    torch.testing.assert_close(
        dense_output.sequence_state.state_nu,
        general_output.sequence_state.state_nu,
    )

    dense_loss = (
        dense_output.probabilities.square().mean()
        + dense_output.q_probabilities.square().mean()
    )
    general_loss = (
        general_output.probabilities.square().mean()
        + general_output.q_probabilities.square().mean()
    )
    dense_loss.backward()
    general_loss.backward()
    torch.testing.assert_close(hidden_dense.grad, hidden_general.grad)
    for (dense_name, dense_parameter), (general_name, general_parameter) in zip(
        dense.named_parameters(),
        general.named_parameters(),
    ):
        assert dense_name == general_name
        assert dense_parameter.grad is not None
        assert general_parameter.grad is not None
        torch.testing.assert_close(dense_parameter.grad, general_parameter.grad)


def test_router_is_causal_and_batch_rows_are_isolated(tiny_config) -> None:
    torch.manual_seed(402)
    router = CPTRouter(tiny_config, layer_index=0).eval()
    hidden_states = torch.randn(2, 6, tiny_config.hidden_size)

    with torch.no_grad():
        full = router(hidden_states)
        prefix = router(hidden_states[:, :4])
        row_zero = router(hidden_states[0:1])
        row_one = router(hidden_states[1:2])

    full_rows = full.probabilities.reshape(
        2,
        6,
        tiny_config.num_local_experts,
    )
    prefix_rows = prefix.probabilities.reshape(
        2,
        4,
        tiny_config.num_local_experts,
    )
    torch.testing.assert_close(full_rows[:, :4], prefix_rows)
    torch.testing.assert_close(full_rows[0], row_zero.probabilities)
    torch.testing.assert_close(full_rows[1], row_one.probabilities)

    changed_future = hidden_states.clone()
    changed_future[:, 4:] = 1000.0 * torch.randn_like(changed_future[:, 4:])
    with torch.no_grad():
        changed = router(changed_future).probabilities.reshape_as(full_rows)
    torch.testing.assert_close(changed[:, :4], full_rows[:, :4])


def test_padding_all_padding_and_explicit_reset_semantics(tiny_config) -> None:
    torch.manual_seed(403)
    router = CPTRouter(tiny_config, layer_index=0).eval()
    valid_text = torch.randn(1, 4, tiny_config.hidden_size)
    invalid_prefix = 500.0 * torch.randn(1, 2, tiny_config.hidden_size)
    padded = torch.cat((invalid_prefix, valid_text), dim=1)
    padded_mask = torch.tensor([[False, False, True, True, True, True]])

    with torch.no_grad():
        direct = router(valid_text)
        padded_output = router(padded, route_valid_mask=padded_mask)

    torch.testing.assert_close(padded_output.q_probabilities, direct.q_probabilities)
    torch.testing.assert_close(padded_output.probabilities, direct.probabilities)
    torch.testing.assert_close(
        padded_output.proposal.load_sum,
        direct.proposal.load_sum,
    )
    assert padded_output.flat_valid_indices.tolist() == [2, 3, 4, 5]

    all_padding_mask = torch.zeros(2, 3, dtype=torch.bool)
    with torch.no_grad():
        all_padding = router(
            torch.randn(2, 3, tiny_config.hidden_size),
            route_valid_mask=all_padding_mask,
        )
    assert all_padding.probabilities.shape == (
        0,
        tiny_config.num_local_experts,
    )
    assert all_padding.q_probabilities.shape == (
        0,
        tiny_config.cpt_num_prototypes,
    )
    assert all_padding.flat_valid_indices.numel() == 0
    assert all_padding.proposal.token_count.item() == 0
    assert all_padding.proposal.valid.item()
    torch.testing.assert_close(
        all_padding.proposal.load_sum,
        torch.zeros(tiny_config.num_local_experts),
    )

    two_segments = torch.randn(1, 6, tiny_config.hidden_size)
    explicit_reset = torch.tensor(
        [[True, False, False, True, False, False]],
        dtype=torch.bool,
    )
    with torch.no_grad():
        combined = router(two_segments, reset_mask=explicit_reset)
        second_segment = router(two_segments[:, 3:])
    torch.testing.assert_close(
        combined.q_probabilities[3:],
        second_segment.q_probabilities,
    )
    torch.testing.assert_close(
        combined.probabilities[3:],
        second_segment.probabilities,
    )

    with pytest.raises(ValueError, match="cannot mark an invalid"):
        router(
            padded,
            route_valid_mask=padded_mask,
            reset_mask=torch.tensor(
                [[True, False, False, False, False, False]],
            ),
        )


def test_router_gradients_preserve_main_path_and_exclude_lambda(tiny_config) -> None:
    torch.manual_seed(404)
    router = CPTRouter(tiny_config, layer_index=0).train()
    hidden_states = torch.randn(2, 5, tiny_config.hidden_size)
    output = router(hidden_states)
    output.probabilities.retain_grad()
    output.q_probabilities.retain_grad()
    output.expert_kernel.retain_grad()
    coefficients = torch.randn_like(output.probabilities)

    loss = (output.probabilities * coefficients).sum()
    loss.backward()

    for parameter in (router.projection, router.anchors, router.energy):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.float().norm() > 0
    for tensor in (
        output.probabilities,
        output.q_probabilities,
        output.expert_kernel,
    ):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.float().norm() > 0

    assert not router.congestion_price.requires_grad
    assert router.congestion_price.grad is None
    assert "congestion_price" not in dict(router.named_parameters())
    assert not output.proposal.load_sum.requires_grad
    assert not output.proposal.token_count.requires_grad
    assert not output.proposal.state_version.requires_grad
    assert not output.proposal.valid.requires_grad


def test_router_forward_is_pure_and_repeatable(tiny_config) -> None:
    torch.manual_seed(405)
    router = CPTRouter(tiny_config, layer_index=0).train()
    hidden_states = torch.randn(2, 5, tiny_config.hidden_size)
    before = {
        name: value.detach().clone()
        for name, value in router.state_dict().items()
    }

    first = router(hidden_states)
    second = router(hidden_states)

    torch.testing.assert_close(first.q_probabilities, second.q_probabilities)
    torch.testing.assert_close(first.expert_kernel, second.expert_kernel)
    torch.testing.assert_close(first.probabilities, second.probabilities)
    torch.testing.assert_close(first.proposal.load_sum, second.proposal.load_sum)
    assert first.proposal.token_count.item() == second.proposal.token_count.item()
    assert first.proposal.state_version.item() == 0
    assert second.proposal.state_version.item() == 0

    after = router.state_dict()
    assert before.keys() == after.keys()
    for name, previous in before.items():
        torch.testing.assert_close(previous, after[name])


def test_model_bfloat16_conversion_keeps_all_cpt_state_fp32(tiny_config) -> None:
    model = TinyMixtralForCausalLM(tiny_config)
    original = {
        (layer_index, name): getattr(layer.moe.cpt_router, name).detach().clone()
        for layer_index, layer in enumerate(model.layers)
        for name in ("projection", "anchors", "energy")
    }
    identities = {
        (layer_index, name): id(getattr(layer.moe.cpt_router, name))
        for layer_index, layer in enumerate(model.layers)
        for name in ("projection", "anchors", "energy")
    }
    model = model.to(dtype=torch.bfloat16)

    assert model.embed_tokens.weight.dtype == torch.bfloat16
    for layer_index, layer in enumerate(model.layers):
        router = layer.moe.cpt_router
        for name in ("projection", "anchors", "energy"):
            parameter = getattr(router, name)
            assert parameter.dtype == torch.float32
            assert id(parameter) == identities[(layer_index, name)]
            assert torch.equal(parameter, original[(layer_index, name)])
        assert router.congestion_price.dtype == torch.float32
        assert router.state_version.dtype == torch.int64
        state = router.state_dict()
        assert state["projection"].dtype == torch.float32
        assert state["anchors"].dtype == torch.float32
        assert state["energy"].dtype == torch.float32
        assert state["congestion_price"].dtype == torch.float32
        assert state["state_version"].dtype == torch.int64


def test_layer_specific_initialization_is_deterministic_but_independent(
    tiny_config,
) -> None:
    first = CPTRouter(tiny_config, layer_index=0)
    repeated = CPTRouter(tiny_config, layer_index=0)
    second_layer = CPTRouter(tiny_config, layer_index=1)

    torch.testing.assert_close(first.projection, repeated.projection)
    torch.testing.assert_close(first.anchors, repeated.anchors)
    torch.testing.assert_close(first.energy, repeated.energy)
    assert not torch.equal(first.projection, second_layer.projection)
    assert not torch.equal(first.anchors, second_layer.anchors)
    assert not torch.equal(first.energy, second_layer.energy)


def test_router_copy_keeps_no_pending_or_sequence_local_state(tiny_config) -> None:
    router = CPTRouter(tiny_config, layer_index=0)
    cloned = copy.deepcopy(router)

    assert set(router.state_dict()) == {
        "projection",
        "anchors",
        "energy",
        "congestion_price",
        "state_version",
        "router_algorithm_version",
    }
    assert set(cloned.state_dict()) == set(router.state_dict())
