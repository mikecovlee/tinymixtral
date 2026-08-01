import copy
from dataclasses import replace

import pytest
import torch

from hf.configuration_tinymixtral import (
    TinyMixtralConfig as HFTinyMixtralConfig,
)
from hf.modeling_tinymixtral import (
    TinyMixtralForCausalLM as HFTinyMixtralForCausalLM,
)
import model.cpt_router as cpt_router_module
from model.config import TinyMixtralConfig
from model.cpt_router import (
    CPTLayerProposal,
    CPTRouter,
    CPTSequenceState,
    normalize_binary_mask,
)
from model.modeling import TinyMixtralForCausalLM


def _config(
    *,
    layers: int = 2,
    prototypes: int = 4,
    projection_dim: int = 4,
    rho_beta: float = 0.95,
) -> TinyMixtralConfig:
    return TinyMixtralConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        expert_intermediate_size=16,
        cpt_num_prototypes=prototypes,
        cpt_projection_dim=projection_dim,
        cpt_rho_beta=rho_beta,
    )


def _manual_proposals(
    model: TinyMixtralForCausalLM,
    load: tuple[float, ...],
    count: int,
) -> tuple[CPTLayerProposal, ...]:
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
                    dtype=torch.bool,
                ),
            )
        )
    return tuple(proposals)


def test_sequence_state_continuation_equals_one_column_vector_scan() -> None:
    torch.manual_seed(801)
    router = CPTRouter(_config(layers=1), layer_index=0).eval()
    hidden_states = torch.randn(2, 7, router.hidden_size)

    with torch.no_grad():
        full = router(hidden_states)
        first = router(hidden_states[:, :3])
        entry_state_copy = replace(
            first.sequence_state,
            state_s=first.sequence_state.state_s.clone(),
            state_nu=first.sequence_state.state_nu.clone(),
            initialized=first.sequence_state.initialized.clone(),
            state_version=first.sequence_state.state_version.clone(),
        )
        second = router(
            hidden_states[:, 3:],
            sequence_state=first.sequence_state,
        )

    combined_q = torch.cat(
        (
            first.q_probabilities.view(2, 3, -1),
            second.q_probabilities.view(2, 4, -1),
        ),
        dim=1,
    ).reshape(14, -1)
    combined_pi = torch.cat(
        (
            first.probabilities.view(2, 3, -1),
            second.probabilities.view(2, 4, -1),
        ),
        dim=1,
    ).reshape(14, -1)
    torch.testing.assert_close(
        full.q_probabilities,
        combined_q,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        full.probabilities,
        combined_pi,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        full.sequence_state.state_s,
        second.sequence_state.state_s,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        full.sequence_state.state_nu,
        second.sequence_state.state_nu,
        rtol=0,
        atol=0,
    )

    # State-in is immutable.  Replay from the same entry state is therefore
    # deterministic and cannot double-advance online state.
    torch.testing.assert_close(
        first.sequence_state.state_s,
        entry_state_copy.state_s,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        first.sequence_state.state_nu,
        entry_state_copy.state_nu,
        rtol=0,
        atol=0,
    )
    with torch.no_grad():
        replay = router(
            hidden_states[:, 3:],
            sequence_state=entry_state_copy,
        )
    torch.testing.assert_close(second.probabilities, replay.probabilities)
    torch.testing.assert_close(
        second.sequence_state.state_s,
        replay.sequence_state.state_s,
    )

    state = second.sequence_state
    assert state.state_s.dtype == torch.float32
    assert state.state_nu.dtype == torch.float32
    assert state.initialized.dtype == torch.bool
    assert state.state_version.dtype == torch.int64
    assert state.layer_index == 0
    assert state.router_identity() is router
    assert not state.state_s.requires_grad
    assert not state.state_nu.requires_grad
    assert state.initialized.all()
    assert torch.isfinite(state.state_s).all()
    assert torch.isfinite(state.state_nu).all()
    assert (state.state_nu >= 0).all()
    assert (
        state.state_nu
        <= router.state_nu_upper_bound + router.state_nu_element_tolerance
    ).all()
    assert (
        state.state_nu.sum(dim=-1, dtype=torch.float64)
        <= router.state_nu_upper_bound + router.state_nu_mass_tolerance
    ).all()
    assert (
        state.state_s.norm(dim=1)
        <= router.state_radius + 1e-5
    ).all()


def test_k_greater_than_projection_dim_initializes_distinct_unit_anchors() -> None:
    router = CPTRouter(
        _config(layers=1, prototypes=6, projection_dim=2),
        layer_index=0,
    )
    anchors = router.anchors.detach().float()
    torch.testing.assert_close(
        anchors.norm(dim=0),
        torch.ones(6),
        rtol=0,
        atol=1e-6,
    )
    gram = anchors.T @ anchors
    off_diagonal = gram[
        ~torch.eye(6, dtype=torch.bool)
    ]
    assert (
        off_diagonal
        <= 1.0 - 8.0 * torch.finfo(torch.float32).eps
    ).all()


def test_router_v1_config_accepts_single_projection_coordinate() -> None:
    config = _config(layers=1, prototypes=2, projection_dim=1)
    router = CPTRouter(config, layer_index=0)

    assert config.cpt_projection_dim == 1
    assert config.cpt_router_version == 1
    torch.testing.assert_close(
        torch.sort(router.anchors.detach().flatten()).values,
        torch.tensor([-1.0, 1.0]),
        rtol=0.0,
        atol=0.0,
    )


def test_binary_mask_validation_is_strict_and_canonical() -> None:
    router = CPTRouter(_config(layers=1), layer_index=0).eval()
    hidden_states = torch.randn(1, 3, router.hidden_size)

    for dtype in (torch.bool, torch.int64, torch.float32):
        route_valid = torch.tensor([[1, 0, 1]], dtype=dtype)
        reset = torch.tensor([[1, 0, 0]], dtype=dtype)
        output = router(hidden_states, route_valid, reset)
        assert output.flat_valid_indices.tolist() == [0, 2]
        assert output.proposal.token_count.item() == 2

    invalid_real_values = (2.0, -1.0, float("nan"), float("inf"))
    for value in invalid_real_values:
        invalid = torch.tensor([[1.0, value, 0.0]])
        with pytest.raises(ValueError, match="binary values 0 or 1"):
            router(hidden_states, invalid)
        with pytest.raises(ValueError, match="binary values 0 or 1"):
            router(hidden_states, torch.ones_like(invalid), invalid)

    for value in (2, -1):
        invalid_integer = torch.tensor([[1, value, 0]], dtype=torch.int64)
        with pytest.raises(ValueError, match="binary values 0 or 1"):
            router(hidden_states, invalid_integer)

    with pytest.raises(TypeError, match="boolean, integer, or real dtype"):
        router(hidden_states, torch.ones(1, 3, dtype=torch.complex64))
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        normalize_binary_mask(
            [[1, 0, 1]],
            (1, 3),
            torch.device("cpu"),
            name="test_mask",
        )


def test_cpu_bfloat16_autocast_keeps_q_and_pi_softmax_contract_fp32() -> None:
    torch.manual_seed(812)
    router = CPTRouter(
        _config(layers=1, prototypes=12, projection_dim=8),
        layer_index=0,
    ).train()
    hidden_states = torch.randn(
        4,
        64,
        router.hidden_size,
        requires_grad=True,
    )
    route_valid = torch.ones(4, 64, dtype=torch.bool)
    reset = torch.zeros_like(route_valid)
    reset[:, 0] = True

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = router(hidden_states, route_valid, reset)
        loss = output.probabilities.square().sum()
    loss.backward()

    assert output.q_probabilities.dtype == torch.float32
    assert output.probabilities.dtype == torch.float32
    torch.testing.assert_close(
        output.q_probabilities.sum(dim=-1),
        torch.ones(output.q_probabilities.shape[0]),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        output.probabilities.sum(dim=-1),
        torch.ones(output.probabilities.shape[0]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert output.proposal.valid.item()
    assert torch.isfinite(hidden_states.grad).all()
    for parameter in (router.projection, router.anchors, router.energy):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 is unavailable",
)
def test_cuda_bfloat16_projection_normalization_has_no_projection_softmax(
    monkeypatch,
) -> None:
    torch.manual_seed(813)
    router = CPTRouter(
        _config(layers=1, prototypes=4, projection_dim=8),
        layer_index=0,
    ).cuda().train()
    hidden_states = torch.randn(
        2,
        7,
        router.hidden_size,
        device="cuda",
        requires_grad=True,
    )
    route_valid = torch.ones(2, 7, dtype=torch.bool, device="cuda")
    reset = torch.zeros_like(route_valid)
    reset[:, 0] = True
    original_softmax = torch.softmax
    projection_calls = []

    def recording_softmax(input_tensor, *args, **kwargs):
        result = original_softmax(input_tensor, *args, **kwargs)
        if input_tensor.ndim == 3 and input_tensor.shape[-1] == 8:
            dim = kwargs.get("dim", args[0] if args else None)
            projection_calls.append(
                (input_tensor.dtype, result.dtype, dim, tuple(input_tensor.shape))
            )
        return result

    monkeypatch.setattr(torch, "softmax", recording_softmax)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = router(hidden_states, route_valid, reset)
        coefficients = torch.randn_like(output.probabilities)
        loss = (output.probabilities * coefficients).sum()
    loss.backward()

    assert projection_calls == []
    assert output.q_probabilities.dtype == torch.float32
    assert output.probabilities.dtype == torch.float32
    assert torch.isfinite(hidden_states.grad).all()
    assert router.projection.grad is not None
    assert torch.isfinite(router.projection.grad).all()
    assert router.projection.grad.float().norm() > 0


def test_energy_initialization_rejects_identical_rows_and_retries(
    monkeypatch,
) -> None:
    identical = torch.tensor(
        [[0.05, -0.05], [0.05, -0.05]],
        dtype=torch.float32,
    )
    distinct = torch.tensor(
        [[0.05, -0.05], [-0.05, 0.05]],
        dtype=torch.float32,
    )
    assert not CPTRouter._valid_initial_energy(identical)
    assert CPTRouter._valid_initial_energy(distinct)

    monkeypatch.setattr(
        CPTRouter,
        "_valid_initial_energy",
        staticmethod(lambda _candidate: False),
    )
    with pytest.raises(RuntimeError, match="non-degenerate CPT energy"):
        CPTRouter(_config(layers=1), layer_index=0)


def test_sequence_state_crosses_successful_step_as_pure_provenance() -> None:
    torch.manual_seed(802)
    router = CPTRouter(_config(layers=1), layer_index=0).eval()
    first_hidden = torch.randn(1, 3, router.hidden_size)
    second_hidden = torch.randn(1, 2, router.hidden_size)

    with torch.no_grad():
        first = router(first_hidden)
        assert first.sequence_state.state_version.item() == 0
        # Simulate a successful model transaction.  S and nu have no optimizer
        # step subscript in the PDF, so old provenance must not force a reset.
        router.state_version.add_(1)
        second = router(
            second_hidden,
            sequence_state=first.sequence_state,
        )

    assert second.sequence_state.state_version.item() == 1
    assert second.sequence_state.initialized.item()
    assert not torch.equal(
        second.sequence_state.state_s,
        torch.zeros_like(second.sequence_state.state_s),
    )


def test_new_rows_force_first_valid_reset_but_continuations_do_not() -> None:
    torch.manual_seed(803)
    router = CPTRouter(_config(layers=1), layer_index=0).eval()
    hidden_states = torch.randn(1, 6, router.hidden_size)

    with torch.no_grad():
        automatic = router(hidden_states[:, :3])
        explicit_false = router(
            hidden_states[:, :3],
            reset_mask=torch.zeros(1, 3, dtype=torch.bool),
        )
        continued = router(
            hidden_states[:, 3:],
            sequence_state=automatic.sequence_state,
            reset_mask=torch.zeros(1, 3, dtype=torch.bool),
        )
        forced_fresh = router(
            hidden_states[:, 3:],
            sequence_state=automatic.sequence_state,
            reset_mask=torch.tensor([[True, False, False]]),
        )
        fresh = router(hidden_states[:, 3:])

    torch.testing.assert_close(automatic.probabilities, explicit_false.probabilities)
    torch.testing.assert_close(forced_fresh.probabilities, fresh.probabilities)
    assert not torch.allclose(continued.q_probabilities, fresh.q_probabilities)


def test_sequence_state_rejects_invalid_shape_dtype_values_and_metadata() -> None:
    router = CPTRouter(_config(layers=1), layer_index=0).eval()
    hidden_states = torch.randn(2, 2, router.hidden_size)
    with torch.no_grad():
        valid_state = router(hidden_states[:, :1]).sequence_state

    invalid_states = [
        replace(valid_state, state_s=valid_state.state_s[:, :-1]),
        replace(valid_state, state_s=valid_state.state_s.to(torch.bfloat16)),
        replace(
            valid_state,
            state_nu=valid_state.state_nu.clone().fill_(float("nan")),
        ),
        replace(
            valid_state,
            state_nu=-torch.ones_like(valid_state.state_nu),
        ),
        replace(
            valid_state,
            state_s=torch.full_like(
                valid_state.state_s,
                2.0 * router.state_radius,
            ),
        ),
        replace(
            valid_state,
            initialized=valid_state.initialized.to(torch.int64),
        ),
        replace(
            valid_state,
            state_version=torch.tensor(-1, dtype=torch.int64),
        ),
        replace(
            valid_state,
            state_s=valid_state.state_s.detach().requires_grad_(True),
        ),
        replace(valid_state, layer_index=1),
        replace(valid_state, router_identity=object()),
    ]
    for invalid_state in invalid_states:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            router(hidden_states, sequence_state=invalid_state)


def test_sequence_state_rejects_cross_layer_and_cross_router_before_softmax(
    monkeypatch,
) -> None:
    config = _config(layers=2)
    source = CPTRouter(config, layer_index=0).eval()
    other_layer = CPTRouter(config, layer_index=1).eval()
    other_model_same_layer = CPTRouter(config, layer_index=0).eval()
    hidden_states = torch.randn(1, 2, source.hidden_size)
    with torch.no_grad():
        state = source(hidden_states[:, :1]).sequence_state
        copied_state = copy.deepcopy(state)
        copied_output = source(
            hidden_states[:, 1:],
            sequence_state=copied_state,
        )
    assert copied_output.sequence_state.router_identity() is source

    softmax_calls = 0

    def forbidden_softmax(*_args, **_kwargs):
        nonlocal softmax_calls
        softmax_calls += 1
        raise AssertionError("identity validation must precede q/B softmax")

    monkeypatch.setattr(torch, "softmax", forbidden_softmax)
    with pytest.raises(ValueError, match="layer identity"):
        other_layer(hidden_states, sequence_state=state)
    with pytest.raises(ValueError, match="different CPT router instance"):
        other_model_same_layer(hidden_states, sequence_state=state)
    assert softmax_calls == 0


@pytest.mark.parametrize(
    "rho_beta",
    [0.0, 0.95, 0.9999999403953552],
)
def test_sequence_state_nu_geometric_bounds_with_fp32_tolerance(
    rho_beta,
) -> None:
    config = _config(layers=1, rho_beta=rho_beta)
    router = CPTRouter(
        config,
        layer_index=0,
    ).eval()
    base = router._new_sequence_state(1, torch.device("cpu"))
    hidden_states = torch.randn(1, 2, router.hidden_size)
    all_padding = torch.zeros(1, 2, dtype=torch.bool)
    upper = router.state_nu_upper_bound
    element_tolerance = router.state_nu_element_tolerance
    mass_tolerance = router.state_nu_mass_tolerance
    effective_rho = float(torch.tensor(rho_beta, dtype=torch.float32).item())
    assert config.cpt_rho_beta_effective == effective_rho
    assert router.rho_beta == effective_rho
    assert upper == 1.0 / (1.0 - effective_rho)
    assert config.cpt_kappa_beta == router.kappa_beta
    assert router.kappa_beta == upper / router.num_prototypes
    assert element_tolerance < 1e-3 * upper
    assert mass_tolerance < 1e-3 * upper

    def state_with_nu(values: torch.Tensor) -> CPTSequenceState:
        return replace(
            base,
            state_s=base.state_s.clone(),
            state_nu=values.to(dtype=torch.float32),
            initialized=torch.ones_like(base.initialized),
            state_version=base.state_version.clone(),
        )

    boundary = torch.zeros_like(base.state_nu)
    boundary[0, 0] = upper
    within_tolerance = torch.full_like(
        base.state_nu,
        (upper + 0.5 * mass_tolerance) / router.num_prototypes,
    )
    with torch.no_grad():
        boundary_output = router(
            hidden_states,
            route_valid_mask=all_padding,
            sequence_state=state_with_nu(boundary),
        )
        tolerance_output = router(
            hidden_states,
            route_valid_mask=all_padding,
            sequence_state=state_with_nu(within_tolerance),
        )
    torch.testing.assert_close(
        boundary_output.sequence_state.state_nu,
        boundary,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        tolerance_output.sequence_state.state_nu,
        within_tolerance,
        rtol=0,
        atol=0,
    )

    component_over = torch.zeros_like(base.state_nu)
    component_over[0, 0] = upper + 2.0 * element_tolerance
    with pytest.raises(ValueError, match="theoretical upper bound"):
        router(
            hidden_states,
            route_valid_mask=all_padding,
            sequence_state=state_with_nu(component_over),
        )

    mass_over = torch.full_like(
        base.state_nu,
        (upper + 2.0 * mass_tolerance) / router.num_prototypes,
    )
    assert torch.all(mass_over < upper)
    with pytest.raises(ValueError, match="row mass exceeds"):
        router(
            hidden_states,
            route_valid_mask=all_padding,
            sequence_state=state_with_nu(mass_over),
        )

    for multiple in (2.0, 10.0):
        impossible = torch.zeros_like(base.state_nu)
        impossible[0, 0] = multiple * upper
        with pytest.raises(ValueError, match="theoretical upper bound"):
            router(
                hidden_states,
                route_valid_mask=all_padding,
                sequence_state=state_with_nu(impossible),
            )

    huge = torch.zeros_like(base.state_nu)
    huge[0, 0] = torch.finfo(torch.float32).max
    with pytest.raises(ValueError, match="theoretical upper bound"):
        router(
            hidden_states,
            route_valid_mask=all_padding,
            sequence_state=state_with_nu(huge),
        )


def test_invalid_rows_never_enter_q_softmax_or_poison_backward(monkeypatch) -> None:
    torch.manual_seed(804)
    router = CPTRouter(
        _config(layers=1, prototypes=3),
        layer_index=0,
    ).train()
    hidden_states = torch.randn(2, 3, router.hidden_size, requires_grad=True)
    route_valid_mask = torch.tensor(
        [[False, True, True], [True, False, True]],
        dtype=torch.bool,
    )
    with torch.no_grad():
        hidden_states[~route_valid_mask] = float("nan")

    original_softmax = torch.softmax
    q_softmax_batch_sizes = []

    def recording_softmax(input_tensor, *args, **kwargs):
        if input_tensor.ndim == 2 and input_tensor.shape[-1] == 3:
            q_softmax_batch_sizes.append(input_tensor.shape[0])
            assert torch.isfinite(input_tensor).all()
        return original_softmax(input_tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "softmax", recording_softmax)
    output = router(
        hidden_states,
        route_valid_mask=route_valid_mask,
    )
    assert q_softmax_batch_sizes == [1, 1, 2]
    assert torch.isfinite(output.q_probabilities).all()
    assert torch.isfinite(output.probabilities).all()
    assert output.proposal.valid.item()

    coefficients = torch.randn_like(output.probabilities)
    (output.probabilities * coefficients).sum().backward()
    for parameter in (router.projection, router.anchors, router.energy):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert torch.equal(
        hidden_states.grad[~route_valid_mask],
        torch.zeros_like(hidden_states.grad[~route_valid_mask]),
    )


def test_microbatch_raw_statistics_aggregate_once_by_identity() -> None:
    model = TinyMixtralForCausalLM(_config()).train()
    first = _manual_proposals(model, (3.0, 1.0, 0.0, 0.0), 4)
    second = _manual_proposals(model, (0.0, 0.0, 1.0, 1.0), 2)
    transaction = model.prepare_cpt_transaction(
        first,
        microbatch_id="micro-0",
    )
    model.accumulate_cpt_transaction(
        transaction,
        second,
        microbatch_id="micro-1",
    )
    # Replay of the same logical micro-batch is idempotent.
    model.accumulate_cpt_transaction(
        transaction,
        second,
        microbatch_id="micro-1",
    )

    assert transaction.microbatch_ids == ("micro-0", "micro-1")
    for proposal in transaction.proposals:
        torch.testing.assert_close(
            proposal.load_sum,
            torch.tensor([3.0, 1.0, 1.0, 1.0]),
        )
        assert proposal.token_count.item() == 6

    conflicting = _manual_proposals(model, (0.0, 2.0, 0.0, 0.0), 2)
    with pytest.raises(RuntimeError, match="reused with different"):
        model.accumulate_cpt_transaction(
            transaction,
            conflicting,
            microbatch_id="micro-1",
        )

    model.validate_cpt_transaction(transaction)
    expected_mean = torch.tensor([3.0, 1.0, 1.0, 1.0]) / 6.0
    for layer, price in zip(model.layers, transaction._prepared_prices):
        router = layer.moe.cpt_router
        expected_price = torch.clamp_min(
            router.price_learning_rate
            * (
                expected_mean
                - router.capacity_factor / router.num_experts
            ),
            0.0,
        )
        torch.testing.assert_close(price, expected_price)

    with pytest.raises(RuntimeError, match="after price preparation"):
        model.accumulate_cpt_transaction(
            transaction,
            second,
            microbatch_id="micro-2",
        )


def test_finite_price_shadow_tampering_is_rejected() -> None:
    model = TinyMixtralForCausalLM(_config()).train()
    transaction = model.prepare_cpt_transaction(
        _manual_proposals(model, (3.0, 1.0, 0.0, 0.0), 4),
        microbatch_id="micro-0",
    )
    model.validate_cpt_transaction(transaction)
    tampered = list(transaction._prepared_prices)
    tampered[0] = torch.full_like(tampered[0], 123.0)
    transaction._prepared_prices = tuple(tampered)

    with pytest.raises(RuntimeError, match="immutable raw-statistics formula"):
        model.commit_cpt_transaction(transaction)
    assert model.get_cpt_state_version() == 0
    assert torch.equal(
        model.layers[0].moe.cpt_router.congestion_price,
        torch.zeros(4),
    )


def test_any_negative_raw_probability_mass_is_rejected() -> None:
    model = TinyMixtralForCausalLM(_config()).train()
    transaction = model.prepare_cpt_transaction(
        _manual_proposals(
            model,
            (-1e-7, 1.0 + 1e-7, 0.0, 0.0),
            1,
        ),
        microbatch_id="negative-mass",
    )

    with pytest.raises(RuntimeError, match="load is negative"):
        model.validate_cpt_transaction(transaction)
    assert model.get_cpt_state_version() == 0


def test_ddp_uses_validation_consensus_then_raw_sum_and_count(monkeypatch) -> None:
    model = TinyMixtralForCausalLM(_config()).train()
    transaction = model.prepare_cpt_transaction(
        _manual_proposals(model, (3.0, 1.0, 0.0, 0.0), 4),
        microbatch_id="rank-local",
    )

    monkeypatch.setattr(cpt_router_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(cpt_router_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(cpt_router_module.dist, "get_world_size", lambda: 2)
    collective_calls = []

    def valid_all_reduce(tensor, op=None):
        collective_calls.append((tuple(tensor.shape), tensor.dtype, op))
        if tensor.ndim == 2:
            # Remote rank contributes two tokens, one to each of experts 2/3.
            tensor.add_(torch.tensor([0.0, 0.0, 1.0, 1.0]).expand_as(tensor))
        elif tensor.ndim == 1 and tensor.dtype == torch.int64:
            tensor.add_(2)

    monkeypatch.setattr(
        cpt_router_module.dist,
        "all_reduce",
        valid_all_reduce,
    )
    model.validate_cpt_transaction(transaction)

    assert len(collective_calls) == 5
    torch.testing.assert_close(
        transaction._prepared_global_loads[0],
        torch.tensor([3.0, 1.0, 1.0, 1.0]),
    )
    assert transaction._prepared_global_counts[0].item() == 6
    global_mean = torch.tensor([3.0, 1.0, 1.0, 1.0]) / 6.0
    equal_rank_mean = 0.5 * (
        torch.tensor([3.0, 1.0, 0.0, 0.0]) / 4.0
        + torch.tensor([0.0, 0.0, 1.0, 1.0]) / 2.0
    )
    assert not torch.allclose(global_mean, equal_rank_mean)

    remote_invalid = model.prepare_cpt_transaction(
        _manual_proposals(model, (3.0, 1.0, 0.0, 0.0), 4),
        microbatch_id="remote-invalid",
    )
    invalid_collectives = []

    def invalid_consensus(tensor, op=None):
        invalid_collectives.append((tuple(tensor.shape), tensor.dtype, op))
        assert tensor.ndim == 0
        tensor.zero_()

    monkeypatch.setattr(
        cpt_router_module.dist,
        "all_reduce",
        invalid_consensus,
    )
    with pytest.raises(RuntimeError, match="another distributed rank rejected"):
        model.validate_cpt_transaction(remote_invalid)
    # No rank enters raw-statistic collectives after the validity MIN fails.
    assert len(invalid_collectives) == 1


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda state: state["congestion_price"].fill_(-1.0),
            "finite and non-negative",
        ),
        (
            lambda state: state["anchors"].mul_(2.0),
            "not column-normalized",
        ),
        (
            lambda state: state["energy"].fill_(float("nan")),
            "energy is non-finite",
        ),
        (
            lambda state: state["state_version"].fill_(-1),
            "state_version must be non-negative",
        ),
    ],
)
def test_router_load_rejects_persistent_invariant_violations(
    mutation,
    message,
) -> None:
    source = CPTRouter(_config(layers=1), layer_index=0)
    state = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
    }
    mutation(state)
    target = CPTRouter(_config(layers=1), layer_index=0)
    with pytest.raises(RuntimeError, match=message):
        target.load_state_dict(state)


def test_model_load_revalidates_every_cpt_layer() -> None:
    source = TinyMixtralForCausalLM(_config())
    state = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
    }
    state["layers.1.moe.cpt_router.congestion_price"][0] = -0.25
    target = TinyMixtralForCausalLM(_config())
    with pytest.raises(RuntimeError, match="layer 1 CPT congestion_price"):
        target.load_state_dict(state)


def test_model_load_rejects_mixed_layer_state_versions() -> None:
    source = TinyMixtralForCausalLM(_config())
    state = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
    }
    state["layers.1.moe.cpt_router.state_version"].fill_(1)
    target = TinyMixtralForCausalLM(_config())
    with pytest.raises(RuntimeError, match="CPT layer versions disagree"):
        target.load_state_dict(state)


def test_hf_model_load_defers_version_consensus_until_meta_state_materializes() -> None:
    # Only the HF implementation permits strict=False because Transformers
    # uses staged low-memory materialization.  The native model now rejects
    # every strict=False load at its public boundary.
    model = HFTinyMixtralForCausalLM(
        HFTinyMixtralConfig(**_config().to_dict())
    )
    second_router = model.layers[1].moe.cpt_router
    second_router.state_version = torch.empty(
        (),
        device="meta",
        dtype=torch.int64,
    )

    # Simulate one intermediate low-memory assignment: layer zero is already
    # version one, while layer one has not left the meta device.  Consensus is
    # intentionally deferred instead of reading a meta scalar or misreporting
    # a transient mixed version.
    model.load_state_dict(
        {
            "layers.0.moe.cpt_router.state_version": torch.tensor(
                1,
                dtype=torch.int64,
            )
        },
        strict=False,
        assign=True,
    )
    assert model.layers[0].moe.cpt_router.state_version.item() == 1
    assert second_router.state_version.is_meta

    # Once the final layer materializes, the same mixed checkpoint must fail.
    second_router.state_version = torch.zeros((), dtype=torch.int64)
    with pytest.raises(RuntimeError, match="CPT layer versions disagree"):
        model.load_state_dict({}, strict=False)
