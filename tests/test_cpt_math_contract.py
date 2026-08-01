import math

import pytest
import torch

from model.config import TinyMixtralConfig
from model.cpt_router import CPT_ROUTER_ALGORITHM_VERSION, CPTRouter


FIRST_VERSION_K = 12
FIRST_VERSION_D_P = 128
NUM_EXPERTS = 6
CPT_ROUTER_VERSION = 1


def _probability_factors(
    token_count: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(201)
    q_logits = torch.randn(
        FIRST_VERSION_K,
        token_count,
        generator=generator,
        dtype=torch.float64,
    )
    b_logits = torch.randn(
        FIRST_VERSION_K,
        NUM_EXPERTS,
        generator=generator,
        dtype=torch.float64,
    )
    q_columns = torch.softmax(q_logits, dim=0)
    expert_kernel = torch.softmax(b_logits, dim=1)
    return q_columns, expert_kernel


def test_first_version_dimensions_and_temperature() -> None:
    assert FIRST_VERSION_K == 12
    assert FIRST_VERSION_D_P == 128
    assert NUM_EXPERTS == 6
    assert math.isclose(
        1.0 / math.sqrt(FIRST_VERSION_D_P),
        0.08838834764831843,
    )


def test_production_defaults_and_derived_hyperparameters() -> None:
    config = TinyMixtralConfig()

    assert config.cpt_router_version == CPT_ROUTER_VERSION
    assert config.cpt_num_prototypes == FIRST_VERSION_K
    assert config.cpt_projection_dim == FIRST_VERSION_D_P
    assert config.num_local_experts == NUM_EXPERTS
    assert math.isclose(config.cpt_rho_beta, 0.95)
    assert math.isclose(config.cpt_beta_max, 0.45)
    assert config.cpt_kappa_beta == 1.0 / (
        FIRST_VERSION_K * (1.0 - config.cpt_rho_beta_effective)
    )
    # In real arithmetic the PDF fixes kappa_beta=20/K. Runtime state
    # arithmetic is FP32, where 0.95 is not exactly representable; deriving
    # kappa from the actual recurrence coefficient keeps the implemented
    # beta/nu system internally consistent and remains within a few FP32 ULPs
    # of the mathematical target.
    target_kappa = torch.tensor(
        20.0 / FIRST_VERSION_K,
        dtype=torch.float32,
    )
    effective_kappa = torch.tensor(
        config.cpt_kappa_beta,
        dtype=torch.float32,
    )
    assert abs(float(effective_kappa - target_kappa)) <= (
        4.0 * torch.finfo(torch.float32).eps * float(target_kappa.abs())
    )
    assert math.isclose(config.cpt_lambda_sa, 1.0 / FIRST_VERSION_K)
    assert math.isclose(
        config.cpt_projection_temperature,
        1.0 / math.sqrt(FIRST_VERSION_D_P),
    )
    assert math.isclose(config.cpt_expert_temperature, 1.0)
    assert math.isclose(
        config.cpt_state_step_size,
        0.1 / (1.0 + 1.0 / FIRST_VERSION_K),
    )
    assert math.isclose(config.cpt_state_radius, 1.0)
    assert math.isclose(config.cpt_eps_z, 1e-6)
    assert math.isclose(config.cpt_eps_m, 1e-6)
    assert math.isclose(config.cpt_eps_init, 1e-8)
    assert math.isclose(config.cpt_energy_init_scale, 0.05)
    assert math.isclose(config.cpt_capacity_factor, 1.25)
    assert math.isclose(config.cpt_price_learning_rate, 0.01)
    assert CPT_ROUTER_ALGORITHM_VERSION == CPT_ROUTER_VERSION


@pytest.mark.parametrize("mutated_version", [2, True, 1.0])
def test_router_implementation_rejects_mutated_config_version(
    mutated_version,
) -> None:
    config = TinyMixtralConfig()
    config.cpt_router_version = mutated_version

    with pytest.raises(
        ValueError,
        match="strict-TeX Router implementation requires cpt_router_version=1",
    ):
        CPTRouter(config, layer_index=0)


@pytest.mark.parametrize(
    "mutated_projection_dim",
    [0, -1, True, 2.0, 8.5],
)
def test_router_implementation_rejects_mutated_projection_dimension(
    mutated_projection_dim,
) -> None:
    config = TinyMixtralConfig()
    config.cpt_projection_dim = mutated_projection_dim

    with pytest.raises(
        ValueError,
        match=(
            "strict-TeX Router implementation requires "
            "integer cpt_projection_dim >= 1"
        ),
    ):
        CPTRouter(config, layer_index=0)


def test_router_rejects_mutated_infeasible_one_dimensional_anchors() -> None:
    config = TinyMixtralConfig()
    config.cpt_projection_dim = 1

    with pytest.raises(
        ValueError,
        match="supports at most 2 distinct unit prototype anchors",
    ):
        CPTRouter(config, layer_index=0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"cpt_projection_dim": 8.5}, "positive integer"),
        ({"cpt_num_prototypes": True}, "positive integer"),
        ({"cpt_router_version": True}, "cpt_router_version=1"),
        ({"cpt_router_version": 0}, "cpt_router_version=1"),
        ({"cpt_router_version": 2}, "cpt_router_version=1"),
        ({"cpt_init_seed": 1.5}, "must be an integer"),
        (
            {"num_local_experts": 1, "num_experts_per_tok": 1},
            "at least 2",
        ),
        ({"cpt_num_prototypes": 1}, "at least 2"),
        (
            {"cpt_projection_dim": 1, "cpt_num_prototypes": 3},
            "supports at most 2 distinct unit prototype anchors",
        ),
    ],
)
def test_native_config_rejects_fractional_boolean_and_degenerate_dimensions(
    overrides,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        TinyMixtralConfig(**overrides)


def test_column_vector_probability_contract() -> None:
    """Q:[K,T], B:[K,N], Pi=B^T Q:[N,T] under column vectors."""
    q_columns, expert_kernel = _probability_factors()
    expert_probabilities = expert_kernel.T @ q_columns

    assert q_columns.shape == (FIRST_VERSION_K, 7)
    assert expert_kernel.shape == (FIRST_VERSION_K, NUM_EXPERTS)
    assert expert_probabilities.shape == (NUM_EXPERTS, 7)
    assert (q_columns >= 0).all()
    assert (expert_kernel >= 0).all()
    assert (expert_probabilities >= 0).all()
    torch.testing.assert_close(
        q_columns.sum(dim=0),
        torch.ones(7, dtype=torch.float64),
    )
    torch.testing.assert_close(
        expert_kernel.sum(dim=1),
        torch.ones(FIRST_VERSION_K, dtype=torch.float64),
    )
    torch.testing.assert_close(
        expert_probabilities.sum(dim=0),
        torch.ones(7, dtype=torch.float64),
    )


def test_row_major_code_equals_column_vector_formula() -> None:
    q_columns, expert_kernel = _probability_factors()
    probability_columns = expert_kernel.T @ q_columns
    probability_rows = q_columns.T @ expert_kernel

    torch.testing.assert_close(probability_rows, probability_columns.T)


def test_closed_form_matrix_gradients() -> None:
    generator = torch.Generator().manual_seed(202)
    q_columns = torch.randn(
        FIRST_VERSION_K,
        5,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    expert_kernel = torch.randn(
        FIRST_VERSION_K,
        NUM_EXPERTS,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    output_gradient = torch.randn(
        NUM_EXPERTS,
        5,
        generator=generator,
        dtype=torch.float64,
    )

    expert_probabilities = expert_kernel.T @ q_columns
    loss = (expert_probabilities * output_gradient).sum()
    gradient_q, gradient_b = torch.autograd.grad(
        loss,
        (q_columns, expert_kernel),
    )

    torch.testing.assert_close(gradient_q, expert_kernel @ output_gradient)
    torch.testing.assert_close(
        gradient_b,
        q_columns @ output_gradient.T,
    )


def test_pi_is_not_softmaxed_a_second_time() -> None:
    q_columns, expert_kernel = _probability_factors(token_count=3)
    direct_probabilities = expert_kernel.T @ q_columns
    incorrectly_softmaxed = torch.softmax(direct_probabilities, dim=0)

    assert not torch.allclose(direct_probabilities, incorrectly_softmaxed)
    torch.testing.assert_close(
        direct_probabilities.sum(dim=0),
        torch.ones(3, dtype=torch.float64),
    )


def test_unchanged_top2_selected_weight_normalization() -> None:
    probability_columns = torch.tensor(
        [
            [0.05, 0.40],
            [0.30, 0.10],
            [0.25, 0.20],
            [0.15, 0.05],
            [0.20, 0.15],
            [0.05, 0.10],
        ],
        dtype=torch.float64,
    )
    probability_rows = probability_columns.T

    selected_weights, selected_experts = torch.topk(
        probability_rows,
        k=2,
        dim=-1,
    )
    normalized_weights = selected_weights / selected_weights.sum(
        dim=-1,
        keepdim=True,
    )

    assert selected_experts.tolist() == [[1, 2], [0, 2]]
    assert (normalized_weights >= 0).all()
    torch.testing.assert_close(
        normalized_weights.sum(dim=-1),
        torch.ones(2, dtype=torch.float64),
    )
