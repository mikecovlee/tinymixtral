import ast
import inspect
import textwrap

import pytest
import torch
import torch.nn.functional as F

from model.config import TinyMixtralConfig
from model.cpt_router import CPT_ROUTER_ALGORITHM_VERSION, CPTRouter
from model.modeling import GQAAttention, SparseMoE, TinyMixtralForCausalLM


LEGACY_SHORT_STATE_CONFIG_FIELDS = frozenset(
    {
        "cpt_rho_beta",
        "cpt_beta_max",
        "cpt_kappa_beta",
        "cpt_lambda_sa",
        "cpt_state_step_size",
        "cpt_state_radius",
    }
)

LEGACY_SHORT_STATE_ROUTER_ATTRIBUTES = frozenset(
    {
        "rho_beta",
        "beta_max",
        "kappa_beta",
        "lambda_sa",
        "state_step_size",
        "state_radius",
    }
)

EXPECTED_CPT_V1_3_CONFIG_FIELDS = frozenset(
    {
        "cpt_router_version",
        "cpt_num_prototypes",
        "cpt_projection_dim",
        "cpt_prototype_temperature",
        "cpt_expert_temperature",
        "cpt_eps_z",
        "cpt_eps_m",
        "cpt_eps_init",
        "cpt_energy_init_scale",
        "cpt_capacity_factor",
        "cpt_price_learning_rate",
        "cpt_init_seed",
    }
)


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
        "cpt_init_seed": 17,
    }
    values.update(overrides)
    return TinyMixtralConfig(**values)


def replace_once(source, old, new):
    assert source.count(old) == 1
    return source.replace(old, new, 1)


def override_audit_sources(
    monkeypatch,
    *,
    router_source=None,
    moe_source=None,
):
    import scripts.audit_projection_l2_v1 as audit_projection_l2_v1

    original_getsource = audit_projection_l2_v1.inspect.getsource
    router_source = router_source or original_getsource(CPTRouter.forward)
    moe_source = moe_source or original_getsource(SparseMoE.forward)

    def get_source(obj):
        if obj is CPTRouter.forward:
            return router_source
        if obj is SparseMoE.forward:
            return moe_source
        return original_getsource(obj)

    monkeypatch.setattr(
        audit_projection_l2_v1.inspect,
        "getsource",
        get_source,
    )
    return audit_projection_l2_v1


def column_vector_oracle(router, hidden_states, valid_mask=None):
    """Literal long-state-only column-vector implementation."""
    x = hidden_states.float()
    batch, sequence, _ = x.shape
    if valid_mask is None:
        valid_mask = torch.ones(batch, sequence, dtype=torch.bool, device=x.device)
    else:
        valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)

    normalized_anchors = router.anchors / torch.linalg.vector_norm(
        router.anchors,
        dim=0,
        keepdim=True,
    ).clamp_min(router.eps_m)
    centered_energy = router.energy - router.energy.mean(dim=-1, keepdim=True)
    expert_kernel = torch.softmax(
        (
            centered_energy
            - router.congestion_price.detach().unsqueeze(0)
        )
        / router.expert_temperature,
        dim=-1,
        dtype=torch.float32,
    )
    result = torch.zeros(
        batch,
        sequence,
        router.num_experts,
        device=x.device,
        dtype=torch.float32,
    )
    for batch_index in range(batch):
        for position in range(sequence):
            if not bool(valid_mask[batch_index, position]):
                continue
            x_column = x[batch_index, position].unsqueeze(1)
            projected = router.projection @ x_column
            z_column = projected / torch.linalg.vector_norm(
                projected,
                dim=0,
                keepdim=True,
            ).clamp_min(router.eps_z)
            q_column = torch.softmax(
                normalized_anchors.T @ z_column / router.prototype_temperature,
                dim=0,
                dtype=torch.float32,
            )
            result[batch_index, position] = (
                expert_kernel.T @ q_column
            ).squeeze(1)
    return result


@torch.no_grad()
def make_expert_kernel_distinct(router):
    """Give different prototypes observably different expert preferences."""
    energy = torch.full_like(router.energy, -1.0)
    prototype_indices = torch.arange(
        router.num_prototypes,
        device=router.energy.device,
    )
    energy[
        prototype_indices,
        prototype_indices.remainder(router.num_experts),
    ] = 2.0
    router.energy.copy_(energy)


def test_config_identifies_long_state_only_router_v3():
    config = tiny_config()
    router = CPTRouter(config, layer_index=0)

    assert config.cpt_router_version == 3
    assert CPT_ROUTER_ALGORITHM_VERSION == 3
    assert int(router.router_algorithm_version.item()) == 3
    for attribute in LEGACY_SHORT_STATE_ROUTER_ATTRIBUTES:
        assert not hasattr(router, attribute)


def test_config_schema_removes_only_short_state_controls():
    config = tiny_config()
    cpt_fields = {
        name
        for name in TinyMixtralConfig.__dataclass_fields__
        if name.startswith("cpt_")
    }

    assert cpt_fields == EXPECTED_CPT_V1_3_CONFIG_FIELDS
    assert set(config.cpt_config_dict()) == EXPECTED_CPT_V1_3_CONFIG_FIELDS
    assert LEGACY_SHORT_STATE_CONFIG_FIELDS.isdisjoint(cpt_fields)
    assert config.cpt_eps_m == pytest.approx(1e-6)


def test_config_derives_k_and_authoritative_long_state_defaults():
    config = tiny_config()
    assert config.cpt_num_prototypes == 2 * config.num_local_experts == 6
    assert config.cpt_prototype_temperature == pytest.approx(0.5)
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
        ({"cpt_router_version": True}, "cpt_router_version=3"),
        ({"cpt_router_version": 1}, "cpt_router_version=3"),
        ({"cpt_router_version": 2}, "cpt_router_version=3"),
        ({"cpt_router_version": 4}, "cpt_router_version=3"),
        ({"cpt_projection_dim": 1}, "more than two"),
        ({"cpt_prototype_temperature": 0.0}, "positive"),
        ({"cpt_eps_m": 0.0}, "positive"),
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
    ],
)
def test_config_rejects_invalid_cpt_contract(updates, message):
    with pytest.raises(ValueError, match=message):
        tiny_config(**updates)


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
    make_expert_kernel_distinct(router)
    hidden = torch.randn(2, 5, router.hidden_size)
    mask = torch.tensor([[1, 1, 0, 1, 0], [0, 1, 1, 1, 1]], dtype=torch.bool)
    actual = router(hidden, mask).probabilities
    expected = column_vector_oracle(router, hidden, mask)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        actual[mask].sum(dim=-1), torch.ones(int(mask.sum())), atol=2e-6, rtol=0
    )
    assert torch.equal(actual[~mask], torch.zeros_like(actual[~mask]))


def test_router_gradients_match_long_state_column_vector_oracle():
    torch.manual_seed(8)
    router = CPTRouter(tiny_config(), layer_index=0)
    make_expert_kernel_distinct(router)
    with torch.no_grad():
        router.anchors.mul_(
            torch.linspace(
                0.7,
                1.3,
                router.num_prototypes,
                dtype=torch.float32,
            ).unsqueeze(0)
        )

    mask = torch.tensor(
        [[1, 1, 0, 1], [1, 0, 1, 1]],
        dtype=torch.bool,
    )
    hidden_values = torch.randn(2, 4, router.hidden_size)
    loss_weights = torch.randn(2, 4, router.num_experts)

    actual_hidden = hidden_values.clone().requires_grad_()
    actual_probabilities = router(actual_hidden, mask).probabilities
    actual_loss = (actual_probabilities * loss_weights).sum()
    actual_gradients = torch.autograd.grad(
        actual_loss,
        (actual_hidden, router.projection, router.anchors, router.energy),
    )

    oracle_hidden = hidden_values.clone().requires_grad_()
    oracle_probabilities = column_vector_oracle(router, oracle_hidden, mask)
    oracle_loss = (oracle_probabilities * loss_weights).sum()
    oracle_gradients = torch.autograd.grad(
        oracle_loss,
        (oracle_hidden, router.projection, router.anchors, router.energy),
    )

    for actual, expected in zip(actual_gradients, oracle_gradients):
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-5)


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


def test_long_state_v3_projection_has_no_projection_softmax():
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
    source = inspect.getsource(CPTRouter.forward)
    assert "flat_prototype_probabilities.index_select" in source
    assert "valid_probabilities = prototype_probabilities @ kernel" in source
    assert ".view(batch_size, sequence_length, self.num_experts)" in source
    assert source.count("@ kernel") == 1


def test_router_forward_ast_has_no_short_state_or_position_scan():
    source = textwrap.dedent(inspect.getsource(CPTRouter.forward))
    tree = ast.parse(source)
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While))
    ]
    identifiers = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    identifiers.update(
        node.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.arg)
    )
    identifiers.update(
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    )
    forbidden_identifiers = {
        "short_state",
        "responsibility",
        "beta",
        "state_old",
        "nu_old",
        "state_gradient",
        "state_candidate",
        "prototype_probability_steps",
    }

    assert loops == []
    assert identifiers.isdisjoint(forbidden_identifiers)
    assert "normalized_anchors" in identifiers


def test_source_audit_is_invariant_to_local_variable_names(monkeypatch):
    import scripts.audit_projection_l2_v1 as audit_projection_l2_v1

    def rename_locals(source, replacements):
        tree = ast.parse(textwrap.dedent(source))

        class LocalNameRenamer(ast.NodeTransformer):
            def visit_Name(self, node):
                replacement = replacements.get(node.id)
                if replacement is not None:
                    node.id = replacement
                return node

        return ast.unparse(LocalNameRenamer().visit(tree))

    original_getsource = audit_projection_l2_v1.inspect.getsource
    router_source = rename_locals(
        original_getsource(CPTRouter.forward),
        {
            "projected": "token_projection",
            "kernel": "expert_mixture",
            "normalized_anchors": "anchor_columns",
            "prototype_logits": "assignment_logits",
            "nominal_prototype_probabilities": "prototype_distribution",
            "flat_prototype_probabilities": "flattened_distribution",
            "valid_token_indices": "active_token_indices",
            "prototype_probabilities": "active_prototype_distribution",
            "valid_probabilities": "active_expert_probabilities",
            "flat_probabilities": "flattened_expert_probabilities",
            "probabilities": "expert_probabilities",
        },
    )
    moe_source = rename_locals(
        original_getsource(SparseMoE.forward),
        {
            "router_output": "cpt_result",
            "all_routing_weights": "dense_routes",
            "valid_token_idx": "active_indices",
            "routing_weights": "active_routes",
            "routing_weights_topk": "selected_route_weights",
            "selected_experts": "selected_expert_indices",
        },
    )

    audit_projection_l2_v1 = override_audit_sources(
        monkeypatch,
        router_source=router_source,
        moe_source=moe_source,
    )
    audit_projection_l2_v1.audit_source_contract(tiny_config())


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "kernel = self.expert_kernel()",
            "kernel = self.expert_kernel()\n"
            "            kernel = torch.zeros_like(kernel)",
            "expert_kernel result must flow directly",
        ),
        (
            "kernel = self.expert_kernel()",
            "kernel = self.expert_kernel()\n"
            "            *kernel, = [torch.zeros_like(kernel)]",
            "expert_kernel result must flow directly",
        ),
        (
            "prototype_probabilities @ kernel",
            "torch.zeros_like(prototype_probabilities) @ kernel",
            "prototype probabilities must flow directly",
        ),
    ],
)
def test_source_audit_rejects_broken_router_dataflow(
    monkeypatch,
    old,
    new,
    message,
):
    import scripts.audit_projection_l2_v1 as audit_projection_l2_v1

    router_source = replace_once(
        audit_projection_l2_v1.inspect.getsource(CPTRouter.forward),
        old,
        new,
    )
    audit_projection_l2_v1 = override_audit_sources(
        monkeypatch,
        router_source=router_source,
    )

    with pytest.raises(AssertionError, match=message):
        audit_projection_l2_v1.audit_source_contract(tiny_config())


def test_source_audit_ignores_annotation_only_statements(monkeypatch):
    import scripts.audit_projection_l2_v1 as audit_projection_l2_v1

    router_source = replace_once(
        audit_projection_l2_v1.inspect.getsource(CPTRouter.forward),
        "kernel = self.expert_kernel()",
        "kernel = self.expert_kernel()\n"
        "            kernel: torch.Tensor",
    )
    audit_projection_l2_v1 = override_audit_sources(
        monkeypatch,
        router_source=router_source,
    )

    audit_projection_l2_v1.audit_source_contract(tiny_config())


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "self.top_k,\n            dim=-1,",
            "1,\n            dim=-1,",
            "Top-2 must use self.top_k",
        ),
        (
            "self.top_k,\n            dim=-1,",
            "self.top_k,\n            dim=0,",
            "Top-2 must operate on the expert dimension",
        ),
    ],
)
def test_source_audit_rejects_invalid_topk_contract(
    monkeypatch,
    old,
    new,
    message,
):
    import scripts.audit_projection_l2_v1 as audit_projection_l2_v1

    moe_source = replace_once(
        audit_projection_l2_v1.inspect.getsource(SparseMoE.forward),
        old,
        new,
    )
    audit_projection_l2_v1 = override_audit_sources(
        monkeypatch,
        moe_source=moe_source,
    )

    with pytest.raises(AssertionError, match=message):
        audit_projection_l2_v1.audit_source_contract(tiny_config())


@pytest.mark.parametrize(
    ("callable_binding", "softmax_expression", "global_softmax_alias"),
    [
        pytest.param(
            "",
            "F.softmax(post_pi_alias, dim=-1)",
            None,
            id="data-alias",
        ),
        pytest.param(
            "apply_probability_normalization = F.softmax\n",
            "apply_probability_normalization(post_pi_alias, dim=-1)",
            None,
            id="callable-alias",
        ),
        pytest.param(
            "apply_probability_normalization = torch.nn.Softmax(dim=-1)\n",
            "apply_probability_normalization(post_pi_alias)",
            None,
            id="module-alias",
        ),
        pytest.param(
            "from torch.nn.functional import softmax as "
            "apply_probability_normalization\n",
            "apply_probability_normalization(post_pi_alias, dim=-1)",
            None,
            id="local-import-alias",
        ),
        pytest.param(
            "",
            "apply_probability_normalization(post_pi_alias, dim=-1)",
            F.softmax,
            id="global-functional-alias",
        ),
        pytest.param(
            "",
            "apply_probability_normalization(post_pi_alias, dim=-1)",
            torch.softmax,
            id="global-torch-function-alias",
        ),
        pytest.param(
            "",
            "apply_probability_normalization(post_pi_alias, dim=-1)",
            torch.Tensor.softmax,
            id="global-tensor-method-alias",
        ),
        pytest.param(
            "",
            "apply_probability_normalization(post_pi_alias, dim=-1)",
            torch.special.softmax,
            id="global-special-function-alias",
        ),
        pytest.param(
            "",
            "apply_probability_normalization(post_pi_alias)",
            torch.nn.Softmax(dim=-1),
            id="global-module-instance-alias",
        ),
    ],
)
def test_source_audit_rejects_post_pi_softmax_through_aliases(
    monkeypatch,
    callable_binding,
    softmax_expression,
    global_softmax_alias,
):
    import scripts.audit_projection_l2_v1 as audit_projection_l2_v1

    if global_softmax_alias is not None:
        monkeypatch.setitem(
            SparseMoE.forward.__globals__,
            "apply_probability_normalization",
            global_softmax_alias,
        )

    moe_tree = ast.parse(
        textwrap.dedent(
            audit_projection_l2_v1.inspect.getsource(SparseMoE.forward)
        )
    )

    class InsertAliasedPostPiSoftmax(ast.NodeTransformer):
        def __init__(self):
            self.mutations = 0

        def visit_Assign(self, node):
            node = self.generic_visit(node)
            if not (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "torch"
                and node.value.func.attr == "topk"
                and len(node.targets) == 1
                and isinstance(node.targets[0], (ast.Tuple, ast.List))
                and len(node.targets[0].elts) == 2
                and isinstance(node.targets[0].elts[0], ast.Name)
            ):
                return node

            self.mutations += 1
            weight_name = node.targets[0].elts[0].id
            mutation_source = (
                callable_binding
                + f"post_pi_alias = {weight_name}\n"
                + f"post_pi_alias = {softmax_expression}\n"
                + f"{weight_name} = post_pi_alias"
            )
            mutation = ast.parse(mutation_source).body
            return [node, *mutation]

    transformer = InsertAliasedPostPiSoftmax()
    moe_tree = transformer.visit(moe_tree)
    assert transformer.mutations == 1
    moe_source = ast.unparse(ast.fix_missing_locations(moe_tree))
    audit_projection_l2_v1 = override_audit_sources(
        monkeypatch,
        moe_source=moe_source,
    )

    with pytest.raises(AssertionError, match="illegal post-Pi softmax"):
        audit_projection_l2_v1.audit_source_contract(tiny_config())


def test_token_permutation_only_permutes_router_probabilities():
    torch.manual_seed(10)
    router = CPTRouter(tiny_config(), layer_index=0)
    make_expert_kernel_distinct(router)
    hidden = torch.randn(2, 6, router.hidden_size)
    permutation = torch.tensor([4, 1, 5, 0, 3, 2])

    original = router(hidden)
    permuted = router(hidden[:, permutation])

    torch.testing.assert_close(
        permuted.probabilities,
        original.probabilities[:, permutation],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        permuted.proposal.load_sum,
        original.proposal.load_sum,
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.equal(
        permuted.proposal.token_count,
        original.proposal.token_count,
    )


def test_same_token_route_is_independent_of_prefix_and_sequence_length():
    torch.manual_seed(12)
    router = CPTRouter(tiny_config(), layer_index=0)
    make_expert_kernel_distinct(router)
    target = torch.randn(1, 1, router.hidden_size)
    prefix_a = torch.randn(1, 2, router.hidden_size)
    prefix_b = torch.randn(1, 4, router.hidden_size)
    suffix = torch.randn(1, 3, router.hidden_size)

    target_alone = router(target).probabilities[:, 0]
    after_prefix_a = router(
        torch.cat((prefix_a, target), dim=1)
    ).probabilities[:, -1]
    after_prefix_b = router(
        torch.cat((prefix_b, target), dim=1)
    ).probabilities[:, -1]
    before_suffix = router(torch.cat((target, suffix), dim=1)).probabilities[:, 0]

    for routed_target in (after_prefix_a, after_prefix_b, before_suffix):
        torch.testing.assert_close(
            routed_target,
            target_alone,
            atol=2e-6,
            rtol=2e-6,
        )


def test_padding_is_not_routed_and_does_not_affect_valid_routes_or_proposal():
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


def test_single_token_router_loss_only_gradients_the_same_token_hidden():
    torch.manual_seed(17)
    router = CPTRouter(tiny_config(), layer_index=0)
    hidden = torch.randn(1, 4, router.hidden_size, requires_grad=True)
    output = router(hidden)
    expert_weights = torch.tensor([0.25, -0.75, 1.25])
    (output.probabilities[:, -1] * expert_weights).sum().backward()

    assert torch.equal(hidden.grad[:, :-1], torch.zeros_like(hidden.grad[:, :-1]))
    assert torch.isfinite(hidden.grad[:, -1]).all()
    assert float(hidden.grad[:, -1].abs().sum()) > 0


def test_batch_rows_are_independent_and_suffix_changes_do_not_affect_prefix():
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
