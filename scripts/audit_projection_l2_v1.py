#!/usr/bin/env python3
# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

"""Audit CPT v1.3 while retaining the strict-v1 projection contract.

The filename is intentionally stable: v1.3 keeps the strict-v1
``P x -> stable L2`` path, but removes all sequence-local Router state.  The
numeric oracle below uses the theory's column-vector convention throughout.
"""

import argparse
import ast
import inspect
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from model.config import TinyMixtralConfig
from model.cpt_router import CPT_ROUTER_ALGORITHM_VERSION, CPTRouter
from model.modeling import SparseMoE


EXPECTED_ROUTER_VERSION = 3
LEGACY_SEQUENCE_STATE_FIELDS = frozenset(
    {
        "cpt_rho_beta",
        "cpt_beta_max",
        "cpt_kappa_beta",
        "cpt_lambda_sa",
        "cpt_state_step_size",
        "cpt_state_radius",
    }
)
LEGACY_SEQUENCE_STATE_TERMS = frozenset(
    {
        "short_state",
        "responsibility",
        "beta",
        "rho_beta",
        "beta_max",
        "kappa_beta",
        "lambda_sa",
        "state_step_size",
        "state_radius",
        "state_gradient",
        "responsibility_quality",
    }
)


def make_config() -> TinyMixtralConfig:
    """Build a small, non-degenerate Router configuration for the audit."""
    return TinyMixtralConfig(
        vocab_size=64,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        num_local_experts=3,
        num_experts_per_tok=2,
        expert_intermediate_size=16,
        cpt_projection_dim=4,
        cpt_init_seed=2026,
    )


@torch.no_grad()
def column_vector_oracle(
    router: CPTRouter,
    hidden_states: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return full-shape probabilities from the column-vector CPT equations.

    For the ``T`` valid tokens, ``X`` has shape ``[d, T]`` and every column is
    one token.  The calculation is deliberately independent from the
    row-major implementation in :meth:`CPTRouter.forward`.
    """
    batch_size, sequence_length, _ = hidden_states.shape
    flat_valid_indices = torch.nonzero(
        valid_mask.reshape(-1),
        as_tuple=False,
    ).flatten()
    full_probabilities = torch.zeros(
        batch_size * sequence_length,
        router.num_experts,
        device=hidden_states.device,
        dtype=torch.float32,
    )
    if flat_valid_indices.numel() == 0:
        return full_probabilities.view(
            batch_size,
            sequence_length,
            router.num_experts,
        )

    # Column-vector convention: X:[d,T], Z:[d_p,T], Q:[K,T].
    x = hidden_states[valid_mask].float().T.contiguous()
    projected = router.projection @ x
    z = projected / torch.linalg.vector_norm(
        projected,
        dim=0,
        keepdim=True,
    ).clamp_min(router.eps_z)
    normalized_anchors = router.anchors / torch.linalg.vector_norm(
        router.anchors,
        dim=0,
        keepdim=True,
    ).clamp_min(router.eps_m)
    q = torch.softmax(
        (normalized_anchors.T @ z) / router.prototype_temperature,
        dim=0,
        dtype=torch.float32,
    )

    # H_N right-centers each prototype row of Theta_C.  B is row-stochastic.
    centered_energy = router.energy - router.energy.mean(dim=-1, keepdim=True)
    b = torch.softmax(
        (
            centered_energy
            - router.congestion_price.detach().unsqueeze(0)
        )
        / router.expert_temperature,
        dim=-1,
        dtype=torch.float32,
    )
    pi = b.T @ q
    return full_probabilities.index_copy(
        0,
        flat_valid_indices,
        pi.T,
    ).view(batch_size, sequence_length, router.num_experts)


def _is_torch_softmax(value: object) -> bool:
    return any(
        value is candidate
        for candidate in (
            torch.softmax,
            torch.nn.functional.softmax,
            torch.nn.Softmax,
            torch.Tensor.softmax,
            torch.special.softmax,
        )
    ) or isinstance(value, torch.nn.Softmax)


def _is_softmax_reference(
    node: ast.AST,
    namespace: dict[str, object] | None = None,
) -> bool:
    if isinstance(node, ast.Attribute):
        identifiers = (node.attr,)
    elif isinstance(node, ast.Name):
        identifiers = (node.id,)
    elif isinstance(node, ast.alias):
        identifiers = (
            node.name.rsplit(".", 1)[-1],
            node.asname,
        )
    else:
        return False

    if any(
        identifier is not None and identifier.lower() == "softmax"
        for identifier in identifiers
    ):
        return True

    return (
        isinstance(node, ast.Name)
        and namespace is not None
        and _is_torch_softmax(namespace.get(node.id))
    )


def _is_softmax_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _is_softmax_reference(node.func)


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _is_self_attribute(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == name
        and _is_name(node.value, "self")
    )


def _is_call(node: ast.AST, owner: str, function: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == function
        and _is_name(node.func.value, owner)
    )


def _is_self_call(node: ast.AST, function: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == function
        and _is_name(node.func.value, "self")
    )


def _integer_literal(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return -node.operand.value
    return None


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _target_names(target: ast.AST) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(
            name
            for element in target.elts
            for name in _target_names(element)
        )
    return ()


@dataclass(frozen=True)
class _NameWrite:
    name: str
    value: ast.AST | None
    position: tuple[int, int]
    is_plain_assignment: bool


def _node_end_position(node: ast.AST) -> tuple[int, int]:
    return (
        getattr(node, "end_lineno", None) or getattr(node, "lineno", -1),
        getattr(node, "end_col_offset", None)
        or getattr(node, "col_offset", -1),
    )


def _node_start_position(node: ast.AST) -> tuple[int, int]:
    return (
        getattr(node, "lineno", -1),
        getattr(node, "col_offset", -1),
    )


def _name_writes(tree: ast.AST) -> list[_NameWrite]:
    writes: list[_NameWrite] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
            is_plain_assignment = True
        elif isinstance(node, ast.AnnAssign):
            if node.value is None:
                continue
            targets = [node.target]
            value = node.value
            is_plain_assignment = True
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
            value = node.value
            is_plain_assignment = False
        elif isinstance(node, ast.NamedExpr):
            targets = [node.target]
            value = node.value
            is_plain_assignment = False
        else:
            continue
        for target in targets:
            writes.extend(
                _NameWrite(
                    name=name,
                    value=value,
                    position=_node_end_position(node),
                    is_plain_assignment=is_plain_assignment,
                )
                for name in _target_names(target)
            )
    return sorted(writes, key=lambda write: write.position)


def _plain_assignments(writes: list[_NameWrite]) -> list[_NameWrite]:
    return [write for write in writes if write.is_plain_assignment]


def _reaches(
    binding: _NameWrite,
    use: ast.AST,
    writes: list[_NameWrite],
) -> bool:
    if not _is_name(use, binding.name):
        return False
    use_position = _node_start_position(use)
    prior_writes = [
        write
        for write in writes
        if write.name == binding.name and write.position < use_position
    ]
    return bool(prior_writes) and prior_writes[-1] is binding


def _call_argument(
    call: ast.Call,
    position: int,
    keyword: str,
) -> ast.AST | None:
    if len(call.args) > position:
        return call.args[position]
    return _keyword(call, keyword)


def _is_stable_l2_assignment(
    value: ast.AST,
    input_name: str | None,
    input_attribute: str | None,
    dim: int,
    eps_attribute: str,
) -> bool:
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "_stable_l2"
        and _is_name(value.func.value, "self")
        and len(value.args) == 1
    ):
        return False
    input_matches = (
        input_name is not None and _is_name(value.args[0], input_name)
    ) or (
        input_attribute is not None
        and _is_self_attribute(value.args[0], input_attribute)
    )
    return (
        input_matches
        and _integer_literal(_keyword(value, "dim")) == dim
        and _is_self_attribute(_keyword(value, "eps"), eps_attribute)
    )


def _is_divided_matmul(
    value: ast.AST,
    left_name: str,
    right_name: str,
    divisor_attribute: str,
) -> bool:
    return (
        isinstance(value, ast.BinOp)
        and isinstance(value.op, ast.Div)
        and isinstance(value.left, ast.BinOp)
        and isinstance(value.left.op, ast.MatMult)
        and _is_name(value.left.left, left_name)
        and _is_name(value.left.right, right_name)
        and _is_self_attribute(value.right, divisor_attribute)
    )


def _is_fp32_softmax(value: ast.AST, input_name: str, dim: int) -> bool:
    dtype = _keyword(value, "dtype") if isinstance(value, ast.Call) else None
    return (
        _is_call(value, "F", "softmax")
        and len(value.args) == 1
        and _is_name(value.args[0], input_name)
        and _integer_literal(_keyword(value, "dim")) == dim
        and isinstance(dtype, ast.Attribute)
        and _is_name(dtype.value, "torch")
        and dtype.attr == "float32"
    )


def audit_source_contract(config: TinyMixtralConfig) -> None:
    """Fail if the source reintroduces sequence-local routing semantics."""
    config_fields = set(config.cpt_config_dict())
    stale_fields = sorted(config_fields & LEGACY_SEQUENCE_STATE_FIELDS)
    if stale_fields:
        raise AssertionError(
            "legacy sequence-state config fields remain: " + ", ".join(stale_fields)
        )
    if config.num_experts_per_tok != 2:
        raise AssertionError("cpt_v1.3 source audit requires the Top-2 contract")

    router_source = textwrap.dedent(inspect.getsource(CPTRouter.forward))
    router_tree = ast.parse(router_source)
    router_identifiers = {
        node.id for node in ast.walk(router_tree) if isinstance(node, ast.Name)
    }
    router_identifiers.update(
        node.attr
        for node in ast.walk(router_tree)
        if isinstance(node, ast.Attribute)
    )
    stale_identifiers = sorted(
        router_identifiers & LEGACY_SEQUENCE_STATE_TERMS
    )
    if stale_identifiers:
        raise AssertionError(
            "legacy sequence-state identifiers remain in CPTRouter.forward: "
            + ", ".join(stale_identifiers)
        )
    lowered_source = router_source.lower()
    stale_source_terms = sorted(
        term for term in LEGACY_SEQUENCE_STATE_TERMS if term in lowered_source
    )
    if stale_source_terms:
        raise AssertionError(
            "legacy sequence-state source terms remain in CPTRouter.forward: "
            + ", ".join(stale_source_terms)
        )

    serial_nodes = (
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.comprehension,
    )
    if any(isinstance(node, serial_nodes) for node in ast.walk(router_tree)):
        raise AssertionError(
            "CPTRouter.forward must not scan or iterate over token positions"
        )
    if "range(sequence_length)" in lowered_source.replace(" ", ""):
        raise AssertionError("a sequence-length position scan remains")

    linear_calls = [
        node for node in ast.walk(router_tree) if _is_call(node, "F", "linear")
    ]
    if len(linear_calls) != 1:
        raise AssertionError("the strict-v1 projection must use exactly one F.linear")

    router_writes = _name_writes(router_tree)
    router_assignments = _plain_assignments(router_writes)
    projection_bindings = [
        binding
        for binding in router_assignments
        if binding.value is linear_calls[0]
    ]
    if len(projection_bindings) != 1:
        raise AssertionError("F.linear must have one direct projection binding")
    raw_projection = projection_bindings[0]

    normalized_projection_bindings = [
        binding
        for binding in router_assignments
        if _is_stable_l2_assignment(
            binding.value,
            input_name=raw_projection.name,
            input_attribute=None,
            dim=-1,
            eps_attribute="eps_z",
        )
        and _reaches(
            raw_projection,
            binding.value.args[0],
            router_writes,
        )
    ]
    if len(normalized_projection_bindings) != 1:
        raise AssertionError("projected tokens must use stable L2 normalization")
    projection = normalized_projection_bindings[0]

    normalized_anchor_bindings = [
        binding
        for binding in router_assignments
        if _is_stable_l2_assignment(
            binding.value,
            input_name=None,
            input_attribute="anchors",
            dim=0,
            eps_attribute="eps_m",
        )
    ]
    if len(normalized_anchor_bindings) != 1:
        raise AssertionError("anchors must use differentiable column L2 normalization")
    normalized_anchor = normalized_anchor_bindings[0]

    prototype_logit_bindings = [
        binding
        for binding in router_assignments
        if _is_divided_matmul(
            binding.value,
            left_name=projection.name,
            right_name=normalized_anchor.name,
            divisor_attribute="prototype_temperature",
        )
        and _reaches(
            projection,
            binding.value.left.left,
            router_writes,
        )
        and _reaches(
            normalized_anchor,
            binding.value.left.right,
            router_writes,
        )
    ]
    if len(prototype_logit_bindings) != 1:
        raise AssertionError("prototype_logits must have one authoritative assignment")
    prototype_logits = prototype_logit_bindings[0]

    prototype_softmax_bindings = [
        binding
        for binding in router_assignments
        if _is_fp32_softmax(
            binding.value,
            input_name=prototype_logits.name,
            dim=-1,
        )
        and _reaches(
            prototype_logits,
            binding.value.args[0],
            router_writes,
        )
    ]
    if len(prototype_softmax_bindings) != 1:
        raise AssertionError("prototype probabilities need one authoritative softmax")
    prototype_probabilities = prototype_softmax_bindings[0]

    for node in ast.walk(router_tree):
        if not _is_softmax_call(node) or not node.args:
            continue
        if _reaches(
            raw_projection,
            node.args[0],
            router_writes,
        ) or _reaches(
            projection,
            node.args[0],
            router_writes,
        ):
            raise AssertionError("projection-softmax was found in the strict-v1 path")

    flat_prototype_bindings = []
    for binding in router_assignments:
        value = binding.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "reshape"
            and _reaches(
                prototype_probabilities,
                value.func.value,
                router_writes,
            )
        ):
            flat_prototype_bindings.append(binding)
    if len(flat_prototype_bindings) != 1:
        raise AssertionError("Q must flow through one direct token reshape")
    flat_prototype_probabilities = flat_prototype_bindings[0]

    selected_prototype_bindings = []
    for binding in router_assignments:
        value = binding.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "index_select"
            and _reaches(
                flat_prototype_probabilities,
                value.func.value,
                router_writes,
            )
        ):
            selected_prototype_bindings.append(binding)
    if len(selected_prototype_bindings) != 1:
        raise AssertionError("Q must flow through one direct valid-token selection")
    selected_prototype_probabilities = selected_prototype_bindings[0]

    kernel_bindings = [
        binding
        for binding in router_assignments
        if _is_self_call(binding.value, "expert_kernel")
    ]
    if len(kernel_bindings) != 1:
        raise AssertionError("expert_kernel must have one direct binding")
    kernel = kernel_bindings[0]

    kernel_products = [
        node
        for node in ast.walk(router_tree)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.MatMult)
        and _reaches(kernel, node.right, router_writes)
    ]
    if len(kernel_products) != 1:
        raise AssertionError(
            "expert_kernel result must flow directly into exactly one matmul"
        )
    if not _reaches(
        selected_prototype_probabilities,
        kernel_products[0].left,
        router_writes,
    ):
        raise AssertionError(
            "prototype probabilities must flow directly into the kernel matmul"
        )

    moe_source = textwrap.dedent(inspect.getsource(SparseMoE.forward))
    moe_tree = ast.parse(moe_source)
    moe_writes = _name_writes(moe_tree)
    moe_assignments = _plain_assignments(moe_writes)
    router_output_bindings = [
        binding
        for binding in moe_assignments
        if _is_self_call(binding.value, "cpt_router")
    ]
    if len(router_output_bindings) != 1:
        raise AssertionError("SparseMoE must call CPT Router exactly once")
    router_output = router_output_bindings[0]

    all_weight_bindings = []
    for binding in moe_assignments:
        value = binding.value
        direct_probability_view = (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "view"
            and isinstance(value.func.value, ast.Attribute)
            and value.func.value.attr == "probabilities"
            and _reaches(
                router_output,
                value.func.value.value,
                moe_writes,
            )
        )
        if direct_probability_view:
            all_weight_bindings.append(binding)
    if len(all_weight_bindings) != 1:
        raise AssertionError("SparseMoE must bind CPT probabilities exactly once")
    all_weights = all_weight_bindings[0]

    routing_weight_bindings = []
    for binding in moe_assignments:
        value = binding.value
        direct_valid_selection = (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "to"
            and isinstance(value.func.value, ast.Call)
            and isinstance(value.func.value.func, ast.Attribute)
            and value.func.value.func.attr == "index_select"
            and _reaches(
                all_weights,
                value.func.value.func.value,
                moe_writes,
            )
        )
        if direct_valid_selection:
            routing_weight_bindings.append(binding)
    if len(routing_weight_bindings) != 1:
        raise AssertionError("routing_weights must have one direct CPT assignment")
    routing_weights = routing_weight_bindings[0]

    topk_calls = []
    for node in ast.walk(moe_tree):
        if not _is_call(node, "torch", "topk"):
            continue
        topk_input = _call_argument(node, 0, "input")
        if _reaches(routing_weights, topk_input, moe_writes):
            topk_calls.append(node)
    if len(topk_calls) != 1:
        raise AssertionError("CPT probabilities must feed exactly one direct Top-2")
    topk_call = topk_calls[0]
    if not _is_self_attribute(_call_argument(topk_call, 1, "k"), "top_k"):
        raise AssertionError("Top-2 must use self.top_k")
    if _integer_literal(_call_argument(topk_call, 2, "dim")) != -1:
        raise AssertionError("Top-2 must operate on the expert dimension")

    topk_assignments = [
        node
        for node in ast.walk(moe_tree)
        if isinstance(node, ast.Assign) and node.value is topk_call
    ]
    if len(topk_assignments) != 1:
        raise AssertionError("Top-2 results must have one direct tuple binding")
    topk_assignment = topk_assignments[0]
    if (
        len(topk_assignment.targets) != 1
        or not isinstance(topk_assignment.targets[0], (ast.Tuple, ast.List))
        or len(topk_assignment.targets[0].elts) != 2
        or not all(
            isinstance(element, ast.Name)
            for element in topk_assignment.targets[0].elts
        )
    ):
        raise AssertionError("Top-2 must bind weights and expert indices directly")

    moe_globals = SparseMoE.forward.__globals__
    if any(
        _is_softmax_reference(node, namespace=moe_globals)
        for node in ast.walk(moe_tree)
    ):
        raise AssertionError("SparseMoE applies an illegal post-Pi softmax")


def assert_finite_nonzero_gradient(name: str, tensor: torch.Tensor) -> None:
    """Require a live, finite, non-zero gradient for a trainable path."""
    gradient = tensor.grad
    if gradient is None:
        raise AssertionError(f"missing {name} gradient")
    if not bool(torch.isfinite(gradient).all()):
        raise AssertionError(f"non-finite {name} gradient")
    if not bool((gradient != 0).any()):
        raise AssertionError(f"zero {name} gradient")


def assert_fp32_router(router: CPTRouter) -> None:
    """Require all floating CPT state to remain in the FP32 precision island."""
    for name in ("projection", "anchors", "energy", "congestion_price"):
        value = getattr(router, name)
        if value.dtype != torch.float32:
            raise AssertionError(f"CPT {name} must remain FP32")


def run(device: torch.device) -> dict[str, object]:
    torch.manual_seed(0)
    config = make_config()
    if config.cpt_router_version != EXPECTED_ROUTER_VERSION:
        raise AssertionError("config must select cpt_router_version=3")
    if CPT_ROUTER_ALGORITHM_VERSION != EXPECTED_ROUTER_VERSION:
        raise AssertionError("CPT Router algorithm constant must equal 3")

    audit_source_contract(config)
    router = CPTRouter(config, layer_index=0).to(device)
    if int(router.router_algorithm_version.item()) != EXPECTED_ROUTER_VERSION:
        raise AssertionError("Router algorithm-version buffer must equal 3")
    assert_fp32_router(router)

    # Non-unit, differently scaled columns make forward anchor normalization
    # observable instead of relying on the unit-norm initialization invariant.
    with torch.no_grad():
        anchor_scales = torch.linspace(
            0.35,
            1.65,
            router.num_prototypes,
            device=device,
            dtype=torch.float32,
        )
        router.anchors.mul_(anchor_scales.unsqueeze(0))
        router.congestion_price.copy_(
            torch.linspace(
                0.0,
                0.25,
                router.num_experts,
                device=device,
                dtype=torch.float32,
            )
        )

    hidden = torch.randn(
        3,
        7,
        config.hidden_size,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    mask = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
        device=device,
    )
    valid_tokens = int(mask.sum().item())

    output = router(hidden, mask)
    expected = column_vector_oracle(router, hidden, mask)
    torch.testing.assert_close(
        output.probabilities,
        expected,
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        output.probabilities[mask].sum(dim=-1),
        torch.ones(valid_tokens, device=device, dtype=torch.float32),
        atol=2e-6,
        rtol=0,
    )
    if not torch.equal(
        output.probabilities[~mask],
        torch.zeros_like(output.probabilities[~mask]),
    ):
        raise AssertionError("padding positions must be exactly zero")
    if int(output.proposal.token_count.item()) != valid_tokens:
        raise AssertionError("padding was included in CPT token_count")
    torch.testing.assert_close(
        output.proposal.load_sum,
        expected[mask].sum(dim=0),
        atol=2e-5,
        rtol=2e-6,
    )

    # A token permutation must only permute the corresponding output rows.
    permutation = torch.tensor([5, 2, 0, 6, 1, 4, 3], device=device)
    inverse_permutation = torch.argsort(permutation)
    permuted_output = router(
        hidden.detach().index_select(1, permutation),
        mask.index_select(1, permutation),
    )
    torch.testing.assert_close(
        permuted_output.probabilities.index_select(1, inverse_permutation),
        output.probabilities.detach(),
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        permuted_output.proposal.load_sum,
        output.proposal.load_sum,
        atol=2e-5,
        rtol=2e-6,
    )

    # Changing a valid prefix cannot change routing of later token positions.
    prefix_length = 3
    prefix_variant = hidden.detach().clone()
    prefix_offset = torch.linspace(
        -2.0,
        3.0,
        config.hidden_size,
        device=device,
        dtype=torch.float32,
    )
    prefix_variant[:, :prefix_length] = (
        -1.75 * prefix_variant[:, :prefix_length]
        + prefix_offset.view(1, 1, -1)
    )
    prefix_output = router(prefix_variant, mask)
    torch.testing.assert_close(
        prefix_output.probabilities[:, prefix_length:],
        output.probabilities.detach()[:, prefix_length:],
        atol=2e-6,
        rtol=2e-6,
    )

    # NaNs in masked positions, including an all-padding row, are quarantined.
    poisoned_hidden = hidden.detach().clone()
    poisoned_hidden[~mask] = torch.nan
    poisoned_output = router(poisoned_hidden, mask)
    torch.testing.assert_close(
        poisoned_output.probabilities,
        output.probabilities.detach(),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        poisoned_output.proposal.load_sum,
        output.proposal.load_sum,
        atol=0,
        rtol=0,
    )
    if not bool(torch.isfinite(poisoned_output.probabilities).all()):
        raise AssertionError("masked NaNs escaped into Router probabilities")

    all_padding_hidden = torch.full(
        (1, 4, config.hidden_size),
        torch.nan,
        device=device,
        dtype=torch.float32,
    )
    all_padding_mask = torch.zeros(1, 4, device=device, dtype=torch.bool)
    all_padding_output = router(all_padding_hidden, all_padding_mask)
    if not torch.equal(
        all_padding_output.probabilities,
        torch.zeros_like(all_padding_output.probabilities),
    ):
        raise AssertionError("all-padding probabilities must be exactly zero")
    if int(all_padding_output.proposal.token_count.item()) != 0:
        raise AssertionError("all-padding token_count must be zero")
    if not torch.equal(
        all_padding_output.proposal.load_sum,
        torch.zeros_like(all_padding_output.proposal.load_sum),
    ):
        raise AssertionError("all-padding load_sum must be exactly zero")

    expert_weights = torch.tensor([0.25, -0.5, 1.0], device=device)
    (output.probabilities * expert_weights).sum().backward()
    for name, tensor in (
        ("hidden", hidden),
        ("projection", router.projection),
        ("anchors", router.anchors),
        ("energy", router.energy),
    ):
        assert_finite_nonzero_gradient(name, tensor)
    if not torch.equal(hidden.grad[~mask], torch.zeros_like(hidden.grad[~mask])):
        raise AssertionError("masked tokens must have exactly zero hidden gradient")

    # ``Module.to(dtype=...)`` must not puncture the Router precision island.
    precision_router = CPTRouter(config, layer_index=0).to(
        device=device,
        dtype=torch.bfloat16,
    )
    assert_fp32_router(precision_router)
    precision_hidden = hidden.detach().to(torch.bfloat16)
    if device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            precision_output = precision_router(precision_hidden, mask)
    else:
        precision_output = precision_router(precision_hidden, mask)
    if precision_output.probabilities.dtype != torch.float32:
        raise AssertionError("CPT probabilities must remain FP32")
    if precision_output.proposal.load_sum.dtype != torch.float32:
        raise AssertionError("CPT load proposal must remain FP32")

    return {
        "anchor_path": "A -> column L2(eps_m)",
        "column_vector_oracle": "Pi = B^T Q",
        "cpt_router_version": config.cpt_router_version,
        "device": str(device),
        "masked_nan_isolated": True,
        "num_experts": config.num_local_experts,
        "num_prototypes": config.cpt_num_prototypes,
        "post_pi_softmax": False,
        "prefix_independent": True,
        "projection_path": "P X -> column L2(eps_z)",
        "router_algorithm_version": int(router.router_algorithm_version.item()),
        "sequence_local_state": False,
        "status": "passed",
        "token_parallel": True,
        "token_permutation_equivariant": True,
        "valid_tokens": valid_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the CPT v1.3 long-state-only Router contract."
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    print(json.dumps(run(device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
