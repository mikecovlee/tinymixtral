from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from hf.configuration_tinymixtral import TinyMixtralConfig as HFTinyMixtralConfig
from hf.modeling_tinymixtral import SparseMoE as HFSparseMoE
from model.config import TinyMixtralConfig as NativeTinyMixtralConfig
from model.modeling import SparseMoE as NativeSparseMoE


def _config_kwargs() -> dict:
    return {
        "vocab_size": 32,
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "max_position_embeddings": 16,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "expert_intermediate_size": 12,
        "cpt_num_prototypes": 4,
        "cpt_projection_dim": 4,
        "cpt_init_seed": 23,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
    }


def _new_module(implementation: str):
    kwargs = _config_kwargs()
    if implementation == "native":
        return NativeSparseMoE(NativeTinyMixtralConfig(**kwargs), layer_index=0)
    if implementation == "hf":
        return HFSparseMoE(HFTinyMixtralConfig(**kwargs), layer_index=0)
    raise AssertionError(f"unknown implementation {implementation!r}")


def _matched_modules(implementation: str):
    torch.manual_seed(20260731)
    reference = _new_module(implementation).train()
    actual = _new_module(implementation).train()
    result = actual.load_state_dict(reference.state_dict(), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    return reference, actual


def _routing_case(case: str):
    hidden_states = torch.linspace(-0.9, 0.8, 48, dtype=torch.float32).reshape(
        2,
        3,
        8,
    )
    valid_indices = torch.tensor([0, 1, 3, 5], dtype=torch.int64)
    if case == "all_experts":
        probabilities = torch.tensor(
            [
                [0.40, 0.30, 0.20, 0.10],
                [0.10, 0.40, 0.30, 0.20],
                [0.20, 0.10, 0.40, 0.30],
                [0.30, 0.20, 0.10, 0.40],
            ],
            dtype=torch.float32,
        )
    elif case == "empty_experts":
        # Experts 2 and 3 receive no Top-2 selection.  The production path must
        # still execute their zero-row matmuls without querying mask.any().
        probabilities = torch.tensor(
            [
                [0.55, 0.35, 0.06, 0.04],
                [0.52, 0.38, 0.06, 0.04],
                [0.51, 0.39, 0.06, 0.04],
                [0.50, 0.40, 0.06, 0.04],
            ],
            dtype=torch.float32,
        )
    else:
        raise AssertionError(f"unknown routing case {case!r}")
    coefficient = torch.linspace(
        -0.7,
        0.9,
        hidden_states.numel(),
        dtype=torch.float32,
    ).reshape_as(hidden_states)
    return hidden_states, probabilities, valid_indices, coefficient


def _reference_expert_forward(module, x, probabilities, valid_indices):
    """Exact pre-refactor dispatch, including the host-synchronizing branch."""
    batch_size, seq_len, hidden_size = x.shape
    if valid_indices.numel() == 0:
        return torch.zeros_like(x)

    x_flat = x.reshape(-1, hidden_size)
    valid_states = x_flat.index_select(0, valid_indices)
    selected_weights, selected_experts = torch.topk(
        probabilities,
        module.top_k,
        dim=-1,
    )
    selected_weights = selected_weights / selected_weights.sum(
        dim=-1,
        keepdim=True,
    )
    valid_output = torch.zeros_like(valid_states)
    for slot in range(module.top_k):
        expert_index = selected_experts[:, slot]
        slot_weight = selected_weights[:, slot]
        for expert in range(module.num_experts):
            token_mask = expert_index == expert
            if not token_mask.any():
                continue
            token_states = valid_states[token_mask]
            gate = F.silu(token_states @ module.gate_proj[expert].T)
            up = token_states @ module.up_proj[expert].T
            expert_output = (gate * up) @ module.down_proj[expert].T
            valid_output[token_mask] += expert_output * slot_weight[
                token_mask
            ].to(expert_output.dtype).unsqueeze(-1)

    output_flat = torch.zeros_like(x_flat).index_copy(
        0,
        valid_indices,
        valid_output,
    )
    return output_flat.reshape(batch_size, seq_len, hidden_size)


def _run_backward(
    module,
    expert_function,
    hidden_states,
    probabilities,
    valid_indices,
    coefficient,
    *,
    checkpointed: bool = False,
    cpu_bf16_autocast: bool = False,
):
    module.zero_grad(set_to_none=True)
    hidden = hidden_states.detach().clone().requires_grad_(True)
    routing = probabilities.detach().clone().requires_grad_(True)

    def apply_experts(hidden_arg, routing_arg):
        return expert_function(module, hidden_arg, routing_arg, valid_indices)

    autocast_context = (
        torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        if cpu_bf16_autocast
        else nullcontext()
    )
    with autocast_context:
        if checkpointed:
            output = checkpoint(
                apply_experts,
                hidden,
                routing,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            output = apply_experts(hidden, routing)
        loss = (output.float() * coefficient).sum()
    loss.backward()

    return {
        "output": output.detach().float(),
        "hidden_grad": hidden.grad.detach().float(),
        "routing_grad": routing.grad.detach().float(),
        "gate_grad": module.gate_proj.grad.detach().float(),
        "up_grad": module.up_proj.grad.detach().float(),
        "down_grad": module.down_proj.grad.detach().float(),
    }


def _production_expert_function(module, x, probabilities, valid_indices):
    return module._expert_forward(x, probabilities, valid_indices)


def _assert_results_close(left, right, *, atol=1e-7, rtol=1e-6):
    assert left.keys() == right.keys()
    for name in left:
        torch.testing.assert_close(
            left[name],
            right[name],
            atol=atol,
            rtol=rtol,
            msg=lambda message, field=name: f"{field}: {message}",
        )


@pytest.mark.parametrize("implementation", ["native", "hf"])
@pytest.mark.parametrize("case", ["all_experts", "empty_experts"])
def test_expert_dispatch_matches_pre_refactor_forward_and_all_gradients(
    implementation,
    case,
):
    reference, actual = _matched_modules(implementation)
    hidden, probabilities, valid_indices, coefficient = _routing_case(case)

    expected = _run_backward(
        reference,
        _reference_expert_forward,
        hidden,
        probabilities,
        valid_indices,
        coefficient,
    )
    observed = _run_backward(
        actual,
        _production_expert_function,
        hidden,
        probabilities,
        valid_indices,
        coefficient,
    )

    _assert_results_close(expected, observed)
    if case == "empty_experts":
        torch.testing.assert_close(
            observed["gate_grad"][2:],
            torch.zeros_like(observed["gate_grad"][2:]),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            observed["up_grad"][2:],
            torch.zeros_like(observed["up_grad"][2:]),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            observed["down_grad"][2:],
            torch.zeros_like(observed["down_grad"][2:]),
            atol=0.0,
            rtol=0.0,
        )


@pytest.mark.parametrize("implementation", ["native", "hf"])
def test_expert_dispatch_all_padding_keeps_zero_output(implementation):
    module = _new_module(implementation).train()
    hidden = torch.randn(2, 3, module.hidden_size)
    probabilities = torch.empty(0, module.num_experts)
    valid_indices = torch.empty(0, dtype=torch.int64)

    expected = _reference_expert_forward(
        module,
        hidden,
        probabilities,
        valid_indices,
    )
    observed = module._expert_forward(hidden, probabilities, valid_indices)

    torch.testing.assert_close(observed, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(observed, torch.zeros_like(hidden), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("implementation", ["native", "hf"])
@pytest.mark.parametrize("empty_valid_set", [False, True])
def test_production_expert_hot_path_never_uses_tensor_any_or_host_bool(
    implementation,
    empty_valid_set,
    monkeypatch,
):
    module = _new_module(implementation).train()
    hidden, probabilities, valid_indices, _ = _routing_case("empty_experts")
    if empty_valid_set:
        probabilities = probabilities[:0]
        valid_indices = valid_indices[:0]

    def forbidden_any(*args, **kwargs):
        raise AssertionError("production expert dispatch called Tensor.any()")

    def forbidden_bool(*args, **kwargs):
        raise AssertionError("production expert dispatch evaluated a Tensor as bool")

    with monkeypatch.context() as patcher:
        patcher.setattr(torch.Tensor, "any", forbidden_any)
        patcher.setattr(torch.Tensor, "__bool__", forbidden_bool)
        observed = module._expert_forward(hidden, probabilities, valid_indices)

    assert observed.shape == hidden.shape


@pytest.mark.parametrize("implementation", ["native", "hf"])
def test_expert_dispatch_activation_checkpoint_recompute_parity(implementation):
    plain, recomputed = _matched_modules(implementation)
    hidden, probabilities, valid_indices, coefficient = _routing_case("empty_experts")

    expected = _run_backward(
        plain,
        _production_expert_function,
        hidden,
        probabilities,
        valid_indices,
        coefficient,
        checkpointed=False,
    )
    observed = _run_backward(
        recomputed,
        _production_expert_function,
        hidden,
        probabilities,
        valid_indices,
        coefficient,
        checkpointed=True,
    )

    _assert_results_close(expected, observed)


def _cpu_bf16_autocast_available() -> bool:
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            value = torch.ones(2, 2) @ torch.ones(2, 2)
        return value.dtype == torch.bfloat16
    except (RuntimeError, TypeError):
        return False


@pytest.mark.skipif(
    not _cpu_bf16_autocast_available(),
    reason="CPU BF16 autocast is unavailable",
)
@pytest.mark.parametrize("implementation", ["native", "hf"])
def test_expert_dispatch_cpu_bf16_autocast_matches_reference(implementation):
    reference, actual = _matched_modules(implementation)
    hidden, probabilities, valid_indices, coefficient = _routing_case("empty_experts")

    expected = _run_backward(
        reference,
        _reference_expert_forward,
        hidden,
        probabilities,
        valid_indices,
        coefficient,
        cpu_bf16_autocast=True,
    )
    observed = _run_backward(
        actual,
        _production_expert_function,
        hidden,
        probabilities,
        valid_indices,
        coefficient,
        cpu_bf16_autocast=True,
    )

    _assert_results_close(expected, observed, atol=2e-3, rtol=2e-3)


def test_native_and_hf_expert_dispatch_remain_exactly_aligned():
    torch.manual_seed(20260731)
    native = _new_module("native").train()
    hf = _new_module("hf").train()
    result = hf.load_state_dict(native.state_dict(), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    hidden, probabilities, valid_indices, coefficient = _routing_case("empty_experts")

    native_result = _run_backward(
        native,
        _production_expert_function,
        hidden,
        probabilities,
        valid_indices,
        coefficient,
    )
    hf_result = _run_backward(
        hf,
        _production_expert_function,
        hidden,
        probabilities,
        valid_indices,
        coefficient,
    )

    _assert_results_close(native_result, hf_result, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("implementation", ["native", "hf"])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_expert_dispatch_cuda_bf16_empty_experts_forward_backward(
    implementation,
    checkpointed,
):
    """Exercise zero-row expert matmuls on the real CUDA BF16 path."""
    reference, actual = _matched_modules(implementation)
    reference = reference.to(device="cuda", dtype=torch.bfloat16)
    actual = actual.to(device="cuda", dtype=torch.bfloat16)
    hidden, probabilities, valid_indices, coefficient = _routing_case(
        "empty_experts"
    )
    hidden = hidden.to(device="cuda", dtype=torch.bfloat16)
    probabilities = probabilities.to(device="cuda")
    valid_indices = valid_indices.to(device="cuda")
    coefficient = coefficient.to(device="cuda")

    expected = _run_backward(
        reference,
        _reference_expert_forward,
        hidden,
        probabilities,
        valid_indices,
        coefficient,
        checkpointed=checkpointed,
    )
    observed = _run_backward(
        actual,
        _production_expert_function,
        hidden,
        probabilities,
        valid_indices,
        coefficient,
        checkpointed=checkpointed,
    )

    _assert_results_close(expected, observed, atol=0.0, rtol=0.0)
    for gradient_name in ("gate_grad", "up_grad", "down_grad"):
        assert torch.isfinite(observed[gradient_name]).all()
        torch.testing.assert_close(
            observed[gradient_name][2:],
            torch.zeros_like(observed[gradient_name][2:]),
            atol=0.0,
            rtol=0.0,
        )
