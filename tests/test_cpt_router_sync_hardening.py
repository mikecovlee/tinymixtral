import re

import pytest
import torch

import model.cpt_router as cpt_router_module
from model.config import TinyMixtralConfig
from model.cpt_router import CPTLayerProposal, CPTRouter
from model.modeling import TinyMixtralForCausalLM


def _config(*, layers: int = 6) -> TinyMixtralConfig:
    return TinyMixtralConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        num_local_experts=4,
        num_experts_per_tok=2,
        expert_intermediate_size=16,
        cpt_num_prototypes=4,
        cpt_projection_dim=4,
    )


def _model(*, layers: int = 6, device: str = "cpu") -> TinyMixtralForCausalLM:
    return TinyMixtralForCausalLM(_config(layers=layers)).to(device).train()


def _manual_proposals(
    model: TinyMixtralForCausalLM,
    *,
    load: tuple[float, ...] = (3.0, 1.0, 0.0, 0.0),
    count: int = 4,
) -> tuple[CPTLayerProposal, ...]:
    proposals = []
    for layer_index, router in enumerate(model._cpt_routers()):
        device = router.congestion_price.device
        proposals.append(
            CPTLayerProposal(
                layer_index=layer_index,
                load_sum=torch.tensor(load, device=device, dtype=torch.float32),
                token_count=torch.tensor(count, device=device, dtype=torch.int64),
                state_version=router.state_version.detach().clone(),
                valid=torch.tensor(True, device=device, dtype=torch.bool),
            )
        )
    return tuple(proposals)


def _clone_proposals(
    proposals: tuple[CPTLayerProposal, ...],
) -> list[CPTLayerProposal]:
    return [
        CPTLayerProposal(
            layer_index=proposal.layer_index,
            load_sum=proposal.load_sum.detach().clone(),
            token_count=proposal.token_count.detach().clone(),
            state_version=proposal.state_version.detach().clone(),
            valid=proposal.valid.detach().clone(),
        )
        for proposal in proposals
    ]


def _count_tensor_items(monkeypatch):
    original_item = torch.Tensor.item
    devices = []

    def counted_item(tensor, *args, **kwargs):
        devices.append(str(tensor.device))
        return original_item(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", counted_item)
    return devices


def _prepared_transaction(*, layers: int = 6, device: str = "cpu"):
    model = _model(layers=layers, device=device)
    transaction = model.prepare_cpt_transaction(
        _manual_proposals(model),
        microbatch_id="prepared",
    )
    model.validate_cpt_transaction(transaction)
    return model, transaction


def test_router_forward_keeps_simplex_validation_on_device(monkeypatch) -> None:
    router = CPTRouter(_config(layers=1), layer_index=0).train()

    def forbidden_allclose(*args, **kwargs):
        raise AssertionError("router forward must not call torch.allclose")

    monkeypatch.setattr(torch, "allclose", forbidden_allclose)
    hidden_states = torch.randn(2, 5, router.hidden_size, requires_grad=True)
    output = router(hidden_states)

    assert output.proposal.valid.ndim == 0
    assert output.proposal.valid.dtype == torch.bool
    assert output.proposal.valid.device == hidden_states.device
    assert bool(output.proposal.valid.item())
    output.probabilities.square().sum().backward()
    assert router.projection.grad is not None
    assert torch.isfinite(router.projection.grad).all()


def test_multilayer_proposal_validation_uses_one_success_sync(monkeypatch) -> None:
    model = _model(layers=8)
    proposals = _manual_proposals(model)
    item_devices = _count_tensor_items(monkeypatch)

    loads, counts = model._validate_proposal_tuple(
        proposals,
        model._cpt_routers(),
        context="sync-test",
    )

    assert loads.shape == (8, 4)
    assert counts.shape == (8,)
    assert item_devices == ["cpu"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_multilayer_cuda_proposal_validation_uses_one_success_sync(
    monkeypatch,
) -> None:
    model = _model(layers=8, device="cuda")
    proposals = _manual_proposals(model)
    item_devices = _count_tensor_items(monkeypatch)

    model._validate_proposal_tuple(
        proposals,
        model._cpt_routers(),
        context="cuda-sync-test",
    )

    assert item_devices == [str(model._cpt_routers()[0].congestion_price.device)]


def test_proposal_tuple_equality_uses_one_success_sync(monkeypatch) -> None:
    model = _model(layers=8)
    left = _manual_proposals(model)
    right = tuple(_clone_proposals(left))
    item_devices = _count_tensor_items(monkeypatch)

    assert model._proposal_tuples_equal(left, right)
    assert item_devices == ["cpu"]


def test_state_version_read_avoids_per_layer_tensor_item(monkeypatch) -> None:
    model = _model(layers=8)

    def forbidden_item(*args, **kwargs):
        raise AssertionError("get_cpt_state_version must not call Tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", forbidden_item)
    assert model.get_cpt_state_version() == 0
    with torch.no_grad():
        model._cpt_routers()[-1].state_version.add_(1)
    with pytest.raises(RuntimeError, match="CPT layer versions disagree"):
        model.get_cpt_state_version()


def test_microbatch_replay_remains_idempotent_with_constant_sync_count(
    monkeypatch,
) -> None:
    model = _model(layers=8)
    proposals = _manual_proposals(model)
    transaction = model.prepare_cpt_transaction(
        proposals,
        microbatch_id="stable",
    )
    item_devices = _count_tensor_items(monkeypatch)

    assert (
        model.accumulate_cpt_transaction(
            transaction,
            proposals,
            microbatch_id="stable",
        )
        is transaction
    )
    assert item_devices == ["cpu", "cpu"]

    conflicting = _clone_proposals(proposals)
    conflicting[0].load_sum[0] -= 0.5
    conflicting[0].load_sum[1] += 0.5
    with pytest.raises(RuntimeError, match="reused with different raw statistics"):
        model.accumulate_cpt_transaction(
            transaction,
            conflicting,
            microbatch_id="stable",
        )
    assert item_devices == ["cpu", "cpu", "cpu", "cpu"]


@pytest.mark.parametrize(
    "case, message",
    [
        ("active_versions", "CPT layer versions disagree before commit"),
        ("layer_index", "proposal layer index 9 does not match expected 1"),
        ("invalid", "layer 1 CPT proposal is invalid"),
        ("proposal_version", "layer 1 proposal version 1 does not match active version 0"),
        ("nan_load", "layer 1 CPT load is non-finite"),
        ("inf_load", "layer 1 CPT load is non-finite"),
        ("negative_load", "layer 1 CPT load is negative"),
        ("negative_count", "layer 1 CPT count is negative"),
        ("mass_mismatch", "layer 1 CPT probability mass does not match token_count"),
        ("layer_count", "CPT layers disagree on route-valid token_count"),
    ],
)
def test_proposal_fast_gate_preserves_ordered_diagnostics(case, message) -> None:
    model = _model(layers=3)
    proposals = _clone_proposals(_manual_proposals(model))

    if case == "active_versions":
        with torch.no_grad():
            model._cpt_routers()[1].state_version.add_(1)
    elif case == "layer_index":
        proposals[1].layer_index = 9
    elif case == "invalid":
        proposals[1].valid.fill_(False)
    elif case == "proposal_version":
        proposals[1].state_version.add_(1)
    elif case == "nan_load":
        proposals[1].load_sum[0] = float("nan")
    elif case == "inf_load":
        proposals[1].load_sum[0] = float("inf")
    elif case == "negative_load":
        proposals[1].load_sum.copy_(torch.tensor([-0.1, 4.1, 0.0, 0.0]))
    elif case == "negative_count":
        proposals[1].load_sum.zero_()
        proposals[1].token_count.fill_(-1)
    elif case == "mass_mismatch":
        proposals[1].load_sum.fill_(0.0)
    elif case == "layer_count":
        proposals[1].load_sum.copy_(torch.tensor([3.0, 0.0, 0.0, 0.0]))
        proposals[1].token_count.fill_(3)
    else:
        raise AssertionError(f"unhandled case {case}")

    with pytest.raises(RuntimeError, match=re.escape(message)):
        model._validate_proposal_tuple(
            proposals,
            model._cpt_routers(),
            context="diagnostic-test",
        )


@pytest.mark.parametrize(
    "case, message",
    [
        ("nan_load", "global CPT load is non-finite"),
        ("inf_load", "global CPT load is non-finite"),
        ("negative_load", "global CPT load is negative"),
        ("negative_count", "global CPT token_count is negative"),
        ("mass_mismatch", "global layer 1 CPT probability mass does not match token_count"),
        ("layer_count", "global CPT layers disagree on route-valid token_count"),
    ],
)
def test_global_statistics_fast_gate_preserves_ordered_diagnostics(
    case,
    message,
) -> None:
    loads = torch.tensor(
        [[3.0, 1.0, 0.0, 0.0], [3.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    counts = torch.tensor([4, 4], dtype=torch.int64)

    if case == "nan_load":
        loads[1, 0] = float("nan")
    elif case == "inf_load":
        loads[1, 0] = float("inf")
    elif case == "negative_load":
        loads[1].copy_(torch.tensor([-0.1, 4.1, 0.0, 0.0]))
    elif case == "negative_count":
        loads[1].zero_()
        counts[1] = -1
    elif case == "mass_mismatch":
        loads[1].zero_()
    elif case == "layer_count":
        loads[1].copy_(torch.tensor([3.0, 0.0, 0.0, 0.0]))
        counts[1] = 3
    else:
        raise AssertionError(f"unhandled case {case}")

    with pytest.raises(RuntimeError, match=re.escape(message)):
        cpt_router_module.CPTModelTransactionMixin._validate_global_statistics(
            loads,
            counts,
        )


def test_initial_transaction_validation_sync_count_is_layer_constant(
    monkeypatch,
) -> None:
    small_model = _model(layers=2)
    large_model = _model(layers=8)
    small_transaction = small_model.prepare_cpt_transaction(
        _manual_proposals(small_model),
        microbatch_id="small",
    )
    large_transaction = large_model.prepare_cpt_transaction(
        _manual_proposals(large_model),
        microbatch_id="large",
    )
    item_devices = _count_tensor_items(monkeypatch)

    small_model.validate_cpt_transaction(small_transaction)
    small_count = len(item_devices)
    item_devices.clear()
    large_model.validate_cpt_transaction(large_transaction)
    large_count = len(item_devices)

    assert small_count == large_count == 6
    assert item_devices == ["cpu"] * 6


def test_prepared_snapshot_success_sync_count_is_layer_constant(
    monkeypatch,
) -> None:
    model, transaction = _prepared_transaction(layers=8)
    item_devices = _count_tensor_items(monkeypatch)

    model._validate_prepared_snapshot(
        transaction,
        model._cpt_routers(),
        transaction._prepared_global_loads,
        transaction._prepared_global_counts,
    )

    assert item_devices == ["cpu"] * 4


@pytest.mark.parametrize(
    "case, message",
    [
        ("price_nan", "layer 0 prepared CPT price is invalid"),
        ("price_negative", "layer 0 prepared CPT price is invalid"),
        ("formula", "layer 0 prepared CPT price does not match its immutable raw-statistics formula"),
        ("base_version", "CPT router versions changed after price preparation"),
        ("active_price", "layer 0 active CPT price changed after preparation"),
        ("global_count", "CPT global token counts changed after preparation"),
        ("global_load", "CPT global raw loads changed after preparation"),
    ],
)
def test_prepared_snapshot_fast_gates_preserve_tamper_diagnostics(
    case,
    message,
) -> None:
    model, transaction = _prepared_transaction(layers=3)
    global_loads = transaction._prepared_global_loads.detach().clone()
    global_counts = transaction._prepared_global_counts.detach().clone()

    if case in {"price_nan", "price_negative", "formula"}:
        prices = list(transaction._prepared_prices)
        prices[0] = prices[0].detach().clone()
        if case == "price_nan":
            prices[0][0] = float("nan")
        elif case == "price_negative":
            prices[0][0] = -0.1
        else:
            prices[0][0] += 0.25
        transaction._prepared_prices = tuple(prices)
    elif case == "base_version":
        versions = list(transaction._prepared_base_versions)
        versions[0] += 1
        transaction._prepared_base_versions = tuple(versions)
    elif case == "active_price":
        with torch.no_grad():
            model._cpt_routers()[0].congestion_price.add_(0.1)
    elif case == "global_count":
        transaction._prepared_global_counts = global_counts.detach().clone()
        transaction._prepared_global_counts[0] += 1
    elif case == "global_load":
        transaction._prepared_global_loads = global_loads.detach().clone()
        transaction._prepared_global_loads[0, 0] += 0.25
    else:
        raise AssertionError(f"unhandled case {case}")

    with pytest.raises(RuntimeError, match=re.escape(message)):
        model._validate_prepared_snapshot(
            transaction,
            model._cpt_routers(),
            global_loads,
            global_counts,
        )


def test_zero_token_price_candidate_is_exact_and_uses_one_success_sync(
    monkeypatch,
) -> None:
    model = _model(layers=8)
    routers = model._cpt_routers()
    loads = torch.zeros(8, 4, dtype=torch.float32)
    counts = torch.zeros(8, dtype=torch.int64)
    base_prices = tuple(
        torch.linspace(0.0, 0.3, 4, dtype=torch.float32)
        for _ in routers
    )
    item_devices = _count_tensor_items(monkeypatch)

    candidates = model._compute_price_candidates(
        routers,
        loads,
        counts,
        base_prices,
    )

    assert item_devices == ["cpu"]
    for candidate, base_price in zip(candidates, base_prices):
        assert torch.equal(candidate, base_price)


def test_post_optimizer_router_checks_have_layer_constant_sync_count(
    monkeypatch,
) -> None:
    small_model = _model(layers=2)
    large_model = _model(layers=8)
    item_devices = _count_tensor_items(monkeypatch)

    small_model._validate_learnable_router_state(small_model._cpt_routers())
    small_normalized = small_model._validated_normalized_anchors(
        small_model._cpt_routers()
    )
    small_count = len(item_devices)
    item_devices.clear()
    large_model._validate_learnable_router_state(large_model._cpt_routers())
    large_normalized = large_model._validated_normalized_anchors(
        large_model._cpt_routers()
    )
    large_count = len(item_devices)

    assert small_count == large_count == 2
    assert item_devices == ["cpu", "cpu"]
    for normalized in (*small_normalized, *large_normalized):
        torch.testing.assert_close(
            normalized.norm(dim=0),
            torch.ones(normalized.shape[1]),
            rtol=1e-6,
            atol=1e-6,
        )


def test_post_optimizer_router_fast_gates_preserve_anchor_diagnostics() -> None:
    model = _model(layers=3)
    with torch.no_grad():
        model._cpt_routers()[1].anchors.zero_()
    with pytest.raises(RuntimeError, match="layer 1 CPT anchor norm is zero"):
        model._validated_normalized_anchors(model._cpt_routers())

    model = _model(layers=3)
    with torch.no_grad():
        model._cpt_routers()[0].anchors.fill_(torch.finfo(torch.float32).max)
    with pytest.raises(RuntimeError, match="layer 0 CPT anchors are non-finite"):
        model._validated_normalized_anchors(model._cpt_routers())


def test_post_optimizer_learnable_state_fast_gate_preserves_first_error() -> None:
    model = _model(layers=3)
    with torch.no_grad():
        model._cpt_routers()[1].projection[0, 0] = float("nan")
        model._cpt_routers()[2].energy[0, 0] = float("inf")
    with pytest.raises(
        RuntimeError,
        match="layer 1 CPT projection is non-finite after optimizer step",
    ):
        model._validate_learnable_router_state(model._cpt_routers())


def test_commit_revalidates_prepared_snapshot_before_state_advance() -> None:
    model, transaction = _prepared_transaction(layers=3)
    tampered = list(transaction._prepared_prices)
    tampered[0] = tampered[0].detach().clone()
    tampered[0][0] += 0.25
    transaction._prepared_prices = tuple(tampered)

    with pytest.raises(RuntimeError, match="immutable raw-statistics formula"):
        model.commit_cpt_transaction(transaction)
    assert model.get_cpt_state_version() == 0
    assert not transaction.closed
