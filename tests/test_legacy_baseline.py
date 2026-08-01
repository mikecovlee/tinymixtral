import copy

import pytest
import torch
import torch.nn.functional as F

import scripts.train_utils as train_utils
from model.config import TinyMixtralConfig
from model.cpt_router import CPTLayerProposal, CPTRouterOutput
from model.modeling import SparseMoE, TinyMixtralForCausalLM
from scripts.train_utils import (
    TRAINING_STATE_SCHEMA_NAME,
    TRAINING_STATE_SCHEMA_VERSION,
    build_code_manifest,
    build_data_manifest,
    build_tokenizer_manifest,
    canonical_config_hash,
    make_adamw,
    make_cosine_schedule,
    save_training_state,
)


def _downstream_reference(
    moe: SparseMoE,
    x: torch.Tensor,
    probabilities: torch.Tensor,
    flat_valid_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference for the unchanged Top-2 and expert-dispatch boundary."""
    batch_size, seq_len, hidden_size = x.shape
    x_flat = x.reshape(-1, hidden_size)
    x_valid = x_flat.index_select(0, flat_valid_indices)
    topk_weights, selected_experts = torch.topk(
        probabilities,
        moe.top_k,
        dim=-1,
    )
    topk_weights = topk_weights / topk_weights.sum(
        dim=-1,
        keepdim=True,
    )

    final_valid = torch.zeros(
        flat_valid_indices.numel(),
        hidden_size,
        device=x.device,
        dtype=x.dtype,
    )
    for slot in range(moe.top_k):
        expert_index = selected_experts[:, slot]
        slot_weight = topk_weights[:, slot]
        for expert in range(moe.num_experts):
            mask = expert_index == expert
            if not mask.any():
                continue
            token_states = x_valid[mask]
            gate = F.silu(token_states @ moe.gate_proj[expert].T)
            up = token_states @ moe.up_proj[expert].T
            expert_out = (gate * up) @ moe.down_proj[expert].T
            final_valid[mask] += expert_out * slot_weight[mask].to(
                expert_out.dtype
            ).unsqueeze(-1)

    final_out = torch.zeros_like(x_flat).index_copy(
        0,
        flat_valid_indices,
        final_valid,
    )
    return (
        final_out.reshape(batch_size, seq_len, hidden_size),
        selected_experts,
        topk_weights,
    )


def test_default_architecture_contract() -> None:
    config = TinyMixtralConfig()
    assert config.hidden_size == 896
    assert config.num_hidden_layers == 10
    assert config.num_local_experts == 6
    assert config.num_experts_per_tok == 2
    assert config.expert_intermediate_size == 2389


def test_config_roundtrip_and_unknown_key_filter(tiny_config) -> None:
    payload = tiny_config.to_dict()
    payload["future_unknown_field"] = "ignored"
    restored = TinyMixtralConfig.from_dict(payload)
    assert restored.to_dict() == tiny_config.to_dict()
    assert restored.cpt_kappa_beta == tiny_config.cpt_kappa_beta
    assert (
        restored.cpt_projection_temperature
        == tiny_config.cpt_projection_temperature
    )


def test_sparse_moe_preserves_top2_dispatch_and_full_pi_price_statistics(
    tiny_config,
    monkeypatch,
) -> None:
    torch.manual_seed(101)
    moe = SparseMoE(tiny_config, layer_index=0).train()
    x = torch.linspace(
        -0.2,
        0.2,
        steps=2 * 3 * tiny_config.hidden_size,
    ).reshape(2, 3, tiny_config.hidden_size)
    route_valid_mask = torch.tensor(
        [
            [True, False, True],
            [False, True, True],
        ]
    )
    flat_valid_indices = route_valid_mask.reshape(-1).nonzero().flatten()
    probabilities = torch.tensor(
        [
            [0.55, 0.25, 0.15, 0.05],
            [0.10, 0.20, 0.60, 0.10],
            [0.05, 0.15, 0.30, 0.50],
            [0.26, 0.40, 0.09, 0.25],
        ],
        dtype=torch.float32,
    )
    proposal = CPTLayerProposal(
        layer_index=0,
        load_sum=probabilities.sum(dim=0).detach(),
        token_count=torch.tensor(4, dtype=torch.int64),
        state_version=torch.tensor(0, dtype=torch.int64),
        valid=torch.tensor(True),
    )
    fixed_output = CPTRouterOutput(
        probabilities=probabilities,
        flat_valid_indices=flat_valid_indices,
        q_probabilities=torch.empty(
            4,
            tiny_config.cpt_num_prototypes,
        ),
        expert_kernel=torch.empty(
            tiny_config.cpt_num_prototypes,
            tiny_config.num_local_experts,
        ),
        proposal=proposal,
    )

    def fixed_router(
        hidden_states,
        route_valid_mask=None,
        reset_mask=None,
    ):
        assert hidden_states is x
        assert torch.equal(route_valid_mask, globals_route_mask)
        return fixed_output

    globals_route_mask = route_valid_mask
    monkeypatch.setattr(moe.cpt_router, "forward", fixed_router)

    (
        actual_out,
        load_sum,
        token_count,
        state_version,
        proposal_valid,
    ) = moe(x, route_valid_mask=route_valid_mask)
    (
        expected_out,
        expected_experts,
        expected_weights,
    ) = _downstream_reference(
        moe,
        x,
        probabilities,
        flat_valid_indices,
    )

    torch.testing.assert_close(actual_out, expected_out)
    assert expected_experts.tolist() == [
        [0, 1],
        [2, 1],
        [3, 2],
        [1, 0],
    ]
    torch.testing.assert_close(
        expected_weights.sum(dim=-1),
        torch.ones(probabilities.size(0)),
    )
    invalid_indices = (~route_valid_mask).reshape(-1).nonzero().flatten()
    torch.testing.assert_close(
        actual_out.reshape(-1, tiny_config.hidden_size).index_select(
            0,
            invalid_indices,
        ),
        torch.zeros(invalid_indices.numel(), tiny_config.hidden_size),
    )
    torch.testing.assert_close(load_sum, probabilities.sum(dim=0))
    assert token_count.item() == 4
    assert state_version.item() == 0
    assert proposal_valid.item()


def test_sparse_moe_all_padding_skips_top2_and_experts(tiny_config) -> None:
    torch.manual_seed(102)
    moe = SparseMoE(tiny_config, layer_index=0).train()
    x = torch.randn(2, 3, tiny_config.hidden_size)
    route_valid_mask = torch.zeros(2, 3, dtype=torch.bool)

    output, load_sum, token_count, state_version, valid = moe(
        x,
        route_valid_mask=route_valid_mask,
    )

    torch.testing.assert_close(output, torch.zeros_like(output))
    torch.testing.assert_close(
        load_sum,
        torch.zeros(tiny_config.num_local_experts),
    )
    assert token_count.item() == 0
    assert state_version.item() == 0
    assert valid.item()


def test_eval_forward_is_deterministic_and_uses_task_loss_only(
    tiny_config,
    fixed_batch,
) -> None:
    torch.manual_seed(103)
    model = TinyMixtralForCausalLM(tiny_config).eval()
    input_ids, labels, attention_mask = fixed_batch

    with torch.no_grad():
        first = model(input_ids, attention_mask=attention_mask, labels=labels)
        second = model(input_ids, attention_mask=attention_mask, labels=labels)

    assert first["logits"].shape == (2, 8, tiny_config.vocab_size)
    assert torch.equal(first["logits"], second["logits"])
    assert "aux_loss" not in first
    assert first["cpt_transaction"] is None
    assert second["cpt_transaction"] is None
    manual_loss = F.cross_entropy(
        first["logits"].reshape(-1, tiny_config.vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )
    torch.testing.assert_close(first["loss"], manual_loss)


def test_train_backward_reaches_cpt_router_expert_and_attention(
    tiny_config,
    fixed_batch,
) -> None:
    torch.manual_seed(104)
    model = TinyMixtralForCausalLM(tiny_config).train()
    input_ids, labels, attention_mask = fixed_batch

    output = model(input_ids, attention_mask=attention_mask, labels=labels)
    assert torch.isfinite(output["loss"])
    manual_loss = F.cross_entropy(
        output["logits"].reshape(-1, tiny_config.vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )
    torch.testing.assert_close(output["loss"], manual_loss)
    assert "aux_loss" not in output
    assert output["cpt_transaction"] is not None
    output["loss"].backward()

    parameters = dict(model.named_parameters())
    checked_names = (
        "layers.0.moe.cpt_router.projection",
        "layers.0.moe.cpt_router.anchors",
        "layers.0.moe.cpt_router.energy",
        "layers.0.moe.gate_proj",
        "layers.0.self_attn.q_proj.weight",
    )
    for name in checked_names:
        gradient = parameters[name].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.float().norm() > 0
    router = model.layers[0].moe.cpt_router
    assert router.congestion_price.grad is None
    assert router.state_version.item() == 0


def test_activation_checkpoint_matches_regular_backward_and_proposals(
    tiny_config,
    fixed_batch,
) -> None:
    torch.manual_seed(105)
    regular = TinyMixtralForCausalLM(tiny_config).train()
    checkpointed = TinyMixtralForCausalLM(tiny_config).train()
    checkpointed.load_state_dict(copy.deepcopy(regular.state_dict()), strict=True)
    checkpointed.gradient_checkpointing_enable()
    input_ids, labels, attention_mask = fixed_batch
    regular_prices = [
        layer.moe.cpt_router.congestion_price.detach().clone()
        for layer in regular.layers
    ]
    checkpointed_prices = [
        layer.moe.cpt_router.congestion_price.detach().clone()
        for layer in checkpointed.layers
    ]

    regular_output = regular(
        input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    checkpointed_output = checkpointed(
        input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    regular_output["loss"].backward()
    checkpointed_output["loss"].backward()

    torch.testing.assert_close(
        regular_output["logits"],
        checkpointed_output["logits"],
    )
    torch.testing.assert_close(
        regular_output["loss"],
        checkpointed_output["loss"],
    )
    assert "aux_loss" not in regular_output
    assert "aux_loss" not in checkpointed_output
    regular_transaction = regular_output["cpt_transaction"]
    checkpointed_transaction = checkpointed_output["cpt_transaction"]
    assert regular_transaction is not None
    assert checkpointed_transaction is not None
    for left, right in zip(
        regular_transaction.proposals,
        checkpointed_transaction.proposals,
    ):
        assert left.layer_index == right.layer_index
        torch.testing.assert_close(left.load_sum, right.load_sum)
        assert left.token_count.item() == right.token_count.item()
        assert left.state_version.item() == right.state_version.item() == 0
        assert left.valid.item() and right.valid.item()

    regular_parameters = dict(regular.named_parameters())
    checkpointed_parameters = dict(checkpointed.named_parameters())
    assert regular_parameters.keys() == checkpointed_parameters.keys()
    for name, parameter in regular_parameters.items():
        other_gradient = checkpointed_parameters[name].grad
        assert parameter.grad is not None, name
        assert other_gradient is not None, name
        torch.testing.assert_close(parameter.grad, other_gradient)

    for model, original_prices in (
        (regular, regular_prices),
        (checkpointed, checkpointed_prices),
    ):
        assert model.get_cpt_state_version() == 0
        for layer, original_price in zip(model.layers, original_prices):
            torch.testing.assert_close(
                layer.moe.cpt_router.congestion_price,
                original_price,
            )


def test_left_padding_preserves_valid_logits(tiny_config) -> None:
    torch.manual_seed(106)
    model = TinyMixtralForCausalLM(tiny_config).eval()
    unpadded = torch.tensor([[1, 3, 5, 7]], dtype=torch.long)
    padded = torch.tensor([[0, 0, 1, 3, 5, 7]], dtype=torch.long)
    unpadded_mask = torch.ones_like(unpadded, dtype=torch.bool)
    padded_mask = torch.tensor(
        [[False, False, True, True, True, True]],
        dtype=torch.bool,
    )

    with torch.no_grad():
        unpadded_logits = model(
            unpadded,
            attention_mask=unpadded_mask,
        )["logits"]
        padded_logits = model(
            padded,
            attention_mask=padded_mask,
        )["logits"][:, -unpadded.size(1):]

    torch.testing.assert_close(unpadded_logits, padded_logits)


def test_strict_in_memory_state_dict_roundtrip_preserves_cpt_state(
    tiny_config,
    fixed_batch,
) -> None:
    torch.manual_seed(107)
    source = TinyMixtralForCausalLM(tiny_config).eval()
    with torch.no_grad():
        for layer_index, layer in enumerate(source.layers):
            router = layer.moe.cpt_router
            router.congestion_price.copy_(
                torch.linspace(
                    0.0,
                    0.03 * (layer_index + 1),
                    tiny_config.num_local_experts,
                )
            )
            router.state_version.fill_(3)

    restored = TinyMixtralForCausalLM(tiny_config).eval()
    state = copy.deepcopy(source.state_dict())
    result = restored.load_state_dict(state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert restored.get_cpt_state_version() == 3

    for source_layer, restored_layer in zip(source.layers, restored.layers):
        source_router = source_layer.moe.cpt_router
        restored_router = restored_layer.moe.cpt_router
        torch.testing.assert_close(
            source_router.congestion_price,
            restored_router.congestion_price,
        )
        assert restored_router.congestion_price.dtype == torch.float32
        assert restored_router.state_version.item() == 3

    input_ids, _, attention_mask = fixed_batch
    with torch.no_grad():
        source_logits = source(input_ids, attention_mask=attention_mask)["logits"]
        restored_logits = restored(
            input_ids,
            attention_mask=attention_mask,
        )["logits"]
    torch.testing.assert_close(source_logits, restored_logits)


def test_missing_cpt_checkpoint_key_is_rejected(tiny_config) -> None:
    model = TinyMixtralForCausalLM(tiny_config)
    incomplete_state = copy.deepcopy(model.state_dict())
    incomplete_state.pop("layers.0.moe.cpt_router.projection")

    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(incomplete_state, strict=True)


def test_legacy_linear_router_checkpoint_is_explicitly_rejected(
    tiny_config,
) -> None:
    model = TinyMixtralForCausalLM(tiny_config)
    legacy_state = copy.deepcopy(model.state_dict())
    legacy_state["layers.0.moe.router.weight"] = torch.randn(
        tiny_config.num_local_experts,
        tiny_config.hidden_size,
    )

    with pytest.raises(RuntimeError, match="Legacy Linear-router checkpoint"):
        model.load_state_dict(legacy_state, strict=True)


def test_optimizer_groups_and_training_state_schema(
    tiny_config,
    monkeypatch,
    tmp_path,
) -> None:
    model = TinyMixtralForCausalLM(tiny_config)
    optimizer = make_adamw(model, lr=3e-4, weight_decay=0.1)
    scheduler = make_cosine_schedule(optimizer, warmup_steps=2, total_steps=8)

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    grouped = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(grouped) == len(list(model.parameters()))
    assert len({id(parameter) for parameter in grouped}) == len(grouped)
    assert all(
        layer.moe.cpt_router.congestion_price.grad is None
        for layer in model.layers
    )

    for _ in range(3):
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    shard = tmp_path / "train_00000.pt"
    shard.write_bytes(b"fixed token shard")
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    code_manifest = build_code_manifest()
    data_manifest = build_data_manifest([shard])
    tokenizer_manifest = build_tokenizer_manifest(tokenizer_dir)

    captured = {}

    def capture_save(state, path) -> None:
        captured["state"] = state
        captured["path"] = path

    monkeypatch.setattr(train_utils, "_torch_save_fsync", capture_save)
    save_training_state(
        "unused.pt",
        optimizer,
        scheduler,
        step=3,
        total_tok=48,
        warmup_steps=2,
        total_steps=8,
        fi=1,
        ptr=54,
        batch_size=2,
        seq_len=8,
        optimizer_kind="adamw",
        schedule_kind="cosine",
        schedule_decay_ratio=None,
        cpt_state_version=3,
        run_id="c" * 32,
        config_sha256=canonical_config_hash(model.config),
        code_manifest=code_manifest,
        data_manifest=data_manifest,
        tokenizer_manifest=tokenizer_manifest,
        model_state_sha256="a" * 64,
        config_file_sha256="b" * 64,
    )
    assert captured["path"] == "unused.pt"
    assert set(captured["state"]) == {
        "schema_name",
        "schema_version",
        "cpt_state_version",
        "optimizer_kind",
        "schedule_kind",
        "schedule_decay_ratio",
        "opt",
        "sched",
        "step",
        "total_tok",
        "warmup_steps",
        "total_steps",
        "fi",
        "ptr",
        "batch_size",
        "seq_len",
        "rng_world_size",
        "rng_states",
        "run_id",
        "config_sha256",
        "code_manifest",
        "data_manifest",
        "tokenizer_manifest",
        "model_state_sha256",
        "config_file_sha256",
    }
    assert captured["state"]["schema_name"] == TRAINING_STATE_SCHEMA_NAME
    assert captured["state"]["schema_version"] == TRAINING_STATE_SCHEMA_VERSION
    assert captured["state"]["cpt_state_version"] == 3
    assert captured["state"]["optimizer_kind"] == "adamw"
    assert captured["state"]["schedule_kind"] == "cosine"
    assert captured["state"]["schedule_decay_ratio"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tiny_cuda_bf16_forward_backward(tiny_config, fixed_batch) -> None:
    assert torch.cuda.is_bf16_supported()
    torch.manual_seed(108)
    torch.cuda.manual_seed_all(108)
    model = TinyMixtralForCausalLM(tiny_config).to(
        device="cuda",
        dtype=torch.bfloat16,
    ).train()
    model.gradient_checkpointing_enable()
    input_ids, labels, attention_mask = (
        tensor.cuda() for tensor in fixed_batch
    )

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    router = model.layers[0].moe.cpt_router
    gradient = router.projection.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.float().norm() > 0
    assert router.congestion_price.dtype == torch.float32
