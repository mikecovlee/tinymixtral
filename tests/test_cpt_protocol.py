"""CPT transaction protocol guards, merge semantics and dual-mode wiring."""
from unittest.mock import patch

import pytest
import torch

from cpt_model import CPTConfig, CPTForCausalLM, config_from_json_file, model_for_config
from model.config import TinyMixtralConfig
from model.modeling import TinyMixtralForCausalLM
from scripts.train_utils import make_adamw

TOKENS = torch.tensor([[1, 2, 3, 4]])
MASK = torch.tensor([[1, 1, 1, 0]])
TOKENS_B = torch.tensor([[5, 6, 7, 8]])
MASK_B = torch.tensor([[1, 1, 0, 0]])


def tiny_cpt(k=2, n=4, chunk=2) -> CPTConfig:
    return CPTConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        num_local_experts=n, num_experts_per_tok=k, expert_intermediate_size=24,
        cpt_state_chunk_size=chunk)


def tiny_linear(k=2, n=4) -> TinyMixtralConfig:
    return TinyMixtralConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        num_local_experts=n, num_experts_per_tok=k, expert_intermediate_size=24)


def build_cpt() -> CPTForCausalLM:
    torch.manual_seed(11)
    return CPTForCausalLM(tiny_cpt()).train()


def test_cpt_config_pins_aux_and_jitter():
    cfg = CPTConfig(router_aux_loss_coef=0.01, router_jitter_noise=0.02)
    assert cfg.router_aux_loss_coef == 0.0
    assert cfg.router_jitter_noise == 0.0


def test_linear_mode_is_default_with_live_aux():
    model = TinyMixtralForCausalLM(tiny_linear()).train()
    out = model(TOKENS, labels=TOKENS)
    assert out.get("cpt_transaction") is None
    assert out["aux_loss"].item() > 0
    assert out["loss"].item() > out["ce_loss"].item()
    keys = set(model.state_dict())
    assert any(key.endswith(".moe.router.weight") for key in keys)
    assert not any("cpt_router" in key for key in keys)


def test_linear_model_has_no_cpt_api():
    model = TinyMixtralForCausalLM(tiny_linear()).train()
    assert not hasattr(model, "commit_cpt_transaction")
    assert not hasattr(model, "no_weight_decay_parameters")


def test_dispatch_selects_model_and_config_classes(tmp_path):
    linear_dir = tmp_path / "linear"
    cpt_dir = tmp_path / "cpt"
    tiny_linear().save_pretrained(str(linear_dir))
    tiny_cpt().save_pretrained(str(cpt_dir))
    linear = config_from_json_file(str(linear_dir / "config.json"))
    cpt = config_from_json_file(str(cpt_dir / "config.json"))
    assert type(linear) is TinyMixtralConfig
    assert type(cpt) is CPTConfig
    assert cpt.cpt_num_prototypes == 2 * cpt.num_local_experts
    assert type(model_for_config(linear)) is TinyMixtralForCausalLM
    assert type(model_for_config(cpt)) is CPTForCausalLM


def test_commit_rejects_eval_forward_transaction():
    model = build_cpt()
    model.eval()
    with torch.no_grad():
        out = model(TOKENS, attention_mask=MASK, labels=TOKENS)
    transaction = out["cpt_transaction"]
    assert transaction is not None and not transaction.training_forward
    with pytest.raises(RuntimeError, match="non-training forward"):
        model.commit_cpt_transaction(transaction)
    assert transaction.consumed


def test_commit_governed_by_forward_mode_not_commit_mode():
    model = build_cpt()
    out = model(TOKENS, attention_mask=MASK, labels=TOKENS)
    model.eval()
    try:
        assert model.commit_cpt_transaction(out["cpt_transaction"]) == 1
    finally:
        model.train()


def test_commit_rejects_distributed_process_group():
    model = build_cpt()
    out = model(TOKENS, attention_mask=MASK, labels=TOKENS)
    with patch.object(torch.distributed, "is_available", return_value=True), \
            patch.object(torch.distributed, "is_initialized", return_value=True):
        with pytest.raises(RuntimeError, match="single-process"):
            model.commit_cpt_transaction(out["cpt_transaction"])


def test_merged_microbatch_commit_updates_price_once():
    model = build_cpt()
    router = model.layers[0].moe.cpt_router
    out_a = model(TOKENS, attention_mask=MASK, labels=TOKENS)
    out_b = model(TOKENS_B, attention_mask=MASK_B, labels=TOKENS_B)
    transaction_a = out_a["cpt_transaction"]
    transaction_b = out_b["cpt_transaction"]
    hits = transaction_a.proposals[0].expert_hits + transaction_b.proposals[0].expert_hits
    old_price = router.congestion_price.clone()

    assert model.commit_cpt_transaction([transaction_a, transaction_b]) == 1

    # One merged price update and exactly one state-version advance.
    assignments = int(hits.sum())
    shares = hits.to(torch.float32) / float(assignments)
    delta = router.capacity_factor - 1.0
    lower = (1.0 - delta) / router.num_experts
    upper = (1.0 + delta) / router.num_experts
    control = torch.where(
        shares < lower,
        shares - lower,
        torch.where(shares > upper, shares - upper, torch.zeros_like(shares)),
    )
    expected = old_price + router.price_learning_rate * control
    expected = expected - expected.mean()
    torch.testing.assert_close(router.congestion_price, expected)
    assert model.get_cpt_state_version() == 1
    assert model.get_cpt_optimizer_step() == 0

    with pytest.raises(RuntimeError, match="already consumed"):
        model.commit_cpt_transaction([transaction_a, transaction_b])


def test_merged_commit_rejects_stale_microbatch():
    model = build_cpt()
    out_a = model(TOKENS, attention_mask=MASK, labels=TOKENS)
    out_b = model(TOKENS_B, attention_mask=MASK_B, labels=TOKENS_B)
    model.commit_cpt_transaction(out_a["cpt_transaction"])
    with pytest.raises(RuntimeError, match="stale or mismatched"):
        model.commit_cpt_transaction(out_b["cpt_transaction"])


def test_anchors_excluded_from_weight_decay():
    model = build_cpt()
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.1)
    decay_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
    no_decay_ids = {id(p) for p in optimizer.param_groups[1]["params"]}
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    for name, parameter in model.named_parameters():
        if name.endswith(".anchors"):
            assert id(parameter) in no_decay_ids
        elif parameter.ndim >= 2:
            assert id(parameter) in decay_ids
    assert set(model.no_weight_decay_parameters()) == {
        layer.moe.cpt_router.anchors for layer in model.layers
    }
