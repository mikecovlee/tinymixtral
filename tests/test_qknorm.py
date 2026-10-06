# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

import torch

from model.config import TinyMixtralConfig
from model.modeling import TinyMixtralForCausalLM


def tiny_cfg(**over) -> TinyMixtralConfig:
    base = dict(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=64,
        max_position_embeddings=64, num_local_experts=4,
        num_experts_per_tok=2, expert_intermediate_size=48,
        router_aux_loss_coef=0.0, router_jitter_noise=0.0,
    )
    base.update(over)
    return TinyMixtralConfig(**base)


def build(cfg) -> TinyMixtralForCausalLM:
    torch.manual_seed(7)
    return TinyMixtralForCausalLM(cfg).eval()


def test_qk_keys_present_only_when_enabled():
    m_true = build(tiny_cfg(use_qk_norm=True))
    m_false = build(tiny_cfg(use_qk_norm=False))
    keys_true = set(m_true.state_dict().keys())
    keys_false = set(m_false.state_dict().keys())
    assert any(".q_norm.weight" in k for k in keys_true)
    assert any(".k_norm.weight" in k for k in keys_true)
    assert not any("q_norm" in k or "k_norm" in k for k in keys_false)


def test_qk_norm_changes_output_and_grads():
    ids = torch.randint(0, 64, (2, 16))
    labels = torch.randint(0, 64, (2, 16))
    m_true = build(tiny_cfg(use_qk_norm=True))
    m_false = build(tiny_cfg(use_qk_norm=False))
    with torch.no_grad():
        o_true = m_true(ids)["logits"]
        o_false = m_false(ids)["logits"]
    assert not torch.allclose(o_true, o_false, atol=1e-6)
    out = m_true(ids, labels=labels)
    out["loss"].backward()
    assert m_true.layers[0].self_attn.q_norm.weight.grad is not None
    assert m_true.layers[0].self_attn.q_norm.weight.grad.abs().sum() > 0


def test_qnorm_is_per_head():
    m = build(tiny_cfg(use_qk_norm=True))
    attn = m.layers[0].self_attn
    assert attn.q_norm.weight.shape[-1] == attn.head_dim
    assert torch.allclose(attn.q_norm.weight, torch.ones_like(attn.q_norm.weight))
