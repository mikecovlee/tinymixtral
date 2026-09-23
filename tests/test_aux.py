# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

import pytest
import torch

from model.config import TinyMixtralConfig
from model.modeling import SparseMoE


def tiny_moe(aux_coef: float = 0.01) -> SparseMoE:
    cfg = TinyMixtralConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=64,
        max_position_embeddings=64, num_local_experts=4,
        num_experts_per_tok=1, expert_intermediate_size=48,
        router_aux_loss_coef=aux_coef, router_jitter_noise=0.0,
    )
    torch.manual_seed(0)
    return SparseMoE(cfg).train()


def test_balanced_aux_near_one():
    moe = tiny_moe()
    with torch.no_grad():
        moe.router.weight.zero_()
    x = torch.randn(1, 256, 32)
    _, aux = moe(x)
    assert 0.95 <= aux.item() <= 1.15


def test_collapsed_aux_near_num_experts():
    moe = tiny_moe()
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0] = 50.0
    x = torch.rand(1, 64, 32) + 0.5
    _, aux = moe(x)
    assert aux.item() > 3.0
    assert int(moe.last_expert_counts[0]) == 64


def test_aux_grad_flows_to_router():
    moe = tiny_moe()
    x = torch.randn(1, 32, 32, requires_grad=True)
    _, aux = moe(x)
    aux.backward(retain_graph=True)
    assert moe.router.weight.grad is not None
    assert moe.router.weight.grad.abs().sum() > 0


def test_aux_reaches_only_router():
    moe = tiny_moe()
    x = torch.randn(1, 32, 32)
    _, aux = moe(x)
    g_router = torch.autograd.grad(aux, moe.router.weight, retain_graph=True)[0]
    assert g_router.abs().sum() > 0
    g_experts = torch.autograd.grad(
        aux, [moe.gate_proj, moe.up_proj, moe.down_proj], allow_unused=True
    )
    assert all(g is None for g in g_experts)
