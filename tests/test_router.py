# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

import pytest
import torch

from model.config import TinyMixtralConfig
from model.modeling import SparseMoE


def tiny_moe(top_k: int, seed: int = 0) -> SparseMoE:
    cfg = TinyMixtralConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=64,
        max_position_embeddings=64, num_local_experts=4,
        num_experts_per_tok=top_k, expert_intermediate_size=48,
        router_aux_loss_coef=0.01, router_jitter_noise=0.0,
    )
    torch.manual_seed(seed)
    return SparseMoE(cfg).eval()


def naive_moe_ref(moe: SparseMoE, x: torch.Tensor) -> torch.Tensor:
    logits = x @ moe.router.weight.T
    probs = torch.softmax(logits, dim=-1)
    top_w, top_i = probs.topk(moe.top_k, dim=-1)
    top_w = top_w / top_w.sum(dim=-1, keepdim=True)
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for j in range(moe.top_k):
            e = int(top_i[t, j])
            h = x[t] @ moe.gate_proj[e].T
            u = x[t] @ moe.up_proj[e].T
            act = torch.nn.functional.silu(h) * u
            out[t] += top_w[t, j] * (act @ moe.down_proj[e].T)
    return out


@pytest.mark.parametrize("top_k", [1, 2])
def test_sortbased_matches_naive(top_k):
    moe = tiny_moe(top_k)
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        out, _ = moe(x)
        ref = naive_moe_ref(moe, x.reshape(-1, 32)).reshape(2, 8, 32)
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()


@pytest.mark.parametrize("top_k", [1, 2])
def test_renorm_weights_and_counts(top_k):
    moe = tiny_moe(top_k)
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        moe.eval()
        out, _ = moe(x)
    moe.train()
    with torch.no_grad():
        out, _ = moe(x)
    counts = moe.last_expert_counts
    assert counts is not None and counts.numel() == 4
    assert int(counts.sum()) == 2 * 8 * top_k


def test_eval_mode_no_counts_update():
    moe = tiny_moe(2)
    assert moe.last_expert_counts is None
    x = torch.randn(1, 4, 32)
    with torch.no_grad():
        moe(x)
    assert moe.last_expert_counts is None
