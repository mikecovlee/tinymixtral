# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

import torch

from model.config import TinyMixtralConfig
from model.modeling import TinyMixtralForCausalLM


def tiny_model(**over) -> TinyMixtralForCausalLM:
    base = dict(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=64,
        max_position_embeddings=64, num_local_experts=16,
        num_experts_per_tok=1, expert_intermediate_size=48,
        router_aux_loss_coef=0.0, router_jitter_noise=0.0,
    )
    base.update(over)
    torch.manual_seed(11)
    return TinyMixtralForCausalLM(TinyMixtralConfig(**base))


def test_keepalive_zero_grad_for_dead_experts():
    model = tiny_model().train()
    ids = torch.randint(0, 64, (1, 8))
    out = model(ids, labels=torch.randint(0, 64, (1, 8)))
    out["loss"].backward()
    moe = model.layers[0].moe
    assert moe.gate_proj.grad is not None
    counts = moe.last_expert_counts
    dead = (counts == 0).nonzero().flatten()
    assert len(dead) >= 1
    assert moe.gate_proj.grad[dead].abs().sum().item() == 0.0
    live = (counts > 0).nonzero().flatten()
    assert moe.gate_proj.grad[live].abs().sum().item() > 0


def test_keepalive_does_not_perturb_output():
    model = tiny_model()
    ids = torch.randint(0, 64, (1, 16))
    model.eval()
    with torch.no_grad():
        out_eval = model(ids)["logits"]
    model.train()
    with torch.no_grad():
        out_train = model(ids)["logits"]
    assert torch.equal(out_eval, out_train)


def test_utilization_after_forward():
    model = tiny_model(use_qk_norm=True).train()
    assert model.expert_utilization() is None
    ids = torch.randint(0, 64, (2, 16))
    model(ids, labels=torch.randint(0, 64, (2, 16)))
    util = model.expert_utilization()
    assert util is not None and len(util) == 16
    assert abs(sum(util) - 1.0) < 1e-5
