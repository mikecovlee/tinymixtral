# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

import torch

from model.config import TinyMixtralConfig
from model.modeling import TinyMixtralForCausalLM


def tiny_model(qk: bool) -> TinyMixtralForCausalLM:
    cfg = TinyMixtralConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=64,
        max_position_embeddings=64, num_local_experts=4,
        num_experts_per_tok=2, expert_intermediate_size=48,
        router_aux_loss_coef=0.0, router_jitter_noise=0.0,
        use_qk_norm=qk,
    )
    torch.manual_seed(5)
    return TinyMixtralForCausalLM(cfg).eval()


def test_roundtrip_qk_true(tmp_path):
    m = tiny_model(True)
    m.save_pretrained(str(tmp_path))
    m2 = TinyMixtralForCausalLM.from_pretrained(str(tmp_path))
    assert m2.config.use_qk_norm is True
    ids = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        assert torch.equal(m(ids)["logits"], m2(ids)["logits"])


def test_roundtrip_qk_false(tmp_path):
    m = tiny_model(False)
    m.save_pretrained(str(tmp_path))
    m2 = TinyMixtralForCausalLM.from_pretrained(str(tmp_path))
    assert m2.config.use_qk_norm is False
    keys = m2.state_dict().keys()
    assert not any("q_norm" in k for k in keys)


def test_strict_load_rejects_mismatch(tmp_path):
    m = tiny_model(True)
    m.save_pretrained(str(tmp_path))
    cfg = TinyMixtralConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=64,
        max_position_embeddings=64, num_local_experts=4,
        num_experts_per_tok=2, expert_intermediate_size=48,
        use_qk_norm=False,
    )
    try:
        TinyMixtralForCausalLM.from_pretrained(str(tmp_path), config=cfg)
        assert False, "expected strict load failure"
    except RuntimeError as e:
        assert "q_norm" in str(e)
