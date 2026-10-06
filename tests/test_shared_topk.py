# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""Smoke: shared_topk (v2.0-beta shared-expert MoE) package loads, forwards, dispatches."""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import peek_mechanism  # noqa: E402
from model.shared_topk.config import TinyMixtralConfig  # noqa: E402
from model.shared_topk.modeling import TinyMixtralForCausalLM  # noqa: E402

TINY = dict(
    vocab_size=64,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    max_position_embeddings=64,
    num_shared_experts=1,
    num_routed_experts=2,
    num_experts_per_tok=1,
    expert_intermediate_size=48,
    router_aux_loss_coef=0.01,
    router_jitter_noise=0.0,
)


def test_shared_topk_forward():
    torch.manual_seed(0)
    cfg = TinyMixtralConfig(**TINY)
    model = TinyMixtralForCausalLM(cfg)
    ids = torch.randint(0, TINY["vocab_size"], (2, 16))
    out = model(ids, labels=ids)
    assert out["logits"].shape == (2, 16, TINY["vocab_size"])
    assert torch.isfinite(out["loss"])
    assert out["aux_loss"] >= 0.0


def test_shared_topk_dispatch_and_real_json():
    assert peek_mechanism(TINY) == "shared_topk"
    real = json.loads((ROOT / "config" / "v2.0-beta" / "config.json").read_text(encoding="utf-8"))
    assert peek_mechanism(real) == "shared_topk"
    cfg = TinyMixtralConfig.from_json_file(str(ROOT / "config" / "v2.0-beta" / "config.json"))
    assert cfg.num_shared_experts == 1
    assert cfg.num_local_experts == cfg.num_shared_experts + cfg.num_routed_experts
