"""v3.0-dense-276m: iso-active dense (FFN inter=4096) parameter-count lock assertions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from model.config import TinyMixtralConfig
from model.modeling import TinyMixtralForCausalLM

CFG = str(Path(__file__).parent.parent / "versions" / "v3.0-dense-276m" / "configs" / "v3.0-dense-276m.json")
EXPECT_TOTAL = 276_073_472


def test_config_keys():
    cfg = TinyMixtralConfig.from_json_file(CFG)
    assert cfg.num_local_experts == 0
    assert cfg.num_experts_per_tok == 0
    assert cfg.expert_intermediate_size == 4096
    assert cfg.hidden_size == 1024
    assert cfg.num_hidden_layers == 16
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 4
    assert cfg.head_dim == 64
    assert cfg.vocab_size == 32000
    assert cfg.use_qk_norm is True
    assert cfg.router_aux_loss_coef == 0.0
    assert cfg.attention_dropout == 0.0
    assert cfg.tie_word_embeddings is True


def test_param_count_exact():
    cfg = TinyMixtralConfig.from_json_file(CFG)
    with torch.device("meta"):
        model = TinyMixtralForCausalLM(cfg)
    n = sum(p.numel() for p in model.parameters())
    assert n == EXPECT_TOTAL, f"expected {EXPECT_TOTAL}, got {n}"


def test_ffn_mac_parity_with_moe_active():
    """dense 3*h*i == MoE top-2 of 4x(3*h*i/2): 3*1024*4096 == 2*(3*1024*2048)."""
    assert 3 * 1024 * 4096 == 2 * (3 * 1024 * 2048) == 12_582_912


def test_iso_active_minus_router():
    """276,073,472 = MoE active 276,139,008 - 16 routers (16*1024*4)."""
    assert EXPECT_TOTAL == 276_139_008 - 16 * 1024 * 4
