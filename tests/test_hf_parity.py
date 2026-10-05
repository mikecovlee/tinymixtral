# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

"""Parity guard: model/ (training) vs hf/ (published) must compute identically.

model/modeling.py is the source of truth; hf/modeling_tinymixtral.py is the
trust_remote_code export shipped in published repos. This test pins their
numerical behaviour so the two copies cannot silently drift.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hf.configuration_tinymixtral import TinyMixtralConfig as HFConfig  # noqa: E402
from hf.modeling_tinymixtral import TinyMixtralForCausalLM as HFModel  # noqa: E402
from model.config import TinyMixtralConfig as TrainConfig  # noqa: E402
from model.modeling import TinyMixtralForCausalLM as TrainModel  # noqa: E402

TINY = dict(
    vocab_size=97,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=64,
    num_local_experts=3,
    num_experts_per_tok=2,
    expert_intermediate_size=48,
    router_aux_loss_coef=0.01,
    router_jitter_noise=0.0,
)


def _models():
    torch.manual_seed(0)
    m_train = TrainModel(TrainConfig(**TINY))
    m_hf = HFModel(HFConfig(**TINY))
    m_hf.load_state_dict(m_train.state_dict(), strict=True)
    m_train.eval()
    m_hf.eval()
    return m_train, m_hf


def test_state_dict_keys_parity():
    torch.manual_seed(0)
    m_train = TrainModel(TrainConfig(**TINY))
    m_hf = HFModel(HFConfig(**TINY))
    assert set(m_train.state_dict().keys()) == set(m_hf.state_dict().keys())


@pytest.mark.parametrize("with_mask", [False, True])
def test_logits_parity(with_mask):
    m_train, m_hf = _models()
    torch.manual_seed(123)
    ids = torch.randint(0, TINY["vocab_size"], (2, 16))
    mask = None
    if with_mask:
        mask = torch.ones(2, 16, dtype=torch.long)
        mask[0, -3:] = 0
        mask[1, -1:] = 0
    with torch.no_grad():
        logits_train = m_train(ids, attention_mask=mask)["logits"]
        logits_hf = m_hf(ids, attention_mask=mask)["logits"]
    torch.testing.assert_close(logits_train, logits_hf, atol=1e-5, rtol=1e-4)


def test_loss_parity():
    m_train, m_hf = _models()
    torch.manual_seed(123)
    ids = torch.randint(0, TINY["vocab_size"], (2, 16))
    labels = ids.clone()
    labels[:, :2] = -100
    with torch.no_grad():
        loss_train = m_train(ids, labels=labels)["loss"]
        loss_hf = m_hf(ids, labels=labels)["loss"]
    torch.testing.assert_close(loss_train, loss_hf, atol=1e-5, rtol=1e-4)


TINY_DENSE = {**TINY, "num_local_experts": 0, "num_experts_per_tok": 0}


def _models_dense():
    torch.manual_seed(0)
    m_train = TrainModel(TrainConfig(**TINY_DENSE))
    m_hf = HFModel(HFConfig(**TINY_DENSE))
    m_hf.load_state_dict(m_train.state_dict(), strict=True)
    m_train.eval()
    m_hf.eval()
    return m_train, m_hf


def test_dense_state_dict_keys_parity():
    m_train = TrainModel(TrainConfig(**TINY_DENSE))
    m_hf = HFModel(HFConfig(**TINY_DENSE))
    assert set(m_train.state_dict()) == set(m_hf.state_dict())


@pytest.mark.parametrize("with_mask", [False, True])
def test_dense_logits_parity(with_mask):
    m_train, m_hf = _models_dense()
    torch.manual_seed(7)
    ids = torch.randint(0, TINY_DENSE["vocab_size"], (2, 16))
    mask = None
    if with_mask:
        mask = torch.ones(2, 16, dtype=torch.long)
        mask[0, -3:] = 0
        mask[1, -1:] = 0
    with torch.no_grad():
        logits_train = m_train(ids, attention_mask=mask)["logits"]
        logits_hf = m_hf(ids, attention_mask=mask)["logits"]
    torch.testing.assert_close(logits_train, logits_hf, atol=1e-5, rtol=1e-4)


def test_dense_loss_parity():
    m_train, m_hf = _models_dense()
    torch.manual_seed(123)
    ids = torch.randint(0, TINY_DENSE["vocab_size"], (2, 16))
    labels = ids.clone()
    labels[:, :2] = -100
    with torch.no_grad():
        loss_train = m_train(ids, labels=labels)["loss"]
        loss_hf = m_hf(ids, labels=labels)["loss"]
    torch.testing.assert_close(loss_train, loss_hf, atol=1e-5, rtol=1e-4)


def test_v3_dense_config_param_count(root):
    cfg = TrainConfig.from_json_file(
        str(root / "versions" / "v3.0-dense-276m" / "configs" / "v3.0-dense-276m.json")
    )
    assert cfg.num_local_experts == 0
    assert cfg.num_experts_per_tok == 0
    assert cfg.expert_intermediate_size == 4096
    model = TrainModel(cfg)
    n = model.num_parameters
    expected = (
        cfg.vocab_size * cfg.hidden_size
        + cfg.hidden_size
        + cfg.num_hidden_layers
        * (
            2 * cfg.hidden_size
            + cfg.hidden_size * cfg.num_attention_heads * cfg.head_dim
            + 2 * cfg.hidden_size * cfg.num_key_value_heads * cfg.head_dim
            + cfg.num_attention_heads * cfg.head_dim * cfg.hidden_size
            + (2 * cfg.head_dim if cfg.use_qk_norm else 0)
            + 3 * cfg.hidden_size * cfg.expert_intermediate_size
        )
    )
    assert n == expected
    assert 276_000_000 < n < 276_100_000
