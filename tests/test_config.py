# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

import json

import pytest

from model.config import TinyMixtralConfig


def test_use_qk_norm_defaults_false():
    assert TinyMixtralConfig().use_qk_norm is False


def test_roundtrip_dict():
    cfg = TinyMixtralConfig(use_qk_norm=True, num_local_experts=4,
                            num_experts_per_tok=1, router_aux_loss_coef=1e-3,
                            router_jitter_noise=0.0)
    cfg2 = TinyMixtralConfig.from_dict(cfg.to_dict())
    assert cfg2.to_dict() == cfg.to_dict()
    assert cfg2.use_qk_norm is True


def test_roundtrip_json(tmp_path):
    cfg = TinyMixtralConfig(use_qk_norm=True)
    d = tmp_path / "cfg"
    cfg.save_pretrained(str(d))
    assert TinyMixtralConfig.from_json_file(str(d / "config.json")).use_qk_norm is True


def test_old_json_without_qk_field_loads():
    d = json.loads(json.dumps(TinyMixtralConfig().to_dict()))
    d.pop("use_qk_norm")
    assert TinyMixtralConfig.from_dict(d).use_qk_norm is False


def test_validation_catches_bad_qk_combo():
    with pytest.raises(ValueError):
        TinyMixtralConfig(hidden_size=1024, num_attention_heads=15, head_dim=64)
    with pytest.raises(ValueError):
        TinyMixtralConfig(num_local_experts=4, num_experts_per_tok=5)


def _load_shared_expert_config(root):
    import importlib.util
    path = root / "shared_expert" / "model" / "config.py"
    spec = importlib.util.spec_from_file_location("shared_expert_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TinyMixtralConfig


def test_repo_configs_load(root):
    import glob
    for path in glob.glob(str(root / "versions/*/configs/*.json")):
        cls = _load_shared_expert_config(root) if "v2.0-beta" in path else TinyMixtralConfig
        cfg = cls.from_json_file(path)
        assert cfg.hidden_size == cfg.num_attention_heads * cfg.head_dim
        assert cfg.num_experts_per_tok <= cfg.num_local_experts
