"""TinyMixtral/TinyMistral model packages.

Self-contained mechanism packages (pick one by config dialect):
  - model.topk       : MoE top-k routing            (tinymixtral family)
  - model.shared_topk: shared + routed experts      (tinymixtral family, v2.0-beta)
  - model.dense      : dense FFN, no router         (tinymistral family)

This module is the only shared code: a tiny dispatcher so training/eval scripts
can run any config x data combination without caring about the mechanism.
"""
from __future__ import annotations

import json
import os

_MECHANISMS = ("topk", "shared_topk", "dense")


def peek_mechanism(data: dict) -> str:
    """Infer mechanism from a config dict (field-presence based)."""
    if "num_shared_experts" in data and int(data["num_shared_experts"]) >= 1:
        return "shared_topk"
    if "num_local_experts" in data and int(data["num_local_experts"]) == 0:
        return "dense"
    return "topk"


def get_classes(mechanism: str):
    """Return (ConfigClass, ModelClass) for a mechanism name."""
    if mechanism == "topk":
        from .topk.config import TinyMixtralConfig as C
        from .topk.modeling import TinyMixtralForCausalLM as M
    elif mechanism == "shared_topk":
        from .shared_topk.config import TinyMixtralConfig as C
        from .shared_topk.modeling import TinyMixtralForCausalLM as M
    elif mechanism == "dense":
        from .dense.config import TinyMistralConfig as C
        from .dense.modeling import TinyMistralForCausalLM as M
    else:
        raise ValueError(f"unknown mechanism {mechanism!r}, expected one of {_MECHANISMS}")
    return C, M


def _as_dict(source) -> dict:
    if isinstance(source, dict):
        return source
    with open(source) as f:
        return json.load(f)


def load_config(source):
    """Build the right config object from a JSON path or dict."""
    data = _as_dict(source)
    C, _ = get_classes(peek_mechanism(data))
    if hasattr(C, "from_dict"):
        return C.from_dict(data)
    import dataclasses

    valid = {f.name for f in dataclasses.fields(C)}
    return C(**{k: v for k, v in data.items() if k in valid})


def _cfg_dict(cfg) -> dict:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    return dict(cfg.__dict__)


def build_model(cfg):
    """Instantiate the model matching an already-built config object."""
    _, M = get_classes(peek_mechanism(_cfg_dict(cfg)))
    return M(cfg)


def from_pretrained(path):
    """Load a saved checkpoint (reads <path>/config.json to pick the mechanism)."""
    with open(os.path.join(path, "config.json")) as f:
        data = json.load(f)
    _, M = get_classes(peek_mechanism(data))
    return M.from_pretrained(path)


def default_config():
    """Top-k config with legacy defaults (was: TinyMixtralConfig())."""
    C, _ = get_classes("topk")
    return C()
