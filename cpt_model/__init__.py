# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

"""TinyMixtral CPT-MoE 变体包。

与主分支代码完全隔离：``model/`` 与 main 保持零差异，本包以薄子类
叠加 CPT 概率路由器与事务协议。配置含 ``cpt_router_version`` 的
checkpoint/配置文件走 CPT 路径，否则回到主线的线性路由器模型。

入口：
- ``model_for_config(config)`` — 按配置类构建对应模型
- ``load_model(path)`` — 按 checkpoint 的 config.json 自动选择并加载
"""

from model.config import TinyMixtralConfig
from model.modeling import TinyMixtralForCausalLM

from .config import CPTConfig, config_from_json_file
from .constants import exact_scalar, inverse_sqrt_fp32, rational_fp32
from .modeling import CPTBlock, CPTForCausalLM, CPTSparseMoE
from .numerics import stable_l2
from .protocol import CPTModelMixin
from .router import CPTLayerProposal, CPTRouter, CPTRouterOutput, CPTTransaction


def model_for_config(config: TinyMixtralConfig) -> TinyMixtralForCausalLM:
    """按配置类构建模型：CPTConfig → CPTForCausalLM，否则主线模型。"""
    if isinstance(config, CPTConfig):
        return CPTForCausalLM(config)
    return TinyMixtralForCausalLM(config)


def load_model(path: str) -> TinyMixtralForCausalLM:
    """按 checkpoint 的 config.json 自动选择并加载对应模型。"""
    config = config_from_json_file(f"{path}/config.json")
    cls = CPTForCausalLM if isinstance(config, CPTConfig) else TinyMixtralForCausalLM
    return cls.from_pretrained(path, config=config)


__all__ = [
    "CPTBlock",
    "CPTConfig",
    "CPTForCausalLM",
    "CPTLayerProposal",
    "CPTModelMixin",
    "CPTRouter",
    "CPTRouterOutput",
    "CPTSparseMoE",
    "CPTTransaction",
    "TinyMixtralConfig",
    "TinyMixtralForCausalLM",
    "config_from_json_file",
    "exact_scalar",
    "inverse_sqrt_fp32",
    "load_model",
    "model_for_config",
    "rational_fp32",
    "stable_l2",
]
