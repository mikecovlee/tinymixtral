# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

"""TinyMixtral 模型配置。"""

import math
import struct
from dataclasses import dataclass
from numbers import Real
from typing import Optional


CPT_ROUTING_ARCHITECTURE_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_local_experts",
    "num_experts_per_tok",
)


def _cpt_fp32(name: str, value: object) -> float:
    """Return the finite FP32 scalar that the CPT Router will actually use."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    try:
        value = float(value)
        effective = struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, TypeError, ValueError, struct.error) as error:
        raise ValueError(f"{name} must be finite in FP32") from error
    if not math.isfinite(value) or not math.isfinite(effective):
        raise ValueError(f"{name} must be finite in FP32")
    return effective


@dataclass
class TinyMixtralConfig:
    """TinyMixtral MoE 因果语言模型配置。

    参考: Qwen2.5-0.5B (896d/24L/dense), Mixtral 8x7B (MoE), Qwen1.5-MoE-A2.7B
    """

    # 基础架构 (Qwen2.5-0.5B 对齐)
    vocab_size: int = 32000
    hidden_size: int = 896
    num_hidden_layers: int = 10
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    head_dim: int = 64
    max_position_embeddings: int = 2048

    # MoE 参数
    num_local_experts: int = 6
    num_experts_per_tok: int = 2
    expert_intermediate_size: int = 2389  # 8/3 × hidden_size
    router_aux_loss_coef: float = 0.01    # 标准 Mixtral 值
    router_jitter_noise: float = 0.01

    # CPT-MoE probability Router (strict v1: P x -> stable L2)
    cpt_router_version: int = 1
    cpt_num_prototypes: Optional[int] = None  # derived as K = 2N
    cpt_projection_dim: int = 128
    cpt_rho_beta: float = 0.95
    cpt_beta_max: float = 0.45
    cpt_kappa_beta: Optional[float] = None
    cpt_lambda_sa: Optional[float] = None
    cpt_prototype_temperature: Optional[float] = None
    cpt_expert_temperature: float = 1.0
    cpt_state_step_size: Optional[float] = None
    cpt_state_radius: float = 1.0
    cpt_state_chunk_size: int = 32
    cpt_eps_z: float = 1e-6
    cpt_eps_m: float = 1e-6
    cpt_eps_init: float = 1e-8
    cpt_energy_init_scale: Optional[float] = None
    cpt_capacity_factor: float = 1.25
    cpt_price_learning_rate: Optional[float] = None
    cpt_init_seed: int = 0

    # 归一化 & 激活
    rms_norm_eps: float = 1e-6

    # RoPE (Qwen2.5 同款 theta)
    rope_theta: float = 1_000_000.0

    # Dropout
    attention_dropout: float = 0.0

    # 权重绑定
    tie_word_embeddings: bool = True

    # 初始化
    initializer_range: float = 0.02

    def __post_init__(self):
        for name in CPT_ROUTING_ARCHITECTURE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be a positive integer")

        positive_fields = (
            "vocab_size", "hidden_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "max_position_embeddings", "num_local_experts",
            "num_experts_per_tok", "expert_intermediate_size",
        )
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

        if (
            isinstance(self.cpt_router_version, bool)
            or not isinstance(self.cpt_router_version, int)
            or self.cpt_router_version != 1
        ):
            raise ValueError("only cpt_router_version=1 is supported")
        if self.num_local_experts < 2:
            raise ValueError(
                "num_local_experts must be an integer of at least 2 for CPT"
            )
        if self.cpt_num_prototypes is None:
            self.cpt_num_prototypes = 2 * self.num_local_experts
        if isinstance(self.cpt_num_prototypes, bool) or not isinstance(
            self.cpt_num_prototypes, int
        ):
            raise ValueError("cpt_num_prototypes must be an integer")
        if self.cpt_num_prototypes != 2 * self.num_local_experts:
            raise ValueError("cpt_num_prototypes must equal 2 * num_local_experts")
        if self.num_experts_per_tok != 2:
            raise ValueError("num_experts_per_tok must remain 2 for CPT v1")
        if (
            isinstance(self.cpt_projection_dim, bool)
            or not isinstance(self.cpt_projection_dim, int)
            or self.cpt_projection_dim <= 0
        ):
            raise ValueError("cpt_projection_dim must be a positive integer")
        if self.cpt_projection_dim == 1 and self.cpt_num_prototypes > 2:
            raise ValueError(
                "cpt_projection_dim=1 cannot initialize more than two distinct "
                "unit anchors"
            )
        if (
            isinstance(self.cpt_init_seed, bool)
            or not isinstance(self.cpt_init_seed, int)
            or self.cpt_init_seed < 0
        ):
            raise ValueError("cpt_init_seed must be a non-negative integer")
        if (
            isinstance(self.cpt_state_chunk_size, bool)
            or not isinstance(self.cpt_state_chunk_size, int)
            or self.cpt_state_chunk_size <= 0
        ):
            raise ValueError("cpt_state_chunk_size must be a positive integer")
        max_layer_seed = self.cpt_init_seed + 104_729 * (
            self.num_hidden_layers - 1
        )
        if max_layer_seed > (1 << 64) - 1:
            raise ValueError("cpt_init_seed and layer offsets must fit uint64")

        base_cpt_scalars = (
            "cpt_rho_beta",
            "cpt_beta_max",
            "cpt_expert_temperature",
            "cpt_state_radius",
            "cpt_eps_z",
            "cpt_eps_m",
            "cpt_eps_init",
            "cpt_capacity_factor",
        )
        for name in base_cpt_scalars:
            setattr(self, name, _cpt_fp32(name, getattr(self, name)))
        if not 0.0 <= self.cpt_rho_beta < 1.0:
            raise ValueError("cpt_rho_beta must be in [0, 1)")
        if self.cpt_state_radius <= 0.0:
            raise ValueError("cpt_state_radius must be positive")
        beta_limit = 1.0 / (1.0 + self.cpt_state_radius)
        if not 0.0 < self.cpt_beta_max < beta_limit:
            raise ValueError(
                "cpt_beta_max must be in (0, 1 / (1 + cpt_state_radius))"
            )
        if self.cpt_expert_temperature <= 0.0:
            raise ValueError("cpt_expert_temperature must be positive")
        for name in ("cpt_eps_z", "cpt_eps_m", "cpt_eps_init"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.cpt_capacity_factor < 1.0:
            raise ValueError("cpt_capacity_factor must be at least 1")

        default_kappa_beta = 1.0 / (
            self.cpt_num_prototypes * (1.0 - self.cpt_rho_beta)
        )
        independent_defaults = {
            "cpt_kappa_beta": default_kappa_beta,
            "cpt_lambda_sa": 1.0 / self.cpt_num_prototypes,
            "cpt_prototype_temperature": self.cpt_projection_dim ** -0.5,
            "cpt_energy_init_scale": 0.05 * self.cpt_expert_temperature,
            "cpt_price_learning_rate": 1e-2 * self.cpt_expert_temperature,
        }
        for name, default in independent_defaults.items():
            value = default if getattr(self, name) is None else getattr(self, name)
            setattr(self, name, _cpt_fp32(name, value))

        if self.cpt_kappa_beta <= 0.0:
            raise ValueError("cpt_kappa_beta must be positive")
        if self.cpt_lambda_sa < 0.0:
            raise ValueError("cpt_lambda_sa must be non-negative")
        if self.cpt_prototype_temperature <= 0.0:
            raise ValueError("cpt_prototype_temperature must be positive")

        default_state_step_size = 0.1 / (1.0 + self.cpt_lambda_sa)
        state_step_size = (
            default_state_step_size
            if self.cpt_state_step_size is None
            else self.cpt_state_step_size
        )
        self.cpt_state_step_size = _cpt_fp32(
            "cpt_state_step_size",
            state_step_size,
        )
        max_state_step = 1.0 / (2.0 * (1.0 + self.cpt_lambda_sa))
        if not 0.0 < self.cpt_state_step_size <= max_state_step:
            raise ValueError(
                "cpt_state_step_size must be in (0, 1 / (2 * (1 + cpt_lambda_sa))]"
            )
        if self.cpt_energy_init_scale <= 0.0:
            raise ValueError("cpt_energy_init_scale must be positive")
        if self.cpt_price_learning_rate <= 0.0:
            raise ValueError("cpt_price_learning_rate must be positive")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.num_experts_per_tok > self.num_local_experts:
            raise ValueError("num_experts_per_tok cannot exceed num_local_experts")
        if self.router_aux_loss_coef < 0 or self.router_jitter_noise < 0:
            raise ValueError("router loss coefficient and jitter noise must be non-negative")
        if self.attention_dropout < 0 or self.attention_dropout >= 1:
            raise ValueError("attention_dropout must be in [0, 1)")
        if self.initializer_range <= 0 or self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("initializer_range, rms_norm_eps, and rope_theta must be positive")

    @classmethod
    def from_dict(cls, d: dict) -> "TinyMixtralConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__class__.__dataclass_fields__}

    def cpt_config_dict(self) -> dict:
        """Return only the fields that define CPT Router behavior."""
        return {
            key: getattr(self, key)
            for key in self.__class__.__dataclass_fields__
            if key.startswith("cpt_")
        }

    @classmethod
    def from_json_file(cls, path: str) -> "TinyMixtralConfig":
        import json
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save_pretrained(self, path: str):
        import json, os
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)
