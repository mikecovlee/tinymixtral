# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

"""TinyMixtral 模型配置。"""

from dataclasses import dataclass
import math
from numbers import Real
import struct


CPT_ROUTER_RECOMPUTE_MODES = ("global", "on", "off")


def _coerce_finite_real(name: str, value) -> float:
    """Return a canonical finite float and reject booleans/non-real values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _float32_effective(name: str, value: float) -> float:
    """Return the exact FP32 scalar used by Router/model tensor operations."""
    try:
        effective = struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(f"{name} must remain finite in FP32") from error
    if not math.isfinite(effective):
        raise ValueError(f"{name} must remain finite in FP32")
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

    # CPT-MoE 概率路由（v1）
    cpt_router_version: int = 1
    cpt_router_recompute: str = "global"
    cpt_num_prototypes: int = 12
    cpt_projection_dim: int = 128
    cpt_rho_beta: float = 0.95
    cpt_beta_max: float = 0.45
    cpt_expert_temperature: float = 1.0
    cpt_state_radius: float = 1.0
    cpt_eps_z: float = 1e-6
    cpt_eps_m: float = 1e-6
    cpt_eps_init: float = 1e-8
    cpt_capacity_factor: float = 1.25
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
        finite_real_fields = (
            "cpt_rho_beta",
            "cpt_beta_max",
            "cpt_expert_temperature",
            "cpt_state_radius",
            "cpt_eps_z",
            "cpt_eps_m",
            "cpt_eps_init",
            "cpt_capacity_factor",
            "rms_norm_eps",
            "rope_theta",
            "attention_dropout",
            "initializer_range",
        )
        for name in finite_real_fields:
            setattr(self, name, _coerce_finite_real(name, getattr(self, name)))

        positive_integer_fields = (
            "vocab_size", "hidden_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "max_position_embeddings", "num_local_experts",
            "num_experts_per_tok", "expert_intermediate_size",
            "cpt_num_prototypes", "cpt_projection_dim",
        )
        for name in positive_integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.cpt_router_version, bool)
            or not isinstance(self.cpt_router_version, int)
            or self.cpt_router_version != 1
        ):
            raise ValueError("only cpt_router_version=1 is supported")
        if not isinstance(self.cpt_router_recompute, str):
            raise ValueError("cpt_router_recompute must be a string")
        if self.cpt_router_recompute not in CPT_ROUTER_RECOMPUTE_MODES:
            raise ValueError(
                "cpt_router_recompute must be one of: global, on, off"
            )
        if isinstance(self.cpt_init_seed, bool) or not isinstance(
            self.cpt_init_seed,
            int,
        ):
            raise ValueError("cpt_init_seed must be an integer")
        if self.cpt_init_seed < 0:
            raise ValueError("cpt_init_seed must be non-negative")
        max_layer_seed = self.cpt_init_seed + 104_729 * (
            self.num_hidden_layers - 1
        )
        if max_layer_seed > (1 << 64) - 1:
            raise ValueError(
                "cpt_init_seed plus the per-layer seed offset must fit in "
                "PyTorch's unsigned 64-bit manual_seed range"
            )
        if not isinstance(self.tie_word_embeddings, bool):
            raise ValueError("tie_word_embeddings must be a boolean")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.num_experts_per_tok > self.num_local_experts:
            raise ValueError("num_experts_per_tok cannot exceed num_local_experts")
        if self.num_local_experts < 2:
            raise ValueError("num_local_experts must be at least 2 for CPT routing")
        if self.cpt_num_prototypes < 2:
            raise ValueError("cpt_num_prototypes must be at least 2")
        if (
            self.cpt_projection_dim == 1
            and self.cpt_num_prototypes > 2
        ):
            raise ValueError(
                "cpt_projection_dim=1 supports at most 2 distinct unit "
                "prototype anchors"
            )
        if not 0 <= self.cpt_rho_beta < 1:
            raise ValueError("cpt_rho_beta must be in [0, 1)")
        rho_beta_fp32 = self.cpt_rho_beta_effective
        if rho_beta_fp32 >= 1:
            raise ValueError(
                "cpt_rho_beta must remain below 1 when represented in FP32"
            )
        if self.cpt_state_radius <= 0:
            raise ValueError("cpt_state_radius must be positive")
        beta_upper_bound = 1.0 / (1.0 + self.cpt_state_radius)
        if not 0 <= self.cpt_beta_max < beta_upper_bound:
            raise ValueError(
                "cpt_beta_max must be in [0, 1 / (1 + cpt_state_radius))"
            )
        if self.cpt_expert_temperature <= 0:
            raise ValueError("cpt_expert_temperature must be positive")
        if self.cpt_capacity_factor < 1:
            raise ValueError("cpt_capacity_factor must be at least 1")
        for name in ("cpt_eps_z", "cpt_eps_m", "cpt_eps_init"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1)")
        if self.initializer_range <= 0 or self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("initializer_range, rms_norm_eps, and rope_theta must be positive")
        self._validate_fp32_contract()

    def _validate_fp32_contract(self) -> None:
        effective = {
            name: _float32_effective(name, getattr(self, name))
            for name in (
                "cpt_rho_beta",
                "cpt_beta_max",
                "cpt_expert_temperature",
                "cpt_state_radius",
                "cpt_eps_z",
                "cpt_eps_m",
                "cpt_eps_init",
                "cpt_capacity_factor",
                "rms_norm_eps",
                "rope_theta",
                "attention_dropout",
                "initializer_range",
            )
        }
        positive = (
            "cpt_expert_temperature",
            "cpt_state_radius",
            "cpt_eps_z",
            "cpt_eps_m",
            "cpt_eps_init",
            "rms_norm_eps",
            "rope_theta",
            "initializer_range",
        )
        for name in positive:
            if effective[name] <= 0:
                raise ValueError(f"{name} must remain positive in FP32")
        if not 0 <= effective["cpt_rho_beta"] < 1:
            raise ValueError("cpt_rho_beta must remain in [0, 1) in FP32")
        if not 0 <= effective["attention_dropout"] < 1:
            raise ValueError("attention_dropout must remain in [0, 1) in FP32")
        if effective["cpt_capacity_factor"] < 1:
            raise ValueError("cpt_capacity_factor must remain at least 1 in FP32")
        beta_limit = 1.0 / (1.0 + effective["cpt_state_radius"])
        if not 0 <= effective["cpt_beta_max"] < beta_limit:
            raise ValueError(
                "cpt_beta_max must remain below 1 / (1 + cpt_state_radius) "
                "in FP32"
            )

        derived = {
            "cpt_kappa_beta": 1.0
            / (
                self.cpt_num_prototypes
                * (1.0 - effective["cpt_rho_beta"])
            ),
            "cpt_lambda_sa": 1.0 / self.cpt_num_prototypes,
            "cpt_projection_temperature": self.cpt_projection_dim ** -0.5,
            "cpt_state_step_size": 0.1
            / (1.0 + 1.0 / self.cpt_num_prototypes),
            "cpt_energy_init_scale": 0.05
            * effective["cpt_expert_temperature"],
            "cpt_price_learning_rate": 1e-2
            * effective["cpt_expert_temperature"],
        }
        for name, value in derived.items():
            if _float32_effective(name, value) <= 0:
                raise ValueError(f"{name} must remain positive in FP32")

    @property
    def cpt_rho_beta_effective(self) -> float:
        """Return the FP32 value used by the sequence-state recurrence."""
        return struct.unpack(
            "f",
            struct.pack("f", self.cpt_rho_beta),
        )[0]

    @property
    def cpt_kappa_beta(self) -> float:
        """Return kappa_beta from the effective FP32 recurrence decay."""
        return 1.0 / (
            self.cpt_num_prototypes
            * (1.0 - self.cpt_rho_beta_effective)
        )

    @property
    def cpt_lambda_sa(self) -> float:
        """Short-term-state anchor regularization coefficient."""
        return 1.0 / self.cpt_num_prototypes

    @property
    def cpt_projection_temperature(self) -> float:
        """Return tau_p = 1 / sqrt(d_p)."""
        return self.cpt_projection_dim ** -0.5

    @property
    def cpt_state_step_size(self) -> float:
        """Return alpha_S = 0.1 / (1 + lambda_sa)."""
        return 0.1 / (1.0 + self.cpt_lambda_sa)

    @property
    def cpt_energy_init_scale(self) -> float:
        """Return epsilon_C = 0.05 tau_e."""
        return 0.05 * self.cpt_expert_temperature

    @property
    def cpt_price_learning_rate(self) -> float:
        """Return eta_lambda = 1e-2 tau_e."""
        return 1e-2 * self.cpt_expert_temperature

    @classmethod
    def from_dict(cls, d: dict) -> "TinyMixtralConfig":
        if not isinstance(d, dict):
            raise TypeError("config payload must be a dictionary")
        payload = dict(d)
        # Older CPT checkpoints predate this execution-only field.  Their
        # historical behavior is exactly the new ``global`` default.
        payload.setdefault("cpt_router_recompute", "global")
        field_names = tuple(cls.__dataclass_fields__)
        missing = [name for name in field_names if name not in payload]
        if missing:
            raise ValueError(
                "config payload is missing behavior-critical fields: "
                + ", ".join(missing)
            )
        # Preserve the repository's forward-compatibility contract for future
        # metadata while refusing any omission of a field that affects model
        # behavior.  Publication performs its own exact canonical comparison.
        return cls(**{name: payload[name] for name in field_names})

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__class__.__dataclass_fields__}

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
