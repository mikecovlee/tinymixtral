# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

import math
from numbers import Real
import struct

from transformers import PretrainedConfig


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


class TinyMixtralConfig(PretrainedConfig):
    model_type = "tinymixtral"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 896,
        num_hidden_layers: int = 10,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        head_dim: int = 64,
        max_position_embeddings: int = 2048,
        num_local_experts: int = 6,
        num_experts_per_tok: int = 2,
        expert_intermediate_size: int = 2389,
        cpt_router_version: int = 1,
        cpt_router_recompute: str = "global",
        cpt_num_prototypes: int = 12,
        cpt_projection_dim: int = 128,
        cpt_rho_beta: float = 0.95,
        cpt_beta_max: float = 0.45,
        cpt_expert_temperature: float = 1.0,
        cpt_state_radius: float = 1.0,
        cpt_eps_z: float = 1e-6,
        cpt_eps_m: float = 1e-6,
        cpt_eps_init: float = 1e-8,
        cpt_capacity_factor: float = 1.25,
        cpt_init_seed: int = 0,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
        attention_dropout: float = 0.0,
        tie_word_embeddings: bool = True,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        # Drop retired Linear-router metadata when loading an older config.
        # Neither value is retained or serialized because CPT routing has no
        # multiplicative input jitter and no load-balancing task loss.
        kwargs.pop("router_aux_loss_coef", None)
        kwargs.pop("router_jitter_noise", None)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.expert_intermediate_size = expert_intermediate_size
        self.cpt_router_version = cpt_router_version
        self.cpt_router_recompute = cpt_router_recompute
        self.cpt_num_prototypes = cpt_num_prototypes
        self.cpt_projection_dim = cpt_projection_dim
        self.cpt_rho_beta = cpt_rho_beta
        self.cpt_beta_max = cpt_beta_max
        self.cpt_expert_temperature = cpt_expert_temperature
        self.cpt_state_radius = cpt_state_radius
        self.cpt_eps_z = cpt_eps_z
        self.cpt_eps_m = cpt_eps_m
        self.cpt_eps_init = cpt_eps_init
        self.cpt_capacity_factor = cpt_capacity_factor
        self.cpt_init_seed = cpt_init_seed
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self._validate()

    @property
    def cpt_rho_beta_effective(self) -> float:
        return struct.unpack(
            "f",
            struct.pack("f", self.cpt_rho_beta),
        )[0]

    @property
    def cpt_kappa_beta(self) -> float:
        return 1.0 / (
            self.cpt_num_prototypes
            * (1.0 - self.cpt_rho_beta_effective)
        )

    @property
    def cpt_lambda_sa(self) -> float:
        return 1.0 / self.cpt_num_prototypes

    @property
    def cpt_projection_temperature(self) -> float:
        return 1.0 / math.sqrt(self.cpt_projection_dim)

    @property
    def cpt_state_step_size(self) -> float:
        return 0.1 / (1.0 + self.cpt_lambda_sa)

    @property
    def cpt_energy_init_scale(self) -> float:
        return 0.05 * self.cpt_expert_temperature

    @property
    def cpt_price_learning_rate(self) -> float:
        return 1e-2 * self.cpt_expert_temperature

    def _validate(self) -> None:
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
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "num_local_experts",
            "num_experts_per_tok",
            "expert_intermediate_size",
            "cpt_num_prototypes",
            "cpt_projection_dim",
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
        if isinstance(self.cpt_init_seed, bool) or not isinstance(self.cpt_init_seed, int):
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
        beta_limit = 1.0 / (1.0 + self.cpt_state_radius)
        if not 0 <= self.cpt_beta_max < beta_limit:
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
