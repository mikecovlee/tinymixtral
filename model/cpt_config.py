# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

"""CPT-MoE 配置：TinyMixtralConfig 的 CPT Router 扩展。

CPTConfig 继承主分支的 TinyMixtralConfig，仅新增 CPT Router 字段与校验，
主分支字段的语义完全不变。auxiliary router loss 与 jitter 不属于 CPT 语义，
初始化时直接写死为 0（传入值被忽略）。

标量参数接受精确分数字符串（如 "19/20"）；序列化只保存数学输入，
绝不保存运行期 FP32 近似值，跨主机可精确重推导。
"""

import json
from dataclasses import dataclass
from fractions import Fraction

from .config import TinyMixtralConfig
from .cpt_constants import exact_scalar, inverse_sqrt_fp32, rational_fp32

CPT_ROUTING_ARCHITECTURE_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_local_experts",
)


@dataclass
class CPTConfig(TinyMixtralConfig):
    """CPT-MoE probability Router 配置 (P x -> stable L2 -> B^T q)。"""

    # CPT-MoE probability Router (P x -> stable L2 -> B^T q)
    cpt_router_version: int = 2
    cpt_num_prototypes: int | None = None  # derived as K = 2N
    cpt_projection_dim: int | None = None  # derived as K - 1
    cpt_rho_beta: float | str = "19/20"
    cpt_beta_max: float | str = "9/20"
    cpt_kappa_beta: float | str | None = None
    cpt_lambda_sa: float | str | None = None
    cpt_prototype_temperature: float | str | None = None
    cpt_expert_temperature: float | str = "3/5"
    cpt_state_step_size: float | str | None = None
    cpt_state_radius: float | str = "1"
    cpt_state_chunk_size: int = 128
    cpt_state_corrector: bool = True
    cpt_eps_z: float | str = "1/1000000"
    cpt_eps_m: float | str = "1/1000000"
    cpt_eps_init: float | str = "1/100000000"
    cpt_energy_init_scale: float | str | None = "2"
    cpt_capacity_factor: float | str = "5/4"
    cpt_price_learning_rate: float | str | None = None
    cpt_init_seed: int = 0

    def __post_init__(self):
        # CPT 没有 auxiliary router loss 与 jitter：初始化时直接写死 0。
        self.router_aux_loss_coef = 0.0
        self.router_jitter_noise = 0.0
        super().__post_init__()

        for name in CPT_ROUTING_ARCHITECTURE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be a positive integer")

        if (
            isinstance(self.cpt_router_version, bool)
            or not isinstance(self.cpt_router_version, int)
            or self.cpt_router_version != 2
        ):
            raise ValueError("only cpt_router_version=2 is supported")
        if self.num_local_experts < 2:
            raise ValueError("num_local_experts must be an integer of at least 2 for CPT")
        if self.cpt_num_prototypes is None:
            self.cpt_num_prototypes = 2 * self.num_local_experts
        if isinstance(self.cpt_num_prototypes, bool) or not isinstance(self.cpt_num_prototypes, int):
            raise ValueError("cpt_num_prototypes must be an integer")
        if self.cpt_num_prototypes != 2 * self.num_local_experts:
            raise ValueError("cpt_num_prototypes must equal 2 * num_local_experts")
        if type(self.num_experts_per_tok) is not int:
            raise ValueError("num_experts_per_tok must be an integer")
        if self.cpt_projection_dim is None:
            self.cpt_projection_dim = self.cpt_num_prototypes - 1
        if self.cpt_projection_dim != self.cpt_num_prototypes - 1:
            raise ValueError("cpt_projection_dim must equal cpt_num_prototypes - 1")
        if (
            isinstance(self.cpt_projection_dim, bool)
            or not isinstance(self.cpt_projection_dim, int)
            or self.cpt_projection_dim <= 0
        ):
            raise ValueError("cpt_projection_dim must be a positive integer")
        if self.cpt_projection_dim == 1 and self.cpt_num_prototypes > 2:
            raise ValueError("cpt_projection_dim=1 cannot initialize more than two distinct unit anchors")
        if isinstance(self.cpt_init_seed, bool) or not isinstance(self.cpt_init_seed, int) or self.cpt_init_seed < 0:
            raise ValueError("cpt_init_seed must be a non-negative integer")
        if (
            isinstance(self.cpt_state_chunk_size, bool)
            or not isinstance(self.cpt_state_chunk_size, int)
            or self.cpt_state_chunk_size <= 0
        ):
            raise ValueError("cpt_state_chunk_size must be a positive integer")
        if not isinstance(self.cpt_state_corrector, bool):
            raise ValueError("cpt_state_corrector must be a boolean")
        max_layer_seed = self.cpt_init_seed + 104_729 * (self.num_hidden_layers - 1)
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
        derived_scalars = (
            "cpt_kappa_beta", "cpt_lambda_sa", "cpt_prototype_temperature",
            "cpt_state_step_size", "cpt_energy_init_scale", "cpt_price_learning_rate",
        )
        # Save mathematical inputs, never their runtime FP32 approximations.
        exact = {
            name: None if getattr(self, name) is None else exact_scalar(name, getattr(self, name))
            for name in base_cpt_scalars + derived_scalars
        }
        self._cpt_scalar_sources = {
            name: None if value is None else str(value) for name, value in exact.items()
        }
        for name in base_cpt_scalars:
            if exact[name] is None:
                raise ValueError(f"{name} requires an exact scalar")
            setattr(self, name, rational_fp32(name, exact[name]))
        if not 0.0 <= self.cpt_rho_beta < 1.0:
            raise ValueError("cpt_rho_beta must be in [0, 1)")
        if self.cpt_state_radius <= 0.0:
            raise ValueError("cpt_state_radius must be positive")
        beta_limit = 1.0 / (1.0 + self.cpt_state_radius)
        if not 0.0 < self.cpt_beta_max < beta_limit:
            raise ValueError("cpt_beta_max must be in (0, 1 / (1 + cpt_state_radius))")
        if self.cpt_expert_temperature <= 0.0:
            raise ValueError("cpt_expert_temperature must be positive")
        for name in ("cpt_eps_z", "cpt_eps_m", "cpt_eps_init"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.cpt_capacity_factor < 1.0:
            raise ValueError("cpt_capacity_factor must be at least 1")

        derived_temperature = inverse_sqrt_fp32("cpt_prototype_temperature", self.cpt_projection_dim)
        if exact["cpt_prototype_temperature"] is not None and rational_fp32(
            "cpt_prototype_temperature", exact["cpt_prototype_temperature"]
        ) != derived_temperature:
            raise ValueError("cpt_prototype_temperature must equal 1 / sqrt(cpt_projection_dim)")
        self.cpt_prototype_temperature = derived_temperature

        default_kappa_beta = 1 / (self.cpt_num_prototypes * (1 - exact["cpt_rho_beta"]))
        independent_defaults = {
            "cpt_kappa_beta": default_kappa_beta,
            "cpt_lambda_sa": Fraction(1, self.cpt_num_prototypes),
            "cpt_energy_init_scale": Fraction(1, 20) * exact["cpt_expert_temperature"],
            "cpt_price_learning_rate": Fraction(1, 100) * exact["cpt_expert_temperature"],
        }
        for name, default in independent_defaults.items():
            exact[name] = default if exact[name] is None else exact[name]
            setattr(self, name, rational_fp32(name, exact[name]))
        if self.cpt_kappa_beta <= 0.0:
            raise ValueError("cpt_kappa_beta must be positive")
        if self.cpt_lambda_sa < 0.0:
            raise ValueError("cpt_lambda_sa must be non-negative")
        if self.cpt_prototype_temperature <= 0.0:
            raise ValueError("cpt_prototype_temperature must be positive")

        default_state_step_size = Fraction(1, 10) / (1 + exact["cpt_lambda_sa"])
        state_step_size = default_state_step_size if exact["cpt_state_step_size"] is None else exact["cpt_state_step_size"]
        self.cpt_state_step_size = rational_fp32("cpt_state_step_size", state_step_size)
        max_state_step = 1.0 / (2.0 * (1.0 + self.cpt_lambda_sa))
        if not 0.0 < self.cpt_state_step_size <= max_state_step:
            raise ValueError("cpt_state_step_size must be in (0, 1 / (2 * (1 + cpt_lambda_sa))]")
        if self.cpt_energy_init_scale <= 0.0:
            raise ValueError("cpt_energy_init_scale must be positive")
        if self.cpt_price_learning_rate <= 0.0:
            raise ValueError("cpt_price_learning_rate must be positive")

    def to_dict(self) -> dict:
        result = super().to_dict()
        result.update(self._cpt_scalar_sources)
        return result

    def cpt_config_dict(self) -> dict:
        """Return only the fields that define CPT Router behavior."""
        return {key: getattr(self, key) for key in self.__class__.__dataclass_fields__ if key.startswith("cpt_")}


def config_from_json_file(path: str) -> TinyMixtralConfig:
    """按 config.json 内容分发配置类。

    含 ``cpt_router_version`` 键 ⇒ CPTConfig，否则主分支的 TinyMixtralConfig。
    主分支的 config.json 不含 CPT 字段，自动以线性路由模式加载。
    """
    with open(path) as f:
        data = json.load(f)
    cls = CPTConfig if "cpt_router_version" in data else TinyMixtralConfig
    return cls.from_dict(data)
