# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

"""TinyMixtral 模型配置。"""

from dataclasses import dataclass


@dataclass
class TinyMixtralConfig:
    """TinyMixtral MoE 因果语言模型配置。

    参考: Qwen2.5-0.5B (896d/24L/dense), Mixtral 8x7B (MoE), Qwen1.5-MoE-A2.7B
    """

    # 基础架构 (Qwen2.5-0.5B 对齐)
    vocab_size: int = 32000
    hidden_size: int = 896
    num_hidden_layers: int = 10
    num_attention_heads: int = 16
    num_key_value_heads: int = 4  # GQA 4:1 (was 14:2 = 7:1)
    head_dim: int = 56  # 16 × 56 = 896
    max_position_embeddings: int = 2048

    # MoE 参数 (DeepSeek-style: shared + routed)
    num_shared_experts: int = 1  # always-on
    num_routed_experts: int = 6  # top-k selected
    num_experts_per_tok: int = 2
    expert_intermediate_size: int = 2389  # 8/3 × hidden_size

    @property
    def num_local_experts(self) -> int:
        return self.num_shared_experts + self.num_routed_experts

    router_aux_loss_coef: float = 0.01  # 标准 Mixtral 值
    router_jitter_noise: float = 0.01

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
        positive_fields = (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "num_shared_experts",
            "num_routed_experts",
            "num_experts_per_tok",
            "expert_intermediate_size",
        )
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.num_experts_per_tok > self.num_routed_experts:
            raise ValueError("num_experts_per_tok cannot exceed num_routed_experts")
        if self.router_aux_loss_coef < 0 or self.router_jitter_noise < 0:
            raise ValueError("router loss coefficient and jitter noise must be non-negative")
        if self.attention_dropout < 0 or self.attention_dropout >= 1:
            raise ValueError("attention_dropout must be in [0, 1)")
        if self.initializer_range <= 0 or self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("initializer_range, rms_norm_eps, and rope_theta must be positive")

    @classmethod
    def from_dict(cls, d: dict) -> "TinyMixtralConfig":
        d = dict(d)
        if "num_local_experts" in d and "num_shared_experts" not in d:
            raise ValueError(
                "Old config format (num_local_experts without num_shared_experts). "
                "This checkpoint was trained with the v1/v1.1 architecture. "
                "Use a current checkpoint or retrain from scratch."
            )
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__class__.__dataclass_fields__}
        d["num_local_experts"] = self.num_local_experts  # for readability
        return d

    @classmethod
    def from_json_file(cls, path: str) -> "TinyMixtralConfig":
        import json

        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save_pretrained(self, path: str):
        import json
        import os

        os.makedirs(path, exist_ok=True)
        with open(f"{path}/config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)
