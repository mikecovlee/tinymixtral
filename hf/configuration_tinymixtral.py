# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

from transformers import PretrainedConfig


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
        router_aux_loss_coef: float = 0.01,
        router_jitter_noise: float = 0.01,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
        attention_dropout: float = 0.0,
        tie_word_embeddings: bool = True,
        initializer_range: float = 0.02,
        **kwargs,
    ):
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
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
