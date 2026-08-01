import pytest
import torch

from model.config import TinyMixtralConfig


@pytest.fixture
def tiny_config() -> TinyMixtralConfig:
    """Small deterministic configuration for CPU/CUDA smoke tests."""
    return TinyMixtralConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        expert_intermediate_size=24,
        cpt_num_prototypes=4,
        cpt_projection_dim=8,
        cpt_init_seed=17,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        initializer_range=0.02,
    )


@pytest.fixture
def fixed_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor(
        [
            [1, 3, 5, 7, 9, 11, 13, 15],
            [2, 4, 6, 8, 10, 12, 14, 16],
        ],
        dtype=torch.long,
    )
    labels = torch.tensor(
        [
            [3, 5, 7, 9, 11, 13, 15, 17],
            [4, 6, 8, 10, 12, 14, 16, 18],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    return input_ids, labels, attention_mask
