import torch

from hf.configuration_tinymixtral import TinyMixtralConfig
from hf.modeling_tinymixtral import TinyMixtralForCausalLM
from scripts.train_utils import execute_training_iteration, make_adamw


def test_hf_model_strict_training_iteration_commits_one_cpt_transaction():
    torch.manual_seed(20260731)
    config = TinyMixtralConfig(
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
    model = TinyMixtralForCausalLM(config).train()
    model.gradient_checkpointing_enable()
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    input_ids = torch.tensor(
        [
            [1, 3, 5, 7, 9, 11, 13, 15],
            [2, 4, 6, 8, 10, 12, 14, 16],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    routers = [layer.moe.cpt_router for layer in model.layers]
    versions_before = [int(router.state_version.item()) for router in routers]

    output, grad_norm = execute_training_iteration(
        model,
        optimizer,
        scheduler=None,
        forward_fn=lambda: model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        ),
        max_grad_norm=1.0,
    )

    transaction = output["cpt_transaction"]
    assert torch.isfinite(grad_norm)
    assert transaction.closed
    assert not transaction.aborted
    assert model.get_cpt_state_version() == versions_before[0] + 1

    for router, version_before in zip(routers, versions_before):
        assert int(router.state_version.item()) == version_before + 1
        torch.testing.assert_close(
            router.anchors.detach().float().norm(dim=0),
            torch.ones(config.cpt_num_prototypes),
            rtol=1e-5,
            atol=1e-6,
        )
        for fp32_state in (
            router.projection,
            router.anchors,
            router.energy,
            router.congestion_price,
        ):
            assert fp32_state.dtype == torch.float32
            assert torch.isfinite(fp32_state.detach()).all()
        assert router.congestion_price.shape == (config.num_local_experts,)
        assert torch.all(router.congestion_price >= 0)
        router.validate_persistent_invariants(layer_index=router.layer_index)

    assert all(parameter.grad is None for parameter in model.parameters())
