"""CUDA-only tests for the compiled CPT Router fast path.

The router compiles its forward only on CUDA during training, so the CPU
suite never exercises that path.  These tests pin the compiled path to the
eager implementation and cover the gating rules, the FP32 precision island
under bf16 autocast, activation checkpointing, transaction commit and device
moves.  They are skipped automatically when CUDA is unavailable.
"""

import pytest
import torch

from model.config import TinyMixtralConfig
from model.cpt_router import CPTRouter
from model.modeling import TinyMixtralForCausalLM

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for compiled-path tests")


def tiny_config(**overrides):
    values = {
        "vocab_size": 41,
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "max_position_embeddings": 32,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "expert_intermediate_size": 16,
        "cpt_projection_dim": 4,
        "cpt_state_chunk_size": 1,
        "cpt_init_seed": 17,
    }
    values.update(overrides)
    return TinyMixtralConfig(**values)


def make_router(**overrides):
    return CPTRouter(tiny_config(**overrides), layer_index=0).cuda()


def random_inputs(seed, batch=2, seq=9, hidden_size=8):
    torch.manual_seed(seed)
    hidden = torch.randn(batch, seq, hidden_size, device="cuda", requires_grad=True)
    mask = torch.randint(0, 2, (batch, seq), device="cuda", dtype=torch.bool)
    mask[:, 0] = True
    return hidden, mask


@pytest.mark.parametrize("corrector", [False, True])
def test_compiled_forward_matches_eager_probabilities_and_gradients(corrector):
    router = make_router(cpt_state_chunk_size=4, cpt_state_corrector=corrector)
    results = {}
    for mode in ("eager", "compiled"):
        hidden, mask = random_inputs(seed=5)
        if mode == "eager":
            output = router._forward_impl(hidden, mask)
        else:
            output = router(hidden, mask)
        output.probabilities.square().sum().backward()
        results[mode] = (
            output.probabilities.detach().clone(),
            output.proposal.load_sum.detach().clone(),
            hidden.grad.detach().clone(),
            tuple(parameter.grad.detach().clone() for parameter in router.trainable_parameters()),
        )
        router.zero_grad()
    eager_result, compiled_result = results["eager"], results["compiled"]
    torch.testing.assert_close(compiled_result[0], eager_result[0], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(compiled_result[1], eager_result[1], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(compiled_result[2], eager_result[2], atol=2e-6, rtol=2e-6)
    for compiled_grad, eager_grad in zip(compiled_result[3], eager_result[3]):
        torch.testing.assert_close(compiled_grad, eager_grad, atol=2e-6, rtol=2e-6)
    assert router._compiled_forward_impl is not None


def test_compiled_forward_under_bf16_autocast_matches_eager_and_keeps_fp32():
    router = make_router(cpt_state_chunk_size=4)
    hidden, mask = random_inputs(seed=7)
    hidden_bf16 = hidden.detach().to(torch.bfloat16)
    eager_output = router._forward_impl(hidden_bf16, mask)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        compiled_output = router(hidden_bf16, mask)
    assert compiled_output.probabilities.dtype == torch.float32
    assert compiled_output.proposal.load_sum.dtype == torch.float32
    assert router.congestion_price.dtype == torch.float32
    for parameter in router.trainable_parameters():
        assert parameter.dtype == torch.float32
    torch.testing.assert_close(compiled_output.probabilities, eager_output.probabilities, atol=2e-6, rtol=2e-6)


def test_compiled_forward_preserves_mass_padding_and_gradients():
    router = make_router(cpt_state_chunk_size=4)
    hidden, mask = random_inputs(seed=11)
    output = router(hidden, mask)
    torch.testing.assert_close(
        output.probabilities[mask].sum(dim=-1),
        torch.ones(int(mask.sum()), device="cuda"),
        atol=2e-6,
        rtol=0,
    )
    assert torch.equal(output.probabilities[~mask], torch.zeros_like(output.probabilities[~mask]))
    assert bool(torch.isfinite(output.probabilities).all())
    output.probabilities.square().sum().backward()
    assert bool(torch.isfinite(hidden.grad).all())
    assert torch.equal(hidden.grad[~mask], torch.zeros_like(hidden.grad[~mask]))
    for parameter in router.trainable_parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())


def test_eval_mode_stays_eager_and_train_mode_compiles():
    router = make_router(cpt_state_chunk_size=4)
    hidden, mask = random_inputs(seed=13)
    router.eval()
    with torch.no_grad():
        router(hidden, mask)
    assert router._compiled_forward_impl is None
    router.train()
    router(hidden, mask)
    assert router._compiled_forward_impl is not None


def test_chunk_count_above_threshold_stays_eager_on_cuda():
    router = make_router(cpt_state_chunk_size=1)
    hidden = torch.randn(1, 200, 8, device="cuda")
    with torch.no_grad():
        router(hidden)
    assert router._compiled_forward_impl is None


def test_compiled_forward_tracks_parameter_updates():
    router = make_router(cpt_state_chunk_size=4)
    hidden, mask = random_inputs(seed=19)
    optimizer = torch.optim.AdamW(router.trainable_parameters(), lr=1e-2)
    with torch.no_grad():
        before = router(hidden, mask).probabilities.detach().clone()
    router(hidden, mask).probabilities.square().sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    compiled_after = router(hidden, mask).probabilities.detach()
    eager_after = router._forward_impl(hidden, mask).probabilities.detach()
    torch.testing.assert_close(compiled_after, eager_after, atol=2e-6, rtol=2e-6)
    assert not torch.allclose(before, compiled_after, atol=1e-7, rtol=1e-7)


def test_compiled_activation_checkpointing_matches_plain_forward_and_gradients():
    torch.manual_seed(23)
    config = tiny_config(cpt_state_chunk_size=4)
    plain = TinyMixtralForCausalLM(config).cuda()
    checkpointed = TinyMixtralForCausalLM(config).cuda()
    checkpointed.load_state_dict(plain.state_dict(), strict=True)
    checkpointed.gradient_checkpointing_enable()
    plain.train()
    checkpointed.train()
    input_ids = torch.randint(0, config.vocab_size, (2, 9), device="cuda")
    labels = torch.randint(0, config.vocab_size, (2, 9), device="cuda")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        plain_output = plain(input_ids, labels=labels)
        checkpoint_output = checkpointed(input_ids, labels=labels)
    torch.testing.assert_close(checkpoint_output["logits"], plain_output["logits"], atol=2e-5, rtol=2e-5)
    plain_output["loss"].backward()
    checkpoint_output["loss"].backward()
    for plain_parameter, checkpoint_parameter in zip(plain.cpt_trainable_parameters(), checkpointed.cpt_trainable_parameters()):
        torch.testing.assert_close(checkpoint_parameter.grad, plain_parameter.grad, atol=2e-5, rtol=2e-5)


def test_compiled_training_step_commits_transaction():
    torch.manual_seed(29)
    model = TinyMixtralForCausalLM(tiny_config(cpt_state_chunk_size=4)).cuda()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 9), device="cuda")
    attention_mask = torch.ones(2, 9, dtype=torch.long, device="cuda")
    attention_mask[1, 6:] = 0

    version = model.get_cpt_state_version()
    for step in range(2):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids, labels=input_ids.clone(), attention_mask=attention_mask)
        assert bool(torch.isfinite(output["loss"]))
        output["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert bool(torch.isfinite(grad_norm))
        optimizer.step()
        new_version = model.commit_cpt_transaction(output["cpt_transaction"], optimizer_step=step + 1)
        optimizer.zero_grad(set_to_none=True)
        assert new_version == version + step + 1
    model.validate_persistent_cpt_state()
    for layer in model.layers:
        assert layer.moe.cpt_router._compiled_forward_impl is not None


def test_device_move_clears_compiled_cache():
    router = make_router(cpt_state_chunk_size=4)
    hidden, mask = random_inputs(seed=31)
    router(hidden, mask)
    assert router._compiled_forward_impl is not None
    router.cpu()
    assert router._compiled_forward_impl is None
    hidden_cpu = torch.randn(2, 9, 8, requires_grad=True)
    router.train()
    router(hidden_cpu, mask.cpu())
    assert router._compiled_forward_impl is None
    router.cuda()
    assert router._compiled_forward_impl is None
