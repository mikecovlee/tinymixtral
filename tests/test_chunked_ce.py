# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

import pytest
import torch

from model.config import TinyMixtralConfig
from model.modeling import TinyMixtralForCausalLM


def tiny_model() -> TinyMixtralForCausalLM:
    cfg = TinyMixtralConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=64,
        max_position_embeddings=64, num_local_experts=4,
        num_experts_per_tok=2, expert_intermediate_size=48,
        router_aux_loss_coef=0.01, router_jitter_noise=0.0,
    )
    torch.manual_seed(13)
    return TinyMixtralForCausalLM(cfg)


def batch(with_ignore: bool = False):
    torch.manual_seed(17)
    ids = torch.randint(0, 64, (2, 11))
    labels = ids.clone()
    if with_ignore:
        labels[:, 3] = -100
        labels[0, 7] = -100
    return ids, labels


def grads(model):
    return {n: (None if p.grad is None else p.grad.clone())
            for n, p in model.named_parameters()}


@pytest.mark.parametrize("with_ignore", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_chunked_matches_monolithic_grads(training, with_ignore):
    model = tiny_model()
    model.train() if training else model.eval()
    if training:
        model.gradient_checkpointing_enable()
    ids, labels = batch(with_ignore)

    model.use_chunked_ce = False
    out_a = model(ids, labels=labels)
    if training:
        out_a["loss"].backward()
        grads_a = grads(model)
        model.zero_grad(set_to_none=True)

    model.use_chunked_ce = True
    model.ce_chunk_size = 5
    out_b = model(ids, labels=labels)
    if training:
        out_b["loss"].backward()
        grads_b = grads(model)
    else:
        grads_b = grads_a = {}

    assert torch.allclose(out_a["loss"], out_b["loss"], atol=1e-5)
    assert out_b["logits"] is None or out_b["logits"].shape == out_a["logits"].shape
    for name in grads_a:
        ga, gb = grads_a[name], grads_b[name]
        assert (ga is None) == (gb is None), name
        if ga is not None:
            assert torch.allclose(ga, gb, atol=1e-5), name


def test_training_path_skips_logits():
    model = tiny_model().train()
    model.use_chunked_ce = True
    ids, labels = batch()
    out = model(ids, labels=labels)
    assert out["logits"] is None
    assert out["ce_loss"] is not None and out["loss"] is not None


def test_chunk_boundaries_full_size():
    model = tiny_model().eval()
    ids, labels = batch(with_ignore=True)
    with torch.no_grad():
        model.use_chunked_ce = False
        ref = model(ids, labels=labels)["loss"]
        model.ce_chunk_size = 1
        out = model(ids, labels=labels)
        tiny = out["loss"]
        model.ce_chunk_size = 10_000
        whole = model(ids, labels=labels)["loss"]
    assert torch.allclose(ref, tiny, atol=1e-5)
    assert torch.allclose(ref, whole, atol=1e-5)
