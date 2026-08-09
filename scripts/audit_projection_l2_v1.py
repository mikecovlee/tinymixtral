#!/usr/bin/env python3
# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

"""Focused audit for the strict CPT v1 projection and final probability path."""

import argparse
import ast
import inspect
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

from model.config import TinyMixtralConfig  # noqa: E402
from model.cpt_router import CPTRouter  # noqa: E402
from model.modeling import SparseMoE  # noqa: E402


def make_config():
    return TinyMixtralConfig(
        vocab_size=64,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        num_local_experts=3,
        num_experts_per_tok=2,
        expert_intermediate_size=16,
        cpt_projection_dim=4,
        cpt_init_seed=2026,
    )


@torch.no_grad()
def first_token_column_oracle(router, hidden):
    x = hidden[0, 0].float().unsqueeze(1)
    projected = router.projection @ x
    z = projected / max(float(torch.linalg.vector_norm(projected)), router.eps_z)
    prototypes = router.anchors / torch.linalg.vector_norm(router.anchors, dim=0, keepdim=True).clamp_min(router.eps_m)
    q = torch.softmax((prototypes.T @ z).squeeze(1) / router.prototype_temperature, dim=0)
    energy = router.energy - router.energy.mean(dim=-1, keepdim=True)
    kernel = torch.softmax(
        (energy - router.congestion_price.unsqueeze(0)) / router.expert_temperature,
        dim=-1,
    )
    return kernel.T @ q


def run(device):
    torch.manual_seed(0)
    config = make_config()
    router = CPTRouter(config, layer_index=0).to(device)
    hidden = torch.randn(2, 7, config.hidden_size, device=device, requires_grad=True)
    mask = torch.tensor(
        [[1, 1, 1, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0]],
        dtype=torch.bool,
        device=device,
    )

    output = router(hidden, mask)
    expected_first = first_token_column_oracle(router, hidden)
    torch.testing.assert_close(output.probabilities[0, 0], expected_first, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        output.probabilities[mask].sum(dim=-1),
        torch.ones(int(mask.sum()), device=device),
        atol=2e-6,
        rtol=0,
    )
    if not torch.equal(output.probabilities[~mask], torch.zeros_like(output.probabilities[~mask])):
        raise AssertionError("padding positions must be zero and must not be routed")
    if int(output.proposal.token_count) != int(mask.sum()):
        raise AssertionError("padding was included in CPT token_count")
    torch.testing.assert_close(
        output.proposal.load_sum.sum(),
        mask.sum(dtype=torch.float32),
        atol=2e-5,
        rtol=0,
    )

    weights = torch.tensor([0.25, -0.5, 1.0], device=device)
    (output.probabilities * weights).sum().backward()
    for name, tensor in (
        ("hidden", hidden),
        ("projection", router.projection),
        ("anchors", router.anchors),
        ("energy", router.energy),
    ):
        if tensor.grad is None or not bool(torch.isfinite(tensor.grad).all()):
            raise AssertionError(f"missing or non-finite {name} gradient")

    router_source = inspect.getsource(CPTRouter._forward_impl)
    chunk_source = inspect.getsource(CPTRouter._route_chunk)
    moe_source = inspect.getsource(SparseMoE.forward)
    required_fragments = (
        "projected = F.linear",
        "self._stable_l2(projected",
        "valid_probabilities = prototype_probabilities @ kernel",
    )
    for fragment in required_fragments:
        if fragment not in router_source:
            raise AssertionError(f"missing strict-v1 source fragment: {fragment}")
    if "q_chunk = F.softmax" not in chunk_source:
        raise AssertionError("missing strict-v1 source fragment: q_chunk = F.softmax")
    if router_source.count("@ kernel") != 1:
        raise AssertionError("final CPT probabilities must use exactly one GEMM")
    projection_block = router_source.split("projected = F.linear", 1)[1].split("kernel = self.expert_kernel", 1)[0]
    if "softmax" in projection_block:
        raise AssertionError("projection-softmax was found in strict CPT v1")
    moe_tree = ast.parse(textwrap.dedent(moe_source))
    has_softmax_call = any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == "softmax")
            or (isinstance(node.func, ast.Name) and node.func.id == "softmax")
        )
        for node in ast.walk(moe_tree)
    )
    if has_softmax_call:
        raise AssertionError("SparseMoE applies an illegal post-Pi softmax")

    if device.type == "cuda":
        bf16_router = CPTRouter(config, layer_index=0).to(device).to(torch.bfloat16)
        bf16_hidden = hidden.detach().to(torch.bfloat16)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            bf16_output = bf16_router(bf16_hidden, mask)
        if bf16_output.probabilities.dtype != torch.float32:
            raise AssertionError("CPT probability path must remain FP32")
        if bf16_router.congestion_price.dtype != torch.float32:
            raise AssertionError("congestion price must remain FP32")

    return {
        "device": str(device),
        "router_algorithm_version": int(router.router_algorithm_version),
        "num_experts": config.num_local_experts,
        "num_prototypes": config.cpt_num_prototypes,
        "valid_tokens": int(mask.sum()),
        "projection_path": "P x -> stable L2",
        "post_pi_softmax": False,
        "status": "passed",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    print(json.dumps(run(device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
