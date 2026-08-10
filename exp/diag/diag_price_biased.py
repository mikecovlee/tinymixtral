"""Level-1b diagnostic: price feedback under injected load imbalance.

At init, CPT routing is deliberately near-balanced (energy_init_scale=0.05),
so price dynamics never engage. This variant injects a strong energy bias
toward experts {0,1} at step 0, then watches whether congestion pricing:
  (a) rebalances load toward capacity (1.25/6 = 0.208),
  (b) overshoots / prices the favored experts out (load -> 0),
  (c) oscillates.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402

from model.config import TinyMixtralConfig  # noqa: E402
from model.cpt_router import CPTRouter  # noqa: E402

SHARD = str(Path(__file__).resolve().parents[2] / "data/pretrain/smollm_blend/train_0000.pt")
OUT = str(Path(__file__).parent / "price_dynamics_biased.jsonl")
B, SEQ, ITERS, LOG_EVERY = 8, 1024, 2500, 50
BIAS = 2.0  # added to energy[:, 0:2]; ~e^2=7.4x preference pre-price


def make_config():
    return TinyMixtralConfig(
        vocab_size=49152,
        hidden_size=896,
        num_hidden_layers=2,
        num_attention_heads=14,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=2048,
        num_local_experts=6,
        num_experts_per_tok=2,
        expert_intermediate_size=2389,
        cpt_projection_dim=128,
        cpt_init_seed=0,
    )


def row_stats(B_kernel):
    entropy = -(B_kernel * B_kernel.clamp_min(1e-12).log()).sum(dim=-1)
    normed = entropy / torch.log(torch.tensor(float(B_kernel.shape[-1])))
    cos = torch.nn.functional.cosine_similarity(B_kernel.unsqueeze(0), B_kernel.unsqueeze(1), dim=-1)
    n = B_kernel.shape[0]
    offdiag = cos[~torch.eye(n, dtype=torch.bool, device=cos.device)]
    return normed.mean().item(), offdiag.mean().item()


def run_variant(name, static):
    torch.manual_seed(0)
    config = make_config()
    router = CPTRouter(config, layer_index=0).cuda()
    if static:
        router.beta_max = 0.0
    with torch.no_grad():
        router.energy[:, 0:2] += BIAS
    embed = torch.nn.Embedding(config.vocab_size, config.hidden_size).cuda()
    torch.nn.init.normal_(embed.weight, std=0.02)

    shard = torch.load(SHARD, weights_only=True)
    ptr = 0
    records = []

    t0 = time.perf_counter()
    for it in range(ITERS):
        ids = shard[ptr : ptr + B * SEQ].view(B, SEQ).cuda()
        ptr += B * SEQ
        hidden = embed(ids)
        with torch.no_grad():
            output = router(hidden)
        proposal = output.proposal
        prepared = router.prepare_commit(proposal, optimizer_step=int(router.optimizer_step.item()))
        router.apply_commit(*prepared)

        if it % LOG_EVERY == 0 or it == ITERS - 1:
            probs = output.probabilities
            load_frac = (proposal.load_sum / proposal.token_count.float()).cpu()
            price = router.congestion_price.cpu()
            B_kernel = router.expert_kernel().detach()
            b_entropy, b_cos = row_stats(B_kernel)
            pi_entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1).mean()
            pi_entropy = (pi_entropy / torch.log(torch.tensor(6.0))).item()
            top2 = probs.topk(2, dim=-1).indices
            pairs = torch.sort(top2, dim=-1).values.reshape(-1, 2)
            distinct = torch.unique(pairs, dim=0).shape[0]
            flat_pairs = pairs[:, 0] * 6 + pairs[:, 1]
            top_pair_frac = (flat_pairs == flat_pairs.mode().values).float().mean().item()
            records.append(
                dict(
                    variant=name,
                    it=it,
                    load_frac=load_frac.tolist(),
                    price=price.tolist(),
                    load_min=load_frac.min().item(),
                    load_max=load_frac.max().item(),
                    price_max=price.max().item(),
                    b_row_entropy=b_entropy,
                    b_row_cos=b_cos,
                    pi_entropy=pi_entropy,
                    pair_diversity=distinct / 15.0,
                    top_pair_frac=top_pair_frac,
                )
            )
    print(f"[{name}] {ITERS} commits in {time.perf_counter() - t0:.1f}s", flush=True)
    return records


def summarize(records):
    idxs = [0, 1, 2, 5, 10, 20, len(records) // 2, len(records) - 1]
    for i in sorted(set(idxs)):
        if i >= len(records):
            continue
        rec = records[i]
        load = " ".join(f"{x:.3f}" for x in rec["load_frac"])
        price = " ".join(f"{x:.2f}" for x in rec["price"])
        print(
            f"  it={rec['it']:5d} load=[{load}] " f"pi_ent={rec['pi_entropy']:.3f} pair_div={rec['pair_diversity']:.2f}",
            flush=True,
        )
        print(f"           price=[{price}]", flush=True)


def main():
    all_records = []
    for name, static in (("static(beta=0)", True), ("full(default)", False)):
        print(f"=== {name} (bias={BIAS} on experts 0,1) ===", flush=True)
        records = run_variant(name, static)
        summarize(records)
        all_records.extend(records)
    with open(OUT, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")
    print(f"records -> {OUT}")


if __name__ == "__main__":
    main()
