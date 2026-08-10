"""Level-2 diagnostic: small-scale training comparison.

Four router variants, identical data/order/budget (1 shard = 100M tokens,
same FineWeb-Edu+Cosmo blend v1.1 used), small model (hidden 256 / 4 layers /
6 experts / top-2):

  legacy      Linear gate + jitter + aux loss (v1.1-style)
  cpt-static  CPT with beta_max -> 0 (mimics the end-to-end experiment)
  cpt-full    CPT default (serial state + corrector)
  cpt-jitter  cpt-static + multiplicative jitter on projections

Logs loss, grad norm, per-layer expert load, prices, routing entropy,
top-2 pair diversity and B-row similarity.  Answers:
  H1 does routing collapse emerge under real training?
  H3 does the full mechanism (serial state) recover vs static?
  jitter ablation.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import model.modeling as mm  # noqa: E402
from model.config import TinyMixtralConfig  # noqa: E402
from model.cpt_router import CPTRouter  # noqa: E402
from model.modeling import SparseMoE, TinyMixtralForCausalLM  # noqa: E402
from scripts.train_utils import make_adamw, make_cosine_schedule  # noqa: E402

SHARD = str(Path(__file__).resolve().parents[2] / "data/pretrain/smollm_blend/train_0000.pt")
OUT_DIR = Path(__file__).parent
B, SEQ = 24, 1024
CHUNK = (SEQ + 1) * B
LR, WD, WARMUP = 7e-4, 0.1, 200
LOG_EVERY, PI_EVERY = 50, 250
import os  # noqa: E402

LOG_EVERY = int(os.environ.get("LOG_EVERY", LOG_EVERY))


def make_config(energy_init_scale=None, expert_temperature=None, capacity_factor=None):
    kwargs = dict(
        vocab_size=32000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=2048,
        num_local_experts=6,
        num_experts_per_tok=2,
        expert_intermediate_size=683,
        cpt_projection_dim=64,
        cpt_state_chunk_size=32,
        cpt_init_seed=0,
    )
    if energy_init_scale is not None:
        kwargs["cpt_energy_init_scale"] = energy_init_scale
    if expert_temperature is not None:
        kwargs["cpt_expert_temperature"] = expert_temperature
    if capacity_factor is not None:
        kwargs["cpt_capacity_factor"] = capacity_factor
    return TinyMixtralConfig(**kwargs)


class LegacySparseMoE(SparseMoE):
    """v1.1-style Linear gate + jitter + aux loss, reusing expert weights."""

    def __init__(self, config, layer_index):
        super().__init__(config, layer_index)
        del self.cpt_router
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.jitter_noise = config.router_jitter_noise
        self.aux_loss_coef = config.router_aux_loss_coef

    def forward(self, x, attention_mask=None):
        Bsz, S, D = x.shape
        x_flat = x.view(-1, D)
        N = Bsz * S
        router_logits = self.router(x_flat)
        if self.training and self.jitter_noise > 0:
            router_logits = router_logits * (1 + torch.randn_like(router_logits) * self.jitter_noise)
        routing_weights = F.softmax(router_logits.float(), dim=-1).to(x.dtype)
        routing_weights_topk, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(dim=-1, keepdim=True)
        aux_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        if self.training and self.aux_loss_coef > 0:
            with torch.no_grad():
                expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).float()
                f_i = expert_mask.mean(dim=(0, 1))
            P_i = routing_weights.mean(dim=0)
            aux_loss = (f_i.detach() * P_i).sum() * self.num_experts

        self._diag_rw = routing_weights.detach().float()

        flat_experts = selected_experts.view(-1)
        flat_weights = routing_weights_topk.view(-1)
        flat_token_idx = torch.arange(N, device=x.device)
        flat_token_idx = flat_token_idx.unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        sorted_indices = flat_experts.argsort(stable=True)
        sorted_token_idx = flat_token_idx[sorted_indices]
        sorted_weights = flat_weights[sorted_indices]
        sorted_experts = flat_experts[sorted_indices]
        expert_counts = torch.bincount(sorted_experts, minlength=self.num_experts).tolist()
        final_out = torch.zeros(N, D, device=x.device, dtype=x.dtype)
        start = 0
        for e in range(self.num_experts):
            count = expert_counts[e]
            if count == 0:
                start += count
                continue
            end = start + count
            idx = sorted_token_idx[start:end]
            w = sorted_weights[start:end]
            token_states = x_flat[idx]
            gate = F.silu(torch.matmul(token_states, self.gate_proj[e].T))
            up = torch.matmul(token_states, self.up_proj[e].T)
            expert_out = torch.matmul(gate * up, self.down_proj[e].T)
            final_out.index_add_(0, idx, (expert_out * w.unsqueeze(-1)).to(x.dtype))
            start = end
        load_sum = routing_weights.detach().float().sum(dim=0)
        token_count = torch.tensor(N, device=x.device, dtype=torch.int64)
        state_version = torch.zeros((), device=x.device, dtype=torch.int64)
        return final_out.view(Bsz, S, D), aux_loss, load_sum, token_count, state_version


class JitteredCPTRouter(CPTRouter):
    def _route_chunk(self, z_chunk, valid_chunk, state_old, nu_old, rho):
        if self.training:
            z_chunk = z_chunk * (1 + torch.randn_like(z_chunk) * 0.01)
        return super()._route_chunk(z_chunk, valid_chunk, state_old, nu_old, rho)


def build_model(variant, energy_init_scale=None, expert_temperature=None, capacity_factor=None):
    torch.manual_seed(0)
    if variant == "legacy":
        original = mm.SparseMoE
        mm.SparseMoE = LegacySparseMoE
        try:
            model = TinyMixtralForCausalLM(make_config(energy_init_scale, expert_temperature, capacity_factor))
        finally:
            mm.SparseMoE = original
    else:
        if variant == "cpt-jitter":
            original = mm.CPTRouter
            mm.CPTRouter = JitteredCPTRouter
            try:
                model = TinyMixtralForCausalLM(make_config(energy_init_scale, expert_temperature, capacity_factor))
            finally:
                mm.CPTRouter = original
        else:
            model = TinyMixtralForCausalLM(make_config(energy_init_scale, expert_temperature, capacity_factor))
        if variant in ("cpt-static", "cpt-jitter"):
            for layer in model.layers:
                layer.moe.cpt_router.beta_max = 0.0
    return model.cuda()


def routing_stats(model):
    stats = []
    for layer in model.layers:
        moe = layer.moe
        if isinstance(moe, LegacySparseMoE):
            pi = getattr(moe, "_diag_rw", None)
            b_cos, b_ent = None, None
        else:
            pi = getattr(moe.cpt_router, "_diag_pi", None)
            B_kernel = moe.cpt_router.expert_kernel().detach()
            ent = -(B_kernel * B_kernel.clamp_min(1e-12).log()).sum(-1)
            b_ent = (ent / torch.log(torch.tensor(6.0))).mean().item()
            cos = F.cosine_similarity(B_kernel.unsqueeze(0), B_kernel.unsqueeze(1), dim=-1)
            n = B_kernel.shape[0]
            b_cos = cos[~torch.eye(n, dtype=torch.bool, device=cos.device)].mean().item()
        if pi is None:
            stats.append(dict(pi_entropy=None, pair_div=None, b_row_cos=b_cos, b_row_ent=b_ent))
            continue
        pi = pi.float().reshape(-1, 6)
        ent = -(pi * pi.clamp_min(1e-12).log()).sum(-1).mean()
        pi_entropy = (ent / torch.log(torch.tensor(6.0))).item()
        top2 = pi.topk(2, dim=-1).indices
        pairs = torch.sort(top2, dim=-1).values
        distinct = torch.unique(pairs, dim=0).shape[0]
        flat_pairs = pairs[:, 0] * 6 + pairs[:, 1]
        top_pair_frac = (flat_pairs == flat_pairs.mode().values).float().mean().item()
        stats.append(
            dict(
                pi_entropy=pi_entropy,
                pair_div=distinct / 15.0,
                top_pair_frac=top_pair_frac,
                b_row_cos=b_cos,
                b_row_ent=b_ent,
            )
        )
    return stats


def train_variant(variant, steps, energy_init_scale=None, expert_temperature=None, capacity_factor=None):
    model = build_model(variant, energy_init_scale, expert_temperature, capacity_factor)
    model.train()
    hooks = []
    if variant != "legacy":
        for layer in model.layers:
            r = layer.moe.cpt_router

            def hook(module, inputs, output, router=r):
                router._diag_pi = output.probabilities.detach()

            hooks.append(r.register_forward_hook(hook))

    opt = make_adamw(model, lr=LR, weight_decay=WD)
    sched = make_cosine_schedule(opt, warmup_steps=WARMUP, total_steps=steps)
    shard = torch.load(SHARD, weights_only=True)
    ptr = 0
    records = []
    losses_tail = []
    t0 = time.perf_counter()

    for step in range(steps):
        if ptr + CHUNK > len(shard):
            ptr = 0
        batch = shard[ptr : ptr + CHUNK].view(B, SEQ + 1).cuda()
        ptr += CHUNK
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch[:, :-1], labels=batch[:, 1:])
        loss = out["loss"]
        if variant == "legacy":
            loss = loss + 0.01 * out["aux_loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{variant}: non-finite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if variant != "legacy":
            model.commit_cpt_transaction(out["cpt_transaction"], optimizer_step=step + 1)
        sched.step()
        opt.zero_grad(set_to_none=True)

        loss_val = loss.item()
        losses_tail.append(loss_val)
        if len(losses_tail) > 100:
            losses_tail.pop(0)

        if step % LOG_EVERY == 0 or step == steps - 1:
            rec = dict(step=step, loss=loss_val, grad_norm=float(grad_norm))
            rec["gpu_mem_mb"] = torch.cuda.memory_allocated() / 1e6
            rec["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1e6
            loads = []
            prices = []
            for proposal in out["cpt_transaction"].proposals:
                lf = proposal.load_sum / proposal.token_count.float()
                loads.append(lf.cpu().tolist())
            for layer in model.layers:
                if hasattr(layer.moe, "cpt_router"):
                    prices.append(layer.moe.cpt_router.congestion_price.cpu().tolist())
            rec["load_frac"] = loads
            rec["prices"] = prices
            if step % PI_EVERY == 0 or step == steps - 1:
                rec["routing"] = routing_stats(model)
            records.append(rec)
            elapsed = time.perf_counter() - t0
            print(
                f"[{variant}] step {step}/{steps} loss={loss_val:.4f} " f"({elapsed:.0f}s)",
                flush=True,
            )

    for h in hooks:
        h.remove()
    summary = dict(
        variant=variant,
        final_loss_avg=sum(losses_tail) / len(losses_tail),
        total_s=time.perf_counter() - t0,
    )
    suffix = f"_{variant}"
    if energy_init_scale is not None:
        suffix += f"_es{energy_init_scale}"
    if expert_temperature is not None:
        suffix += f"_et{expert_temperature}"
    if capacity_factor is not None:
        suffix += f"_cap{capacity_factor}"
    out_path = OUT_DIR / f"small_train{suffix}.jsonl"
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"[{variant}] done: avg last-100 loss = {summary['final_loss_avg']:.4f}")
    return summary, records


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4069)
    p.add_argument(
        "--variants",
        nargs="+",
        default=["legacy", "cpt-static", "cpt-full", "cpt-jitter"],
    )
    p.add_argument("--energy-init-scale", type=float, default=None)
    p.add_argument("--expert-temperature", type=float, default=None)
    p.add_argument("--capacity-factor", type=float, default=None)
    args = p.parse_args()
    summaries = []
    for variant in args.variants:
        summary, _ = train_variant(
            variant,
            args.steps,
            energy_init_scale=args.energy_init_scale,
            expert_temperature=args.expert_temperature,
            capacity_factor=args.capacity_factor,
        )
        summaries.append(summary)
    print("\n=== summary (avg last-100 loss) ===")
    for s in summaries:
        print(f"  {s['variant']:12s} {s['final_loss_avg']:.4f}  ({s['total_s']:.0f}s)")


if __name__ == "__main__":
    main()
