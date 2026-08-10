"""Attention-context routing prototype: quantify fidelity vs strict v1 and speed.

Variants:
- blockwise (corrector off) / corrector: existing baselines
- AC-gamma: pass-1 blockwise routing -> SDPA context over z (k heads, V_k=q_k*z)
  -> pass-2 per-position prototypes from blend (1-g)*S1_traj + g*context
- AC-par: gamma=1 with pass-1 from pure anchors (fully parallel, no chunk loop)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from model.config import TinyMixtralConfig  # noqa: E402
from model.cpt_router import CPTRouter  # noqa: E402


def cfg(**kw):
    base = dict(
        vocab_size=41,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        num_local_experts=4,
        num_experts_per_tok=2,
        expert_intermediate_size=32,
        cpt_projection_dim=16,
        cpt_init_seed=3,
    )
    base.update(kw)
    return TinyMixtralConfig(**base)


def stable_l2(t, dim, eps):
    return t / torch.linalg.vector_norm(t, dim=dim, keepdim=True).clamp_min(eps)


def attn_context(z, q, mask, scale):
    """Causal attention-weighted context: C_t[:,:,k] = sum_s w_ts*q_sk*z_s."""
    b, s, d = z.shape
    k = q.shape[-1]
    queries = z.unsqueeze(1).expand(b, k, s, d).contiguous()
    keys = z.unsqueeze(1).expand(b, k, s, d).contiguous()
    values = (z.unsqueeze(2) * q.unsqueeze(-1)).transpose(1, 2).contiguous()
    if mask is None:
        ctx = F.scaled_dot_product_attention(queries, keys, values, is_causal=True, scale=scale)
    else:
        causal = torch.tril(torch.ones(s, s, device=z.device, dtype=torch.bool))
        combined = causal[None, None] & mask[:, None, None, :]
        ctx = F.scaled_dot_product_attention(queries, keys, values, attn_mask=combined, scale=scale)
    return ctx.permute(0, 2, 3, 1)  # [b,s,d,k]


def ac_forward(router, x, mask, gamma, parallel_pass1=False):
    with torch.autocast(device_type=x.device.type, enabled=False):
        hidden = torch.where(mask.unsqueeze(-1), x.float(), torch.zeros_like(x, dtype=torch.float32))
        projected = F.linear(hidden, router.projection)
        projected = stable_l2(projected, dim=-1, eps=router.eps_z)
        kernel = router.expert_kernel()
        b, seq, _ = x.shape
        d = router.projection_dim
        k = router.num_prototypes
        anchors = router.anchors

        # ---- pass 1: prototype probabilities q1 ----
        if parallel_pass1:
            logits1 = torch.bmm(projected, anchors.unsqueeze(0).expand(b, d, k)) / router.prototype_temperature
            q1 = F.softmax(logits1, dim=-1, dtype=torch.float32) * mask.unsqueeze(-1)
        else:
            chunks = []
            S = torch.zeros(b, d, k, device=x.device)
            nu = torch.zeros(b, k, device=x.device)
            rho = nu.new_tensor(router.rho_beta)
            for start in range(0, seq, router.state_chunk_size):
                end = min(start + router.state_chunk_size, seq)
                zc = projected[:, start:end]
                vc = mask[:, start:end]
                valid_rows = vc.any(dim=1, keepdim=True)
                beta = router.beta_max * nu / (nu + router.kappa_beta)
                mixed = anchors.unsqueeze(0) * (1.0 - beta.detach().unsqueeze(1)) + S.detach() * beta.detach().unsqueeze(1)
                M = stable_l2(mixed, dim=1, eps=router.eps_m)
                logits1 = torch.bmm(zc, M) / router.prototype_temperature
                qc = F.softmax(logits1, dim=-1, dtype=torch.float32) * vc.unsqueeze(-1)
                chunks.append(qc)
                with torch.no_grad():
                    qd = qc.detach()
                    count_int = vc.sum(dim=1)
                    count = count_int.to(torch.float32)
                    mass = qd.sum(dim=1)
                    wp = torch.bmm(zc.detach().transpose(1, 2), qd)
                    grad = (
                        S * mass.unsqueeze(1)
                        - wp
                        + router.lambda_sa * count.view(-1, 1, 1) * (S - anchors.detach().unsqueeze(0))
                    )
                    cand = S - router.state_step_size * grad
                    norm = torch.linalg.vector_norm(cand, dim=1, keepdim=True)
                    cand = cand / torch.clamp_min(norm / router.state_radius, 1.0)
                    valid_after = count_int.unsqueeze(-1) - vc.cumsum(dim=1)
                    decay = rho.pow(valid_after).unsqueeze(-1)
                    retain = rho.pow(count).unsqueeze(-1)
                    nu_cand = retain * nu + (qd * decay).sum(dim=1)
                    S = torch.where(valid_rows.unsqueeze(-1), cand, S)
                    nu = torch.where(valid_rows, nu_cand, nu)
            q1 = torch.cat(chunks, dim=1)

        # ---- attention context from pass-1 responsibilities ----
        ctx = attn_context(projected, q1.detach(), mask, scale=1.0 / router.prototype_temperature)

        with torch.no_grad():
            # per-position responsibility trajectory (closed form, nu_old=0)
            valid_float = mask.to(torch.float32)
            valid_before = valid_float.cumsum(dim=1) - valid_float
            rho = q1.new_tensor(router.rho_beta)
            inv_decay = rho.pow(-(valid_before + 1.0)).unsqueeze(-1)
            weighted = q1.detach() * inv_decay
            excl = weighted.cumsum(dim=1) - weighted
            nu_traj = rho.pow(valid_before).unsqueeze(-1) * excl
            beta_traj = router.beta_max * nu_traj / (nu_traj + router.kappa_beta)

            # first-order recurrent trajectory S1 (per chunk, entry-state grads)
            S1_parts = []
            S_entry = torch.zeros(b, d, k, device=x.device)
            nu_entry = torch.zeros(b, k, device=x.device)
            for start in range(0, seq, router.state_chunk_size):
                end = min(start + router.state_chunk_size, seq)
                zc = projected[:, start:end]
                vc = mask[:, start:end]
                valid_rows = vc.any(dim=1, keepdim=True)
                qd = q1[:, start:end].detach()
                g = (
                    (S_entry.unsqueeze(1) - zc.unsqueeze(-1)) * qd.unsqueeze(2)
                    + router.lambda_sa * (S_entry.unsqueeze(1) - anchors.detach().unsqueeze(0).unsqueeze(0))
                ) * vc.unsqueeze(-1).unsqueeze(-1)
                excl_g = g.cumsum(dim=1) - g
                traj = S_entry.unsqueeze(1) - router.state_step_size * excl_g
                tnorm = torch.linalg.vector_norm(traj, dim=2, keepdim=True)
                traj = traj / torch.clamp_min(tnorm / router.state_radius, 1.0)
                S1_parts.append(traj)
                # advance entry state with q1 (predictor consistency)
                count_int = vc.sum(dim=1)
                count = count_int.to(torch.float32)
                mass = qd.sum(dim=1)
                wp = torch.bmm(zc.detach().transpose(1, 2), qd)
                grad = (
                    S_entry * mass.unsqueeze(1)
                    - wp
                    + router.lambda_sa * count.view(-1, 1, 1) * (S_entry - anchors.detach().unsqueeze(0))
                )
                cand = S_entry - router.state_step_size * grad
                norm = torch.linalg.vector_norm(cand, dim=1, keepdim=True)
                cand = cand / torch.clamp_min(norm / router.state_radius, 1.0)
                valid_after = count_int.unsqueeze(-1) - vc.cumsum(dim=1)
                decay = rho.pow(valid_after).unsqueeze(-1)
                retain = rho.pow(count).unsqueeze(-1)
                nu_cand = retain * nu_entry + (qd * decay).sum(dim=1)
                S_entry = torch.where(valid_rows.unsqueeze(-1), cand, S_entry)
                nu_entry = torch.where(valid_rows, nu_cand, nu_entry)
            S1 = torch.cat(S1_parts, dim=1)

        S_blend = (1.0 - gamma) * S1 + gamma * ctx
        mixed2 = anchors.unsqueeze(0).unsqueeze(0) * (
            1.0 - beta_traj.detach().unsqueeze(2)
        ) + S_blend.detach() * beta_traj.detach().unsqueeze(2)
        M2 = stable_l2(mixed2, dim=2, eps=router.eps_m)
        logits2 = torch.einsum("bsd,bsdk->bsk", projected, M2) / router.prototype_temperature
        q2 = F.softmax(logits2, dim=-1, dtype=torch.float32) * mask.unsqueeze(-1)

        flat = q2.reshape(-1, k)
        idx = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
        probs = flat.index_select(0, idx) @ kernel
        out = torch.zeros(b * seq, router.num_experts, device=x.device)
        return out.index_copy(0, idx, probs).view(b, seq, router.num_experts)


def bench(fn, iters=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


dev = "cuda"

# ================= fidelity vs strict v1 =================
print("=== fidelity vs strict v1 (proj_dim=16, seq=64, chunk=32) ===")
ref_router = CPTRouter(cfg(cpt_state_chunk_size=1, cpt_state_corrector=False), layer_index=0).to(dev)

variants = [
    (
        "blockwise corrector=off",
        lambda r, x, m: r(x, m).probabilities,
        dict(cpt_state_chunk_size=32, cpt_state_corrector=False),
    ),
    (
        "corrector (current default)",
        lambda r, x, m: r(x, m).probabilities,
        dict(cpt_state_chunk_size=32, cpt_state_corrector=True),
    ),
]
for gamma in (0.25, 0.5, 0.75, 1.0):
    variants.append(
        (
            f"AC gamma={gamma}",
            lambda r, x, m, g=gamma: ac_forward(r, x, m, g),
            dict(cpt_state_chunk_size=32, cpt_state_corrector=False),
        ),
    )
variants.append(
    (
        "AC-par gamma=1 (parallel pass1)",
        lambda r, x, m: ac_forward(r, x, m, 1.0, True),
        dict(cpt_state_chunk_size=32, cpt_state_corrector=False),
    ),
)

print(f"{'variant':34s} {'mean err':>10s} {'max err':>10s}")
for name, fn, kw in variants:
    errs = []
    for seed in range(5):
        torch.manual_seed(100 + seed)
        router = CPTRouter(cfg(**kw), layer_index=0).to(dev)
        with torch.no_grad():
            router.projection.copy_(ref_router.projection)
            router.anchors.copy_(ref_router.anchors)
            router.energy.copy_(ref_router.energy)
        x = torch.randn(2, 64, 64, device=dev)
        mask = torch.randint(0, 2, (2, 64), device=dev, dtype=torch.bool)
        mask[:, 0] = True
        with torch.no_grad():
            pi_ref = ref_router(x, mask).probabilities
            pi = fn(router, x, mask)
        mm = mask.unsqueeze(-1).expand_as(pi_ref)
        errs.append((pi[mm] - pi_ref[mm]).abs())
    e = torch.cat(errs)
    print(f"{name:34s} {e.mean().item():10.2e} {e.max().item():10.2e}")

# ================= speed =================
print("\n=== speed (proj_dim=128, seq=2048, b=4, no padding) ===")
xb = torch.randn(4, 2048, 64, device=dev)
mb = torch.ones(4, 2048, device=dev, dtype=torch.bool)

r_def = CPTRouter(cfg(cpt_projection_dim=128, cpt_state_chunk_size=32), layer_index=0).to(dev)
r_c128 = CPTRouter(cfg(cpt_projection_dim=128, cpt_state_chunk_size=128), layer_index=0).to(dev)
r_ac = CPTRouter(cfg(cpt_projection_dim=128, cpt_state_chunk_size=128, cpt_state_corrector=False), layer_index=0).to(dev)
r_acp = CPTRouter(cfg(cpt_projection_dim=128, cpt_state_chunk_size=128, cpt_state_corrector=False), layer_index=0).to(dev)

print(f"{'variant':38s} {'eager ms':>9s} {'compiled ms':>12s}")
print(f"{'default chunk=32 (compiled)':38s} {'-':>9s} {bench(lambda: r_def(xb)):12.2f}")
print(f"{'corrector chunk=128 (compiled)':38s} {'-':>9s} {bench(lambda: r_c128(xb)):12.2f}")

ms_e = bench(lambda: ac_forward(r_ac, xb, mb, 1.0))
ac_comp = torch.compile(ac_forward, dynamic=False)
ms_c = bench(lambda: ac_comp(r_ac, xb, mb, 1.0))
print(f"{'AC gamma=1 (chunked pass1)':38s} {ms_e:9.2f} {ms_c:12.2f}")

ms_e2 = bench(lambda: ac_forward(r_acp, xb, mb, 1.0, True))
ms_c2 = bench(lambda: ac_comp(r_acp, xb, mb, 1.0, True))
print(f"{'AC-par gamma=1 (parallel pass1)':38s} {ms_e2:9.2f} {ms_c2:12.2f}")
