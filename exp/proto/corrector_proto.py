"""Predictor-corrector prototype: order-2 blockwise routing.

Pass 1 routes the chunk from the entry state (current behavior).
Pass 2 corrects per-position prototypes using a first-order state
trajectory estimate built from pass-1 responsibilities via an
exclusive prefix sum of per-token gradients.

chunk_size=1 limit: prefix sum is empty -> identical to strict v1.
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


def corrected_route_chunk(router, z_chunk, valid_chunk, S0, nu0, rho):
    valid_rows = valid_chunk.any(dim=1, keepdim=True)
    b, t, d = z_chunk.shape

    anchors = router.anchors  # [d,k]

    # ---- pass 1: route from entry state ----
    beta0 = router.beta_max * nu0 / (nu0 + router.kappa_beta)  # [b,k]
    mixed0 = anchors.unsqueeze(0) * (1.0 - beta0.detach().unsqueeze(1)) + S0.detach() * beta0.detach().unsqueeze(1)
    M0 = stable_l2(mixed0, dim=1, eps=router.eps_m)
    logits1 = torch.bmm(z_chunk, M0) / router.prototype_temperature
    q1 = F.softmax(logits1, dim=-1, dtype=torch.float32)
    q1 = q1 * valid_chunk.unsqueeze(-1)

    # ---- first-order trajectory estimates from pass-1 q ----
    with torch.no_grad():
        # per-token gradient at entry state, [b,t,d,k]; invalid tokens contribute 0
        g = (
            (S0.unsqueeze(1) - z_chunk.unsqueeze(-1)) * q1.unsqueeze(2)
            + router.lambda_sa * (S0.unsqueeze(1) - anchors.unsqueeze(0).unsqueeze(0))
        ) * valid_chunk.unsqueeze(-1).unsqueeze(-1)
        exclusive_prefix = g.cumsum(dim=1) - g  # sum of g_s for s < t
        S_traj = S0.unsqueeze(1) - router.state_step_size * exclusive_prefix
        traj_norm = torch.linalg.vector_norm(S_traj, dim=2, keepdim=True)
        S_traj = S_traj / torch.clamp_min(traj_norm / router.state_radius, 1.0)

        # per-position responsibility before processing t:
        # nu_t = rho^p_t * (nu0 + sum_{s<t} q_s * rho^-(p_s+1)), p = valid prefix count
        valid_int = valid_chunk.to(torch.float32)
        p_excl = valid_int.cumsum(dim=1) - valid_int  # valid tokens before t
        inv_decay = rho.pow(-(p_excl + 1.0)).unsqueeze(-1)  # [b,t,1]
        weighted_q = q1.detach() * inv_decay
        cum_weighted = weighted_q.cumsum(dim=1) - weighted_q  # exclusive
        nu_traj = rho.pow(p_excl).unsqueeze(-1) * (nu0.unsqueeze(1) + cum_weighted)
        beta_traj = router.beta_max * nu_traj / (nu_traj + router.kappa_beta)

    # ---- pass 2: corrected per-position prototypes ----
    mixed2 = anchors.unsqueeze(0).unsqueeze(0) * (
        1.0 - beta_traj.detach().unsqueeze(2)
    ) + S_traj.detach() * beta_traj.detach().unsqueeze(2)
    M2 = stable_l2(mixed2, dim=2, eps=router.eps_m)  # [b,t,d,k]
    logits2 = torch.einsum("btd,btdk->btk", z_chunk, M2) / router.prototype_temperature
    q2 = F.softmax(logits2, dim=-1, dtype=torch.float32)
    q2 = q2 * valid_chunk.unsqueeze(-1)

    # ---- state/nu update using corrected q (same batched formulas) ----
    with torch.no_grad():
        qd = q2.detach()
        count_int = valid_chunk.sum(dim=1)
        count = count_int.to(torch.float32)
        mass = qd.sum(dim=1)
        wp = torch.bmm(z_chunk.detach().transpose(1, 2), qd)
        grad = S0 * mass.unsqueeze(1) - wp + router.lambda_sa * count.view(-1, 1, 1) * (S0 - anchors.detach().unsqueeze(0))
        cand = S0 - router.state_step_size * grad
        norm = torch.linalg.vector_norm(cand, dim=1, keepdim=True)
        cand = cand / torch.clamp_min(norm / router.state_radius, 1.0)
        valid_after = count_int.unsqueeze(-1) - valid_chunk.cumsum(dim=1)
        decay = rho.pow(valid_after).unsqueeze(-1)
        retain = rho.pow(count).unsqueeze(-1)
        nu_cand = retain * nu0 + (qd * decay).sum(dim=1)
        next_S = torch.where(valid_rows.unsqueeze(-1), cand, S0)
        next_nu = torch.where(valid_rows, nu_cand, nu0)
    return q2, next_S, next_nu


def corrected_forward(router, x, mask):
    """Mimic CPTRouter._forward_impl with the corrected chunk body."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        hidden = torch.where(mask.unsqueeze(-1), x.float(), torch.zeros_like(x, dtype=torch.float32))
        projected = F.linear(hidden, router.projection)
        projected = stable_l2(projected, dim=-1, eps=router.eps_z)
        kernel = router.expert_kernel()
        b, seq, _ = x.shape
        S = torch.zeros(b, router.projection_dim, router.num_prototypes, device=x.device)
        nu = torch.zeros(b, router.num_prototypes, device=x.device)
        rho = nu.new_tensor(router.rho_beta)
        chunks = []
        for start in range(0, seq, router.state_chunk_size):
            end = min(start + router.state_chunk_size, seq)
            q, S, nu = corrected_route_chunk(router, projected[:, start:end], mask[:, start:end], S, nu, rho)
            chunks.append(q)
        Q = torch.cat(chunks, dim=1)
        flat = Q.reshape(-1, router.num_prototypes)
        idx = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
        probs = flat.index_select(0, idx) @ kernel
        out = torch.zeros(b * seq, router.num_experts, device=x.device)
        out = out.index_copy(0, idx, probs).view(b, seq, router.num_experts)
    return out


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
torch.manual_seed(13)

# ---- fidelity vs strict v1 (chunk=1 reference) ----
print("=== fidelity vs strict v1 (lower = closer) ===")
print(f"{'chunk':>6} {'blockwise mean/max':>28} {'corrected mean/max':>28}")
for proj_dim in (16,):
    ref_router = CPTRouter(cfg(cpt_state_chunk_size=1, cpt_projection_dim=proj_dim), layer_index=0).to(dev)
    for chunk in (8, 32, 128):
        router = CPTRouter(cfg(cpt_state_chunk_size=chunk, cpt_projection_dim=proj_dim), layer_index=0)
        # share parameters with ref for a fair comparison
        with torch.no_grad():
            router.projection.copy_(ref_router.projection)
            router.anchors.copy_(ref_router.anchors)
            router.energy.copy_(ref_router.energy)
        router = router.to(dev)
        errs_b, errs_c = [], []
        for seed in range(5):
            torch.manual_seed(100 + seed)
            x = torch.randn(2, 64, 64, device=dev)
            mask = torch.randint(0, 2, (2, 64), device=dev, dtype=torch.bool)
            mask[:, 0] = True
            with torch.no_grad():
                pi_ref = ref_router(x, mask).probabilities
                pi_b = router(x, mask).probabilities
                pi_c = corrected_forward(router, x, mask)
            m = mask.unsqueeze(-1).expand_as(pi_ref)
            errs_b.append((pi_b[m] - pi_ref[m]).abs())
            errs_c.append((pi_c[m] - pi_ref[m]).abs())
        eb = torch.cat(errs_b)
        ec = torch.cat(errs_c)
        print(f"{chunk:>6} {eb.mean().item():13.2e}/{eb.max().item():.2e}" f" {ec.mean().item():13.2e}/{ec.max().item():.2e}")

# ---- speed ----
print("\n=== speed (seq=2048, b=4, proj_dim=128) ===")
for chunk in (32, 128):
    router = CPTRouter(cfg(cpt_state_chunk_size=chunk, cpt_projection_dim=128), layer_index=0).to(dev)
    xb = torch.randn(4, 2048, 64, device=dev)
    mask = torch.ones(4, 2048, device=dev, dtype=torch.bool)
    ms_std = bench(lambda: router._forward_impl(xb, mask))
    ms_cor = bench(lambda: corrected_forward(router, xb, mask))
    print(f"chunk={chunk:4d}: standard {ms_std:8.2f} ms | corrected {ms_cor:8.2f} ms | x{ms_cor / ms_std:.2f} cost")

# ---- corrected + torch.compile ----
print("\n=== corrected + torch.compile (seq=2048, b=4, proj_dim=128) ===")
compiled_corrected = torch.compile(corrected_route_chunk, dynamic=False)


def corrected_forward_compiled(router, x, mask):
    with torch.autocast(device_type=x.device.type, enabled=False):
        hidden = torch.where(mask.unsqueeze(-1), x.float(), torch.zeros_like(x, dtype=torch.float32))
        projected = F.linear(hidden, router.projection)
        projected = stable_l2(projected, dim=-1, eps=router.eps_z)
        kernel = router.expert_kernel()
        b, seq, _ = x.shape
        S = torch.zeros(b, router.projection_dim, router.num_prototypes, device=x.device)
        nu = torch.zeros(b, router.num_prototypes, device=x.device)
        rho = nu.new_tensor(router.rho_beta)
        chunks = []
        for start in range(0, seq, router.state_chunk_size):
            end = min(start + router.state_chunk_size, seq)
            q, S, nu = compiled_corrected(router, projected[:, start:end], mask[:, start:end], S, nu, rho)
            chunks.append(q)
        Q = torch.cat(chunks, dim=1)
        flat = Q.reshape(-1, router.num_prototypes)
        idx = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
        probs = flat.index_select(0, idx) @ kernel
        out = torch.zeros(b * seq, router.num_experts, device=x.device)
        out = out.index_copy(0, idx, probs).view(b, seq, router.num_experts)
    return out


for chunk in (32, 128):
    router = CPTRouter(cfg(cpt_state_chunk_size=chunk, cpt_projection_dim=128), layer_index=0).to(dev)
    xb = torch.randn(4, 2048, 64, device=dev)
    mask = torch.ones(4, 2048, device=dev, dtype=torch.bool)
    ms = bench(lambda: corrected_forward_compiled(router, xb, mask))
    print(f"chunk={chunk:4d}: corrected+compiled {ms:8.2f} ms")
