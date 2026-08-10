import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402

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


torch.manual_seed(0)

# --- 1) responsibility closed form vs sequential recurrence ---
rho = torch.tensor(0.95)
nu = torch.randn(3, 8).abs()
valid = torch.tensor([[1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 1], [1, 0, 1, 0, 0, 0]])
q = torch.randn(3, 6, 8).softmax(-1)
q = torch.where(valid.unsqueeze(-1).bool(), q, torch.zeros_like(q))

seq_nu = nu.clone()
for t in range(6):
    seq_nu = torch.where(valid[:, t].bool().unsqueeze(-1), rho * seq_nu + q[:, t], seq_nu)

valid_after = valid.flip(dims=(1,)).cumsum(dim=1).flip(dims=(1,)) - valid
count = valid.sum(dim=1, keepdim=True).float()
closed = rho.pow(count) * nu + (q * rho.pow(valid_after).unsqueeze(-1)).sum(dim=1)
print("responsibility closed form max err:", (seq_nu - closed).abs().max().item())

# --- 2) chunk state gradient == sum of per-token gradients (entry state) ---
b, t, d, k = 2, 5, 16, 8
S = torch.randn(b, d, k)
z = torch.randn(b, t, d)
qq = torch.randn(b, t, k).softmax(-1)
lam = 0.25
anchors = torch.randn(d, k)

per_token = torch.zeros_like(S)
for i in range(t):
    per_token += (S - z[:, i].unsqueeze(-1)) * qq[:, i].unsqueeze(1) + lam * (S - anchors.unsqueeze(0))
mass = qq.sum(dim=1)
wp = torch.einsum("btd,btk->bdk", z, qq)
batched = S * mass.unsqueeze(1) - wp + lam * t * (S - anchors.unsqueeze(0))
print("state gradient batch form max err:", (per_token - batched).abs().max().item())

# --- 3) chunk=1 router == manual strict sequential v1 ---
r1 = CPTRouter(cfg(cpt_state_chunk_size=1), layer_index=0)
x = torch.randn(2, 9, 64)
mask = torch.tensor([[1, 1, 0, 1, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 1, 1, 0, 1]], dtype=torch.bool)
out = r1(x, mask).probabilities

# manual strict sequential replay
with torch.autocast("cpu", enabled=False):
    hidden = torch.where(mask.unsqueeze(-1), x.float(), torch.zeros_like(x, dtype=torch.float32))
    proj = torch.nn.functional.linear(hidden, r1.projection)
    proj = proj / torch.linalg.vector_norm(proj, dim=-1, keepdim=True).clamp_min(r1.eps_z)
    kernel = r1.expert_kernel()
    S = torch.zeros(2, 16, 8)
    nu = torch.zeros(2, 8)
    qs = []
    for pos in range(9):
        beta = r1.beta_max * nu / (nu + r1.kappa_beta)
        mixed = r1.anchors.unsqueeze(0) * (1 - beta.unsqueeze(1)) + S * beta.unsqueeze(1)
        M = mixed / torch.linalg.vector_norm(mixed, dim=1, keepdim=True).clamp_min(r1.eps_m)
        logits = torch.einsum("bd,bdk->bk", proj[:, pos], M) / r1.prototype_temperature
        qt = torch.softmax(logits, dim=-1)
        if not mask[0, pos] and not mask[1, pos]:
            pass
        qt = torch.where(mask[:, pos].unsqueeze(-1), qt, torch.zeros_like(qt))
        qs.append(qt)
        with torch.no_grad():
            grad = (S - proj[:, pos].unsqueeze(-1)) * qt.unsqueeze(1) + r1.lambda_sa * (S - r1.anchors.unsqueeze(0))
            cand = S - r1.state_step_size * grad
            cand = cand / torch.maximum(
                torch.ones_like(torch.linalg.vector_norm(cand, dim=1, keepdim=True)),
                torch.linalg.vector_norm(cand, dim=1, keepdim=True) / r1.state_radius,
            )
            nu_cand = r1.rho_beta * nu + qt
            rows = mask[:, pos]
            S = torch.where(rows.view(-1, 1, 1), cand, S)
            nu = torch.where(rows.unsqueeze(-1), nu_cand, nu)
    Q = torch.stack(qs, dim=1)
    flat = Q.reshape(-1, 8)
    idx = torch.nonzero(mask.reshape(-1)).flatten()
    manual = torch.zeros(18, 4)
    manual[idx] = flat[idx] @ kernel
    manual = manual.view(2, 9, 4)
print("chunk=1 vs manual strict v1 max err:", (out - manual).abs().max().item())


# --- 4) timing: router forward cost vs sequence length / chunk size ---
def bench(fn, iters=20):
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000


dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)
for seq in (512, 2048):
    for chunk in (1, 32, 128):
        r = CPTRouter(cfg(cpt_state_chunk_size=chunk, cpt_projection_dim=128), layer_index=0).to(dev)
        xb = torch.randn(4, seq, 64, device=dev)
        ms = bench(lambda: r(xb))
        print(f"seq={seq} chunk={chunk:4d}: {ms:8.3f} ms/forward  ({seq // ((seq + chunk - 1) // chunk)} tok/step)")
