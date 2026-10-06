"""Audit: (1) is routing exactly scale-invariant in `projection`?
(2) how active is the congestion-price controller?
(3) do prices actually move in the trained HP-sweep checkpoints?
"""
import sys
import torch

sys.path.insert(0, "/home/mikecovlee/work/tinymixtral-improve/git-cpt-v3.0")
from cpt_model.config import CPTConfig  # noqa: E402
from cpt_model.router import CPTRouter  # noqa: E402

torch.manual_seed(0)

cfg = CPTConfig(
    vocab_size=32000, hidden_size=1024, num_hidden_layers=2,
    num_attention_heads=16, num_key_value_heads=4, num_local_experts=4,
    num_experts_per_tok=2, expert_intermediate_size=2048,
    cpt_energy_init_scale="3/10", cpt_price_learning_rate="3/125",
)
r = CPTRouter(cfg, 0).eval()

x = torch.randn(2, 64, 1024)
with torch.no_grad():
    p0 = r(x).probabilities
    r.projection.mul_(0.15)
    p1 = r(x).probabilities
    r.projection.mul_(1e-4)
    p2 = r(x).probabilities
    r.projection.div_(1e-4 * 0.15)
    p3 = r(x).probabilities

print("=== (1) scale-invariance of `projection` ===")
print(f"  ||P||_F initial      = {2.6458:.4f} (orthonormal rows, sqrt(7))")
print(f"  max|dp| after P*=0.15   = {(p1 - p0).abs().max().item():.3e}")
print(f"  max|dp| after P*=1.5e-5 = {(p2 - p0).abs().max().item():.3e}")
print(f"  max|dp| after restore   = {(p3 - p0).abs().max().item():.3e}")
print(f"  => forward is scale-invariant in P: {bool((p1 - p0).abs().max() < 1e-5)}")

print("\n=== (2) price-controller dead zone ===")
N, cf, plr = 4, cfg.cpt_capacity_factor, cfg.cpt_price_learning_rate
lo, hi = (1 - (cf - 1)) / N, (1 + (cf - 1)) / N
print(f"  capacity_factor={cf} -> active band is util < {lo:.4f} or > {hi:.4f}")
print(f"  price_lr={plr:.4f}, tau_e={cfg.cpt_expert_temperature:.3f}")
for excess in (0.02, 0.05, 0.10, 0.30):
    per_commit = plr * excess / cfg.cpt_expert_temperature
    print(f"  util excess {excess:+.2f} -> logit shift {per_commit:.5f}/commit"
          f" -> 1.0 logit in {1 / per_commit:6.1f} commits")

print("\n=== (3) congestion prices in trained HP-sweep checkpoints ===")
import glob  # noqa: E402
import os  # noqa: E402

for run in ("base", "price_strong", "price_off", "T_sharp"):
    hits = sorted(glob.glob(f"checkpoints/hp_sweep/{run}/step_*"))
    if not hits:
        print(f"  {run}: no checkpoint")
        continue
    ck = hits[-1]
    sd = torch.load(os.path.join(ck, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    ks = sorted(k for k in sd if k.endswith("congestion_price"))
    pk = sorted(k for k in sd if k.endswith("cpt_router.projection"))
    norms = [sd[k].abs().max().item() for k in ks]
    pnorm = [sd[k].norm().item() for k in pk]
    print(f"  {run:14s} {os.path.basename(ck)}")
    print(f"     price |max| over {len(ks)} layers: {min(norms):.2e} .. {max(norms):.2e}")
    print(f"     ||P||_F (init 2.6458):            {min(pnorm):.4f} .. {max(pnorm):.4f}")
