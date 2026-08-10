import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402
from attn_context_proto import ac_forward, cfg  # noqa: E402

from model.cpt_router import CPTRouter  # noqa: E402

dev = "cuda"
ref = CPTRouter(cfg(cpt_state_chunk_size=1, cpt_state_corrector=False), layer_index=0).to(dev)
errs_ac0, errs_corr = [], []
for seed in range(5):
    torch.manual_seed(100 + seed)
    r_ac = CPTRouter(cfg(cpt_state_chunk_size=32, cpt_state_corrector=False), layer_index=0).to(dev)
    r_co = CPTRouter(cfg(cpt_state_chunk_size=32, cpt_state_corrector=True), layer_index=0).to(dev)
    for r in (r_ac, r_co, ref):
        with torch.no_grad():
            r.projection.copy_(ref.projection)
            r.anchors.copy_(ref.anchors)
            r.energy.copy_(ref.energy)
    x = torch.randn(2, 64, 64, device=dev)
    mask = torch.randint(0, 2, (2, 64), device=dev, dtype=torch.bool)
    mask[:, 0] = True
    with torch.no_grad():
        pi_ref = ref(x, mask).probabilities
        pi_ac0 = ac_forward(r_ac, x, mask, 0.0)
        pi_co = r_co(x, mask).probabilities
    mm = mask.unsqueeze(-1).expand_as(pi_ref)
    errs_ac0.append((pi_ac0[mm] - pi_ref[mm]).abs())
    errs_corr.append((pi_co[mm] - pi_ref[mm]).abs())
ea = torch.cat(errs_ac0)
ec = torch.cat(errs_corr)
print(f"AC gamma=0:  mean {ea.mean().item():.2e} max {ea.max().item():.2e}")
print(f"corrector:   mean {ec.mean().item():.2e} max {ec.max().item():.2e}")
print(f"AC0 vs corrector direct diff: {(pi_ac0 - pi_co)[mm].abs().max().item():.2e}")
