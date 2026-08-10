import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from diag_small_train import make_config  # noqa: E402

from model.cpt_router import CPTRouter  # noqa: E402

for es in (None, 0.5, 1.0, 2.0):
    cfg = make_config(es)
    r = CPTRouter(cfg, layer_index=0)
    print(f"es={cfg.cpt_energy_init_scale:.3f}: ", end="")
    B = r.expert_kernel()
    cos = F.cosine_similarity(B.unsqueeze(0), B.unsqueeze(1), dim=-1)
    n = B.shape[0]
    off = cos[~torch.eye(n, dtype=torch.bool)]
    print(f"B row cos = {off.mean().item():.4f}, B row min/max = {B.min().item():.3f}/{B.max().item():.3f}")
