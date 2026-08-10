import sys
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


def capture(tag):
    torch.manual_seed(7)
    results = {}
    for chunk in (1, 3, 8):
        router = CPTRouter(cfg(cpt_state_chunk_size=chunk), layer_index=0)
        x = torch.randn(3, 10, 64, requires_grad=True)
        mask = torch.tensor(
            [
                [1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
                [0, 1, 1, 0, 1, 1, 1, 0, 1, 1],
                [1, 0, 1, 0, 0, 0, 1, 1, 0, 1],
            ],
            dtype=torch.bool,
        )
        out = router(x, mask)
        out.probabilities.square().sum().backward()
        results[chunk] = {
            "prob": out.probabilities.detach().clone(),
            "xgrad": x.grad.detach().clone(),
            "proj_grad": router.projection.grad.detach().clone(),
            "anchor_grad": router.anchors.grad.detach().clone(),
            "energy_grad": router.energy.grad.detach().clone(),
            "load_sum": out.proposal.load_sum.detach().clone(),
        }
    torch.save(results, str(Path(__file__).parent / f"ref_{tag}.pt"))
    print(f"saved {Path(__file__).parent / f'ref_{tag}.pt'}")


if __name__ == "__main__":
    capture(sys.argv[1] if len(sys.argv) > 1 else "before")
