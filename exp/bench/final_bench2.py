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
        cpt_projection_dim=128,
        cpt_init_seed=3,
    )
    base.update(kw)
    return TinyMixtralConfig(**base)


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
print("方案 (seq=2048, b=4)                    compiled(ms)")
for label, kw in (
    ("原始默认: chunk=32", dict(cpt_state_chunk_size=32)),
    ("校正器:   chunk=32", dict(cpt_state_chunk_size=32, cpt_state_corrector=True)),
    ("校正器:   chunk=128", dict(cpt_state_chunk_size=128, cpt_state_corrector=True)),
):
    r = CPTRouter(cfg(**kw), layer_index=0).to(dev)
    xb = torch.randn(4, 2048, 64, device=dev)
    ms = bench(lambda: r(xb))  # train mode -> compiled
    print(f"{label:38s} {ms:10.2f}")
