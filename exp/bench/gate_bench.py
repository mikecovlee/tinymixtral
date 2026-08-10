import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

dev = "cuda"
b, seq, d, n_exp, top_k = 4, 2048, 64, 8, 2
x = torch.randn(b, seq, d, device=dev)
gate = torch.nn.Linear(d, n_exp, bias=False, device=dev)


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def legacy():
    logits = gate(x.view(-1, d))
    w = F.softmax(logits.float(), dim=-1)
    wk, idx = torch.topk(w, top_k, dim=-1)
    return wk / wk.sum(-1, keepdim=True)


print(f"legacy Linear gate: {bench(legacy):.3f} ms")
