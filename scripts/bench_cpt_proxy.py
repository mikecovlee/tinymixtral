#!/usr/bin/env python3
"""CPT proxy benchmark: batch-size / grad-ckpt sweep → tok/s + peak GB.

torch.compile of the CPT router chunk-loop is CPU-bound and takes ~2-3 min per
shape (8 layers × Triton kernels). Real training uses a fixed batch and compiles
once, so steady-state throughput is what matters here.
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from cpt_model import config_from_json_file, model_for_config
from scripts.train_utils import make_adamw


def bench_one(cfg_path, bs, seq, grad_ckpt, steps=5, warmup=3):
    cfg = config_from_json_file(cfg_path)
    model = model_for_config(cfg)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    model = model.to("cuda").to(torch.bfloat16)
    model.use_chunked_ce = True
    opt = make_adamw(model, lr=3e-4, weight_decay=0.1, bf16_states=True)

    x = torch.randint(0, cfg.vocab_size, (bs, seq + 1), device="cuda")

    # warmup (absorb torch.compile)
    for i in range(warmup):
        print(f"    warmup {i+1}/{warmup} (compile may take minutes)...", flush=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x[:, :-1], labels=x[:, 1:])
        out["loss"].backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x[:, :-1], labels=x[:, 1:])
        out["loss"].backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    tok = bs * seq * steps
    tok_s = tok / elapsed
    n_params = sum(p.numel() for p in model.parameters())

    del model, opt, x, out
    gc.collect()
    torch.cuda.empty_cache()
    return dict(bs=bs, seq=seq, grad_ckpt=grad_ckpt, tok_s=round(tok_s),
                peak_gb=round(peak_gb, 2), step_ms=round(elapsed / steps * 1000, 1),
                params_M=round(n_params / 1e6, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--bs-list", default="48,96,128,192,256")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    cfg_path = args.config or str(Path(__file__).parent.parent / "configs/cpt_hp_sweep/proxy_base_8l.json")
    bs_list = [int(x) for x in args.bs_list.split(",")]

    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu.name}  mem={gpu.total_memory/1e9:.1f}GB", flush=True)
    print(f"Config: {cfg_path}", flush=True)

    results = []
    for grad_ckpt in (True, False):
        for bs in bs_list:
            label = f"bs={bs} grad_ckpt={'on' if grad_ckpt else 'off'}"
            print(f"\n>>> {label}", flush=True)
            try:
                r = bench_one(cfg_path, bs, args.seq, grad_ckpt, args.steps, args.warmup)
                results.append(r)
                print(f"  {label}: {r['tok_s']} tok/s  peak={r['peak_gb']}GB  "
                      f"step={r['step_ms']}ms  params={r['params_M']}M", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  {label}: OOM", flush=True)
                gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"  {label}: ERROR {type(e).__name__}: {e}", flush=True)
                gc.collect(); torch.cuda.empty_cache()

    # summary table
    print("\n=== SUMMARY ===")
    print(f"{'bs':>5} {'grad_ckpt':>10} {'tok/s':>8} {'step_ms':>8} {'peak_GB':>8}")
    for r in results:
        print(f"{r['bs']:>5} {'on' if r['grad_ckpt'] else 'off':>10} {r['tok_s']:>8} "
              f"{r['step_ms']:>8} {r['peak_gb']:>8}")

    best = max(results, key=lambda r: r["tok_s"], default=None)
    if best:
        print(f"\nBest: bs={best['bs']} grad_ckpt={'on' if best['grad_ckpt'] else 'off'} "
              f"→ {best['tok_s']} tok/s  peak={best['peak_gb']}GB")


if __name__ == "__main__":
    main()
