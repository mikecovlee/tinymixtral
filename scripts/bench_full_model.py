"""Benchmark full 16L CPT model — stop at first OOM."""
import json
import os
import sys
import time
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cpt_model import config_from_json_file, model_for_config  # noqa: E402
from scripts.train_utils import make_adamw  # noqa: E402


def main():
    config_path = os.path.join(REPO, "configs", "cpt_from_scratch", "cpt_v3_0_from_scratch.json")
    data_dir = "/mnt/nas/jetpack-hdd/tinymixtral-improve/v3.0-dense-active/data/pretrain/main_s1"
    seq_len = 1024
    warmup, steps = 2, 3

    config = config_from_json_file(config_path)
    model = model_for_config(config)
    model.use_chunked_ce = True
    model = model.cuda()
    # No torch.compile: recompiling per-batch-size accumulates graph caches and OOMs.
    # Memory footprint is nearly identical; throughput is slightly lower than compiled.

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} params")

    shard = torch.load(
        sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith("train_") and f.endswith(".pt")
        )[0],
        weights_only=True,
    )
    print(f"Data shard: {shard.shape}")

    results = []
    print(f"\n{'bs':>4} {'grad_ckpt':>10} {'tok/s':>10} {'step_ms':>10} {'peak_GB':>10}")
    print("-" * 50)

    for bs in [16, 32, 40, 48, 56, 64]:
        for gc in [True, False]:
            try:
                if gc:
                    model.gradient_checkpointing_enable()
                else:
                    model.gradient_checkpointing_disable()
                model.use_chunked_ce = True

                opt = make_adamw(model, lr=1e-4, weight_decay=0.1, bf16_states=True)

                max_needed = (warmup + steps) * bs + 4
                seqs = shard.size(0) // seq_len
                batches = []
                for i in range(min(seqs, max_needed)):
                    s = i * seq_len
                    batches.append(shard[s : s + seq_len].unsqueeze(0).repeat(bs, 1))

                it = iter(batches)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

                for _ in range(warmup):
                    x = next(it).cuda()
                    out = model(x, labels=x)
                    (out["loss"] if isinstance(out, dict) else out.loss).backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(steps):
                    x = next(it).cuda()
                    out = model(x, labels=x)
                    (out["loss"] if isinstance(out, dict) else out.loss).backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0

                peak = torch.cuda.max_memory_allocated() / 1e9
                tok_s = bs * seq_len * steps / dt
                step_ms = dt / steps * 1000
                tag = "on" if gc else "off"
                print(f"{bs:>4} {tag:>10} {tok_s:>10,.0f} {step_ms:>10,.0f} {peak:>10.2f}")
                results.append({"bs": bs, "grad_ckpt": gc, "tok_s": tok_s, "step_ms": step_ms, "peak_gb": peak})

                del opt, batches, it, x, out
                torch.cuda.empty_cache()

            except RuntimeError as e:
                tag = "on" if gc else "off"
                if "out of memory" in str(e).lower():
                    print(f"{bs:>4} {tag:>10} {'OOM':>10}")
                    results.append({"bs": bs, "grad_ckpt": gc, "tok_s": 0, "step_ms": 0, "peak_gb": 0})
                    torch.cuda.empty_cache()
                    break
                else:
                    raise

    print("\n=== BEST ===")
    valid = [r for r in results if r["tok_s"] > 0]
    if valid:
        best = max(valid, key=lambda r: r["tok_s"])
        print(
            f"bs={best['bs']} gc={'on' if best['grad_ckpt'] else 'off'}: "
            f"{best['tok_s']:,.0f} tok/s  {best['step_ms']:,.0f} ms/step  {best['peak_gb']:.2f} GB"
        )

    os.makedirs(os.path.join(REPO, "checkpoints"), exist_ok=True)
    with open(os.path.join(REPO, "checkpoints", "bench_full_model.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
