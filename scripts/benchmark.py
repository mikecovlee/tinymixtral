#!/usr/bin/env python3
"""硬件适配 Benchmark：测显存 + GPU 利用率，找模型/batch 最佳平衡点。

用法:
    python scripts/benchmark.py [--quick] [--output configs/hardware_profile.json]
"""

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.config import TinyMixtralConfig
from model.modeling import TinyMixtralForCausalLM
from scripts.train_utils import make_adamw


def get_gpu_info():
    """获取 GPU 信息。"""
    p = torch.cuda.get_device_properties(0)
    return {
        "name": p.name,
        "total_memory_gb": round(p.total_memory / 1e9, 1),
        "compute_capability": f"{p.major}.{p.minor}",
        "multi_processor_count": p.multi_processor_count,
    }


def get_gpu_util():
    """通过 pynvml 获取 GPU 利用率。失败时返回 (-1, -1)。"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        pynvml.nvmlShutdown()
        return util.gpu, util.memory
    except Exception:
        return -1, -1


def test_config(hs, nl, ne, bs, sl, steps=5):
    """测试一个配置：返回 (ok, peak_gb, avg_step_ms, avg_gpu_util%, params_M)。"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    n_heads = hs // 64
    n_kv = max(2, n_heads // 4)
    while n_kv > 1 and n_heads % n_kv != 0:
        n_kv -= 1
    if n_heads % n_kv != 0:
        n_kv = 2

    try:
        config = TinyMixtralConfig(
            hidden_size=hs, num_hidden_layers=nl,
            num_attention_heads=n_heads, num_key_value_heads=n_kv, head_dim=64,
            num_local_experts=ne, num_experts_per_tok=min(2, ne),
            expert_intermediate_size=int(hs * 8//3), max_position_embeddings=sl, vocab_size=32000,
        )
        model = TinyMixtralForCausalLM(config)
        model.gradient_checkpointing_enable()
        model = model.to("cuda").to(torch.bfloat16)
        nM = sum(p.numel() for p in model.parameters()) / 1e6

        optimizer = make_adamw(model, lr=3e-4, weight_decay=0.1)
        x = torch.randint(0, 1000, (bs, sl + 1), device="cuda")

        # Warmup
        for _ in range(2):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(x[:, :-1], labels=x[:, 1:])
            out["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()

        # 测量：记录每步时间和利用率
        torch.cuda.synchronize()
        step_times = []
        gpu_utils = []

        for _ in range(steps):
            util_before = get_gpu_util()[0]

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(x[:, :-1], labels=x[:, 1:])
            out["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            step_times.append((t1 - t0) * 1000)

            util_after = get_gpu_util()[0]
            gpu_utils.append(max(util_before, util_after, 0))

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        avg_step_ms = sum(step_times) / len(step_times)
        avg_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils[0] > 0 else -1

        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        mem_pct = peak_gb / total_mem * 100

        del model, optimizer, x, out
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "ok": True,
            "hidden_size": hs, "num_layers": nl, "num_experts": ne,
            "batch_size": bs, "seq_len": sl,
            "params_M": round(nM, 1),
            "peak_memory_gb": round(peak_gb, 2),
            "memory_pct": round(mem_pct, 1),
            "avg_step_ms": round(avg_step_ms, 1),
            "gpu_util_pct": round(avg_gpu_util, 1) if avg_gpu_util > 0 else None,
            "tokens_per_sec": round(bs * sl / (avg_step_ms / 1000), 0),
        }

    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return {"ok": False, "hidden_size": hs, "num_layers": nl, "num_experts": ne,
                "batch_size": bs, "seq_len": sl}
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        return {"ok": False, "hidden_size": hs, "num_layers": nl, "num_experts": ne,
                "batch_size": bs, "seq_len": sl, "error": str(e)[:80]}


def compute_score(r, gpu_info):
    """综合评分：模型大小 × 吞吐 × GPU利用率 （越高越好）。"""
    params_score = math.log10(r["params_M"] + 1)  # 对数避免大模型主导
    speed_score = math.log10(r["tokens_per_sec"] + 1)
    util_score = r.get("gpu_util_pct") or 80
    util_score = util_score / 100.0

    # 显存利用率奖励：越接近上限分越高，但不能 OOM
    mem_score = r["memory_pct"] / 85.0  # 85% 是最佳点
    mem_score = min(mem_score, 1.2)  # cap

    return round(params_score * 0.3 + speed_score * 0.25 + util_score * 0.25 + mem_score * 0.2, 4)


def run_benchmark(output_path, quick=False):
    gpu_info = get_gpu_info()
    total_gb = gpu_info["total_memory_gb"]
    target_gb = total_gb - 2.0

    print("=" * 65)
    print("TinyMixtral Hardware Benchmark")
    print(f"GPU: {gpu_info['name']} ({total_gb:.1f}GB)")
    print(f"Target: <{target_gb:.1f}GB (留2GB), bf16 + AdamW + act_ckpt")
    print("=" * 65)

    # 搜索网格
    if quick:
        hidden_sizes = [768, 896, 1024]
        layer_counts = [8, 10, 12]
        expert_counts = [6]
        batch_sizes = [8, 16, 22]
    else:
        hidden_sizes = [512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048]
        layer_counts = [6, 8, 10, 12, 16, 20]
        expert_counts = [4, 6, 8]
        batch_sizes = [1, 2, 4, 8, 16, 22]

    seq_len = 1024

    total_tests = len(hidden_sizes) * len(layer_counts) * len(expert_counts) * len(batch_sizes)
    results = []
    tested = 0

    print(f"\nSearching {total_tests} configs...")
    print(f"{'Config':<20} {'Params':>7} {'bs':>3} {'Peak':>6} {'Util':>5} {'tok/s':>7} {'Score':>6}")
    print("-" * 60)

    for hs in hidden_sizes:
        for nl in layer_counts:
            if hs * nl > 25000:  # 剪枝：太极端跳过
                continue
            for ne in expert_counts:
                for bs in batch_sizes:
                    tested += 1
                    r = test_config(hs, nl, ne, bs, seq_len)

                    if r["ok"] and r["peak_memory_gb"] < target_gb:
                        r["score"] = compute_score(r, gpu_info)
                        results.append(r)
                        util_str = f'{r.get("gpu_util_pct", "?"):.0f}%' if r.get("gpu_util_pct") else "N/A"
                        print(f'{hs}d/{nl}L/{ne}E      {r["params_M"]:>6.0f}M {bs:>3} '
                              f'{r["peak_memory_gb"]:>5.1f}G {util_str:>5} {r["tokens_per_sec"]:>6.0f} {r["score"]:>6.3f}')

                    if tested % 30 == 0:
                        print(f"  ... {tested}/{total_tests}, {len(results)} feasible")

    print(f"\n--- Done: {len(results)} feasible ---")

    if not results:
        print("No config found! Running minimal fallback...")
        r = test_config(512, 6, 4, 2, 1024)
        if r["ok"]:
            r["score"] = 0
            results = [r]
        else:
            print("FATAL: even minimal config OOMs. Try reducing seq_len or batch_size.")
            return None

    # 按 score 排序
    results.sort(key=lambda x: x.get("score", 0), reverse=True)

    # 找最优
    best = results[0]
    # 找 OOM 边界附近的 top configs
    high_mem = [r for r in results if r["memory_pct"] > 70]
    high_util = [r for r in results if (r.get("gpu_util_pct") or 0) > 70]

    output = {
        "hardware": gpu_info,
        "benchmark_config": {
            "search_grid": f"{len(hidden_sizes)}hs × {len(layer_counts)}L × {len(expert_counts)}E × {len(batch_sizes)}bs",
            "seq_len": seq_len,
            "precision": "bf16",
            "optimizer": "AdamW",
            "activation_checkpointing": True,
            "target_memory_gb": round(target_gb, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "recommended": best,
        "alternatives": {
            "max_model": max(results, key=lambda r: r["params_M"]),
            "max_throughput": max(results, key=lambda r: r["tokens_per_sec"]),
            "high_utilization": high_util[:3] if high_util else [],
            "near_oom": high_mem[:3] if high_mem else [],
        },
        "all_feasible_top30": results[:30],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 打印摘要
    print(f"\n{'=' * 65}")
    print(f"[RECOMMENDED] score={best['score']:.3f}")
    print(f"  Config: {best['hidden_size']}d/{best['num_layers']}L/{best['num_experts']}E")
    print(f"  Batch: {best['batch_size']}, Params: {best['params_M']}M")
    print(f"  Memory: {best['peak_memory_gb']}GB ({best['memory_pct']}%)")
    print(f"  GPU Util: {best.get('gpu_util_pct', 'N/A')}%")
    print(f"  Throughput: {best['tokens_per_sec']} tok/s")
    print(f"\n  Output: {output_path}")
    print(f"{'=' * 65}")

    return output


def main():
    p = argparse.ArgumentParser(description="TinyMixtral Hardware Benchmark")
    p.add_argument("--output", default="configs/hardware_profile.json")
    p.add_argument("--quick", action="store_true", help="快速模式（更小的搜索网格）")
    args = p.parse_args()
    run_benchmark(args.output, args.quick)


if __name__ == "__main__":
    main()
