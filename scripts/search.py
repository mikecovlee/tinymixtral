#!/usr/bin/env python3
"""超参搜索：进程内 pretrain → GLUE eval 循环，最快迭代速度。

用法:
    python scripts/search.py --trials 20 --steps-per-trial 2000
"""

import argparse, csv, gc, glob, json, os, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.config import TinyMixtralConfig
from model.modeling import TinyMixtralForCausalLM
from scripts.train_utils import make_cosine_schedule, make_adamw
from transformers import AutoTokenizer
from datasets import load_dataset


# ============================================================
# 数据加载（一次性，所有 trial 共享）
# ============================================================

def load_local_data(data_dir="data/c4/tokenized", max_tokens=10_000_000):
    """从本地 shard 加载有限 token，避免搜索时读入完整训练集。"""
    files = sorted(glob.glob(f"{data_dir}/train_*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files in {data_dir}")
    parts = []
    remaining = max_tokens
    for path in files:
        shard = torch.load(path, weights_only=True)
        parts.append(shard[:remaining])
        remaining -= len(parts[-1])
        if remaining <= 0:
            break
    return torch.cat(parts)


# ============================================================
# 训练
# ============================================================

def train_trial(config_dict, all_data, tokenizer, steps, batch_size, seq_len,
                lr, warmup_ratio, weight_decay, aux_coef, seed):
    """训练一个 trial，返回 (model, final_loss, tokens_per_sec)。"""
    torch.manual_seed(seed)

    config = TinyMixtralConfig(
        hidden_size=config_dict["hidden_size"],
        num_hidden_layers=config_dict["num_layers"],
        num_attention_heads=config_dict["hidden_size"] // 64,
        num_key_value_heads=2, head_dim=64,
        num_local_experts=config_dict["num_experts"],
        num_experts_per_tok=2,
        expert_intermediate_size=int(config_dict["hidden_size"] * 8 / 3),
        max_position_embeddings=seq_len, vocab_size=32000,
        router_aux_loss_coef=aux_coef,
    )

    model = TinyMixtralForCausalLM(config)
    model.gradient_checkpointing_enable()
    model = model.to("cuda").to(torch.bfloat16)

    warmup = max(1, int(steps * warmup_ratio))
    optimizer = make_adamw(model, lr=lr, weight_decay=weight_decay)
    scheduler = make_cosine_schedule(optimizer, warmup, steps)

    chunk_size = (seq_len + 1) * batch_size
    if len(all_data) < chunk_size:
        print(f"  Dataset too small ({len(all_data)} tokens) for batch, skipping", flush=True)
        return None, float("inf"), 0
    total_tokens = 0
    data_ptr = 0
    losses = []

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    for step in range(1, steps + 1):
        if data_ptr + chunk_size > len(all_data):
            data_ptr = 0
        chunk = all_data[data_ptr:data_ptr + chunk_size].view(batch_size, seq_len + 1).to("cuda")
        data_ptr += chunk_size  # advance by one full batch of raw tokens
        total_tokens += batch_size * seq_len

        ids, labels = chunk[:, :-1], chunk[:, 1:]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(ids, labels=labels)
        if not torch.isfinite(out["loss"]):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"Non-finite loss at search step {step}: {out['loss'].item()}")
        out["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"Non-finite gradient norm at search step {step}: {grad_norm.item()}"
            )
        optimizer.step(); scheduler.step(); optimizer.zero_grad()

        losses.append(out["loss"].item())

    elapsed = time.time() - t0
    avg_loss = sum(losses[-50:]) / min(50, len(losses))  # 最后 50 步平均
    tok_s = total_tokens / elapsed

    return model, avg_loss, tok_s


# ============================================================
# GLUE 评测（进程内，不加载/卸载模型）
# ============================================================

def eval_trial(model, tokenizer, tasks, limit, batch_size, max_length):
    """进程内 GLUE zero-shot 评测。"""
    from evals.glue_tasks import (GLUE_TEMPLATES, get_task, load_glue_dataset,
                                   format_prompt, get_verbalizer_labels)
    from evals.prompt_scoring import score_answers
    from evals.metrics import accuracy_score, f1_score, matthews_corrcoef

    results = {}
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for task_name in tasks:
        if task_name not in GLUE_TEMPLATES:
            continue
        tpl = GLUE_TEMPLATES[task_name]
        task_def = get_task(task_name)
        ds = load_glue_dataset(task_def, split="validation", limit=limit)

        prompts = [format_prompt(ex, tpl["prompt"], task_def.input_fields) for ex in ds]
        gold = [ex["label"] for ex in ds]
        label_ids, answers = get_verbalizer_labels(tpl["verbalizer"])

        sr = score_answers(model, tokenizer, prompts,
                           [answers]*len(prompts), [label_ids]*len(prompts),
                           batch_size=batch_size, max_length=max_length)

        y_pred = [r.predicted_label for r in sr]
        metric = task_def.metric_fn

        if metric == "f1":
            results[task_name] = {"accuracy": round(accuracy_score(gold, y_pred), 4),
                                  "f1": round(f1_score(gold, y_pred), 4)}
        elif metric == "matthews_corrcoef":
            results[task_name] = {"matthews_correlation": round(matthews_corrcoef(gold, y_pred), 4)}
        else:
            results[task_name] = {"accuracy": round(accuracy_score(gold, y_pred), 4)}

    # Aggregate
    scores = []
    for t, r in results.items():
        if "accuracy" in r: scores.append(r["accuracy"])
        elif "matthews_correlation" in r: scores.append((r["matthews_correlation"] + 1) / 2)
    mean_score = round(sum(scores) / len(scores), 4) if scores else 0.0

    return results, mean_score


# ============================================================
# 超参生成
# ============================================================

def generate_trials(n):
    """生成搜索网格。"""
    # 架构搜索
    archs = [
        {"hidden_size": 896, "num_layers": 8,  "num_experts": 6},
        {"hidden_size": 896, "num_layers": 10, "num_experts": 6},
        {"hidden_size": 896, "num_layers": 12, "num_experts": 6},
        {"hidden_size": 768, "num_layers": 12, "num_experts": 6},
        {"hidden_size": 1024, "num_layers": 8, "num_experts": 6},
    ]
    # 训练超参
    train_hparams = [
        {"lr": 3e-4, "warmup_ratio": 0.1, "weight_decay": 0.1, "aux_coef": 0.01},
        {"lr": 5e-4, "warmup_ratio": 0.1, "weight_decay": 0.1, "aux_coef": 0.01},
        {"lr": 1e-4, "warmup_ratio": 0.1, "weight_decay": 0.1, "aux_coef": 0.01},
        {"lr": 3e-4, "warmup_ratio": 0.2, "weight_decay": 0.1, "aux_coef": 0.01},
        {"lr": 3e-4, "warmup_ratio": 0.1, "weight_decay": 0.05, "aux_coef": 0.01},
        {"lr": 3e-4, "warmup_ratio": 0.1, "weight_decay": 0.1, "aux_coef": 0.05},
        {"lr": 3e-4, "warmup_ratio": 0.05,"weight_decay": 0.1, "aux_coef": 0.01},
        {"lr": 2e-4, "warmup_ratio": 0.15,"weight_decay": 0.08,"aux_coef": 0.02},
    ]

    trials = []
    for arch in archs[:3]:  # 前 3 个架构
        for hp in train_hparams[:5]:  # 前 5 组超参
            trials.append({**arch, **hp})
    for arch in archs[3:]:  # 大架构用保守超参
        trials.append({**arch, "lr": 3e-4, "warmup_ratio": 0.1,
                       "weight_decay": 0.1, "aux_coef": 0.01})

    if n < len(trials):
        return trials[:n]
    return trials


# ============================================================
# 主循环
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--steps-per-trial", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--tasks", type=str, default="sst2,mrpc,qnli,rte,cola")
    p.add_argument("--eval-limit", type=int, default=200)
    p.add_argument("--eval-batch", type=int, default=8)
    p.add_argument("--skip-eval", action="store_true", help="仅训练，跳过 GLUE（用 train_loss 初筛）")
    p.add_argument("--output-dir", type=str, default="search_results")
    p.add_argument("--tokenizer", type=str, default="tokenizer/")
    p.add_argument("--data-dir", type=str, default="data/c4/tokenized")
    p.add_argument("--max-data-tokens", type=int, default=10_000_000,
                   help="搜索时最多载入内存的 token 数")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    if args.max_data_tokens <= 0:
        p.error("max-data-tokens must be positive")

    os.makedirs(args.output_dir, exist_ok=True)
    tasks = [t.strip() for t in args.tasks.split(",")]

    # 一次性加载数据和 tokenizer
    print("Loading data...")
    all_data = load_local_data(args.data_dir, args.max_data_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, legacy=False)
    expected_vocab_size = TinyMixtralConfig().vocab_size
    if len(tokenizer) != expected_vocab_size:
        p.error(
            f"tokenizer vocab size is {len(tokenizer)}, expected {expected_vocab_size}"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trials = generate_trials(args.trials)
    print(f"{len(trials)} trials, {args.steps_per_trial} steps each")
    print(f"Eval: {tasks} × {args.eval_limit} examples each\n")

    csv_path = Path(args.output_dir) / "results.csv"
    fieldnames = ["trial", "hs", "layers", "experts", "lr", "warmup", "wd", "aux",
                  "train_loss", "tok_s", "gpu_gb",
                  "sst2", "mrpc_acc", "mrpc_f1", "qnli", "rte", "cola",
                  "mean_score", "time_s"]

    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames).writeheader()

    best_score = float("-inf")
    best_trial = None

    for i, hp in enumerate(trials):
        t0 = time.time()
        print(f"[Trial {i+1}/{len(trials)}] {hp['hidden_size']}d/{hp['num_layers']}L/"
              f"{hp['num_experts']}E lr={hp['lr']:.0e} wd={hp['weight_decay']} aux={hp['aux_coef']}")

        # Train
        torch.manual_seed(args.seed + i)
        try:
            model, train_loss, tok_s = train_trial(
                hp, all_data, tokenizer, args.steps_per_trial,
                args.batch_size, args.seq_len,
                hp["lr"], hp["warmup_ratio"], hp["weight_decay"], hp["aux_coef"],
                args.seed + i)
        except (torch.cuda.OutOfMemoryError, FloatingPointError):
            print(f"  Failed! Skipping.")
            torch.cuda.empty_cache()
            continue

        if model is None:
            continue  # dataset too small

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Train: loss={train_loss:.4f} tok/s={tok_s:.0f} peak={peak_gb:.1f}GB")

        # Eval
        if args.skip_eval:
            eval_results, mean_score = {}, 0.0
        else:
            model.eval()
            eval_results, mean_score = eval_trial(
                model, tokenizer, tasks, args.eval_limit, args.eval_batch, 256)
        del model; gc.collect(); torch.cuda.empty_cache()

        elapsed = time.time() - t0

        row = {
            "trial": i, "hs": hp["hidden_size"], "layers": hp["num_layers"],
            "experts": hp["num_experts"], "lr": hp["lr"], "warmup": hp["warmup_ratio"],
            "wd": hp["weight_decay"], "aux": hp["aux_coef"],
            "train_loss": round(train_loss, 4), "tok_s": round(tok_s, 0),
            "gpu_gb": round(peak_gb, 1), "mean_score": mean_score, "time_s": round(elapsed, 1),
        }
        for t, r in eval_results.items():
            for k, v in r.items():
                row[f"{t}_{k.split('_')[0]}"] = v
            if "accuracy" in r:
                row[t] = r["accuracy"]
            elif "matthews_correlation" in r:
                row[t] = r["matthews_correlation"]
            elif "f1" in r:
                row[t] = r["f1"]

        print(f"  Eval: mean={mean_score:.4f} | { {t: eval_results[t].get('accuracy', eval_results[t].get('matthews_correlation','?')) for t in eval_results} }")
        print(f"  Time: {elapsed:.0f}s")

        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames, extrasaction="ignore").writerow(row)

        selection_score = -train_loss if args.skip_eval else mean_score
        if selection_score > best_score:
            best_score = selection_score
            best_trial = {**hp, "mean_score": mean_score, "train_loss": row["train_loss"]}
            print(f"  ★ NEW BEST!")

        print()

    print(f"{'='*50}")
    if args.skip_eval:
        print(f"Best: train_loss={best_trial['train_loss']:.4f}" if best_trial else "Best: none")
    else:
        print(f"Best: score={best_score:.4f}")
    if best_trial:
        print(f"  {best_trial}")
    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
