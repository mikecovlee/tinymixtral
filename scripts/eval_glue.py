#!/usr/bin/env python3
"""Zero-shot GLUE 评测脚本。

用法:
    # 快速评测
    python scripts/eval_glue.py \
        --checkpoint checkpoints/run_001/step_5000 \
        --tokenizer tokenizer/ \
        --tasks quick --limit 500 \
        --output evals/run_001/step_5000.json

    # 全量评测
    python scripts/eval_glue.py \
        --checkpoint checkpoints/run_001/step_5000 \
        --tokenizer tokenizer/ \
        --tasks all \
        --output evals/run_001/step_5000_full.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.glue_tasks import (
    GLUE_TASKS,
    GLUE_TEMPLATES,
    QUICK_TASKS,
    SKIP_TASKS,
    get_task,
    load_glue_dataset,
    format_prompt,
    get_verbalizer_labels,
)
from evals.prompt_scoring import score_answers
from evals.metrics import accuracy_score, f1_score, matthews_corrcoef

METRIC_FUNCS = {
    "accuracy": lambda yt, yp: accuracy_score(yt, yp),
    "f1": lambda yt, yp: (accuracy_score(yt, yp), f1_score(yt, yp)),
    "matthews_corrcoef": lambda yt, yp: matthews_corrcoef(yt, yp),
}


def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot GLUE evaluation for TinyMixtral")

    # 模型
    p.add_argument("--checkpoint", type=str, required=True, help="模型路径或 HF model ID")
    p.add_argument("--tokenizer", type=str, default=None, help="Tokenizer 路径（默认同 checkpoint）")
    p.add_argument("--trust-remote-code", action="store_true", help="允许 HF remote code")

    # 任务
    p.add_argument("--tasks", type=str, default="quick",
                   help='逗号分隔任务名、"all" 或 "quick"')
    p.add_argument("--split", type=str, default="validation", help="数据集 split")

    # 数据限制
    p.add_argument("--limit", type=int, default=None, help="每个任务最多评测样例数")
    p.add_argument("--max-length", type=int, default=512, help="最大序列长度")
    p.add_argument("--batch-size", type=int, default=8, help="批大小")

    # 推理
    p.add_argument("--precision", type=str, default="bf16",
                   choices=["fp16", "bf16", "fp32"], help="推理精度")
    p.add_argument("--device", type=str, default=None, help="设备")
    p.add_argument("--seed", type=int, default=1234, help="随机种子")
    # 输出
    p.add_argument("--output", type=str, default=None, help="输出 JSON 路径")
    p.add_argument("--debug-examples", type=int, default=0, help="打印 N 个样例的调试信息")
    p.add_argument("--save-predictions", action="store_true", help="保存预测详情 JSONL")

    return p.parse_args()


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(precision):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]


def resolve_tasks(task_arg):
    if task_arg == "all":
        return [t for t in GLUE_TASKS if t not in SKIP_TASKS]
    elif task_arg == "quick":
        return QUICK_TASKS
    else:
        tasks = [t.strip() for t in task_arg.split(",")]
        for t in tasks:
            if t not in GLUE_TASKS:
                print(f"Warning: unknown task '{t}', skipping")
        return [t for t in tasks if t in GLUE_TASKS]


def load_model_and_tokenizer(args):
    """加载模型和 tokenizer。支持 HF 和自定义格式。"""
    from transformers import AutoTokenizer

    tokenizer_path = args.tokenizer or args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=args.trust_remote_code, legacy=False
    )

    # 设置 tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 尝试加载模型
    checkpoint_path = Path(args.checkpoint)

    # 先检查是否是自定义 TinyMixtral 格式
    config_file = checkpoint_path / "config.json" if checkpoint_path.is_dir() else None
    is_custom = False
    if config_file and config_file.exists():
        with open(config_file) as f:
            config_dict = json.load(f)
        # 检测自定义字段
        if "num_local_experts" in config_dict or "expert_intermediate_size" in config_dict:
            is_custom = True

    precision_dtype = resolve_dtype(args.precision)

    if is_custom:
        from model import TinyMixtralForCausalLM
        model = TinyMixtralForCausalLM.from_pretrained(str(checkpoint_path))
        model = model.to(dtype=precision_dtype)
        print(f"Loaded TinyMixtral model from {args.checkpoint} (dtype={args.precision})")
    elif checkpoint_path.is_dir() and (checkpoint_path / "config.json").exists():
        # 标准 HuggingFace 格式
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=precision_dtype,
        )
        print(f"Loaded HF model from {args.checkpoint}")
    elif checkpoint_path.is_dir() and (checkpoint_path / "pytorch_model.bin").exists():
        from model import TinyMixtralForCausalLM
        model = TinyMixtralForCausalLM.from_pretrained(str(checkpoint_path))
        model = model.to(dtype=precision_dtype)
        print(f"Loaded TinyMixtral model from {args.checkpoint} (dtype={args.precision})")
    else:
        raise FileNotFoundError(
            f"Cannot load model from {args.checkpoint}. "
            f"Expected HF config.json or pytorch_model.bin."
        )

    model_vocab_size = getattr(model.config, "vocab_size", None)
    if model_vocab_size is not None and len(tokenizer) != model_vocab_size:
        raise ValueError(
            f"Tokenizer vocab size is {len(tokenizer)}, model expects {model_vocab_size}"
        )

    return model, tokenizer


def evaluate_task(model, tokenizer, task_name, template, verbalizer, args):
    """评测单个 GLUE 任务。"""
    task_def = get_task(task_name)

    # 确定 split
    split = args.split
    if task_name == "mnli_mismatched":
        split = "validation_mismatched"
    elif task_name == "mnli":
        split = "validation_matched"

    # 加载数据
    dataset = load_glue_dataset(task_def, split=split, limit=args.limit)
    print(f"  {task_name}: {len(dataset)} examples (split={split})")

    # 格式化 prompts
    prompts = []
    gold_labels = []
    for ex in dataset:
        prompt = format_prompt(ex, template, task_def.input_fields)
        prompts.append(prompt)
        gold_labels.append(ex["label"])

    # 构建 answer choices
    label_ids, answer_strings = get_verbalizer_labels(verbalizer)
    answer_choices = [answer_strings for _ in prompts]
    label_id_lists = [label_ids for _ in prompts]

    # 评分
    t0 = time.time()
    score_results = score_answers(
        model, tokenizer,
        prompts=prompts,
        answer_choices=answer_choices,
        label_ids=label_id_lists,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    elapsed = time.time() - t0

    y_pred = [r.predicted_label for r in score_results]

    # 调试输出
    if args.debug_examples > 0:
        print(f"\n  --- Debug: {task_name} ---")
        for i in range(min(args.debug_examples, len(prompts))):
            print(f"  [{i}] Prompt: {prompts[i][:120]}...")
            print(f"      Gold: {gold_labels[i]}, Pred: {y_pred[i]}")
            print(f"      Scores: {score_results[i].label_scores}")
            print()

    return y_pred, gold_labels, elapsed, task_def, score_results


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Precision: {args.precision}")

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args)
    model.eval()
    model.to(device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # 解析任务列表
    tasks = resolve_tasks(args.tasks)
    print(f"Tasks: {tasks}")

    # 获取模板
    templates = GLUE_TEMPLATES  # 目前只有 default

    results = {}
    all_predictions = []
    total_start = time.time()

    for task_name in tasks:
        print(f"\n[{task_name}]")
        if task_name not in templates:
            print(f"  Skipping: no template")
            continue

        tpl = templates[task_name]
        y_pred, y_true, elapsed, task_def, score_results = evaluate_task(
            model, tokenizer, task_name, tpl["prompt"], tpl["verbalizer"], args
        )

        # 计算指标
        metric_name = task_def.metric_fn
        metric_fn = METRIC_FUNCS[metric_name]

        if metric_name == "f1":
            acc, f1 = metric_fn(y_true, y_pred)
            task_result = {
                "num_examples": len(y_true),
                "accuracy": round(acc, 4),
                "f1": round(f1, 4),
                "time_seconds": round(elapsed, 1),
            }
        elif metric_name == "matthews_corrcoef":
            mcc = metric_fn(y_true, y_pred)
            task_result = {
                "num_examples": len(y_true),
                "matthews_correlation": round(mcc, 4),
                "time_seconds": round(elapsed, 1),
            }
        else:
            acc = metric_fn(y_true, y_pred)
            task_result = {
                "num_examples": len(y_true),
                "accuracy": round(acc, 4),
                "time_seconds": round(elapsed, 1),
            }

        results[task_name] = task_result
        print(f"  Result: {task_result}")

        # 保存预测
        if args.save_predictions:
            for i in range(len(y_true)):
                all_predictions.append({
                    "task": task_name,
                    "example_id": i,
                    "gold_label": y_true[i],
                    "predicted_label": y_pred[i],
                    "label_scores": score_results[i].label_scores,
                    "correct": y_true[i] == y_pred[i],
                })

    total_time = time.time() - total_start

    # 计算 aggregate
    scores_list = []
    for t in tasks:
        if t in results:
            r = results[t]
            task_def = get_task(t)
            if task_def.metric_fn == "f1" and "f1" in r:
                scores_list.append(r["f1"])
            elif task_def.metric_fn == "matthews_corrcoef" and "matthews_correlation" in r:
                scores_list.append((r["matthews_correlation"] + 1) / 2)
            elif "accuracy" in r:
                scores_list.append(r["accuracy"])
    mean_score = round(sum(scores_list) / len(scores_list), 4) if scores_list else 0.0

    # 构建输出
    output = {
        "checkpoint": str(args.checkpoint),
        "tokenizer": str(args.tokenizer or args.checkpoint),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "tasks": tasks,
            "split": args.split,
            "limit": args.limit,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "precision": args.precision,
            "seed": args.seed,
        },
        "results": results,
        "aggregate": {"mean_score": mean_score},
    }

    # 保存
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # 保存预测
    if args.save_predictions and args.output:
        pred_path = str(Path(args.output).with_suffix(".predictions.jsonl"))
        with open(pred_path, "w") as f:
            for pred in all_predictions:
                f.write(json.dumps(pred) + "\n")
        print(f"Predictions saved to {pred_path}")

    # 打印汇总
    print(f"\n{'=' * 50}")
    print(f"Summary (total: {total_time:.0f}s)")
    for t, r in results.items():
        print(f"  {t}: {r}")
    print(f"  aggregate mean_score: {mean_score}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
