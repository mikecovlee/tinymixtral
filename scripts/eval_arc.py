#!/usr/bin/env python3
"""ARC (AI2 Reasoning Challenge) 评测，支持 zero-shot / few-shot。

用法:
    # zero-shot
    python scripts/eval_arc.py --checkpoint checkpoints/run/step_0005000 --tokenizer tokenizer/
    # few-shot
    python scripts/eval_arc.py --checkpoint checkpoints/run/step_0010000 --tasks arc_c --shots 5 --limit 100
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.prompt_scoring import score_answers
from evals.metrics import accuracy_score
from datasets import load_dataset
from transformers import AutoTokenizer


TASKS = {
    "arc_c": {
        "dataset_config": "ARC-Challenge",
        "prompt": "Question: {question}\nAnswer:",
    },
    "arc_e": {
        "dataset_config": "ARC-Easy",
        "prompt": "Question: {question}\nAnswer:",
    },
}


def load_arc_data(config, split="test", limit=None):
    """加载 ARC 数据。返回 [(question, choices, answer_idx), ...]"""
    ds = load_dataset("ai2_arc", config, split=split)
    if limit is not None and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    examples = []
    for ex in ds:
        choices = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        answer_key = ex["answerKey"]
        try:
            gold_idx = labels.index(answer_key)
        except ValueError:
            continue
        examples.append((ex["question"], choices, gold_idx))
    return examples


def build_few_shot_prompt(demos, target_question):
    """用 k 个 (question, answer_text) demonstrations 构建 few-shot prompt。"""
    parts = []
    for q, a in demos:
        parts.append(f"Question: {q}\nAnswer: {a}")
    parts.append(f"Question: {target_question}\nAnswer:")
    return "\n\n".join(parts)


def load_model_and_tokenizer(checkpoint, tokenizer_path, precision="bf16", trust_remote_code=False):
    """加载模型和 tokenizer。"""
    from model import TinyMixtralForCausalLM

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[precision]

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path or checkpoint, trust_remote_code=trust_remote_code, legacy=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = TinyMixtralForCausalLM.from_pretrained(str(checkpoint))
    model = model.to(dtype=dtype)
    model.eval()
    return model, tokenizer


def main():
    p = argparse.ArgumentParser(description="Zero-shot ARC evaluation for TinyMixtral")
    p.add_argument("--checkpoint", required=True, help="模型 checkpoint 路径")
    p.add_argument("--tokenizer", default=None, help="Tokenizer 路径（默认同 checkpoint）")
    p.add_argument("--tasks", default="arc_c", help="逗号分隔: arc_c, arc_e")
    p.add_argument("--limit", type=int, default=None, help="最多评测样例数")
    p.add_argument("--shots", type=int, default=0, help="few-shot 示例数（0 = zero-shot）")
    p.add_argument("--max-length", type=int, default=None,
                   help="最大序列长度（默认 zero-shot 256, few-shot 512）")
    p.add_argument("--batch-size", type=int, default=8, help="批大小")
    p.add_argument("--precision", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", default=None, help="设备 (cuda/cpu)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output", default=None, help="输出 JSON 路径")
    p.add_argument("--debug-examples", type=int, default=0, help="打印 N 个样例")
    args = p.parse_args()

    if args.max_length is None:
        args.max_length = 512 if args.shots > 0 else 256

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(
        args.checkpoint, args.tokenizer, args.precision
    )
    model = model.to(device)
    nM = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {nM:.0f}M params")

    # 解析任务
    task_names = [t.strip() for t in args.tasks.split(",")]
    for t in task_names:
        if t not in TASKS:
            print(f"Warning: unknown task '{t}', skipping")
    task_names = [t for t in task_names if t in TASKS]
    print(f"Tasks: {task_names}")

    # 加载 few-shot 示例池
    demo_pools = {}
    if args.shots > 0:
        for task_name in task_names:
            task_info = TASKS[task_name]
            pool = load_arc_data(task_info["dataset_config"], split="train")
            demo_pools[task_name] = pool
            print(f"  Few-shot demos for {task_name}: {len(pool)} available")

    all_results = {}
    total_start = time.time()

    for task_name in task_names:
        task_info = TASKS[task_name]
        print(f"\n[{task_name}]")

        # 加载数据
        data = load_arc_data(task_info["dataset_config"], limit=args.limit)
        print(f"  {len(data)} examples ({'%d-shot' % args.shots if args.shots > 0 else 'zero-shot'})")
        rng = random.Random(args.seed)

        # 构建 prompts + answer choices
        prompts = []
        gold_labels = []
        answer_choices = []
        label_id_lists = []

        for question, choices, gold_idx in data:
            if args.shots > 0:
                # 随机采样 k 个 demonstration
                pool = demo_pools[task_name]
                demos_raw = rng.sample(pool, min(args.shots, len(pool)))
                demos = [(q, c[gold]) for q, c, gold in demos_raw]
                prompt = build_few_shot_prompt(demos, question)
            else:
                prompt = task_info["prompt"].format(question=question)
            prompts.append(prompt)
            gold_labels.append(gold_idx)
            # ARC 每个问题有 3-5 个选项，用索引 0..N-1 作为 label
            label_ids = list(range(len(choices)))
            # 前导空格与 GLUE verbalizer 风格一致
            answer_strings = [" " + c for c in choices]
            answer_choices.append(answer_strings)
            label_id_lists.append(label_ids)

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
        acc = accuracy_score(gold_labels, y_pred)

        print(f"  Accuracy: {acc:.4f} ({elapsed:.0f}s, {elapsed/len(data):.2f}s/example)")

        # 调试
        if args.debug_examples > 0:
            print(f"\n  --- Debug: {task_name} ---")
            for i in range(min(args.debug_examples, len(prompts))):
                print(f"  [{i}] Q: {data[i][0][:100]}...")
                print(f"      Choices: {dict(zip(range(len(data[i][1])), data[i][1]))}")
                print(f"      Gold: {gold_labels[i]}, Pred: {y_pred[i]}")
                print(f"      Scores: {score_results[i].label_scores}")
                print()

        all_results[task_name] = {
            "accuracy": round(acc, 4),
            "num_examples": len(data),
            "time_seconds": round(elapsed, 1),
        }

    total_time = time.time() - total_start
    print(f"\n{'=' * 50}")
    for t, r in all_results.items():
        print(f"  {t}: accuracy={r['accuracy']:.4f} ({r['num_examples']} examples)")
    print(f"  Total: {total_time:.0f}s")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output = {
            "checkpoint": str(args.checkpoint),
            "results": all_results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
