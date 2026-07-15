# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

"""GLUE 任务定义、模板和 verbalizer。"""

from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# 任务注册
# ============================================================

@dataclass
class GlueTask:
    name: str
    dataset_config: str
    num_labels: int = 2
    metric_fn: str = "accuracy"
    train_split: str = "train"
    eval_split: str = "validation"
    input_fields: list[str] = field(default_factory=list)


GLUE_TASKS: dict[str, GlueTask] = {
    "sst2": GlueTask(
        name="sst2", dataset_config="sst2", num_labels=2,
        metric_fn="accuracy", input_fields=["sentence"],
    ),
    "mrpc": GlueTask(
        name="mrpc", dataset_config="mrpc", num_labels=2,
        metric_fn="f1", input_fields=["sentence1", "sentence2"],
    ),
    "qqp": GlueTask(
        name="qqp", dataset_config="qqp", num_labels=2,
        metric_fn="f1", input_fields=["question1", "question2"],
    ),
    "qnli": GlueTask(
        name="qnli", dataset_config="qnli", num_labels=2,
        metric_fn="accuracy", input_fields=["question", "sentence"],
    ),
    "rte": GlueTask(
        name="rte", dataset_config="rte", num_labels=2,
        metric_fn="accuracy", input_fields=["sentence1", "sentence2"],
    ),
    "cola": GlueTask(
        name="cola", dataset_config="cola", num_labels=2,
        metric_fn="matthews_corrcoef", input_fields=["sentence"],
    ),
    "mnli": GlueTask(
        name="mnli", dataset_config="mnli", num_labels=3,
        metric_fn="accuracy", eval_split="validation_matched",
        input_fields=["premise", "hypothesis"],
    ),
    "mnli_mismatched": GlueTask(
        name="mnli_mismatched", dataset_config="mnli", num_labels=3,
        metric_fn="accuracy", eval_split="validation_mismatched",
        input_fields=["premise", "hypothesis"],
    ),
}

QUICK_TASKS = ["sst2", "mrpc", "qnli", "rte", "cola"]
SKIP_TASKS = {"wnli"}


# ============================================================
# 模板与 Verbalizer
# ============================================================

GLUE_TEMPLATES = {
    "sst2": {
        "prompt": "Sentence: {sentence}\nSentiment:",
        "verbalizer": {0: " negative", 1: " positive"},
    },
    "mrpc": {
        "prompt": "Sentence 1: {sentence1}\nSentence 2: {sentence2}\nEquivalent?",
        "verbalizer": {0: " no", 1: " yes"},
    },
    "qqp": {
        "prompt": "Question 1: {question1}\nQuestion 2: {question2}\nDuplicate?",
        "verbalizer": {0: " no", 1: " yes"},
    },
    "qnli": {
        "prompt": "Question: {question}\nSentence: {sentence}\nAnswers the question?",
        "verbalizer": {0: " yes", 1: " no"},
    },
    "rte": {
        "prompt": "Premise: {sentence1}\nHypothesis: {sentence2}\nEntailment?",
        "verbalizer": {0: " yes", 1: " no"},
    },
    "cola": {
        "prompt": "Sentence: {sentence}\nGrammatically acceptable?",
        "verbalizer": {0: " no", 1: " yes"},
    },
    "mnli": {
        "prompt": "Premise: {premise}\nHypothesis: {hypothesis}\nRelation:",
        "verbalizer": {0: " entailment", 1: " neutral", 2: " contradiction"},
    },
    "mnli_mismatched": {
        "prompt": "Premise: {premise}\nHypothesis: {hypothesis}\nRelation:",
        "verbalizer": {0: " entailment", 1: " neutral", 2: " contradiction"},
    },
}


# ============================================================
# 辅助函数
# ============================================================

def get_task(task_name: str) -> GlueTask:
    if task_name not in GLUE_TASKS:
        raise KeyError(f"Unknown task: {task_name}. Available: {list(GLUE_TASKS.keys())}")
    return GLUE_TASKS[task_name]


def load_glue_dataset(task: GlueTask, split: str = "validation", limit: Optional[int] = None):
    from datasets import load_dataset
    dataset = load_dataset("glue", task.dataset_config, split=split)
    if limit is not None and limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def format_prompt(example: dict, template: str, input_fields: list[str]) -> str:
    fields = {f: example[f] for f in input_fields}
    return template.format(**fields)


def get_verbalizer_labels(verbalizer: dict) -> tuple:
    items = sorted(verbalizer.items(), key=lambda x: x[0])
    return [i[0] for i in items], [i[1] for i in items]
