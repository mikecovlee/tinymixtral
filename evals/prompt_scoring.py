# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

"""Zero-shot prompt scoring：用条件 log-likelihood 给 label 打分。

核心逻辑：
- 仅对 answer/verbalizer token 计算 log-likelihood
- Prompt token 不计分
- 按 answer token 数归一化（减少长度偏差）
- 使用 left-padding 进行批处理

注意: 训练数据使用 add_special_tokens=False + 显式追加 EOS (无 BOS)，
      此处评测同样禁用 special tokens (add_special_tokens=False)。
      这是为了避免 BOS token 导致 answer span 计算偏移。
"""

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class ScoreResult:
    """单个样例的评分结果。"""
    label_scores: dict  # label_id -> normalized log-likelihood
    predicted_label: int
    metadata: Optional[dict] = None


def score_answers(
    model,
    tokenizer,
    prompts: list[str],
    answer_choices: list[list[str]],
    label_ids: list[list[int]],
    batch_size: int = 8,
    max_length: int = 512,
    device: Optional[torch.device] = None,
) -> list[ScoreResult]:
    """配置评分所需的 tokenizer 状态，并保证异常时恢复。"""
    original_padding_side = tokenizer.padding_side
    original_truncation_side = tokenizer.truncation_side
    original_pad_token = tokenizer.pad_token
    try:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        return _score_answers(
            model, tokenizer, prompts, answer_choices, label_ids,
            batch_size, max_length, device,
        )
    finally:
        tokenizer.padding_side = original_padding_side
        tokenizer.truncation_side = original_truncation_side
        tokenizer.pad_token = original_pad_token


def _score_answers(
    model,
    tokenizer,
    prompts: list[str],
    answer_choices: list[list[str]],
    label_ids: list[list[int]],
    batch_size: int = 8,
    max_length: int = 512,
    device: Optional[torch.device] = None,
) -> list[ScoreResult]:
    """
    对每个 prompt，计算所有 candidate answer 的 conditional log-likelihood。

    Args:
        model: 因果 LM（有 forward 方法，返回 dict with "logits"）
        tokenizer: HF tokenizer
        prompts: 每个样例的 prompt 字符串
        answer_choices: prompts[i] 的候选 answer 列表
        label_ids: answer_choices[i][k] 对应的标签 ID
        batch_size: 批大小
        max_length: 最大序列长度
        device: torch device

    Returns:
        list of ScoreResult，每个样例一个
    """
    if device is None:
        device = next(model.parameters()).device
    if max_length < 2:
        raise ValueError("max_length must be at least 2")

    N = len(prompts)
    if len(answer_choices) != N or len(label_ids) != N:
        raise ValueError("prompts, answer_choices, and label_ids must have equal lengths")

    flat_pairs = []  # [(ex_idx, label_id, prompt_ids, answer_ids)]
    for i in range(N):
        if len(answer_choices[i]) != len(label_ids[i]):
            raise ValueError(f"answer/label count mismatch at example {i}")
        prompt_ids = tokenizer.encode(prompts[i], add_special_tokens=False)
        for k, answer in enumerate(answer_choices[i]):
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
            flat_pairs.append((i, label_ids[i][k], prompt_ids, answer_ids))

    scores = [{} for _ in range(N)]

    for batch_start in range(0, len(flat_pairs), batch_size):
        batch_pairs = flat_pairs[batch_start : batch_start + batch_size]

        batch_sequences = []
        batch_prompt_lens = []
        batch_answer_lens = []

        for _, _, prompt_ids, answer_ids in batch_pairs:
            if not prompt_ids or not answer_ids or len(answer_ids) >= max_length:
                batch_sequences.append([])
                batch_prompt_lens.append(0)
                batch_answer_lens.append(0)
                continue
            kept_prompt = prompt_ids[-(max_length - len(answer_ids)):]
            batch_sequences.append(kept_prompt + answer_ids)
            batch_prompt_lens.append(len(kept_prompt))
            batch_answer_lens.append(len(answer_ids))

        width = max(1, max(map(len, batch_sequences)))
        padded_ids = []
        masks = []
        for sequence in batch_sequences:
            pad_len = width - len(sequence)
            padded_ids.append([tokenizer.pad_token_id] * pad_len + sequence)
            masks.append([0] * pad_len + [1] * len(sequence))
        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=device)

        model_dtype = next(model.parameters()).dtype
        use_autocast = model_dtype in (torch.bfloat16, torch.float16)
        autocast_dtype = model_dtype if use_autocast else torch.bfloat16

        with torch.inference_mode():
            with torch.amp.autocast(device_type=str(device).split(":")[0],
                                    dtype=autocast_dtype, enabled=use_autocast):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits_output = (
                outputs.logits
                if hasattr(outputs, "logits")
                else outputs["logits"]
            )
            logits = logits_output.float()

        log_probs = F.log_softmax(logits, dim=-1)

        for j, (ex_idx, label_id, _, _) in enumerate(batch_pairs):
            total_len = attention_mask[j].sum().item()
            actual_answer_len = batch_answer_lens[j]
            actual_prompt_len = batch_prompt_lens[j]

            if actual_answer_len <= 0 or actual_prompt_len <= 0:
                scores[ex_idx][label_id] = float("-inf")
                continue

            pad_len = input_ids.shape[1] - total_len
            total_ll = 0.0

            for t in range(actual_answer_len):
                token_pos = pad_len + actual_prompt_len + t
                pred_pos = token_pos - 1

                if pred_pos < 0:
                    continue

                token_id = input_ids[j, token_pos].item()
                total_ll += log_probs[j, pred_pos, token_id].item()

            normalized_ll = total_ll / actual_answer_len
            scores[ex_idx][label_id] = normalized_ll

    results = []
    for i in range(N):
        if not scores[i] or not any(math.isfinite(score) for score in scores[i].values()):
            predicted = label_ids[i][0] if label_ids[i] else 0
            results.append(ScoreResult(
                label_scores={},
                predicted_label=predicted,
                metadata={"error": "no valid scores"},
            ))
        else:
            best_label = max(scores[i], key=scores[i].get)
            results.append(ScoreResult(
                label_scores=scores[i],
                predicted_label=best_label,
            ))

    return results
