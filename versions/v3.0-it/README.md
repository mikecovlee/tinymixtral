# TinyMixtral v3.0-it

**Instruction-tuned release.** One-epoch supervised fine-tune of the v3.0 base on
2.17M decontaminated English conversations (10 public sources), produced entirely on
a single GPU. Shipped as `mikecovlee/tinymixtral-it` on the HF Hub.

- HF repo: `mikecovlee/tinymixtral-it`
- Base model: `mikecovlee/tinymixtral` (v3.0, see `versions/v3.0/`)
- Training config: `versions/v3.0-it/configs/3m.json` (the shipped run)

## What's in this bundle

| Path | Purpose |
|------|---------|
| `REPRODUCE.md` | Full runbook: data build → training → evaluation chain |
| `REPORT.md` | Report: methodology, results, lessons learned |
| `configs/` | Training configs per scale tier (200k / 1m / 3m / polish) |
| `data/` | Dataset tooling: `prefetch_sources.py`, `build_dataset.py`, `sample_subset.py` |
| `train_sft.py` | SFT trainer (packing, label masking, `--seed/--resume/--grad-accum/--keep-last`) |
| `run_sft.sh` | Parameterized training entry (`200k|1m|3m|polish`) |
| `eval/` | Evaluation chain: publish → generate → score → lm-eval → summary |
| `judge/` | LLM rubric scoring + paired statistics |
| `eval_prompts/` | Held-out rubric prompt set (4,955 rows, sha256-pinned) |

## Data

`data/build_dataset.py` blends 10 public instruction sources (per-source licenses in
`docs/DATA_LICENSES.md`), with exact-hash + MinHash-LSH (Jaccard ≥ 0.8) dedup, global
shuffle, per-source share caps, and 10-gram decontamination against 6 eval sets plus
the held-out prompt set. No Chinese data. Scale ladder:

| Tier | Rows | Packed seqs | Steps | Wall time |
|------|------|-------------|-------|-----------|
| 200k | 195,170 | 106,718 | 4,447 | 99 min |
| 1m | 856,805 | 644,557 | 26,857 | 10.0 h |
| **3m (shipped)** | **2,168,835** | **1,443,804** | **60,159** | **22.3 h** |
| polish | 50,000 | 26,365 | 1,099 | 24.5 min |

Training: from the v3.0 base (no warm start), lr 2e-5 cosine + 100-step warmup, bf16 +
gradient checkpointing, seq 1024, batch 24 (the max stable at 21.5 GB on a 32 GB
RTX PRO 4500). The polish tier is a low-lr (5e-6) stratified subset run evaluated but
**not shipped** (no gain over the 3M run).

## Results

| Metric | base v3.0 | 3M SFT (v3.0-it) |
|--------|-----------|------------------|
| IFEval prompt-strict | — | **0.1701** |
| IFEval inst-strict | — | **0.2794** |
| GSM8K flexible | — | 0.0227 |
| 7-task harness (canonical) | 0.3979 | **0.4002** |
| LLM rubric (0-100, n=4,955) | 6.4 ± 0.18* | **15.0 ± 0.28** |
| MMLU (acc, 5-shot) | 0.234 | **0.243** |
| TruthfulQA MC2 (0-shot) | **0.417** | 0.412 |

*baseline SFT arm on the same held-out prompts; paired delta +8.61 ± 0.28 (t = +30.7).
Instruction following and answer quality rise sharply, and the 7-task harness edges up
(+0.23pp vs the base) — only ARC-Easy dips (0.478 → 0.448); the other six tasks are flat
or better. Canonical harness mean = acc_norm for hellaswag/piqa/arc_challenge/openbookqa,
acc for winogrande/arc_easy/lambada (BoolQ was dropped from the suite as an unstable
sentinel — see `REPORT.md` §8.5). Full tables and per-task numbers in `REPORT.md`.
MMLU (5-shot) and TruthfulQA (MC2, 0-shot) are supplementary metrics: all arms sit near
the 25% four-choice chance line on MMLU and around 0.41 on TruthfulQA MC2 — knowledge is
capacity/data-budget-limited at this scale, and SFT does not move either metric
(`REPORT.md` §5.4).

### Comparison with similar models

Same 7-task suite, measured locally (lm-evaluation-harness v0.4.12, 0-shot, cuda, bf16):

| Task | Metric | v3.0-it (477M) | SmolLM2-360M | Qwen3-0.6B |
|------|--------|:---:|:---:|:---:|
| HellaSwag | acc_norm | 0.340 | 0.563 | 0.473 |
| PIQA | acc | 0.634 | 0.719 | 0.673 |
| WinoGrande | acc | 0.528 | 0.587 | 0.563 |
| ARC-Easy | acc | 0.448 | 0.705 | 0.609 |
| ARC-Challenge | acc_norm | 0.260 | 0.383 | 0.340 |
| OpenBookQA | acc_norm | 0.302 | 0.372 | 0.316 |
| LAMBADA | acc | 0.289 | 0.532 | 0.401 |
| **Mean** | — | 0.4002 | 0.5516 | 0.4821 |
| MMLU | acc (5-shot) | 0.243 | 0.358* | 0.528† |
| TruthfulQA | MC2 (0-shot) | 0.412 | — | — |

SmolLM2-360M (4T tokens) and Qwen3-0.6B (36T tokens) are far above v3.0-it's 8.05B-token
budget (~500× / ~4500×) — the gap is primarily a data-budget difference. External MMLU
cells come from published reports: `*` SmolLM2-360M, MMLU (cloze, 0-shot, lighteval) —
[official model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M); `†` Qwen3-0.6B
(base), MMLU (5-shot) — [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388).
Neither model publishes an official TruthfulQA MC2 score, so those cells stay blank.

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("mikecovlee/tinymixtral-it",
                                        trust_remote_code=True).to("cuda").to_bfloat16()
t = AutoTokenizer.from_pretrained("mikecovlee/tinymixtral-it")
```

To retrain from scratch see `REPRODUCE.md`; to reproduce the evaluation see
§3 of `REPRODUCE.md` (`eval/response_eval.py` + `judge/rubric_judge.py` + lm-eval).
