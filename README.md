> This branch contains the independent native **tinymixtral-v3.0-downdp-adaptive-codeupdate** model.
> See [CPT_MODEL.md](CPT_MODEL.md) for its configuration, training entry point and validation.
> The upstream documentation below describes the original models and their results.

# TinyMixtral

A Mixtral-style Mixture-of-Experts causal language model for pretraining research on a single
consumer GPU. The current flagship is **[v3.0](versions/v3.0/README.md)** — a ~477.5M total /
~276.1M active MoE (top-2 of 4 experts) trained on 8.05B tokens, matching the previous 1B MoE
with ~2.5× fewer parameters. Its instruction-tuned release is
**[v3.0-it](versions/v3.0-it/README.md)** — [tinymixtral-it](https://huggingface.co/mikecovlee/tinymixtral-it).

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-mikecovlee%2Ftinymixtral-blue)](https://huggingface.co/mikecovlee/tinymixtral)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-mikecovlee%2Ftinymixtral--it-green)](https://huggingface.co/mikecovlee/tinymixtral-it)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-mikecovlee%2Ftinymixtral--v1.1--1b-yellow)](https://huggingface.co/mikecovlee/tinymixtral-v1.1-1b)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-mikecovlee%2Ftinymixtral--v1.1--0.5b-orange)](https://huggingface.co/mikecovlee/tinymixtral-v1.1-0.5b)

## Model Versions

| Version | Params (total / active) | Experts | Data | Harness mean | Card |
|---------|-------------------------|---------|------|:---:|---|
| **v3.0-it** | 477.5M / 276.1M | 4 routed, top-2 | v3.0 + 2.17M SFT | **0.4002** | [versions/v3.0-it](versions/v3.0-it/README.md) |
| **v3.0** | 477.5M / 276.1M | 4 routed, top-2 | 8.05B (6-source blend) | 0.3979 | [versions/v3.0](versions/v3.0/README.md) |
| v2.0 beta | 498M / 241M | 1 shared + 6 routed | 4B (SmolLM blend) | 0.389 | [versions/v2.0-beta](versions/v2.0-beta/README.md) |
| v1.1-1b | 1182M / 352M | 8 routed, top-2 | 4B → 8B (SmolLM blend) | 0.397 | [versions/v1.1-1b](versions/v1.1-1b/README.md) |
| v1.1 | 432M / 176M | 6 routed, top-2 | 4B (SmolLM blend) | 0.381 | [versions/v1.1](versions/v1.1/README.md) |
| v1.0 | 432M / 176M | 6 routed, top-2 | 4B (C4-en, legacy) | 0.378 | [versions/v1.0](versions/v1.0/README.md) |

Harness mean = lm-evaluation-harness v0.4.12, 0-shot, 7-task suite (BoolQ excluded — it swung
abnormally across our own runs while the other seven tasks stayed stable; see
[the report](versions/v3.0-it/REPORT.md)). See each card for full
architecture tables, training recipes, per-task results and negative results.
v3.0-it is instruction-tuned: IFEval 0.1701/0.2794 and LLM rubric 15.0±0.3 rise sharply;
on the 7-task suite it also edges the base (+0.23pp), with only ARC-Easy down (0.478 → 0.448) —
see the [Instruction Tuning](#instruction-tuning) section.

Dataset sources: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`),
[Cosmopedia v2](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-v2),
[DCLM](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0),
[OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) (code),
[OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math),
[Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (`20231101.en`); v1.0 used
[C4](https://huggingface.co/datasets/allenai/c4) (`en`).
Per-source licenses and caveats: [docs/DATA_LICENSES.md](docs/DATA_LICENSES.md).

## Benchmark Comparison

0-shot, 7-task suite (`acc_norm` for HellaSwag / ARC-Challenge / OpenBookQA, `acc` otherwise).
Best per row in **bold**; `v1.1-1b` is the 1B MoE at 4B and 8B tokens.

| Task | Metric | v3.0-it | v3.0 | v2.0 beta | v1.1-1b (8B) | v1.1-1b (4B) | v1.1 | v1.0¹ |
|------|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| HellaSwag | acc_norm | **0.340** | 0.335 | 0.326 | 0.329 | 0.311 | 0.308 | 0.310 |
| PIQA | acc | 0.634 | **0.638** | 0.631 | 0.630 | 0.620 | 0.616 | 0.613 |
| WinoGrande | acc | **0.528** | 0.515 | 0.506 | 0.523 | 0.510 | 0.524 | 0.508 |
| ARC-Easy | acc | 0.448 | 0.478 | 0.474 | **0.479** | 0.463 | 0.456 | 0.422 |
| ARC-Challenge | acc_norm | 0.260 | 0.255 | 0.272 | **0.279** | 0.273 | 0.247 | 0.247 |
| OpenBookQA | acc_norm | 0.302 | 0.296 | 0.290 | 0.306 | 0.288 | 0.288 | **0.308** |
| LAMBADA | acc | **0.289** | 0.268 | 0.224 | 0.234 | 0.200 | 0.227 | 0.240 |
| **Mean** | — | **0.4002** | 0.3979 | 0.3890 | 0.3971 | 0.3807 | 0.3809 | 0.3783 |
| MMLU | acc (5-shot) | **0.243** | 0.234 | — | — | — | — | — |
| TruthfulQA | MC2 (0-shot) | 0.412 | **0.417** | — | — | — | — | — |

¹ v1.0 is the legacy C4 baseline (earlier evaluation setup); shown for reference.
BoolQ was dropped from the suite: it swung abnormally across our own runs (−15pp on the v2.0
architecture change, −19pp after 3M-row SFT) while the other seven tasks stayed stable, so it
is a poor sentinel. v3.0 reaches the 1B MoE's accuracy with **~2.5× fewer total** and
**~1.28× fewer active** parameters.

MMLU (5-shot) and TruthfulQA (MC2, 0-shot) are supplementary metrics, not part of the 7-task
mean; the historical columns were not re-run. All arms sit near the 25% four-choice chance line
on MMLU — knowledge is capacity/data-budget-limited at this scale, and SFT does not move
MMLU/TruthfulQA measurably.

### Comparison with similar models

Same suite and settings, measured locally (lm-evaluation-harness v0.4.12, 0-shot, cuda, bf16):

| Task | Metric | v3.0-it (477M) | v3.0 (477M) | SmolLM2-360M | Qwen3-0.6B |
|------|--------|:---:|:---:|:---:|:---:|
| HellaSwag | acc_norm | 0.340 | 0.335 | 0.563 | 0.473 |
| PIQA | acc | 0.634 | 0.638 | 0.719 | 0.673 |
| WinoGrande | acc | 0.528 | 0.515 | 0.587 | 0.563 |
| ARC-Easy | acc | 0.448 | 0.478 | 0.705 | 0.609 |
| ARC-Challenge | acc_norm | 0.260 | 0.255 | 0.383 | 0.340 |
| OpenBookQA | acc_norm | 0.302 | 0.296 | 0.372 | 0.316 |
| LAMBADA | acc | 0.289 | 0.268 | 0.532 | 0.401 |
| **Mean** | — | **0.4002** | 0.3979 | 0.5516 | 0.4821 |
| MMLU | acc (5-shot) | 0.243 | 0.234 | 0.358* | 0.528† |
| TruthfulQA | MC2 (0-shot) | 0.412 | 0.417 | — | — |

SmolLM2-360M was trained on 4T tokens and Qwen3-0.6B on 36T tokens, versus 8.05B tokens
(~500× and ~4500× less) for v3.0 on a single consumer GPU — the gap is primarily a data-budget
difference. The instruction-tuned v3.0-it improves over the base on this suite (0.4002 vs
0.3979 mean) and matches or beats it on six of seven tasks. External MMLU cells come from
published reports: `*` SmolLM2-360M, MMLU (cloze, 0-shot, lighteval) — [official model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M); `†` Qwen3-0.6B (base), MMLU (5-shot) — [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388). Neither model publishes an official TruthfulQA MC2 score, so those cells stay blank; our metrics were not re-run on the external models.

## Quick Start

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "mikecovlee/tinymixtral", trust_remote_code=True
)
```

Environment:

```bash
conda create -n tinymixtral python=3.13 -y
conda activate tinymixtral
pip install -r requirements.txt
```

Training: `scripts/train.py` (single run) / `scripts/resume.py` (resume + continuation) /
`versions/v3.0/scripts/run_segment.ps1` (the v3.0 4-segment launcher). Data is tokenized once to `.pt` shards
(`scripts/prepare_data.py`), optionally combined (`scripts/mix_data.py`,
`scripts/make_blend_shards.py`). Per-version recipes are in the version cards above; see
[`REPRODUCE.md`](REPRODUCE.md) for the end-to-end walkthrough.

## Evaluation

All reported numbers use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
(v0.4.x) with conditional log-likelihood scoring. Publish a checkpoint first
(`scripts/publish_hf.py`), then:

```bash
# 7-task 0-shot suite
lm_eval --model hf \
  --model_args "pretrained=publish/<run>,tokenizer=tokenizer/,trust_remote_code=True,dtype=bfloat16" \
  --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa,lambada_openai \
  --batch_size 16 --device cuda --output_path evals/harness_0shot

# IFEval (instruction following, generative)
lm_eval --model hf \
  --model_args "pretrained=publish/<run>,tokenizer=tokenizer/,trust_remote_code=True,dtype=bfloat16" \
  --tasks ifeval --apply_chat_template --batch_size 8 --device cuda \
  --output_path evals/ifeval_publish
```

Add `--num_fewshot N` for few-shot variants.

## Publishing & Chat

```bash
# Export a checkpoint to a self-contained HF directory (config auto_map + modeling code + tokenizer)
python scripts/publish_hf.py --checkpoint checkpoints/<run>/<step>_final --output publish/<run> --tokenizer tokenizer/

# Interactive chat
python scripts/chat_hf.py mikecovlee/tinymixtral   # from HF Hub
python scripts/chat.py --checkpoint checkpoints/<run>/<step>_final --tokenizer tokenizer/   # native loader
```

## Instruction Tuning

One epoch of SFT from the v3.0 base on 2.17M decontaminated English conversations
(10 public sources, `versions/v3.0-it/data/build_dataset.py`, lr 2e-5 cosine, seq 1024) produces
**TinyMixtral v3.0 Instruction-tuned** (`v3.0-it`, released as [tinymixtral-it](https://huggingface.co/mikecovlee/tinymixtral-it)):

| Metric | base v3.0 | v3.0-it |
|---|---|---|
| LLM rubric (4,955 held-out, 0-100) | — | 15.0 ± 0.3 (paired +8.61, t=+30.7 vs the prior 50k-row SFT) |
| IFEval prompt-strict / inst-strict | — | 0.1701 / 0.2794 |
| GSM8K flexible | 0.0167* | 0.0227 |
| 7-task harness (canonical) | 0.3979 | 0.4002 |

\* prior 50k-row SFT baseline (the paired reference); the pretrained base was not evaluated on GSM8K/rubric.

Instruction following rose sharply while the 7-task harness edged up (+0.23pp); only ARC-Easy
dipped (0.478 → 0.448) and the other six tasks are flat or up.
Full report, methodology and lessons: `versions/v3.0-it/REPORT.md`;
reproduction: `versions/v3.0-it/REPRODUCE.md`. (*prior-arm reference)

## Project Structure

```
tinymixtral/
├── REPRODUCE.md    # end-to-end reproduction guide (all versions)
├── model/          # training model code (config.py, modeling.py)
├── hf/             # HuggingFace compatibility layer (PreTrainedModel / PretrainedConfig)
├── shared_expert/  # v2.0 shared-expert ablation (model + scripts)
├── versions/       # per-version bundles: card + config + version-specific scripts
│   ├── v3.0/       # README.md, configs/, scripts/ (run_segment.ps1, run_pilot.ps1, analyze_pilot.py)
│   ├── v3.0-it/    # instruction-tuned release: REPORT/REPRODUCE, train_sft.py, run_sft.sh, eval/, judge/, data/, configs/, eval_prompts/
│   ├── v1.1-1b/    # README.md, configs/v1b_moe.json
│   ├── v2.0-beta/  # README.md
│   ├── v1.1/       # README.md
│   └── v1.0/       # README.md
├── scripts/        # shared tooling: train/resume/prepare_data(+_local)/download_{parquets,jsonl_zst}/mix_data/make_blend_shards/publish_hf/chat/val_ppl/…
├── configs/        # hardware profile + shared config dirs
├── evals/          # legacy eval outputs (run-local lm-eval results)
├── docs/           # DATA_LICENSES.md (dataset provenance)
├── tests/
├── requirements.txt
└── LICENSE
```

## Key Design Decisions

1. **Simple training code, HF conversion on export** — training uses plain `nn.Module` + a
   dataclass config; `hf/` + `publish_hf.py` convert to `PreTrainedModel`/`PretrainedConfig`.
2. **Pre-tokenized shards** — tokenize once to disk; shards are cycled/combined at the file
   level, no re-tokenization (`mix_data.py`, `make_blend_shards.py`).
3. **bf16 model, flexible optimizer** — bf16 weights + autocast; fp32 optimizer states by
   default, `--bf16-optim` stores them in bf16.
4. **Auxiliary loss (Mixtral-style)** — `L_aux = N × Σ(f_i × P_i)`, routing fractions detached.
5. **GQA with SDPA** — 16 query heads share 4 key/value heads (4:1). Training path uses
   `enable_gqa=True` with `is_causal=True`; with a padding mask (eval/inference), KV heads are
   expanded and a 4D boolean mask is built.
6. **Atomic checkpoint saves** — write to a temp dir then rename.
7. **Signal handling** — SIGINT/SIGTERM triggers a clean save after the current step.

## Hardware

NVIDIA RTX PRO 4500 Blackwell 32GB (development) and RTX A5000 24GB (also supported; lower the
batch size — the 1B MoE uses ~12–16GB at `--batch-size 16`).

## License

MIT License. Copyright (C) 2026 Michael Lee (李登淳).
