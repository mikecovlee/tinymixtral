# TinyMixtral

A Mixtral-style Mixture-of-Experts causal language model for pretraining research on a single
consumer GPU. The current flagship is **[v3.0](versions/v3.0/README.md)** — a ~477.5M total /
~276.1M active MoE (top-2 of 4 experts) trained on 8.05B tokens, matching the previous 1B MoE
with ~2.5× fewer parameters.

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-mikecovlee%2Ftinymixtral-blue)](https://huggingface.co/mikecovlee/tinymixtral)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-mikecovlee%2Ftinymixtral--v1.1--1b-yellow)](https://huggingface.co/mikecovlee/tinymixtral-v1.1-1b)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-mikecovlee%2Ftinymixtral--v1.1--0.5b-orange)](https://huggingface.co/mikecovlee/tinymixtral-v1.1-0.5b)

## Model Versions

| Version | Params (total / active) | Experts | Data | Harness mean | Card |
|---------|-------------------------|---------|------|:---:|---|
| **v3.0** | 477.5M / 276.1M | 4 routed, top-2 | 8.05B (6-source blend) | **0.4250** | [versions/v3.0](versions/v3.0/README.md) |
| v1.1-1b | 1182M / 352M | 8 routed, top-2 | 4B → 8B (SmolLM blend) | 0.425 | [versions/v1.1-1b](versions/v1.1-1b/README.md) |
| v2.0 beta | 498M / 241M | 1 shared + 6 routed | 4B (SmolLM blend) | 0.397 | [versions/v2.0-beta](versions/v2.0-beta/README.md) |
| v1.1 | 432M / 176M | 6 routed, top-2 | 4B (SmolLM blend) | 0.409 | [versions/v1.1](versions/v1.1/README.md) |
| v1.0 | 432M / 176M | 6 routed, top-2 | 4B (C4-en, legacy) | 0.403 | [versions/v1.0](versions/v1.0/README.md) |

Harness mean = lm-evaluation-harness v0.4.12, 0-shot, 8-task suite. See each card for full
architecture tables, training recipes, per-task results and negative results.

## Benchmark Comparison

0-shot, 8-task suite (`acc_norm` for HellaSwag / ARC-Challenge / OpenBookQA, `acc` otherwise).
Best per row in **bold**; `v1.1-1b` is the 1B MoE at 4B and 8B tokens.

| Task | Metric | v3.0 | v1.1-1b (8B) | v1.1-1b (4B) | v1.1 | v2.0 beta | v1.0¹ |
|------|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| HellaSwag | acc_norm | **0.335** | 0.329 | 0.311 | 0.308 | 0.326 | 0.310 |
| PIQA | acc | **0.638** | 0.630 | 0.620 | 0.616 | 0.631 | 0.613 |
| WinoGrande | acc | 0.515 | 0.523 | 0.510 | **0.524** | 0.506 | 0.508 |
| ARC-Easy | acc | 0.478 | **0.479** | 0.463 | 0.456 | 0.474 | 0.422 |
| ARC-Challenge | acc_norm | 0.255 | **0.279** | 0.273 | 0.247 | 0.272 | 0.247 |
| OpenBookQA | acc_norm | 0.296 | 0.306 | 0.288 | 0.288 | 0.290 | **0.308** |
| BoolQ | acc | 0.615 | **0.620** | 0.548 | 0.606 | 0.455 | 0.579 |
| LAMBADA | acc | **0.268** | 0.234 | 0.200 | 0.227 | 0.224 | 0.240 |
| **Mean** | — | **0.4250** | 0.425 | 0.402 | 0.409 | 0.397 | 0.403 |

¹ v1.0 is the legacy C4 baseline (earlier evaluation setup); shown for reference.
v3.0 reaches the 1B MoE's accuracy with **~2.5× fewer total** and **~1.28× fewer active** parameters.

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
# 8-task 0-shot suite
lm_eval --model hf \
  --model_args "pretrained=publish/<run>,tokenizer=tokenizer/,trust_remote_code=True,dtype=bfloat16" \
  --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq,lambada_openai \
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

## Project Structure

```
tinymixtral/
├── REPRODUCE.md    # end-to-end reproduction guide (all versions)
├── model/          # training model code (config.py, modeling.py)
├── hf/             # HuggingFace compatibility layer (PreTrainedModel / PretrainedConfig)
├── shared_expert/  # v2.0 shared-expert ablation (model + scripts)
├── versions/       # per-version bundles: card + config + version-specific scripts
│   ├── v3.0/       # README.md, configs/, scripts/ (run_segment.ps1, run_pilot.ps1, analyze_pilot.py)
│   ├── v1.1-1b/    # README.md, configs/v1b_moe.json
│   ├── v2.0-beta/  # README.md
│   ├── v1.1/       # README.md
│   └── v1.0/       # README.md
├── scripts/        # shared tooling: train/resume/prepare_data(+_local)/download_{parquets,jsonl_zst}/mix_data/make_blend_shards/publish_hf/chat/val_ppl/…
├── configs/        # hardware profile + shared config dirs
├── evals/          # evaluation outputs
├── docs/
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
