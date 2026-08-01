# TinyMixtral

A small Mixtral-style Mixture-of-Experts causal language model (~432M total, ~176M active parameters) for pretraining research on a single consumer GPU.

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-mikecovlee%2Ftinymixtral-yellow)](https://huggingface.co/mikecovlee/tinymixtral)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-mikecovlee%2Ftinymixtral--v2.0--beta-orange)](https://huggingface.co/mikecovlee/tinymixtral-v2.0-beta)

## Pretrained Model

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "mikecovlee/tinymixtral", trust_remote_code=True
)
```

The latest model (SmolLM blend pretrain + Wiki/Cosmopedia post-train) is available at [mikecovlee/tinymixtral](https://huggingface.co/mikecovlee/tinymixtral).

The v2.0 beta (shared expert architecture, WSD schedule) is available at [mikecovlee/tinymixtral-v2.0-beta](https://huggingface.co/mikecovlee/tinymixtral-v2.0-beta).

The legacy v1 model (C4 pretrain) has been moved to [mikecovlee/tinymixtral-v1.0](https://huggingface.co/mikecovlee/tinymixtral-v1.0).

## Model Architecture

| Parameter | Value |
|-----------|-------|
| hidden_size | 896 |
| num_layers | 10 |
| Attention | Grouped Query Attention (14 heads / 2 KV heads) |
| Head dim | 64 |
| RoPE theta | 1,000,000 |
| Norm | RMSNorm |
| Experts | 6 (top-2 routing) |
| Expert FFN | SwiGLU, intermediate = 2389 (8/3 × hidden_size) |
| Vocab size | 32,000 |
| Max position | 2,048 |
| **Total params** | **~432M** |
| **Active params** | **~176M** |

## Hardware & Environment

- GPU: NVIDIA RTX A5000 24GB
- CPU: AMD Ryzen 7 5800X
- RAM: 32GB

```bash
conda create -n tinymixtral python=3.13 -y
conda activate tinymixtral
pip install -r requirements.txt
```

## Quick Start

The recommended recipe uses a SmolLM-inspired data blend (FineWeb-Edu + Cosmopedia v2, 89:11) with LR=7e-4:

```bash
# 1. Download tokenizer
python scripts/prepare_tokenizer.py --from-hf TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output tokenizer/

# 2. Tokenize FineWeb-Edu (3.56B tokens)
python scripts/prepare_data.py --dataset HuggingFaceFW/fineweb-edu --subset sample-10BT \
  --tokenizer tokenizer/ --output data/pretrain/fineweb --max-tokens 3560000000 --force

# 3. Tokenize Cosmopedia v2 (440M tokens)
python scripts/prepare_data.py --dataset HuggingFaceTB/cosmopedia-v2 --subset cosmopedia-v2 \
  --tokenizer tokenizer/ --output data/pretrain/cosmopedia --max-tokens 440000000 --force

# 4. Mix shards (89:11)
python scripts/mix_data.py data/pretrain/fineweb data/pretrain/cosmopedia \
  --output data/pretrain/smollm_blend --weights 36 4

# 5. Pretrain (4B tokens, LR=7e-4, batch=24)
python scripts/train.py --cache-dir data/pretrain/smollm_blend \
  --output-dir checkpoints/smollm_blend \
  --batch-size 24 --max-tokens 4000000000 --lr 7e-4 \
  --keep-last-checkpoints 5 2>&1 | tee train.log
```

## Training Details

> [!WARNING]
> CPT training is transactional. Stock Hugging Face `Trainer`, a bare
> `loss.backward()` / `optimizer.step()` loop, and stock gradient accumulation
> are not supported: they can bypass CPT proposal validation, all-or-none
> commit/abort, and exact-rollback/fail-stop handling. Run every optimizer
> update through `scripts.train_utils.execute_training_iteration` (preferred),
> or through `execute_training_step` only when the caller explicitly manages
> the forward output and pre-forward snapshot. The repository `train.py` and
> `resume.py` entry points already use the strict transaction path.

- **Precision**: bf16 for ordinary model parameters and autocast compute; CPT projection/anchor/energy parameters, congestion prices, and recurrent routing state remain FP32. Optimizer moments preserve this parameter-dtype boundary.
- **CPT projection preprocessing (strict TeX v1)**: for each token column vector, the Router computes `P x` and directly performs per-token FP32 L2 stable normalization. There is no softmax between `P x` and `z`. The later prototype and expert-kernel softmax operations remain unchanged; no softmax is applied after the final `Pi` probabilities.
- **Optimizer**: AdamW (β=0.9,0.95, wd=0.1) by default, with weight decay only on ≥2D parameters. `--bf16-optim` selects BF16AdamW: update arithmetic is performed in FP32, moments for ordinary bf16 parameters are stored in bf16, and moments for FP32 CPT parameters remain FP32.
- **LR schedule**: cosine decay with linear warmup by default; `--schedule wsd` selects Warmup-Stable-Decay with a decay ratio of 0.1. Schema-v5 checkpoints persist both the schedule kind and its WSD decay ratio for strict reconstruction.
- **Gradient clipping**: 1.0
- **Batch**: 24 × 1024 = 24,576 tokens/step
- **Activation checkpointing**: enabled (required for 24GB VRAM)
- **Data**: FineWeb-Edu + Cosmopedia v2 (89:11), pre-tokenized to `.pt` shards (100M tokens each), cycled round-robin
- **LR sweep**: 4 × 100M-token runs at {1e-4, 3e-4, 5e-4, 7e-4}; 7e-4 selected based on lowest final loss

`prepare_data.py` explicitly appends EOS to each document and validates tokenizer vocab size (32K). Shards are written atomically via staging → replace. `mix_data.py` interleaves shards from multiple tokenized datasets at the file level — no re-tokenization needed.

## Checkpoints

Periodic checkpoints saved every `--save-every-min` minutes, plus a final checkpoint at completion:

```
checkpoints/smollm_blend/
├── step_0144718/          # periodic
│   ├── config.json
│   ├── pytorch_model.bin
│   └── training_state.pt  # strict schema-v5 training state
├── ...
└── step_0162761_final/    # final checkpoint
```

`training_state.pt` uses strict schema v5. In addition to optimizer/scheduler states, step, token count, and shard position (`fi`/`ptr`), it records `optimizer_kind` (`adamw` or `bf16_adamw`), `schedule_kind` (`cosine` or `wsd`), `schedule_decay_ratio`, CPT state version, RNG state, run identity, and code/data/tokenizer plus model/config hashes. Resume validates this identity before reconstructing the exact optimizer and scheduler.

## Resume Training

```bash
python scripts/resume.py \
  --checkpoint-dir checkpoints/smollm_blend \
  --cache-dir data/pretrain/smollm_blend \
  --tokenizer-dir tokenizer/ \
  --batch-size 24 --seq-len 1024
```

The script locates the latest valid checkpoint, verifies its schema-v5 hashes and manifests, restores model/optimizer/scheduler/RNG state, and continues from the exact data position. Optimizer kind, schedule kind, WSD ratio, and training target are inherited from the checkpoint. On resume, `--schedule` and `--bf16-optim`/`--no-bf16-optim` are optional identity assertions, not recipe overrides; an explicit mismatch is rejected. Batch size and sequence length must match the checkpoint.

## Post-Training

The strict CPT v1 `resume.py` entry point is continuation-only. It rejects `--max-tokens`, `--lr`, `--wd`, and `--warmup-steps` overrides because resetting the target, optimizer recipe, or schedule while retaining committed CPT state would break the schema-v5 training identity. A domain-adaptive phase therefore requires a separately designed new-run/state-transition protocol and is not represented as a `resume.py` command in this branch.

## Evaluation

```bash
# GLUE (8 tasks)
python scripts/eval_glue.py --checkpoint checkpoints/smollm_blend/step_0162761_final \
  --tokenizer tokenizer/ --tasks all --limit 500 --batch-size 16

# ARC (0-shot + 5-shot)
python scripts/eval_arc.py --checkpoint checkpoints/smollm_blend/step_0162761_final \
  --tokenizer tokenizer/ --tasks arc_c,arc_e --shots 0
python scripts/eval_arc.py --checkpoint checkpoints/smollm_blend/step_0162761_final \
  --tokenizer tokenizer/ --tasks arc_c,arc_e --shots 5
```

Zero-shot evaluation uses conditional log-likelihood scoring over answer spans. Supported GLUE tasks: `sst2`, `mrpc`, `qqp`, `qnli`, `rte`, `cola`, `mnli`, `mnli_mismatched`.

## Publishing to HuggingFace

A pretrained model is already available at [mikecovlee/tinymixtral](https://huggingface.co/mikecovlee/tinymixtral). To export your own trained checkpoint:

```bash
python scripts/publish_hf.py \
  --checkpoint checkpoints/smollm_blend/step_0162761_final \
  --output publish/ --tokenizer tokenizer/
```

This creates a self-contained `publish/` directory with:

- `pytorch_model.bin` — model weights
- `config.json` — HF-compatible config with `auto_map` for `trust_remote_code`
- `configuration_tinymixtral.py` — `TinyMixtralConfig` (PretrainedConfig subclass)
- `modeling_tinymixtral.py` — Full model code (PreTrainedModel subclass)
- Tokenizer files
- LICENSE

Load local or push to Hub:

```python
from transformers import AutoModelForCausalLM

# Local directory
model = AutoModelForCausalLM.from_pretrained("publish/", trust_remote_code=True)

# Push to your own HF Hub repo
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("your-username/tinymixtral", exist_ok=True)
api.upload_folder(repo_id="your-username/tinymixtral", folder_path="publish/")
```

## Interactive Chat

```bash
# From HuggingFace Hub
python scripts/chat_hf.py mikecovlee/tinymixtral

# From local checkpoint (native model loader)
python scripts/chat.py --checkpoint checkpoints/smollm_blend/step_0162761_final --tokenizer tokenizer/

# From local publish directory
python scripts/chat_hf.py publish/
```

Both support interactive conversation with temperature and top-p sampling. Type `quit` to exit.

## Project Structure

```
tinymixtral/
├── model/                          # Training model code
│   ├── config.py                   # TinyMixtralConfig (plain dataclass)
│   └── modeling.py                 # MoE + GQA + RoPE + RMSNorm (nn.Module)
├── hf/                             # HuggingFace compatibility layer
│   ├── configuration_tinymixtral.py  # PretrainedConfig subclass
│   └── modeling_tinymixtral.py     # PreTrainedModel subclass
├── evals/                          # Evaluation framework
│   ├── prompt_scoring.py           # Conditional log-likelihood scoring
│   ├── glue_tasks.py               # 8 GLUE task definitions + templates
│   └── metrics.py                  # Accuracy, F1, Matthews correlation
├── scripts/
│   ├── train.py                    # Training entry point
│   ├── resume.py                   # Checkpoint resume
│   ├── train_utils.py              # Shared training logic
│   ├── eval_glue.py                # GLUE evaluation CLI
│   ├── eval_arc.py                 # ARC evaluation CLI (zero/few-shot)
│   ├── chat.py                     # Interactive chat (native model)
│   ├── chat_hf.py                  # Interactive chat (HF AutoModel)
│   ├── publish_hf.py               # HF-format export
│   ├── prepare_data.py             # Dataset pre-tokenization
│   ├── prepare_tokenizer.py        # Tokenizer download / training
│   ├── mix_data.py                 # Interleave shards from multiple datasets
│   ├── benchmark.py                # GPU memory/throughput profiler
│   └── search.py                   # Hyperparameter search
├── shared_expert/                  # Architecture ablation: shared expert variant
│   ├── model/                      # v2 model code (1 shared + 6 routed experts)
│   ├── hf/                         # v2 HF compatibility layer
│   └── scripts/                    # v2 training scripts (WSD schedule, BF16AdamW)
├── configs/                        # Config templates
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT
└── README.md
```

## Key Design Decisions

1. **Simple training code, HF conversion on export** — Training uses plain `nn.Module` and a dataclass config for simplicity. A separate `hf/` compatibility layer + `publish_hf.py` script converts to `PreTrainedModel`/`PretrainedConfig` for HuggingFace Hub.

2. **Pre-tokenized shards** — Data is tokenized once to disk, eliminating CPU bottleneck during training. Shards cycle round-robin; each shard is loaded on demand.

3. **Mixed precision with FP32 CPT state** — Ordinary model parameters and autocast compute use bf16. CPT projection/anchor/energy parameters, congestion prices, and recurrent routing state remain FP32. Both standard AdamW and `--bf16-optim` preserve FP32 optimizer moments for FP32 CPT parameters; BF16AdamW stores moments for ordinary bf16 parameters in bf16 while performing update arithmetic in FP32.

   Strict TeX CPT Router v1 preprocesses every token as `x -> P x -> per-token L2 stable normalization`. In token-major code, normalization is applied independently along the final projection dimension. No projection softmax is inserted between `P x` and `z`.

4. **Strict optimizer/schedule identity** — Schema-v5 checkpoints bind `adamw|bf16_adamw`, `cosine|wsd`, and the WSD decay ratio to the saved run. Resume reconstructs these values from the checkpoint and treats CLI optimizer/schedule flags only as optional consistency assertions.

5. **CPT congestion-price control** — Expert soft loads are detached from the task graph, accumulated from the complete CPT probability matrix, and used only to prepare the next congestion-price candidate after an accepted optimizer step. No expert load-balancing auxiliary loss is added to the language-model objective.

6. **GQA with SDPA** — 14 query heads share 2 key/value heads (7:1 ratio). Training path uses `enable_gqa=True` with `is_causal=True` (no KV expansion). When a padding mask is present (eval/inference), KV heads are explicitly expanded and a 4D boolean mask is constructed.

7. **Atomic checkpoint saves** — Checkpoints are written to a temporary directory and atomically renamed, preventing corruption from interrupted saves.

8. **Signal handling** — SIGINT/SIGTERM triggers a clean save after the current step completes.

## Results

### Data Quality Ablation

The original model trained on C4-en (noisy web text). We ran an ablation replacing C4 with a SmolLM-inspired blend: **FineWeb-Edu (89%) + Cosmopedia v2 (11%)**, 4B tokens total. Batch size increased to 24. LR swept on 100M-token runs; 7e-4 was optimal.

### Training Summary

| Phase | Data | LR | Tokens | Steps | Time | End Loss |
|-------|------|----|:------:|:-----:|:----:|:--------:|
| Pretrain (C4) | C4-en | 3e-4 | 4B | 177,557 | 77.1 h | 3.0 |
| Pretrain (SmolLM) | FineWeb-Edu + Cosmopedia v2 (89:11) | 7e-4 | 4B | 162,761 | 83.6 h | 2.5 |
| Post-train | Wiki + Cosmopedia v2 (50:50) | 2e-5 | 1B | 40,691 | 20.5 h | 2.5 |

### Standard Benchmarks (lm-evaluation-harness, 0-shot)

| Task | Metric | v1.1 (432M) | v2.0 beta (498M) | SmolLM2-360M | Qwen3-0.6B |
|------|--------|:---:|:---:|:---:|:---:|
| HellaSwag | acc_norm | 0.308 | 0.326 | **0.563** | 0.473 |
| PIQA | acc | 0.616 | 0.631 | **0.719** | 0.673 |
| WinoGrande | acc | 0.524 | 0.506 | **0.587** | 0.563 |
| ARC-Easy | acc | 0.456 | 0.474 | **0.705** | 0.609 |
| ARC-Challenge | acc_norm | 0.247 | 0.272 | **0.383** | 0.340 |
| OpenBookQA | acc_norm | 0.288 | 0.290 | **0.372** | 0.316 |
| BoolQ | acc | 0.606 | 0.455 | 0.620 | **0.643** |
| LAMBADA | acc | 0.227 | 0.224 | **0.532** | 0.401 |

All numbers measured locally with identical settings (lm-eval-harness v0.4.12, 0-shot, cuda, bf16). Batch size does not affect log-likelihood evaluation results.

SmolLM2-360M trained on 4T tokens; Qwen3-0.6B trained on 36T tokens. TinyMixtral trained on 4B tokens (~1000× less) on a single consumer GPU. The performance gap is primarily a data budget difference. v2.0 beta improves over v1.1 on most tasks (HellaSwag +1.8pp, PIQA +1.5pp, ARC-C +2.5pp) but regresses on BoolQ.

### GLUE (zero-shot)

| Task | Metric | C4 4B | SmolLM 4B | + Post-train (5B) |
|------|--------|:---:|:---:|:---:|
| SST2 | accuracy | 0.470 | 0.556 | **0.568** |
| MRPC | accuracy / f1 | 0.338 / 0.069 | 0.686 / 0.813 | 0.686 / 0.813 |
| QQP | accuracy / f1 | 0.470 / 0.412 | 0.350 / 0.519 | 0.350 / 0.519 |
| QNLI | accuracy | 0.494 | 0.460 | 0.458 |
| RTE | accuracy | 0.520 | 0.527 | **0.534** |
| MNLI | accuracy | 0.348 | 0.350 | **0.352** |
| MNLI-mm | accuracy | 0.368 | 0.366 | 0.364 |
| **Mean** | — | 0.383 | 0.513 | **0.515** |

The data quality switch (C4 → SmolLM blend) drove the major improvement (+34% GLUE mean). Post-training on Wiki + Cosmopedia produced marginal gains (GLUE mean +0.002), suggesting that at 432M scale, 4B tokens of high-quality pretrain data already saturates the model's capacity.

### ARC

| Task | C4 4B | SmolLM 4B | + Post-train (5B) |
|------|:---:|:---:|:---:|
| ARC-C 0-shot | 0.220 | **0.256** | 0.249 |
| ARC-C 5-shot | 0.223 | **0.259** | 0.254 |
| ARC-E 0-shot | 0.311 | 0.356 | **0.365** |
| ARC-E 5-shot | 0.320 | 0.362 | **0.368** |

ARC improved consistently from data quality alone (+3–4pp). Post-training nudged ARC-E slightly higher but regressed ARC-C marginally — net neutral.

### Key Finding

Switching from C4 to a curated high-quality blend (FineWeb-Edu + Cosmopedia v2) improved GLUE mean by **34%** (+0.130) at the same 4B token budget. The largest gain came from MRPC (paraphrase detection), which went from random guessing to 0.813 F1 — proving that small MoE models can learn meaningful language understanding given clean data. Post-training with domain-specific data provides negligible additional benefit at this scale, indicating that the pretrain data recipe is the dominant factor for model quality.

All evaluations use conditional log-likelihood scoring over answer spans, identical settings for fair comparison (`--limit 500 --batch-size 16 --max-length 512`).

---

## Ablation: Shared Expert Architecture

The `shared_expert/` directory contains a variant architecture experiment (v2) that adds a DeepSeek-style shared expert to the MoE layer. This section documents the experiment for reproducibility.

### Architecture Changes (vs baseline)

| Parameter | Baseline (v1.1) | Shared Expert (v2) |
|-----------|:---:|:---:|
| Attention | GQA 14Q / 2KV (7:1) | GQA 16Q / 4KV (4:1) |
| Head dim | 64 | 56 |
| Experts | 6 routed (top-2) | 1 shared + 6 routed (top-2) |
| LR schedule | Cosine | WSD (Warmup-Stable-Decay) |
| Batch size | 24 | 22 |
| **Total params** | **~432M** | **~498M** |
| **Active params** | **~176M** | **~241M** |

The shared expert is always active (no routing), providing general-purpose features. Routed experts specialize via top-2 gating. Output is the sum of both.

### Training

```bash
# Uses the same tokenized data as baseline
python shared_expert/scripts/train.py --cache-dir data/pretrain/smollm_blend \
  --output-dir checkpoints/v2 --batch-size 22 \
  --max-tokens 4000000000 --lr 7e-4 --schedule wsd \
  --warmup-steps 2000 --save-every-min 120

# Post-train
python shared_expert/scripts/resume.py --checkpoint-dir checkpoints/v2 \
  --output-dir checkpoints/v2_posttrain \
  --cache-dir data/posttrain2/knowledge_blend \
  --max-tokens 1000000000 --lr 2e-5 --warmup-steps 300 \
  --batch-size 22 --schedule wsd --save-every-min 60
```

### Results

| Task | Baseline (v1.1) | Shared Expert pretrain | Shared Expert post-train |
|------|:---:|:---:|:---:|
| SST2 | 0.568 | 0.518 | **0.586** |
| MRPC F1 | **0.813** | 0.811 | 0.809 |
| QQP F1 | 0.519 | 0.519 | 0.519 |
| QNLI | 0.458 | 0.458 | 0.458 |
| RTE | **0.534** | 0.520 | 0.520 |
| MNLI | **0.352** | 0.348 | 0.348 |
| MNLI-mm | 0.364 | 0.368 | **0.368** |
| **GLUE Mean** | 0.513 | 0.505 | **0.515** |
| ARC-C 0-shot | 0.249 | **0.265** | 0.264 |
| ARC-C 5-shot | 0.254 | 0.256 | **0.259** |
| ARC-E 0-shot | 0.365 | 0.384 | **0.388** |
| ARC-E 5-shot | 0.368 | 0.379 | **0.387** |

### Conclusion

The shared expert variant does **not** provide substantial improvement over the baseline at this scale:

- GLUE mean is within evaluation noise (0.515 vs 0.513, +0.002)
- ARC improves consistently (+1–2pp), suggesting better knowledge capacity
- SST2 improves after post-train (+6.8pp vs baseline post-train), but other GLUE tasks are flat or slightly worse
- Total params increase 15% (432M→498M) for marginal gains

At ~241M active parameters, the MoE routing overhead and representational fragmentation outweigh the knowledge capacity benefit. The shared expert design is more likely to pay off at 1B+ active params (cf. DeepSeek, Mixtral). The baseline v1.1 architecture remains the recommended model.

---

## Legacy (v1) — Original C4 Training

The first version of TinyMixtral trained on C4-en, a general-purpose web corpus. The v1 weights are available at [mikecovlee/tinymixtral-v1.0](https://huggingface.co/mikecovlee/tinymixtral-v1.0). This section is kept for historical reference and reproducibility. The current recommended recipe (SmolLM blend) is documented in the sections above.

### Data Preparation

```bash
# 1. Download tokenizer
python scripts/prepare_tokenizer.py --from-hf TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output tokenizer/

# 2. Tokenize C4-en (4B tokens)
python scripts/prepare_data.py --dataset allenai/c4 --subset en \
  --tokenizer tokenizer/ --output data/c4/tokenized \
  --max-tokens 4000000000 --force
```

### Pretrain (4B tokens)

```bash
python scripts/train.py --cache-dir data/c4/tokenized \
  --batch-size 22 --max-tokens 4000000000 --lr 3e-4 --warmup-steps 2000 \
  --keep-last-checkpoints 5 2>&1 | tee train.log
```

| Parameter | Value |
|-----------|-------|
| Data | C4-en |
| Batch size | 22 |
| Sequence length | 1,024 |
| Tokens/step | 22,528 |
| Steps | 177,557 |
| Learning rate | 3e-4 |
| Warmup steps | 2,000 |
| Weight decay | 0.1 |
| Grad clip | 1.0 |
| Time | ~77 h |

### Historical post-train result (1B tokens)

The table below records the original upstream v1.0 experiment. Its former command reset the target, learning rate, warmup, and data cursor through `resume.py`; that workflow is intentionally unsupported by the current strict CPT resume path and is therefore omitted here. Reproducing the historical run requires the corresponding historical upstream revision rather than the current CPT v1 entry point.

| Parameter | Value |
|-----------|-------|
| Data | FineWeb-Edu + Cosmopedia v2 (50:50) |
| Tokens | 1B |
| Steps | 44,390 |
| Learning rate | 5e-5 |
| Warmup steps | 300 |
| Time | ~20.8 h |

### Results

| Task | Metric | Pretrain (C4 4B) | Post-train (+1B, 5B total) |
|------|--------|:---:|:---:|
| SST2 | accuracy | 0.470 | 0.554 |
| MRPC | accuracy / f1 | 0.338 / 0.069 | 0.706 / 0.815 |
| QQP | accuracy / f1 | 0.470 / 0.412 | 0.530 / 0.342 |
| QNLI | accuracy | 0.494 | 0.452 |
| RTE | accuracy | 0.520 | 0.484 |
| MNLI | accuracy | 0.348 | 0.348 |
| MNLI-mm | accuracy | 0.368 | 0.368 |
| **GLUE Mean** | — | 0.383 | 0.480 |
| ARC-C 0-shot | accuracy | 0.220 | 0.233 |
| ARC-C 5-shot | accuracy | 0.223 | 0.246 |
| ARC-E 0-shot | accuracy | 0.311 | 0.342 |
| ARC-E 5-shot | accuracy | 0.320 | 0.348 |

Post-training improved GLUE mean from 0.383 to 0.480, with the largest gain on MRPC (F1: 0.069 → 0.815). ARC improved modestly (+1–3 pp). These v1 results serve as the baseline for the data quality ablation documented above.

## License

MIT License. Copyright (C) 2026 Michael Lee (李登淳).
