# TinyMixtral

A Mixtral-style Mixture-of-Experts causal language model for pretraining research on a single consumer GPU. The current **1B MoE** release is ~1.18B total / ~352M active params (top-2 of 8 experts). Smaller earlier variants (v1.1 ~432M, v2.0 beta ~498M) are kept for comparison and ablation.

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-mikecovlee%2Ftinymixtral--1B-blue)](https://huggingface.co/mikecovlee/tinymixtral-1B)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-mikecovlee%2Ftinymixtral-yellow)](https://huggingface.co/mikecovlee/tinymixtral)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-mikecovlee%2Ftinymixtral--v2.0--beta-orange)](https://huggingface.co/mikecovlee/tinymixtral-v2.0-beta)

## Pretrained Model

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "mikecovlee/tinymixtral-1B", trust_remote_code=True
)
```

The flagship **1B MoE** (SmolLM-blend pretrain; 4B tokens, continued to 8B) is available at [mikecovlee/tinymixtral-1B](https://huggingface.co/mikecovlee/tinymixtral-1B).

Earlier smaller models:
- [mikecovlee/tinymixtral](https://huggingface.co/mikecovlee/tinymixtral) — v1.1, SmolLM blend (~432M)
- [mikecovlee/tinymixtral-v2.0-beta](https://huggingface.co/mikecovlee/tinymixtral-v2.0-beta) — shared-expert variant (~498M)
- [mikecovlee/tinymixtral-v1.0](https://huggingface.co/mikecovlee/tinymixtral-v1.0) — legacy C4 pretrain

## Model Architecture

| Parameter | Value |
|-----------|-------|
| hidden_size | 1024 |
| num_layers | 16 |
| Attention | Grouped Query Attention (16 heads / 4 KV heads) |
| Head dim | 64 |
| RoPE theta | 1,000,000 |
| Norm | RMSNorm |
| Experts | 8 routed (top-2) |
| Expert FFN | SwiGLU, intermediate = 2816 |
| Vocab size | 32,000 |
| Max position | 2,048 |
| **Total params** | **~1.18B** |
| **Active params** | **~352M** |

## Hardware & Environment

- GPU: NVIDIA RTX PRO 4500 Blackwell 32GB
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
  --batch-size 24 --max-tokens 4000000000 --lr 7e-4 \
  --keep-last-checkpoints 5 2>&1 | tee train.log
```

## Training Details

- **Precision**: bf16 (model + autocast forward/backward), fp32 optimizer states
- **Optimizer**: AdamW (β=0.9,0.95, wd=0.1), weight decay only on ≥2D parameters; `--bf16-optim` stores moments in bf16
- **LR schedule**: Cosine decay with linear warmup (warmup_steps=2000); WSD also available via `--schedule wsd`
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
│   └── training_state.pt  # optimizer, scheduler, data position
├── ...
└── step_0162761_final/    # final checkpoint
```

`training_state.pt` contains optimizer/scheduler states, step, token count, shard position (`fi`/`ptr`), and schedule parameters — enabling exact training resumption.

## Resume Training

```bash
python scripts/resume.py --batch-size 24
```

The script automatically locates the latest checkpoint, restores model/optimizer/scheduler state, and continues from the exact data position. Batch size, sequence length, and training target are read from the checkpoint.

## Post-Training

Continue training on domain-specific data to address weaknesses. The example below uses Wikipedia + Cosmopedia v2 (50:50, 1B tokens) to improve formal grammar and factual knowledge.

```bash
# 1. Tokenize Wikipedia (500M tokens)
python scripts/prepare_data.py \
  --dataset wikimedia/wikipedia --subset 20231101.en \
  --tokenizer tokenizer/ --output data/posttrain/wiki \
  --max-tokens 500000000 --force

# 2. Tokenize Cosmopedia v2 (500M tokens)
python scripts/prepare_data.py \
  --dataset HuggingFaceTB/cosmopedia-v2 --subset cosmopedia-v2 \
  --tokenizer tokenizer/ --output data/posttrain/cosmopedia \
  --max-tokens 500000000 --force

# 3. Mix shards (50/50)
python scripts/mix_data.py data/posttrain/wiki data/posttrain/cosmopedia \
  --output data/posttrain/knowledge_blend

# 4. Post-train from best pretrain checkpoint
python scripts/resume.py \
  --checkpoint-dir checkpoints/smollm_blend \
  --output-dir checkpoints/knowledge_posttrain \
  --cache-dir data/posttrain/knowledge_blend \
  --max-tokens 1000000000 --lr 2e-5 --warmup-steps 300 \
  --batch-size 24 --save-every-min 60
```

Key differences from pretraining:

| Aspect | Pretrain | Post-train |
|--------|----------|------------|
| Data | FineWeb-Edu + Cosmopedia (89:11) | Wiki + Cosmopedia (50:50) |
| LR | 7e-4 | 2e-5 |
| Warmup | 2,000 steps | 300 steps |
| Schedule | Cosine from scratch | Fresh cosine, momentum preserved |
| Target | 4B tokens | 1B tokens |

`--max-tokens` triggers post-training mode: the step counter and data position reset to zero, the scheduler starts a fresh warmup+cosine cycle, but optimizer momentum (AdamW β₁/β₂ states) carries over from pretraining. The `--lr` flag correctly overrides the checkpoint's saved LR in post-training mode.

### Evaluation

All reported benchmark numbers use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (v0.4.x) with conditional log-likelihood scoring. Publish the checkpoint first (`scripts/publish_hf.py`), then evaluate:

```bash
# lm-eval harness 0-shot (standard 8-task suite)
lm_eval --model hf \
  --model_args "pretrained=publish/,tokenizer=tokenizer/,trust_remote_code=True,dtype=bfloat16" \
  --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq,lambada_openai \
  --batch_size 16 --device cuda --output_path evals/harness_0shot

# IFEval (instruction following, generative)
lm_eval --model hf \
  --model_args "pretrained=publish/,tokenizer=tokenizer/,trust_remote_code=True,dtype=bfloat16" \
  --tasks ifeval --apply_chat_template --batch_size 8 --device cuda \
  --output_path evals/ifeval_publish
```

For few-shot variants, append `--num_fewshot N` (e.g. HellaSwag 10-shot, WinoGrande 5-shot, ARC 25-shot).

## Publishing to HuggingFace

A pretrained model is already available at [mikecovlee/tinymixtral](https://huggingface.co/mikecovlee/tinymixtral). To export your own trained checkpoint:

```bash
python scripts/publish_hf.py \
  --checkpoint checkpoints/knowledge_posttrain/step_0040691_final \
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
python scripts/chat.py --checkpoint checkpoints/knowledge_posttrain/step_0040691_final --tokenizer tokenizer/

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
├── evals/                          # Evaluation output/results (harness, IFEval, …)
├── scripts/
│   ├── train.py                    # Training entry point
│   ├── resume.py                   # Checkpoint resume
│   ├── train_utils.py              # Shared training logic
│   ├── chat.py                     # Interactive chat (native model)
│   ├── chat_hf.py                  # Interactive chat (HF AutoModel)
│   ├── publish_hf.py               # HF-format export
│   ├── prepare_data.py             # Dataset pre-tokenization
│   ├── prepare_tokenizer.py        # Tokenizer download / training
│   ├── mix_data.py                 # Interleave shards from multiple datasets
│   ├── benchmark.py                # GPU memory/throughput profiler
│   └── eval_summarization.py       # SAMSum summarization evaluation (ROUGE)
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

3. **bf16 model, flexible optimizer** — Model parameters and forward/backward passes use bf16. Optimizer states default to fp32; `--bf16-optim` stores them in bf16 (~50% optimizer VRAM savings).

4. **Auxiliary loss (Mixtral-style)** — `L_aux = N × sum_i(f_i × P_i)` where `f_i` (fraction of routed tokens) is detached and `P_i` (mean router softmax) retains gradient. Aux losses are averaged across layers before applying the coefficient.

5. **GQA with SDPA** — 14 query heads share 2 key/value heads (7:1 ratio). Training path uses `enable_gqa=True` with `is_causal=True` (no KV expansion). When a padding mask is present (eval/inference), KV heads are explicitly expanded and a 4D boolean mask is constructed.

6. **Atomic checkpoint saves** — Checkpoints are written to a temporary directory and atomically renamed, preventing corruption from interrupted saves.

7. **Signal handling** — SIGINT/SIGTERM triggers a clean save after the current step completes.

## Results

### Data Quality Ablation

The original model trained on C4-en (noisy web text). We ran an ablation replacing C4 with a SmolLM-inspired blend: **FineWeb-Edu (89%) + Cosmopedia v2 (11%)**, 4B tokens total. Batch size increased to 24. LR swept on 100M-token runs; 7e-4 was optimal.

### Training Summary

| Phase | Data | LR | Tokens | Steps | Time | End Loss |
|-------|------|----|:------:|:-----:|:----:|:--------:|
| Pretrain (C4) | C4-en | 3e-4 | 4B | 177,557 | 77.1 h | 3.0 |
| Pretrain (SmolLM) | FineWeb-Edu + Cosmopedia v2 (89:11) | 7e-4 | 4B | 162,761 | 83.6 h | 2.5 |
| Post-train | Wiki + Cosmopedia v2 (50:50) | 2e-5 | 1B | 40,691 | 20.5 h | 2.5 |
| 1B MoE (v1b_moe) | SmolLM blend (89:11) | 7e-4 | 4B | 244,141 | ~102.5 h | 1.9* |
| 1B MoE 续训 (v1b_moe_cont) | smollm_blend2（非重叠新段 89:11）| 4.2e-4 | 4B | 121,921 | 60.4 h | 1.80 |

*Decay-phase loss plateaued at ~1.9 then spiked to 3.4 in the final ~300 steps as the data stream wrapped back to the start of the corpus (shard 0).

### Standard Benchmarks (lm-evaluation-harness, 0-shot)

| Task | Metric | v1.1 (432M) | v2.0 beta (498M) | 1B MoE (1182M) | 1B MoE 8B (1182M) | SmolLM2-360M | Qwen3-0.6B |
|------|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| HellaSwag | acc_norm | 0.308 | 0.326 | 0.311 | **0.329** | 0.563 | 0.473 |
| PIQA | acc | 0.616 | **0.631** | 0.620 | 0.630 | 0.719 | 0.673 |
| WinoGrande | acc | **0.524** | 0.506 | 0.510 | 0.523 | 0.587 | 0.563 |
| ARC-Easy | acc | 0.456 | 0.474 | 0.463 | **0.479** | 0.705 | 0.609 |
| ARC-Challenge | acc_norm | 0.247 | 0.272 | 0.273 | **0.279** | 0.383 | 0.340 |
| OpenBookQA | acc_norm | 0.288 | 0.290 | 0.288 | **0.306** | 0.372 | 0.316 |
| BoolQ | acc | 0.606 | 0.455 | 0.548 | **0.620** | 0.620 | 0.643 |
| LAMBADA | acc | 0.227 | 0.224 | 0.200 | **0.234** | 0.532 | 0.401 |

All numbers measured locally with identical settings (lm-eval-harness v0.4.12, 0-shot, cuda, bf16). Batch size does not affect log-likelihood evaluation results. **Bold = best among the TinyMixtral variants**; SmolLM2-360M / Qwen3-0.6B are shown for reference only.

**Few-shot (1B MoE, lm-eval-harness):** HellaSwag 10-shot acc_norm 0.312 · WinoGrande 5-shot acc 0.524 · ARC-Easy 25-shot acc_norm 0.468 · ARC-Challenge 25-shot acc_norm 0.262

SmolLM2-360M trained on 4T tokens; Qwen3-0.6B trained on 36T tokens. TinyMixtral trained on 4–8B tokens (~500–1000× less) on a single consumer GPU. The performance gap is primarily a data budget difference. v2.0 beta improves over v1.1 on most tasks (HellaSwag +1.8pp, PIQA +1.5pp, ARC-C +2.5pp) but regresses on BoolQ.

**Continuation (4B → 8B):** the 8B column is the same 1182M MoE continued on a new, non-overlapping 4B of the same data recipe. Doubling pretrain tokens improved **every** harness task, with the largest gains on BoolQ (+7.2pp) and LAMBADA (+3.5pp); mean 0-shot rose ~+2.3pp (0.402 → 0.425).

### 1B MoE Post-training (negative result)

We post-trained the 1B MoE from its lowest-loss pretrain checkpoint (step 243,038) with the v1.1 recipe (Wiki + Cosmopedia v2 50:50, 1B tokens, lr 2e-5, warmup 300, WSD schedule, 60,975 steps, ~26 h). Training loss fell 2.98 → 1.8, but **downstream performance (lm-eval-harness 0-shot) was unchanged**:

| Task | Pretrain (4B) | Post-train (5B) |
|------|:---:|:---:|
| HellaSwag (acc_norm) | 0.311 | 0.313 |
| PIQA (acc) | 0.620 | 0.623 |
| WinoGrande (acc) | 0.510 | 0.505 |
| ARC-Easy (acc) | 0.463 | 0.465 |
| ARC-Challenge (acc_norm) | 0.273 | 0.272 |
| OpenBookQA (acc_norm) | 0.288 | 0.290 |
| BoolQ (acc) | 0.548 | 0.528 |
| LAMBADA (acc) | 0.200 | 0.195 |

All changes are within evaluation noise (±0.02). At this scale, low-LR post-training on a knowledge blend lowers the loss but does not transfer to task ability.

### Instruction Following (IFEval)

IFEval (instruction-level loose accuracy) via lm-eval-harness with `--apply_chat_template`:

| Model | IFEval inst-level loose |
|-------|:---:|
| 1B MoE 4B | 0.222 |
| 1B MoE 5B (post-train) | 0.234 |
| 1B MoE 8B (continuation) | 0.221 |

Instruction-following stays essentially flat (~0.22) across pretraining and continued pretraining — more pretraining tokens do not improve it, consistent with the common finding that instruction-following requires targeted SFT rather than more LM pretraining.

### Key Finding

The reliable signal across TinyMixtral variants is the lm-eval-harness suite. The C4 → SmolLM data-quality switch improved the knowledge/multiple-choice tasks (e.g. ARC-C/Easy +3–4pp). Doubling the 1B MoE's pretrain tokens (4B → 8B, same data recipe) improved **every** harness task (+2.3pp mean 0-shot), most notably BoolQ and LAMBADA. However, neither post-training nor continued pretraining moves instruction-following (IFEval ≈ 0.22) — at ~1.2B params, knowledge grows with more pretrain data, while instruction-following requires targeted SFT.

ARC-C / ARC-Easy (0-shot) are covered by the `arc_challenge` / `arc_easy` rows in the harness table above.

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

lm-eval-harness 0-shot (see the main benchmark table above; "v2.0 beta" is this shared-expert variant):

| Task | Metric | Baseline (v1.1, 432M) | Shared Expert (v2.0, 498M) |
|------|--------|:---:|:---:|
| HellaSwag | acc_norm | 0.308 | **0.326** |
| PIQA | acc | 0.616 | **0.631** |
| WinoGrande | acc | **0.524** | 0.506 |
| ARC-Easy | acc | 0.456 | **0.474** |
| ARC-Challenge | acc_norm | 0.247 | **0.272** |
| OpenBookQA | acc_norm | 0.288 | **0.290** |
| BoolQ | acc | **0.606** | 0.455 |
| LAMBADA | acc | **0.227** | 0.224 |

### Conclusion

The shared expert variant does **not** provide substantial improvement over the baseline at this scale:

- Harness results are mixed: v2.0 improves HellaSwag / ARC (+1.5–2.5pp) but sharply regresses BoolQ (−15pp)
- Total params increase 15% (432M→498M) for marginal, inconsistent gains

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

### Post-train (1B tokens)

Continue from the C4 checkpoint on higher-quality data:

```bash
# 3. Tokenize FineWeb-Edu (500M tokens)
python scripts/prepare_data.py \
  --dataset HuggingFaceFW/fineweb-edu --subset sample-10BT \
  --tokenizer tokenizer/ --output data/posttrain/fineweb \
  --max-tokens 500000000 --force

# 4. Tokenize Cosmopedia v2 (500M tokens)
python scripts/prepare_data.py \
  --dataset HuggingFaceTB/cosmopedia-v2 --subset cosmopedia-v2 \
  --tokenizer tokenizer/ --output data/posttrain/cosmopedia \
  --max-tokens 500000000 --force

# 5. Mix shards (50/50)
python scripts/mix_data.py data/posttrain/fineweb data/posttrain/cosmopedia \
  --output data/posttrain/mixed

# 6. Post-train
python scripts/resume.py \
  --checkpoint-dir checkpoints/run \
  --output-dir checkpoints/posttrain \
  --cache-dir data/posttrain/mixed \
  --max-tokens 1000000000 --lr 5e-5 --warmup-steps 300 \
  --batch-size 22 --save-every-min 60
```

| Parameter | Value |
|-----------|-------|
| Data | FineWeb-Edu + Cosmopedia v2 (50:50) |
| Tokens | 1B |
| Steps | 44,390 |
| Learning rate | 5e-5 |
| Warmup steps | 300 |
| Time | ~20.8 h |

### Results

lm-eval-harness 0-shot on the released `v1.0` checkpoint ([mikecovlee/tinymixtral-v1.0](https://huggingface.co/mikecovlee/tinymixtral-v1.0), C4-pretrained, ~432M):

| Task | Metric | v1.0 (C4 4B) |
|------|--------|:---:|
| HellaSwag | acc_norm | 0.310 |
| PIQA | acc | 0.613 |
| WinoGrande | acc | 0.508 |
| ARC-Easy | acc | 0.422 |
| ARC-Challenge | acc_norm | 0.247 |
| OpenBookQA | acc_norm | 0.308 |
| BoolQ | acc | 0.579 |
| LAMBADA | acc | 0.240 |

These serve as the (weak) baseline for the data-quality ablation documented above.

## License

MIT License. Copyright (C) 2026 Michael Lee (李登淳).
