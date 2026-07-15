# TinyMixtral

A small Mixtral-style Mixture-of-Experts causal language model (~432M total, ~176M active parameters) for pretraining research on a single consumer GPU.

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-mikecovlee%2Ftinymixtral-yellow)](https://huggingface.co/mikecovlee/tinymixtral)

## Pretrained Model

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "mikecovlee/tinymixtral", trust_remote_code=True
)
```

The model is available on HuggingFace Hub at [mikecovlee/tinymixtral](https://huggingface.co/mikecovlee/tinymixtral). It was trained on 4B tokens of C4-en.

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

```bash
# 1. Download tokenizer
python scripts/prepare_tokenizer.py --from-hf TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output tokenizer/

# 2. Tokenize C4-en dataset (4B tokens)
python scripts/prepare_data.py --dataset allenai/c4 --subset en \
  --tokenizer tokenizer/ --output data/c4/tokenized \
  --max-tokens 4000000000 --force

# 3. Pretrain
python scripts/train.py --cache-dir data/c4/tokenized \
  --batch-size 22 --max-tokens 4000000000 \
  --keep-last-checkpoints 5 --eval-on-save 2>&1 | tee train.log
```

## Training Details

- **Precision**: bf16 (model, AdamW states, autocast forward/backward)
- **Optimizer**: AdamW (β=0.9,0.95, wd=0.1), weight decay only on ≥2D parameters
- **LR schedule**: Cosine decay with linear warmup (warmup_steps=2000)
- **Gradient clipping**: 1.0
- **Batch**: 22 × 1024 = 22,528 tokens/step
- **Activation checkpointing**: enabled (required for 24GB VRAM)
- **Data**: C4-en, pre-tokenized to `.pt` shards (100M tokens each), cycled round-robin

`prepare_data.py` explicitly appends EOS to each document and validates tokenizer vocab size (32K). Shards are written atomically via staging → replace.

## Checkpoints

Periodic checkpoints saved every `--save-every-min` minutes, plus a final checkpoint at completion:

```
checkpoints/run/
├── step_0159189/          # periodic
│   ├── config.json
│   ├── pytorch_model.bin
│   └── training_state.pt  # optimizer, scheduler, data position
├── ...
└── step_0177557_final/    # final checkpoint
```

`training_state.pt` contains optimizer/scheduler states, step, token count, shard position (`fi`/`ptr`), and schedule parameters — enabling exact training resumption.

## Resume Training

```bash
python scripts/resume.py --batch-size 22
```

The script automatically locates the latest checkpoint, restores model/optimizer/scheduler state, and continues from the exact data position. Batch size, sequence length, and training target are read from the checkpoint.

## Post-Training

Continue training on higher-quality data to boost model capabilities. The example below uses the SmolLM blend (FineWeb-Edu + Cosmopedia v2, 50/50) for 1B tokens.

```bash
# 1. Tokenize FineWeb-Edu (~500M tokens)
python scripts/prepare_data.py \
  --dataset HuggingFaceFW/fineweb-edu --subset sample-10BT \
  --tokenizer tokenizer/ --output data/posttrain/fineweb \
  --max-tokens 500000000 --force

# 2. Tokenize Cosmopedia v2 (~500M tokens)
python scripts/prepare_data.py \
  --dataset HuggingFaceTB/cosmopedia-v2 --subset auto \
  --tokenizer tokenizer/ --output data/posttrain/cosmopedia \
  --max-tokens 500000000 --force

# 3. Interleave shards (50/50)
python scripts/mix_data.py data/posttrain/fineweb data/posttrain/cosmopedia \
  --output data/posttrain/mixed

# 4. Post-train from pretrained checkpoint
python scripts/resume.py \
  --checkpoint-dir checkpoints/run \
  --output-dir checkpoints/posttrain \
  --cache-dir data/posttrain/mixed \
  --max-tokens 1000000000 --lr 5e-5 --warmup-steps 300 \
  --batch-size 22 --save-every-min 60
```

Key differences from pretraining:

| Aspect | Pretrain | Post-train |
|--------|----------|------------|
| Data | C4-en (noisy) | FineWeb-Edu + Cosmopedia (curated) |
| LR | 3e-4 | 5e-5 (lower, to avoid forgetting) |
| Warmup | 2,000 steps | 300 steps (short re-warmup) |
| Schedule | Cosine from scratch | Fresh cosine, AdamW momentum preserved |
| Target | 4B tokens | 1–4B tokens |

`--max-tokens` triggers post-training mode: the step counter and data position reset to zero, the scheduler starts a fresh warmup+cosine cycle, but optimizer momentum (AdamW β₁/β₂ states) carries over from pretraining. Output goes to `--output-dir`, keeping pretrain checkpoints untouched.

## Evaluation

### GLUE Benchmark (zero-shot)

```bash
# Quick evaluation (5 tasks, limited samples)
python scripts/eval_glue.py --checkpoint checkpoints/run/step_0177557_final \
  --tokenizer tokenizer/ --tasks quick --limit 500

# Full evaluation (8 tasks)
python scripts/eval_glue.py --checkpoint checkpoints/run/step_0177557_final \
  --tokenizer tokenizer/ --tasks all --output results.json
```

Supported tasks: `sst2`, `mrpc`, `qqp`, `qnli`, `rte`, `cola`, `mnli`, `mnli_mismatched`.

### ARC Challenge (zero-shot / few-shot)

```bash
# Zero-shot
python scripts/eval_arc.py --checkpoint checkpoints/run/step_0177557_final \
  --tokenizer tokenizer/ --tasks arc_c,arc_e

# 5-shot
python scripts/eval_arc.py --checkpoint checkpoints/run/step_0177557_final \
  --tokenizer tokenizer/ --tasks arc_c,arc_e --shots 5
```

### Evaluation Method

Zero-shot evaluation uses conditional log-likelihood scoring: each candidate answer string is appended to the prompt, and the model's per-token log-probability over only the answer span is averaged. The highest-scoring answer is selected. Labels are never leaked into the scoring context.

## Publishing to HuggingFace

A pretrained model is already available at [mikecovlee/tinymixtral](https://huggingface.co/mikecovlee/tinymixtral). To export your own trained checkpoint:

```bash
python scripts/publish_hf.py \
  --checkpoint checkpoints/run/step_0177557_final \
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
python scripts/chat.py --checkpoint checkpoints/run/step_0177557_final --tokenizer tokenizer/

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
├── configs/                        # Config templates
├── requirements.txt                # Python dependencies
├── start.sh                        # Training launch script
├── resume.sh                       # Resume launch script
├── LICENSE                         # MIT
└── README.md
```

## Key Design Decisions

1. **Simple training code, HF conversion on export** — Training uses plain `nn.Module` and a dataclass config for simplicity. A separate `hf/` compatibility layer + `publish_hf.py` script converts to `PreTrainedModel`/`PretrainedConfig` for HuggingFace Hub.

2. **Pre-tokenized shards** — Data is tokenized once to disk, eliminating CPU bottleneck during training. Shards cycle round-robin; each shard is loaded on demand.

3. **bf16 throughout** — Model parameters, AdamW first/second moments, and forward/backward passes all use bf16.

4. **Auxiliary loss (Mixtral-style)** — `L_aux = N × sum_i(f_i × P_i)` where `f_i` (fraction of routed tokens) is detached and `P_i` (mean router softmax) retains gradient. Aux losses are averaged across layers before applying the coefficient.

5. **GQA with SDPA** — 14 query heads share 2 key/value heads (7:1 ratio). Causal and padding masks are merged into a 4D boolean mask for `scaled_dot_product_attention` (PyTorch 2.x disallows simultaneous `attn_mask` + `is_causal`).

6. **Atomic checkpoint saves** — Checkpoints are written to a temporary directory and atomically renamed, preventing corruption from interrupted saves.

7. **Signal handling** — SIGINT/SIGTERM triggers a clean save after the current step completes.

## Results (4B tokens, C4-en)

Training converged from loss ~10.5 to ~3.0 with stable aux_loss (~1.0). As expected for a 176M-active-parameter model evaluated zero-shot, GLUE and ARC scores reflect the model's limited capacity — these results establish the baseline for future iterations.

## License

MIT License. Copyright (C) 2026 Michael Lee (李登淳).
