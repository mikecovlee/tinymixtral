# TinyMixtral v3.0

**Flagship release.** A Mixtral-style sparse MoE language model: ~477.5M total / ~276.1M active
parameters (top-2 of 4 routed experts), trained from scratch on **8.05B tokens** on a single GPU.

- HF repo: `mikecovlee/tinymixtral`
- Config: `versions/v3.0/configs/improve_v05b.json`

## Architecture

| Parameter | Value |
|-----------|-------|
| hidden_size | 1024 |
| num_layers | 16 |
| Attention | Grouped Query Attention (16 heads / 4 KV heads) |
| Head dim | 64 |
| RoPE theta | 1,000,000 |
| Norm | RMSNorm + per-head QK-Norm (pre-RoPE) |
| Experts | 4 routed (top-2), aux loss 1e-3 |
| Expert FFN | SwiGLU, intermediate = 2048 |
| Vocab size | 32,000 (tied embeddings) |
| Max position | 2,048 |
| **Total params** | **~477.5M** |
| **Active params** | **~276.1M** |

Design decisions (validated by the P4a/P4b ablation matrix):
- **top-2 routing** — top-1 was +12.2–19.5% worse in val PPL at 400M tokens; maintain top-2.
- **QK-Norm kept** — neutral at 400M tokens, retained as the trunk default.
- **aux = 1e-3** — auxiliary-loss coefficient is insensitive under E4/top-2.
- **LR ladder** `5e-4 / 5e-4 / 4e-4 / 3e-4` across the four segments (5e-4 was the sole
  significant factor, −5.3% val PPL in the matrix).

## Data

8.05B unique tokens, zero repetition, split into **four strictly disjoint pools**
(`main_s1..s4` = 2.00 / 1.94 / 2.20 / 1.91B) built by `scripts/make_blend_shards.py`
(hard-linked, inode-verified). Validation always uses `pilot_blend30_val` (2 held-out shards).

| Source | Share |
|---|---|
| FineWeb-Edu | 44% |
| DCLM web | 20% |
| Cosmopedia (synthetic) | 12.5% |
| Code | 12.5% |
| Math | 6% |
| Wikipedia | 6% |

### Reproduce the data pools

Each pool is assembled from 100M-token `.pt` shards. Two acquisition routes are used: HF `datasets`
streaming via `scripts/prepare_data.py` (clean parquet corpora), and bulk file download via
`scripts/download_*.py` → `scripts/prepare_data_local.py` (large web/code corpora, avoiding slow
streaming reads).

```bash
# tokenizer (shared across versions)
python scripts/prepare_tokenizer.py --from-hf TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output tokenizer/

# FineWeb-Edu (44%) — 3rd 3.56B slice of sample-10BT -> fineweb3
python scripts/prepare_data.py --dataset HuggingFaceFW/fineweb-edu --subset sample-10BT \
  --tokenizer tokenizer/ --output data/pretrain/fineweb3 \
  --skip-tokens 7120000000 --max-tokens 3560000000 --force

# Cosmopedia v2 (synthetic) — 2nd 440M slice -> cosmopedia3
python scripts/prepare_data.py --dataset HuggingFaceTB/cosmopedia-v2 --subset cosmopedia-v2 \
  --tokenizer tokenizer/ --output data/pretrain/cosmopedia3 \
  --skip-tokens 880000000 --max-tokens 440000000 --force

# Wikipedia -> r5_wiki
python scripts/prepare_data.py --dataset wikimedia/wikipedia --subset 20231101.en \
  --tokenizer tokenizer/ --output data/pretrain/r5_wiki --max-tokens 500000000 --force

# DCLM web (20%) — .jsonl.zst shards -> parquet -> .pt shards
python scripts/download_jsonl_zst.py --repo mlfoundations/dclm-baseline-1.0 \
  --subdir global-shard_01_of_10/local-shard_0_of_10 --output data/raw/r5_web --suffix .jsonl.zst
python scripts/zst_jsonl_to_parquet.py --input data/raw/r5_web --output data/raw/r5_web_pq
python scripts/prepare_data_local.py --input data/raw/r5_web_pq --tokenizer tokenizer/ \
  --output data/pretrain/r5_web --max-tokens 1600000000 --force --workers 8

# Code (12.5%) — OpenCodeInstruct (text = input + "\n\n" + output)
python scripts/download_parquets.py --repo nvidia/OpenCodeInstruct --subdir <files-dir> \
  --output data/raw/r5_code --workers 4
python scripts/columns_to_text_parquet.py --input data/raw/r5_code \
  --output data/raw/r5_code_text --columns input output --sep "\n\n"
python scripts/prepare_data_local.py --input data/raw/r5_code_text --tokenizer tokenizer/ \
  --output data/pretrain/r5_code --max-tokens 1000000000 --force --workers 8

# Math (~6%) — web-math corpus (columns url/text/date/metadata), same download route:
#   download_parquets.py --repo <web-math> --subdir <files-dir> --output data/raw/r5_math
#   prepare_data_local.py --input data/raw/r5_math --output data/pretrain/r5_math --max-tokens 500000000
```

> Conventions: `download_parquets.py` / `zst_jsonl_to_parquet.py` emit `<name>.parquet.parquet`
> and `prepare_data_local.py` globs `*.parquet.parquet` (keep the double extension when
> normalizing); `prepare_data_local.py` reads the `text` column only, so multi-column pools (code)
> must go through `columns_to_text_parquet.py` first; `download_jsonl_zst.py` takes one subdir per
> run.

Then assemble the four strictly disjoint pools (Bresenham-interleaved, hard-linked, inode-verified).
`--start/--take` select each source's shard range; every source passes `--val-take 0` because
validation uses the separate `pilot_blend30_val` directory. Shown for `main_s1`:

```bash
python scripts/make_blend_shards.py --output data/pretrain/main_s1 \
  --source data/pretrain/fineweb3    --start 8 --take 7 --val-take 0 \
  --source data/pretrain/p5_web      --start 0 --take 2 --val-take 0 \
  --source data/pretrain/r5_web      --start 0 --take 4 --val-take 0 \
  --source data/pretrain/cosmopedia3 --start 0 --take 3 --val-take 0 \
  --source data/pretrain/r5_code     --start 0 --take 2 --val-take 0 \
  --source data/pretrain/r5_math     --start 0 --take 1 --val-take 0 \
  --source data/pretrain/r5_wiki     --start 0 --take 1 --val-take 0
```

| Pool | Source shard ranges (100M-token shards) |
|---|---|
| main_s1 (2.00B) | fineweb3[8–14], p5_web[0–1], r5_web[0–3], cosmopedia3[0–2], r5_code[0–1], r5_math[0], r5_wiki[0] |
| main_s2 (1.94B) | fineweb3[15–21], p5_web[2–3], r5_web[4–7], cosmopedia3[4], p5_synth[0], r5_code[2–4], r5_math[1], r5_wiki[1] |
| main_s3 (2.20B) | fineweb3[22–28], p5_web[4–5], r5_web[8–11], p5_synth[1–3], r5_code[5–7], r5_math[2], r5_wiki[2–3] |
| main_s4 (1.91B) | fineweb3[29–35], p5_web[6–7], r5_web[12–15], p5_synth[4–5], r5_code[8–9], r5_math[3–4], r5_wiki[4] |

`p5_web` is an additional DCLM slice; `p5_synth` shares the Cosmopedia v2 schema.

Source datasets: FineWeb-Edu [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
(`sample-10BT`); Cosmopedia v2 [`HuggingFaceTB/cosmopedia-v2`](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-v2);
DCLM web [`mlfoundations/dclm-baseline-1.0`](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) (`.jsonl.zst`);
code [`nvidia/OpenCodeInstruct`](https://huggingface.co/datasets/nvidia/OpenCodeInstruct);
Wikipedia [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) (`20231101.en`).

## Training

Four ~2B-token segments, each with its own complete WSD schedule (warmup 700 → stable →
linear decay over the final 10%); the segment boundary is the anneal point.

| Seg | Tokens (cum.) | Steps | Wall | Peak LR | val PPL |
|---|---|---|---|---|---|
| S1 | 2.00B | 40,640 | 38.8 h | 5e-4 | 17.16 @40k |
| S2 | 3.94B | 39,421 | 37.6 h | 5e-4 | 16.22 @38k |
| S3 | 6.14B | 44,704 | 42.8 h | 4e-4 | 15.81 @44k |
| S4 | 8.05B | 38,811 | 37.1 h | 3e-4 | 15.59 @38k |

- bf16 weights + autocast, bf16 optimizer states (`--bf16-optim`), chunked cross-entropy, seed 42
- batch 48 × 1024 = 49,152 tokens/step, ~14.3k tok/s
- gradient clipping 1.0, AdamW (β=0.9, 0.95, wd 0.1), hourly keep-last-2 checkpoints

## Results (lm-evaluation-harness v0.4.12, 0-shot)

v3.0 (S4, 8.05B tokens) vs the previous releases:

| Task | Metric | v3.0 (477M/276M) | 1B MoE 8B (1182M/352M) | v1.1 (432M) | v2.0 beta (498M) |
|------|--------|:---:|:---:|:---:|:---:|
| HellaSwag | acc_norm | **0.335** | 0.329 | 0.308 | 0.326 |
| PIQA | acc | **0.638** | 0.630 | 0.616 | 0.631 |
| WinoGrande | acc | 0.515 | 0.523 | **0.524** | 0.506 |
| ARC-Easy | acc | 0.478 | **0.479** | 0.456 | 0.474 |
| ARC-Challenge | acc_norm | 0.255 | **0.279** | 0.247 | 0.272 |
| OpenBookQA | acc_norm | 0.296 | **0.306** | 0.288 | 0.290 |
| BoolQ | acc | 0.615 | **0.620** | 0.606 | 0.455 |
| LAMBADA | acc | **0.268** | 0.234 | 0.227 | 0.224 |
| **Mean** | — | **0.4250** | 0.425 | 0.409 | 0.397 |

**Data efficiency:** v3.0 matches the 1B MoE trained on the same 8B tokens using **2.5× fewer
total parameters and 1.28× fewer active parameters** (~1.3× fewer FLOPs per token).

Notes:
- Persistent weak spot: ARC-Challenge (0.255 vs 0.279 for the 1B MoE) — long-training sensitive,
  but it narrowed from −3.2pp at 6B to −2.4pp at 8B.
- v3.0 vs 6B (S3, mean 0.4257) is within noise; S4 is best on the most tasks.
- Early (10k–14k) and late (32k–34k) val-PPL bumps were transient low-entropy shard effects
  that self-healed (36k=16.31 → 38k=15.59); no data-pool change was needed.

## Quick Start (training)

See `versions/v3.0/scripts/run_segment.ps1` (Windows) for the 4-segment launcher, or drive `scripts/train.py`
/ `scripts/resume.py` directly. Publishes to HF with `scripts/publish_hf.py`.

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "mikecovlee/tinymixtral", trust_remote_code=True
)
```
