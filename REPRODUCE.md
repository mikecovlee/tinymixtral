# Reproducing TinyMixtral

End-to-end reproduction guide for all released versions. Each version has a self-contained card
under [`versions/`](versions/) (architecture, config, recipe, per-task results); this file covers
the shared pipeline and the per-version differences.

## 0. Environment

```bash
conda create -n tinymixtral python=3.13 -y
conda activate tinymixtral
pip install -r requirements.txt
```

## 1. Tokenizer

All versions use the same 32k SentencePiece tokenizer (trained on the TinyLlama chat corpus):

```bash
python scripts/prepare_tokenizer.py --from-hf TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output tokenizer/
```

## 2. Data

Two steps: **tokenize** each source to 100M-token `.pt` shards (`scripts/prepare_data.py`), then
**blend** the shards — `scripts/mix_data.py` for a simple 2-source interleave, or
`scripts/make_blend_shards.py` for exact-ratio, slice-controlled, validation-isolated pools.

Large web/code corpora are fetched as whole files rather than streamed:
`scripts/download_parquets.py` (HF parquet dirs) or `scripts/download_jsonl_zst.py` (DCLM
`.jsonl.zst`), then `scripts/zst_jsonl_to_parquet.py` / `scripts/columns_to_text_parquet.py`
normalize them to a single `text` column before `scripts/prepare_data_local.py` tokenizes locally.

**Data-prep conventions (important):**

- The local route uses a fixed double extension: `download_parquets.py` and
  `zst_jsonl_to_parquet.py` write `<name>.parquet.parquet`, and `prepare_data_local.py` globs
  `*.parquet.parquet`. `columns_to_text_parquet.py` preserves its input filename, so only feed it
  outputs from those two tools — a hand-supplied single-extension `*.parquet` becomes a file that
  `prepare_data_local.py` silently ignores.
- `prepare_data_local.py` reads the `text` column only. Pools whose text is split across columns
  (e.g. code: `input` + `output`) must be normalized first with
  `columns_to_text_parquet.py --columns input output`.
- `download_jsonl_zst.py` downloads one subdir per run; for nested trees (DCLM) pass the full
  local-shard path, e.g. `global-shard_01_of_10/local-shard_0_of_10`.
- `mix_data.py` copies shards by default; `make_blend_shards.py` hard-links (and can hold out
  validation shards via `--val-take`).

| Version | Data | Sources |
|---|---|---|
| v3.0 | 8.05B tokens, four disjoint pools | FineWeb-Edu, DCLM web, Cosmopedia v2, code, math (OpenWebMath), Wikipedia (6-source blend) |
| v1.1-1b | 8B (4B + 4B continuation) | FineWeb-Edu + Cosmopedia v2 (89:11) |
| v2.0-beta | 4B (+1B post-train) | FineWeb-Edu + Cosmopedia v2 (89:11) |
| v1.1 | 4B | FineWeb-Edu + Cosmopedia v2 (89:11) |
| v1.0 | 4B (+1B post-train) | C4-en (legacy) |

Exact dataset ids and commands are in each version card (`versions/<ver>/README.md`) and in the
v3.0 card's "Reproduce the data pools" section.

### Dataset sources

| Source | HF id | Used by |
|---|---|---|
| FineWeb-Edu | [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`) | v3.0, v1.1-1b, v2.0-beta, v1.1, v1.0 post-train |
| Cosmopedia v2 | [`HuggingFaceTB/cosmopedia-v2`](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-v2) | v3.0, v1.1-1b, v2.0-beta, v1.1, v1.0 post-train |
| DCLM | [`mlfoundations/dclm-baseline-1.0`](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | v3.0 |
| Code | [`nvidia/OpenCodeInstruct`](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | v3.0 |
| Math | [`open-web-math/open-web-math`](https://huggingface.co/datasets/open-web-math/open-web-math) | v3.0 |
| Wikipedia | [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) (`20231101.en`) | v3.0, v1.1-1b post-train, v2.0-beta post-train |
| C4 | [`allenai/c4`](https://huggingface.co/datasets/allenai/c4) (`en`) | v1.0 pretrain |

## 3. Train

```bash
python scripts/train.py --config versions/<ver>/configs/<config>.json \
  --cache-dir data/pretrain/<blend> \
  --batch-size <bs> --max-tokens <n> --lr <lr> --warmup-steps <w>
```

- Config file names differ per version: `config.json` (v1.0, v1.1, v2.0-beta),
  `v1b_moe.json` (v1.1-1b), `improve_v05b*.json` (v3.0).
- Resume / continue from a checkpoint with `scripts/resume.py` (the architecture is read from the
  checkpoint's saved `config.json`).
- v3.0 additionally ships its 4-segment launcher `versions/v3.0/scripts/run_segment.ps1`.
- v2.0-beta uses the frozen [`shared_expert/`](shared_expert/) stack (self-contained snapshot).

## 4. Publish + evaluate

```bash
python scripts/publish_hf.py --checkpoint checkpoints/<run>/<step>_final \
  --output publish/<name> --tokenizer tokenizer/

lm_eval --model hf \
  --model_args "pretrained=publish/<name>,tokenizer=tokenizer/,trust_remote_code=True,dtype=bfloat16" \
  --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq,lambada_openai \
  --batch_size 16 --device cuda
```

## Per-version notes

- **v3.0** — mainline; four ~2B-token segments each with its own WSD schedule, LR ladder
  5e-4 / 5e-4 / 4e-4 / 3e-4; QK-Norm kept; 477.5M total / 276.1M active.
- **v1.1-1b** — 8 routed experts (top-2), intermediate 2816; 8B tokens; previous flagship.
- **v2.0-beta** — shared-expert architecture ablation; negative result (see the card).
- **v1.1 / v1.0** — identical 6-expert trunk; differ only in data (SmolLM blend vs C4-en).
