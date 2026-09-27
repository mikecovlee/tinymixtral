# v3.0-it Reproduction Guide (v3.0 base → v3.0-it)

This document lists every command and parameter needed to reproduce **v3.0-it**
from scratch. All scripts live under `versions/v3.0-it/` (branch `sft`); run every
command from the repo root.

## 0. Environment

- **Training box**: Linux, single GPU ≥24 GB (measured: RTX PRO 4500 32 GB), 60 GB RAM,
  conda environment per `.env.example` (`CONDA_ENV`), with torch 2.14.0+cu130,
  transformers 4.57.6, pyarrow, numpy, safetensors 0.8.0, lm_eval 0.4.12.
- **Evaluation**: the same box is enough (any 24 GB-class GPU; the campaign also
  used an A5000 on a second machine - see the optional note in §3). Needs the same
  Python env plus `lm_eval`, and a DeepSeek API key in `DEEPSEEK_API_KEY`. Evaluation is plain shell commands (see §3).
- **Base model**: v3.0 base, either source works:
  - HF cache snapshot of the public repo `mikecovlee/tinymixtral` at revision
    `6e0792c1781d3c704c9f9a3844662998795b306c` (the tokenizer comes from here too;
    the scripts resolve it via `TOKENIZER_SNAP`), or
  - raw checkpoint `checkpoints/base_v3_raw/` (`config.json` + `pytorch_model.bin`).
- HF downloads may need a proxy: `export HTTPS_PROXY=http://<proxy-host>:<port>`.

## 1. Data build (versions/v3.0-it/data/, on the training box)

```bash
python versions/v3.0-it/data/prefetch_sources.py --out data/sft_src  # stage all 10 sources (large HF downloads)

PROMPTS=versions/v3.0-it/eval_prompts/heldout_prompts_5k.parquet
python versions/v3.0-it/data/build_dataset.py --out-dir data/sft_200k --scale 200k --extra-holdout $PROMPTS  # -> 195,170 rows
python versions/v3.0-it/data/build_dataset.py --out-dir data/sft_1m  --scale 1m  --extra-holdout $PROMPTS  # -> 856,805 rows
python versions/v3.0-it/data/build_dataset.py --out-dir data/sft_3m  --scale 3m  --extra-holdout $PROMPTS  # -> 2,168,835 rows

# optional: 50k polish-tier slice (stratified from the 3M train set; --scale v1|v2|v3 also accepted)
python versions/v3.0-it/data/sample_subset.py --src data/sft_3m/train.parquet --out data/sft_polish
```

Key points (`data/build_dataset.py`):
- Quota shares (SHARES): tulu3 .32 / openhermes .16 / slimorca .12 / openorca .12 /
  ultrachat .11 / metamath .06 / orcamath .04 / omi2 .03 / squad2 .02 / trivia .02;
  seed 42; no Chinese data.
- Filtering: assistant reply 40–12,000 chars (~10–2048 tokens); per-source cap ≤15%.
- Dedup: exact hash + MinHash-LSH (Jaccard ≥ 0.8).
- Decontamination: 10-gram removal against 6 eval sets (gsm8k, ARC-Challenge,
  ARC-Easy, OpenBookQA, HELLASWAG, PIQA) plus the fixed eval prompt set passed via
  `--extra-holdout` (`--decontam` on by default).
- Outputs: `train.parquet / dev.parquet / heldout_prompts.parquet / stats.json`
  (per-source dataset licenses: `docs/DATA_LICENSES.md`).
- Runtime reference: v2 ≈ 40 min, v3 ≈ 42 min (see `elapsed_s` in stats.json).
- The fixed evaluation prompt set ships with the repo:
  **`eval_prompts/heldout_prompts_5k.parquet`** (4,955 rows, columns `id/prompt/ntok`,
  sha256 `6f8e48e67c592f830615ecc942b46aee3090eec3f827838227db0a5c12b2fdc8`).
  Every model is evaluated on this same file so the paired comparisons stay valid.

## 2. Training (versions/v3.0-it/run_sft.sh, started from the base — no warm start)

```bash
CONDA_ENV=<env> bash versions/v3.0-it/run_sft.sh 200k   # 195,170 rows -> 106,718 packed 1024-seqs -> 4,447 steps, ~99 min
CONDA_ENV=<env> bash versions/v3.0-it/run_sft.sh 1m   # 856,805 rows -> 644,557 seqs -> 26,857 steps, ~10.0 h
CONDA_ENV=<env> bash versions/v3.0-it/run_sft.sh 3m     # 2,168,835 rows -> 1,443,804 seqs -> 60,159 steps, ~22.3 h
# optional polish tier on top of a finished run:
# CONDA_ENV=<env> bash versions/v3.0-it/run_sft.sh polish checkpoints/sft_3m/step_*_final
```

`run_sft.sh` requires `CONDA_ENV`; `TOKENIZER_SNAP` (tokenizer dir or HF snapshot)
is auto-resolved from the local `mikecovlee/tinymixtral` snapshot if unset.

Unified hyperparameters (train_sft.py): `--epochs 1 --seq-len 1024 --lr 2e-5`
(cosine + 100-step warmup, `--warmup-steps 100`), `--batch-size 24 --wd 0.1`, bf16 autocast, gradient
checkpointing, `--save-every 1000 --log-every 100`; output under
`checkpoints/sft_{200k,1m,3m}/step_*_final`.

Notes (see lessons 8.2 in the report):
- `versions/v3.0-it/train_sft.py` carries the **numpy int32 memory patch** (tokenize/pack no
  longer builds Python int lists); without it 1M/3M blew past 40 GB at
  `Packing...` and thrashed swap. RSS went from ~40 GB to ~21 GB with the patch.
- Late-training process RSS of ~42 GB is glibc arena retention — normal, do not kill.
- Measured throughput ~0.745 steps/s (seq 1024, bs 24). The runner prints
  `SFT_<scale>_DONE` markers (e.g. `SFT_3m_DONE`) that can be used to chain jobs
  in any scheduler.
- `--seed`, `--resume`, `--grad-accum` and `--keep-last` are supported by
  train_sft.py (verified by an on-GPU smoke: seeded reruns produce identical loss
  sequences; resume replays the exact data position).

## 3. Evaluation chain (publish -> gen/rubric/lm-eval -> table, single machine)

```bash
# 1) Export the trained checkpoint to HF format (tokenizer whitelist; asserts no model.safetensors)
python scripts/publish_hf.py --checkpoint checkpoints/sft_3m/step_0060159_final \
  --output publish/v3.0-it --tokenizer <tokenizer-dir>

# 2) Generate responses on the fixed 4,955-prompt held-out set
python versions/v3.0-it/eval/response_eval.py gen --model publish/v3.0-it \
  --prompts versions/v3.0-it/eval_prompts/heldout_prompts_5k.parquet \
  --out data/eval/gen_v3.0-it.jsonl --batch-size 8 --max-new-tokens 448

# 3) Rubric scoring (deepseek-flash, 0-100, 4 dimensions; needs DEEPSEEK_API_KEY)
python versions/v3.0-it/judge/rubric_judge.py --responses data/eval/gen_v3.0-it.jsonl \
  --out data/eval/rubric_v3.0-it.jsonl --limit 5000 --concurrency 8

# 4) lm-eval: 7-task harness / ifeval / gsm8k
python -m lm_eval --model hf \
  --model_args pretrained=publish/v3.0-it,tokenizer=publish/v3.0-it,trust_remote_code=True,dtype=bfloat16 \
  --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa,lambada_openai \
  --batch_size 16 --device cuda --output_path evals/harness/v3.0-it
#   ... same with --tasks ifeval --apply_chat_template --batch_size 8 -> evals/ifeval/v3.0-it
#   ... and with --tasks gsm8k --batch_size 8 -> evals/gsm8k/v3.0-it

# 5) Consolidate (copy each newest evals/<task>/<model>/.../results_*.json to
#    data/eval/<task>_<model>.json first)
python versions/v3.0-it/eval/final_table.py --dir data/eval                # markdown comparison table (paired rubric t-test + 3 lm-evals + canonical harness)
python versions/v3.0-it/eval/summarize_evals.py --dir data/eval --detailed # per-task detail
```

Optional pairwise win-rate between two models:
`response_eval.py judge --a data/eval/gen_<A>.jsonl --b data/eval/gen_<B>.jsonl`
(reports B win-rate with both judge orders to control position bias).

**Optional: multi-machine evaluation.** If the training box has no GPU to spare, copy
the `publish/<model>` directory (~1.9 GB) to any CUDA box (24 GB class is enough; the
campaign used a Windows A5000), run steps 2-4 there (step 3 needs `DEEPSEEK_API_KEY`
on that box), then copy the `data/eval` artifacts back and run step 5 locally. No
special scripts are required for this split.

Preregistered methodology (see lessons 8.3 in the report):
- **Rubric**: same 4,955 held-out prompts, same judge (`rubric_judge.py` + deepseek-flash),
  **per-item paired** t-test against the reference model.
- **Canonical harness formula**: acc_norm for hellaswag/piqa/arc_challenge/openbookqa,
  acc for winogrande/arc_easy/lambada, simple mean of the 7 (base v3.0 = 0.3979).
- `rubric_judge.py` **silently skips** items whose API call fails after 4 retries — always
  verify output row count == prompt count (polish once missed 2,036 rows; root cause API 402;
  fix: `--resume` fills exactly the missing ids).

## 4. Expected results (v3.0 base → v3.0-it)

| Metric | base v3.0 | baseline | it-200k | it-1m | **v3.0-it** |
|---|---|---|---|---|---|
| rubric (4,955, paired vs baseline) | not run (new set) | 6.40±0.18 | 11.44±0.26 (+5.04, t=+19.2) | 11.97±0.24 (+5.56, t=+22.4) | **15.01±0.28 (+8.61, t=+30.7)** |
| IFEval prompt/inst-strict | — | 0.0924/0.1894 | 0.0961/0.2014 | 0.1091/0.2026 | **0.1701/0.2794** |
| GSM8K strict/flexible | — | —/0.0167 | 0.0159/0.0265 | 0.0174/0.0197 | 0.0205/**0.0227** |
| 7-task harness (canonical) | 0.3979 | not run* | 0.3983 | 0.3962 | **0.4002** |

\* the only recorded baseline harness number used an unrecoverable legacy formula (pre-dating
the BoolQ removal); not comparable with this table.

Instruction following and open-ended quality (rubric, IFEval) improve sharply with scale
while the 7-task harness stays within ±0.23pp of the base (the 3M tier leads slightly).
BoolQ was dropped from the suite as an unstable sentinel (see `REPORT.md` §8.5). The
50k-polish (50k rows, lr 5e-6, init=3M) verified as no
gain (paired −0.15, t=−0.8) and is not needed to reproduce 3M. See
`REPORT.md` (incl. §8 Lessons Learned).

## 5. Chaining steps

The runner's `SFT_<scale>_DONE` markers (plus the `rc=` lines each script prints)
can be consumed by any scheduler or a few lines of polling glue to run the ladder
unattended: run_sft.sh 200k → 1m → 3m → publish → gen/rubric/lm-eval → final_table.
Each step can equally be run manually.
