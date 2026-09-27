# v3.0-it Reproduction Guide (v3.0 base → v3.0-it)

This document lists every command and parameter needed to reproduce **v3.0-it**
(the delivered arm of the v3.0-it SFT campaign) from scratch. All scripts live on
branch `sft` (commits 966fead / af313a3 / 66728d6 plus the later tooling commits).

## 0. Environment

- **Training box**: Linux, single GPU ≥24 GB (measured: RTX PRO 4500 32 GB), 60 GB RAM,
  conda environment per `.env.example` (`CONDA_ENV`), with torch 2.14.0+cu130,
  transformers 4.57.6, pyarrow, numpy, safetensors 0.8.0, lm_eval 0.4.12.
- **Evaluation box (offload)**: any box with a 24 GB-class GPU (measured: A5000),
  a Python env with torch + lm_eval (`EVAL_PY`), and DeepSeek API access via the
  `DEEPSEEK_API_KEY` environment variable (optional fallback:
  `~/.local/share/opencode/auth.json`, `deepseek` entry). Run long jobs under tmux
  with shipped `.ps1`/`.sh` files (never inline nested quotes).
- **Base model**: v3.0 base, either source works:
  - HF cache snapshot of the public repo `mikecovlee/tinymixtral` at revision
    `6e0792c1781d3c704c9f9a3844662998795b306c` (the tokenizer comes from here too;
    the scripts resolve it via `TOKENIZER_SNAP`), or
  - raw checkpoint `checkpoints/base_v3_raw/` (`config.json` + `pytorch_model.bin`).
- HF downloads may need a proxy: `export HTTPS_PROXY=http://<EVAL_HOST>:<PROXY_PORT>`.

## 1. Data build (scripts/, on the training box)

```bash
python scripts/data/prefetch_sources.py --out data/sft_src            # stage all 10 sources (large HF downloads)
python scripts/data/build_dataset.py --out-dir data/sft_200k --scale v1 # target 200k -> actual 195,170 rows
python scripts/data/build_dataset.py --out-dir data/sft_1m --scale v2 # target 1M   -> actual 856,805 rows
python scripts/data/build_dataset.py --out-dir data/sft_3m --scale v3 # target 3M   -> actual 2,168,835 rows
```

Key points (`data/build_dataset.py`):
- Quota shares (SHARES): tulu3 .32 / openhermes .16 / slimorca .12 / openorca .12 /
  ultrachat .11 / metamath .06 / orcamath .04 / omi2 .03 / squad2 .02 / trivia .02;
  seed 42; no Chinese data.
- Filtering: assistant reply 40–12,000 chars (~10–2048 tokens); per-source cap ≤15%.
- Dedup: exact hash + MinHash-LSH (Jaccard ≥ 0.8).
- Decontamination: n-gram removal against gsm8k / arc / openbookqa / hellaswag / piqa /
  ifeval / mmlu / ceval(cmmlu) and the held-out set (`--decontam` on by default).
- Outputs: `train.parquet / dev.parquet / heldout_prompts.parquet / stats.json / LICENSE_NOTES`.
- Runtime reference: v2 ≈ 40 min, v3 ≈ 42 min (see `elapsed_s` in stats.json).
- The fixed evaluation prompt set ships with the repo:
  **`eval_prompts/heldout_prompts_id_1k5.parquet`** (4,955 rows, columns `id/prompt/ntok`,
  sha256 `6f8e48e67c592f830615ecc942b46aee3090eec3f827838227db0a5c12b2fdc8`).
  Copy it to the eval box under `data/sft_200k/`; every arm uses the same file so the
  paired comparisons stay valid.

## 2. Training (versions/v3.0-it/run_sft.sh, started from the base — no warm start)

```bash
CONDA_ENV=<env> bash versions/v3.0-it/run_sft.sh 200k   # 195,170 rows -> 106,718 packed 1024-seqs -> 4,447 steps, ~99 min
CONDA_ENV=<env> bash versions/v3.0-it/run_sft.sh 1m   # 856,805 rows -> 644,557 seqs -> 26,857 steps, ~10.0 h
CONDA_ENV=<env> bash versions/v3.0-it/run_sft.sh 3m   # 2,168,835 rows -> 1,443,804 seqs -> 60,159 steps, ~22.3 h
```

Unified hyperparameters (train_sft.py): `--epochs 1 --seq-len 1024 --lr 2e-5`
(cosine + 3% warmup), `--batch-size 24 --wd 0.1`, bf16 autocast, gradient
checkpointing, `--save-every 1000 --log-every 100`; output under
`checkpoints/sft_{200k,1m,3m}/step_*_final`.

Notes (see lessons 8.2 in the report):
- `versions/v3.0-it/train_sft.py` carries the **numpy int32 memory patch** (tokenize/pack no
  longer builds Python int lists); without it 1M/3M blew past 40 GB at
  `Packing...` and thrashed swap. RSS went from ~40 GB to ~21 GB with the patch.
- Late-training process RSS of ~42 GB is glibc arena retention — normal, do not kill.
- Measured throughput ~0.745 steps/s (seq 1024, bs 24). The runner prints
  `SFTV<N>_DONE` markers; chaining these markers with small polling waiters gives an
  unattended cascade (that is how the campaign actually ran; the ad-hoc waiters were
  campaign scaffolding and are not shipped).
- `--seed`, `--resume`, `--grad-accum` and `--keep-last` are supported by
  train_sft.py (verified by an on-GPU smoke: seeded reruns produce identical loss
  sequences; resume replays the exact data position).

## 3. Evaluation chain (publish → transfer → eval box gen/rubric/lm-eval → pull back → table)

```bash
EVAL_USER=<user> EVAL_HOST=<host> EVAL_REPO=<remote-repo-root> \
  bash versions/v3.0-it/eval/offload_arm.sh v3.0-it sftv2v3 checkpoints/sft_3m/step_0060159_final
#   -> publish/v3.0-it (pytorch_model.bin + whitelisted tokenizer files; asserts no model.safetensors)
#   -> scp to the eval box publish/ and print the tmux launch command
# On the eval box (Windows example):
tmux new-session -d -s sftv2v3 "powershell -NoProfile -ExecutionPolicy Bypass -File <repo>\versions\v3.0-it\eval\run_offload_arm.ps1 -Arm v3.0-it -Tag sftv2v3"
# On the eval box (Linux equivalent):
bash versions/v3.0-it/eval/run_offload_arm.sh v3.0-it sftv2v3
#   Both run: GEN3 generation (dpo_eval_judge.py gen, 4,955 held-out prompts) -> RUBRIC3
#   (rubric_judge2.py, deepseek-flash 0-100, 4 dimensions, concurrency 8)
#   -> HARNESS / IFEVAL / GSM8K (lm_eval); completion marker OFFLOAD_ARM_DONE.
#   Poll the marker, then copy data/dpo + evals JSONs back.
python versions/v3.0-it/eval/final_table.py --dir data/dpo                # markdown comparison table (paired rubric t-test + 3 lm-evals + canonical harness)
python versions/v3.0-it/eval/summarize_evals.py --dir data/dpo --detailed # per-task detail
```

Preregistered methodology (see lessons 8.3 in the report):
- **Rubric**: same 4,955 held-out prompts, same judge (rubric_judge2 + deepseek-flash),
  **per-item paired** t-test against the reference arm.
- **Canonical harness formula**: acc_norm for hellaswag/piqa/arc_challenge/openbookqa,
  acc for winogrande/arc_easy/boolq/lambada, simple mean of the 8 (base v3.0 = 0.4250).
- rubric_judge2 **silently skips** items whose API call fails after 4 retries — always
  verify output row count == prompt count (polish once missed 2,036 rows; root cause API 402;
  fix: `--resume` fills exactly the missing ids).

## 4. Expected results (v3.0 base → v3.0-it)

| Metric | base v3.0 | baseline | it-200k | it-1m | **v3.0-it** |
|---|---|---|---|---|---|
| rubric (4,955, paired vs baseline) | not run (new set) | 6.40±0.18 | 11.44±0.26 (+5.04, t=+19.2) | 11.97±0.24 (+5.56, t=+22.4) | **15.01±0.28 (+8.61, t=+30.7)** |
| IFEval prompt/inst-strict | — | 0.0924/0.1894 | 0.0961/0.2014 | 0.1091/0.2026 | **0.1701/0.2794** |
| GSM8K strict/flexible | — | —/0.0167 | 0.0159/0.0265 | 0.0174/0.0197 | 0.0205/**0.0227** |
| 8-task harness (canonical) | 0.4250 | 0.4260* | 0.4210 | 0.4158 | 0.4034 (boolq 0.4263 is the main drag) |

\* baseline 0.4260 is a pre-campaign recorded value whose formula is unrecoverable; reference only.

Known trade-off: data scaling greatly improves instruction following and open-ended
quality (rubric, IFEval) but regresses basic discrimination tasks; boolq is the most
sensitive (0.615→0.426). The 50k-polish (50k rows, lr 5e-6, init=3M) verified as no
gain (paired −0.15, t=−0.8) and is not needed to reproduce 3M. See
`docs/SFT_V3_REPORT.md` (incl. §8 Lessons Learned).

## 5. One-click cascade reference

The campaign ran unattended by chaining marker → waiter: run_sft.sh 1m → (SFT_1M_DONE)
→ v3 → (SFT_3M_DONE) → publish+scp+tmux eval → (OFFLOAD_ARM_DONE) → pull artifacts →
final_table. Reproduction can simply run each step above manually; the ad-hoc waiter
scripts were campaign scaffolding and were removed from the tree (kept in git history).
