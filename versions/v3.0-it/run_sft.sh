#!/usr/bin/env bash
set -u
# Usage: run_sft.sh <200k|1m|3m|polish> [init_ckpt_for_polish]
#   200k/1m/3m: SFT ladder from checkpoints/base_v3_raw (195k / 857k / 2.17M rows)
#   polish: 50k high-quality polish at lr 5e-6, requires an init checkpoint
# Env: CONDA_BASE (default $HOME/miniconda3), CONDA_ENV (required),
#      TOKENIZER_SNAP (default: latest local mikecovlee/tinymixtral snapshot)
SCALE="${1:?usage: run_sft.sh <200k|1m|3m|polish> [init_ckpt_for_polish]}"
INIT="${2:-}"
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:?set CONDA_ENV to the training conda environment}"
cd "$REPO_ROOT"
mkdir -p logs
if [ -z "${TOKENIZER_SNAP:-}" ]; then
  TOKENIZER_SNAP=$(ls -d "$HOME"/.cache/huggingface/hub/models--mikecovlee--tinymixtral/snapshots/*/ | head -1)
fi

case "$SCALE" in
  200k) CKPT="checkpoints/base_v3_raw"; DATA="data/sft_200k/train.parquet"; MAXS=195170;  LR=2e-5; OUT="checkpoints/sft_200k"; SAVE=1000; LOG=100 ;;
  1m) CKPT="checkpoints/base_v3_raw"; DATA="data/sft_1m/train.parquet"; MAXS=856805;  LR=2e-5; OUT="checkpoints/sft_1m"; SAVE=1000; LOG=100 ;;
  3m) CKPT="checkpoints/base_v3_raw"; DATA="data/sft_3m/train.parquet"; MAXS=2168835; LR=2e-5; OUT="checkpoints/sft_3m"; SAVE=1000; LOG=100 ;;
  polish)
    [ -n "$INIT" ] || { echo "run_sft.sh polish requires <init_ckpt>" >&2; exit 1; }
    CKPT="$INIT"; DATA="data/sft_polish/train.parquet"; MAXS=50000; LR=5e-6; OUT="checkpoints/sft_polish"; SAVE=500; LOG=50 ;;
  *) echo "unknown scale '$SCALE' (expected 200k|1m|3m|polish)" >&2; exit 1 ;;
esac

MARK="SFT_${SCALE}"
echo "=== ${MARK} START $(date +%T) ==="
python "$REPO_ROOT/versions/v3.0-it/train_sft.py" --checkpoint "$CKPT" --tokenizer-path "$TOKENIZER_SNAP" \
  --dataset "$DATA" --max-samples "$MAXS" --epochs 1 --seq-len 1024 \
  --lr "$LR" --batch-size 24 --output-dir "$OUT" --save-every "$SAVE" --log-every "$LOG" \
  > "logs/sft_${SCALE}_train.log" 2>&1
echo "=== ${MARK} rc=$? $(date +%T) ==="
echo "${MARK}_DONE"
