#!/usr/bin/env bash
set -u
# Usage: run_sft.sh <v1|v2|v3|v4> [init_ckpt_for_v4]
#   v1-v3: SFT ladder from checkpoints/base_v3_raw (195k / 857k / 2.17M rows)
#   v4:    50k high-quality polish at lr 5e-6, requires an init checkpoint
SCALE="${1:?usage: run_sft.sh <v1|v2|v3|v4> [init_ckpt_for_v4]}"
INIT="${2:-}"
source /home/mikecovlee/miniconda3/etc/profile.d/conda.sh
conda activate tinymixtral-cudev
cd /home/mikecovlee/work/tinymixtral
SNAP=$(ls -d /home/mikecovlee/.cache/huggingface/hub/models--mikecovlee--tinymixtral/snapshots/*/ | head -1)

case "$SCALE" in
  v1) CKPT="checkpoints/base_v3_raw"; DATA="data/sft_v2_v1/train.parquet"; MAXS=195170;  LR=2e-5; OUT="checkpoints/sft_v2_v1"; SAVE=1000; LOG=100 ;;
  v2) CKPT="checkpoints/base_v3_raw"; DATA="data/sft_v2_v2/train.parquet"; MAXS=856805;  LR=2e-5; OUT="checkpoints/sft_v2_v2"; SAVE=1000; LOG=100 ;;
  v3) CKPT="checkpoints/base_v3_raw"; DATA="data/sft_v2_v3/train.parquet"; MAXS=2168835; LR=2e-5; OUT="checkpoints/sft_v2_v3"; SAVE=1000; LOG=100 ;;
  v4)
    [ -n "$INIT" ] || { echo "run_sft.sh v4 requires <init_ckpt>" >&2; exit 1; }
    CKPT="$INIT"; DATA="data/sft_v4/train.parquet"; MAXS=50000; LR=5e-6; OUT="checkpoints/sft_v4"; SAVE=500; LOG=50 ;;
  *) echo "unknown scale '$SCALE' (expected v1|v2|v3|v4)" >&2; exit 1 ;;
esac

MARK="SFT$(echo "$SCALE" | tr '[:lower:]' '[:upper:]')"
echo "=== ${MARK} START $(date +%T) ==="
python scripts/train_sft.py --checkpoint "$CKPT" --tokenizer-path "$SNAP" \
  --dataset "$DATA" --max-samples "$MAXS" --epochs 1 --seq-len 1024 \
  --lr "$LR" --batch-size 24 --output-dir "$OUT" --save-every "$SAVE" --log-every "$LOG" \
  > "dpo_local2/sft${SCALE}_train.log" 2>&1
echo "=== ${MARK} rc=$? $(date +%T) ==="
echo "${MARK}_DONE"
