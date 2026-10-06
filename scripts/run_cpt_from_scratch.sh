#!/usr/bin/env bash
set -euo pipefail

# CPT-from-scratch: v3.0 recipe + CPT router (energy_init_scale=0.3)
# Reference: versions/v3.0/README.md COMMON + segment commands
# Data: NAS v3.0-dense-active/data/pretrain/ (main_s1..s4 + pilot_blend30_val)

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-/mnt/nas/jetpack-hdd/tinymixtral-improve/v3.0-dense-active/data/pretrain}"
VAL="${VAL:-$DATA/pilot_blend30_val}"
CFG="$REPO/configs/cpt_from_scratch/cpt_v3_0_from_scratch.json"
OUT="${OUT:-$REPO/checkpoints/cpt_from_scratch}"
PYTHON="${PYTHON:-/home/mikecovlee/miniconda3/envs/tinymixtral-cudev/bin/python}"

COMMON="--schedule wsd --warmup-steps 700 --batch-size 48 --bf16-optim --chunked-ce \
  --save-every-min 60 --log-every 100 --eval-every-steps 2000 --keep-last-checkpoints 2 \
  --val-dir $VAL --seq-len 1024 --wd 0.1"

mkdir -p "$OUT"

case "${1:-s1}" in
  s1)
    echo "=== CPT S1: fresh start, main_s1 (2.00B), lr 5e-4 ==="
    $PYTHON "$REPO/scripts/train.py" \
      --config "$CFG" \
      --cache-dir "$DATA/main_s1" \
      --output-dir "$OUT/s1" \
      --max-tokens 2000000000 --lr 5e-4 --seed 42 $COMMON
    ;;
  s2)
    echo "=== CPT S2: resume from s1, main_s2 (1.94B), lr 5e-4 ==="
    $PYTHON "$REPO/scripts/resume.py" \
      --checkpoint-dir "$OUT/s1" --cache-dir "$DATA/main_s2" \
      --output-dir "$OUT/s2" \
      --max-tokens 1940000000 --lr 5e-4 $COMMON
    ;;
  s3)
    echo "=== CPT S3: resume from s2, main_s3 (2.20B), lr 4e-4 ==="
    $PYTHON "$REPO/scripts/resume.py" \
      --checkpoint-dir "$OUT/s2" --cache-dir "$DATA/main_s3" \
      --output-dir "$OUT/s3" \
      --max-tokens 2200000000 --lr 4e-4 $COMMON
    ;;
  s4)
    echo "=== CPT S4: resume from s3, main_s4 (1.91B), lr 3e-4 ==="
    $PYTHON "$REPO/scripts/resume.py" \
      --checkpoint-dir "$OUT/s3" --cache-dir "$DATA/main_s4" \
      --output-dir "$OUT/s4" \
      --max-tokens 1910000000 --lr 3e-4 $COMMON
    ;;
  all)
    echo "=== CPT full 4-segment run ==="
    "$0" s1 && "$0" s2 && "$0" s3 && "$0" s4
    ;;
  *)
    echo "Usage: $0 {s1|s2|s3|s4|all}"
    exit 1
    ;;
esac
