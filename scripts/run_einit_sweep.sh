#!/usr/bin/env bash
set -euo pipefail
# CPT energy_init_scale scan: 4 runs on the 8L proxy, ~50min each, total ~3.5h.
#
# Motivation: the CPT routing probability is pi = B^T q with no final softmax, so pi is a
# convex combination of B's 8 rows and pi_max is hard-capped by max B[k,j]. At the current
# energy_init_scale=0.3 that cap is 0.385 at init, versus v3.0's measured 0.994. This scan
# varies only energy_init_scale to find where the cap stops binding.
#
# Reference (already run): checkpoints/hp_sweep/price_strong = energy_init 0.3, price_lr 0.024.
# Every variant here differs from it in exactly one key: cpt_energy_init_scale.

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-/mnt/nas/jetpack-hdd/tinymixtral-improve/v3.0-dense-active/data/pretrain}"
VAL="${VAL:-$DATA/pilot_blend30_val}"
CFG_DIR="$REPO/configs/cpt_hp_sweep"
OUT="${OUT:-$REPO/checkpoints/einit_sweep}"
PYTHON="${PYTHON:-/home/mikecovlee/miniconda3/envs/tinymixtral-cudev/bin/python}"

COMMON="--cache-dir $DATA/main_s1 --val-dir $VAL \
  --max-tokens 100000000 --batch-size 48 --seq-len 1024 \
  --lr 5e-4 --wd 0.1 --warmup-steps 100 --schedule wsd \
  --bf16-optim --chunked-ce --seed 42 \
  --eval-every-steps 400 --eval-max-tokens 2000000 \
  --save-every-min 999 --keep-last-checkpoints 1 --log-every 100"

mkdir -p "$OUT"

VARIANTS=(
  "einit06:proxy_einit06_8l.json"
  "einit12:proxy_einit12_8l.json"
  "einit20:proxy_einit20_8l.json"
  "einit30:proxy_einit30_8l.json"
)

for entry in "${VARIANTS[@]}"; do
  name="${entry%%:*}"
  cfg="${entry##*:}"
  out_dir="$OUT/$name"
  log="$OUT/${name}.log"
  if [ -d "$out_dir" ] && compgen -G "$out_dir/step_*" > /dev/null 2>&1; then
    echo "SKIP $name (already has checkpoints)"
    continue
  fi
  echo "=== [$name] $cfg ==="
  $PYTHON "$REPO/scripts/train.py" \
    --config "$CFG_DIR/$cfg" \
    --output-dir "$out_dir" \
    $COMMON 2>&1 | tee "$log"
  echo ""
done

echo "=== ALL DONE ==="
echo "Results in $OUT/"
