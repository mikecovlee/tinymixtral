#!/usr/bin/env bash
set -euo pipefail
# CPT hyperparameter sweep: 8 runs on 8L proxy, ~45min each, total ~6h.
# Primary metric: val PPL. Secondary: expert util balance.

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-/mnt/nas/jetpack-hdd/tinymixtral-improve/v3.0-dense-active/data/pretrain}"
VAL="${VAL:-$DATA/pilot_blend30_val}"
CFG_DIR="$REPO/configs/cpt_hp_sweep"
OUT="${OUT:-$REPO/checkpoints/hp_sweep}"
PYTHON="${PYTHON:-/home/mikecovlee/miniconda3/envs/tinymixtral-cudev/bin/python}"

COMMON="--cache-dir $DATA/main_s1 --val-dir $VAL \
  --max-tokens 100000000 --batch-size 48 --seq-len 1024 \
  --lr 5e-4 --wd 0.1 --warmup-steps 100 --schedule wsd \
  --bf16-optim --chunked-ce --seed 42 \
  --eval-every-steps 400 --eval-max-tokens 2000000 \
  --save-every-min 999 --keep-last-checkpoints 1 --log-every 100"

mkdir -p "$OUT"

VARIANTS=(
  "base:proxy_base_8l.json"
  "T_sharp:proxy_T_sharp_8l.json"
  "T_smooth:proxy_T_smooth_8l.json"
  "rho_short:proxy_rho_short_8l.json"
  "rho_long:proxy_rho_long_8l.json"
  "init_flat:proxy_init_flat_8l.json"
  "price_off:proxy_price_off_8l.json"
  "price_strong:proxy_price_strong_8l.json"
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
