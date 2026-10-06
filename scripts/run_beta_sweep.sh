#!/usr/bin/env bash
set -euo pipefail
# CPT beta_max scan (CPT_AUDIT.md 8 Q2): the never-scanned strength knob of CPT's
# signature sequential state. beta_max sets how hard the (S, nu) state can modulate
# the prototypes; the realised saturation is beta_max * nu / (nu + kappa) which at the
# steady state nu = kappa = 2.5 equals beta_max / 2.
#
#   beta_off 1/20   -> saturation 0.025  (state effectively OFF: ablation)
#   beta_max 49/100 -> saturation 0.245  (state at the hard ceiling 1/(1+radius) = 0.5)
#   reference: price_strong 9/20 -> saturation 0.225
#
# Each variant differs from proxy_price_strong_8l.json in exactly one key: cpt_beta_max.
# Runs LOCALLY: the remote A5000 box cannot run CPT configs at all because the router's
# compiled path needs Triton, which has no Windows support (TritonMissing).

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-/mnt/nas/jetpack-hdd/tinymixtral-improve/v3.0-dense-active/data/pretrain}"
VAL="${VAL:-$DATA/pilot_blend30_val}"
CFG_DIR="$REPO/configs/cpt_hp_sweep"
OUT="${OUT:-$REPO/checkpoints/beta_sweep}"
PYTHON="${PYTHON:-/home/mikecovlee/miniconda3/envs/tinymixtral-cudev/bin/python}"

COMMON="--cache-dir $DATA/main_s1 --val-dir $VAL \
  --max-tokens 100000000 --batch-size 48 --seq-len 1024 \
  --lr 5e-4 --wd 0.1 --warmup-steps 100 --schedule wsd \
  --bf16-optim --chunked-ce --seed 42 \
  --eval-every-steps 400 --eval-max-tokens 2000000 \
  --save-every-min 999 --keep-last-checkpoints 1 --log-every 100"

mkdir -p "$OUT"

VARIANTS=(
  "beta_off:proxy_beta_off_8l.json"
  "beta_max:proxy_beta_max_8l.json"
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
