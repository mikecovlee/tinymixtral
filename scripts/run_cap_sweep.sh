#!/usr/bin/env bash
set -euo pipefail
# CPT capacity_factor scan: 2 runs on the 8L proxy, ~50min each.
#
# capacity_factor is NOT a token-dropping knob in CPT (CPTSparseMoE drops nothing, so FLOPs
# match v3.0 exactly). It only sets the width of the congestion-price controller's dead zone:
#   delta = cap - 1 ; band = [(1-delta)/N, (1+delta)/N] with N = num_local_experts = 4
#   cap 1.25 (current) -> [0.1875, 0.3125]  i.e. util tolerated within +-6.25pp of uniform
#   cap 1.05           -> [0.2375, 0.2625]  +-1.25pp
#   cap 1.00           -> [0.2500, 0.2500]  pure integral control toward exact uniformity
#
# This closes CPT_AUDIT.md 8 Q4, the last never-scanned knob. Note that expert balance is the
# SECONDARY objective (PPL is primary), so a tighter band is only worth adopting if it does not
# cost PPL. The prior is that this is a weak lever: einit06 already reached a 1.4pp util spread
# at cap 1.25, i.e. the system already sits near the tight bands, so little control effort changes.
#
# Reference (already run): checkpoints/hp_sweep/price_strong.
# Every variant here differs from it in exactly one key: cpt_capacity_factor.

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-/mnt/nas/jetpack-hdd/tinymixtral-improve/v3.0-dense-active/data/pretrain}"
VAL="${VAL:-$DATA/pilot_blend30_val}"
CFG_DIR="$REPO/configs/cpt_hp_sweep"
OUT="${OUT:-$REPO/checkpoints/cap_sweep}"
PYTHON="${PYTHON:-/home/mikecovlee/miniconda3/envs/tinymixtral-cudev/bin/python}"

COMMON="--cache-dir $DATA/main_s1 --val-dir $VAL \
  --max-tokens 100000000 --batch-size 48 --seq-len 1024 \
  --lr 5e-4 --wd 0.1 --warmup-steps 100 --schedule wsd \
  --bf16-optim --chunked-ce --seed 42 \
  --eval-every-steps 400 --eval-max-tokens 2000000 \
  --save-every-min 999 --keep-last-checkpoints 1 --log-every 100"

mkdir -p "$OUT"

VARIANTS=(
  "cap_tight:proxy_cap_tight_8l.json"
  "cap_exact:proxy_cap_exact_8l.json"
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
