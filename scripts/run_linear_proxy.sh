#!/usr/bin/env bash
set -euo pipefail
# Matched linear-router baseline on the 8L proxy.
#
# Purpose: remove the budget confound from the expressiveness comparison. The published
# v3.0 routing statistics were measured on a 16L checkpoint trained on 8.05B tokens, while
# the CPT statistics come from an 8L proxy trained on 100M tokens. Routing sharpness grows
# during training, so that comparison is not like-for-like.
#
# This run uses configs/cpt_hp_sweep/proxy_linear_8l.json: identical backbone, identical
# data, identical batch/schedule/seed/budget as proxy_base_8l.json, but with no
# cpt_router_version key, so model_for_config() dispatches to TinyMixtralForCausalLM and
# the router is the plain nn.Linear(hidden, num_experts) of v3.0. router_aux_loss_coef is
# set to 0.001, the value v3.0 actually trained with (CPT pins it to 0 and uses congestion
# pricing instead).
#
# Read the result with scripts/audit_expressiveness.py --linear-ckpt <out>/step_*_final.

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-/mnt/nas/jetpack-hdd/tinymixtral-improve/v3.0-dense-active/data/pretrain}"
VAL="${VAL:-$DATA/pilot_blend30_val}"
CFG_DIR="$REPO/configs/cpt_hp_sweep"
OUT="${OUT:-$REPO/checkpoints/linear_proxy}"
PYTHON="${PYTHON:-/home/mikecovlee/miniconda3/envs/tinymixtral-cudev/bin/python}"

COMMON="--cache-dir $DATA/main_s1 --val-dir $VAL \
  --max-tokens 100000000 --batch-size 48 --seq-len 1024 \
  --lr 5e-4 --wd 0.1 --warmup-steps 100 --schedule wsd \
  --bf16-optim --chunked-ce --seed 42 \
  --eval-every-steps 400 --eval-max-tokens 2000000 \
  --save-every-min 999 --keep-last-checkpoints 1 --log-every 100"

mkdir -p "$OUT"

name="linear"
out_dir="$OUT/$name"
log="$OUT/${name}.log"
if [ -d "$out_dir" ] && compgen -G "$out_dir/step_*" > /dev/null 2>&1; then
  echo "SKIP $name (already has checkpoints)"
  exit 0
fi

echo "=== [$name] proxy_linear_8l.json ==="
$PYTHON "$REPO/scripts/train.py" \
  --config "$CFG_DIR/proxy_linear_8l.json" \
  --output-dir "$out_dir" \
  $COMMON 2>&1 | tee "$log"

echo "=== DONE ==="
