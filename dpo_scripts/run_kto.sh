#!/usr/bin/env bash
# Relaunch KTO locally (crash-recovery, survives reboot because it lives on disk).
set -u
source /home/mikecovlee/miniconda3/etc/profile.d/conda.sh
conda activate tinymixtral-cudev
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
cd /home/mikecovlee/work/tinymixtral
PY=/home/mikecovlee/miniconda3/envs/tinymixtral-cudev/bin/python
OUT=dpo_local2/kto
mkdir -p "$OUT"
{
  echo "=== KTO START $(date +%T) ==="
  $PY dpo_scripts/rl_kto.py \
    --base publish/imp-sft --data data/dpo/kto_pairs.parquet \
    --output-dir "$OUT" --lr 1e-5 --beta 0.1 --epochs 3 \
    --batch-size 4 --grad-accum 4 --max-length 1024 > "$OUT.train.out" 2>&1
  echo "=== KTO rc=$? $(date +%T) ==="
  echo KTO_DONE
} > "$OUT.log" 2>&1
