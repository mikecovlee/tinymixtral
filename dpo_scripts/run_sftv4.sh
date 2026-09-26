#!/usr/bin/env bash
set -u
INIT=${1:?usage: run_sftv4.sh <init_checkpoint>}
source /home/mikecovlee/miniconda3/etc/profile.d/conda.sh
conda activate tinymixtral-cudev
cd /home/mikecovlee/work/tinymixtral
SNAP=$(ls -d /home/mikecovlee/.cache/huggingface/hub/models--mikecovlee--tinymixtral/snapshots/*/ | head -1)
echo "=== SFTV4 START $(date +%T) init=$INIT ==="
python scripts/train_sft.py --checkpoint "$INIT" --tokenizer-path "$SNAP" \
  --dataset data/sft_v4/train.parquet --max-samples 50000 --epochs 1 --seq-len 1024 \
  --lr 5e-6 --batch-size 24 --output-dir checkpoints/sft_v4 --save-every 500 --log-every 50 \
  > dpo_local2/sftv4_train.log 2>&1
echo "=== SFTV4 rc=$? $(date +%T) ==="
echo SFTV4_DONE
