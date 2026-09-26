#!/usr/bin/env bash
set -u
source /home/mikecovlee/miniconda3/etc/profile.d/conda.sh
conda activate tinymixtral-cudev
cd /home/mikecovlee/work/tinymixtral
SNAP=$(ls -d /home/mikecovlee/.cache/huggingface/hub/models--mikecovlee--tinymixtral/snapshots/*/ | head -1)
echo "=== SFTV2 START $(date +%T) ==="
python scripts/train_sft.py --checkpoint checkpoints/base_v3_raw --tokenizer-path "$SNAP" \
  --dataset data/sft_v2_v2/train.parquet --max-samples 856805 --epochs 1 --seq-len 1024 \
  --lr 2e-5 --batch-size 24 --output-dir checkpoints/sft_v2_v2 --save-every 1000 --log-every 100 \
  > dpo_local2/sftv2_train.log 2>&1
echo "=== SFTV2 rc=$? $(date +%T) ==="
echo SFTV2_DONE
