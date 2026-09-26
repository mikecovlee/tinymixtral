#!/usr/bin/env bash
# Batch-size calibration for the SFT trainer on the local GPU.
# For each bs: sample nvidia-smi memory while training 1500 samples for up to 240 s.
set -u
source /home/mikecovlee/miniconda3/etc/profile.d/conda.sh
conda activate tinymixtral-cudev
cd /home/mikecovlee/work/tinymixtral
PY=/home/mikecovlee/miniconda3/envs/tinymixtral-cudev/bin/python
SNAP=$(ls -d /home/mikecovlee/.cache/huggingface/hub/models--mikecovlee--tinymixtral/snapshots/*/ | head -1)
for bs in 8 16 24 32; do
  log="dpo_local2/calib_bs${bs}.log"
  out="checkpoints/calib_bs${bs}"
  echo "=== CALIB bs=${bs} START $(date +%T) ==="
  ( nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 \
      > "dpo_local2/calib_bs${bs}.mem" 2>/dev/null & echo $! > "dpo_local2/calib_bs${bs}.mempid" ) 
  timeout 260 $PY scripts/train_sft.py \
    --checkpoint checkpoints/base_v3_raw --tokenizer-path "$SNAP" \
    --dataset data/sft_v2_v1 --max-samples 1500 --epochs 1 --seq-len 1024 \
    --lr 2e-5 --batch-size "$bs" --output-dir "$out" \
    --save-every 100000 --log-every 20 > "$log" 2>&1
  rc=$?
  kill "$(cat dpo_local2/calib_bs${bs}.mempid)" 2>/dev/null
  peak=$(sort -n "dpo_local2/calib_bs${bs}.mem" 2>/dev/null | tail -1)
  last=$(grep -a "loss=" "$log" | tail -1)
  oom=$(grep -ac "out of memory" "$log")
  echo "=== CALIB bs=${bs} rc=${rc} peakMiB=${peak} oom=${oom} last='${last}' ==="
done
echo CALIB_DONE
