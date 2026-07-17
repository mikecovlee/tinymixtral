#!/bin/bash
export http_proxy=http://127.0.0.1:7890/
export https_proxy=http://127.0.0.1:7890/
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tinymixtral
exec python -u scripts/resume.py \
  --checkpoint-dir checkpoints/run \
  --output-dir checkpoints/posttrain \
  --cache-dir data/posttrain/mixed \
  --max-tokens 1000000000 --lr 5e-5 --warmup-steps 300 \
  --batch-size 22 --save-every-min 60 \
  --log-every 20 --keep-last-checkpoints 3 \
  2>&1  | tee -a train.log
