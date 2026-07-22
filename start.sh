export http_proxy=http://127.0.0.1:7890/
export https_proxy=http://127.0.0.1:7890/
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tinymixtral
python scripts/train.py \
  --cache-dir data/pretrain/smollm_blend \
  --output-dir checkpoints/smollm_blend \
  --batch-size 24 \
  --seq-len 1024 \
  --max-tokens 4000000000 \
  --lr 7e-4 \
  --wd 0.1 \
  --warmup-steps 2000 \
  --save-every-min 120 \
  --keep-last-checkpoints 5 \
  --log-every 100 2>&1 | tee train.log