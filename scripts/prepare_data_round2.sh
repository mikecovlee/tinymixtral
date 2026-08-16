#!/usr/bin/env bash
# 第二轮数据准备：tokenize 新的 4B（非重叠，跳过第一批已用的数据）
# 供 4B→8B 续训
#
# 关键：加 --skip-tokens 跳过第一批已 tokenize 的数据
#   - FineWeb-Edu: 第一批用了 3.56B，第二批从 3.56B 之后继续
#   - Cosmopedia:  第一批用了 440M，第二批从 440M 之后继续
#
# 用法: bash scripts/prepare_data_round2.sh
# 建议: nohup bash scripts/prepare_data_round2.sh > prepare_data2.log 2>&1 &

set -euo pipefail
export http_proxy=http://127.0.0.1:7890/
export https_proxy=http://127.0.0.1:7890/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tinymixtral

cd "$(dirname "$0")/.."

echo "===== [1/3] FineWeb-Edu 再 tokenize 3.56B（跳过前 3.56B，取 3.56B~7.12B 段）====="
python scripts/prepare_data.py --dataset HuggingFaceFW/fineweb-edu --subset sample-10BT \
  --tokenizer tokenizer/ --output data/pretrain/fineweb2 \
  --skip-tokens 3560000000 --max-tokens 3560000000 --force

echo "===== [2/3] Cosmopedia v2 再 tokenize 440M（跳过前 440M）====="
python scripts/prepare_data.py --dataset HuggingFaceTB/cosmopedia-v2 --subset cosmopedia-v2 \
  --tokenizer tokenizer/ --output data/pretrain/cosmopedia2 \
  --skip-tokens 440000000 --max-tokens 440000000 --force

echo "===== [3/3] 混排成新 4B (89:11) ====="
python scripts/mix_data.py data/pretrain/fineweb2 data/pretrain/cosmopedia2 \
  --output data/pretrain/smollm_blend2 --weights 36 4

echo "===== 完成 ====="
ls data/pretrain/smollm_blend2/ | head
echo "shard 数: $(ls data/pretrain/smollm_blend2/train_*.pt | wc -l)"
