#!/usr/bin/env bash
# 第三轮数据准备：tokenize 新的 4B（非重叠，供 8B→12B 续训）
#
# 关键：--skip-tokens 跳过前两批已用的数据
#   - FineWeb-Edu: 前两批用了 7.12B，第三批从 7.12B 之后继续（取 7.12B~10.68B 段）
#   - Cosmopedia:  前两批用了 880M，第三批从 880M 之后继续（取 880M~1.32B 段）
#
# 注意: sample-10BT 约 10B tokens，第三批 3.56B 可能不足 4B（实际 shard 数可能 < 41）。
#       训练脚本现已支持"数据耗尽→warning→保存退出"，短数据也能安全运行。
#
# 用法: bash scripts/prepare_data_round3.sh
# 建议: nohup bash scripts/prepare_data_round3.sh > prepare_data3.log 2>&1 &

set -euo pipefail
export http_proxy=http://127.0.0.1:7890/
export https_proxy=http://127.0.0.1:7890/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tinymixtral

cd "$(dirname "$0")/.."

echo "===== [1/3] FineWeb-Edu 再 tokenize 3.56B（跳过前 7.12B，取 7.12B~10.68B 段）====="
python scripts/prepare_data.py --dataset HuggingFaceFW/fineweb-edu --subset sample-10BT \
  --tokenizer tokenizer/ --output data/pretrain/fineweb3 \
  --skip-tokens 7120000000 --max-tokens 3560000000 --force

echo "===== [2/3] Cosmopedia v2 再 tokenize 440M（跳过前 880M）====="
python scripts/prepare_data.py --dataset HuggingFaceTB/cosmopedia-v2 --subset cosmopedia-v2 \
  --tokenizer tokenizer/ --output data/pretrain/cosmopedia3 \
  --skip-tokens 880000000 --max-tokens 440000000 --force

echo "===== [3/3] 混排成新 4B (89:11) ====="
python scripts/mix_data.py data/pretrain/fineweb3 data/pretrain/cosmopedia3 \
  --output data/pretrain/smollm_blend3 --weights 36 4

echo "===== 完成 ====="
nshards=$(ls data/pretrain/smollm_blend3/train_*.pt 2>/dev/null | wc -l)
echo "shard 数: $nshards (预期 41，若不足说明 FineWeb-Edu sample-10BT 已接近耗尽)"
if (( nshards < 41 )); then
    echo "WARNING: smollm_blend3 不足 41 shards，续训会在数据耗尽时自动 warning+保存退出"
fi
