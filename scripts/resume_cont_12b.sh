#!/usr/bin/env bash
# 从 emergency checkpoint 恢复续训（v1.1 8B→12B）
# 用法: bash scripts/resume_cont_12b.sh
# 说明: 非 post-train 模式精确恢复 step/data 位置，WSD schedule 从 last_epoch 继续

set -euo pipefail
export http_proxy=http://127.0.0.1:7890/
export https_proxy=http://127.0.0.1:7890/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tinymixtral

cd "$(dirname "$0")/.."

EXPERIMENT="v1.1_12b"
LOCAL_CKPT_DIR="checkpoints/$EXPERIMENT"
KEEP_LOCAL=2
SAVE_EVERY_MIN=120
LOG_EVERY=100

echo "===== 前置检查 ====="
if ps aux | grep -E "resume.py|train.py|prepare_data" | grep -v grep >/dev/null; then
    echo "ERROR: 已有训练/数据进程在运行" >&2; exit 1
fi
if [[ ! -d "$LOCAL_CKPT_DIR" ]] || [[ -z "$(find "$LOCAL_CKPT_DIR" -maxdepth 1 -type d -name 'step_*' | head -1)" ]]; then
    echo "ERROR: 本地输出目录无 checkpoint: $LOCAL_CKPT_DIR" >&2; exit 1
fi
nshards=$(ls data/pretrain/smollm_blend3/train_*.pt 2>/dev/null | wc -l)
if (( nshards < 1 )); then
    echo "ERROR: smollm_blend3 数据不完整: $nshards" >&2; exit 1
fi

echo "===== 恢复续训（非 post-train 模式，精确恢复 step/data 位置）====="
echo "实验: $EXPERIMENT | 本地保留: $KEEP_LOCAL"

python scripts/resume.py \
  --checkpoint-dir "$LOCAL_CKPT_DIR" \
  --output-dir "$LOCAL_CKPT_DIR" \
  --cache-dir data/pretrain/smollm_blend3 \
  --schedule wsd --lr 3e-4 --warmup-steps 2000 \
  --batch-size 24 --bf16-optim \
  --save-every-min "$SAVE_EVERY_MIN" \
  --log-every "$LOG_EVERY" \
  --keep-last-checkpoints "$KEEP_LOCAL"

echo "===== 训练完成，同步到 NAS ====="
bash scripts/sync_nas.sh --experiment "$EXPERIMENT" --checkpoint-dir "$LOCAL_CKPT_DIR" --keep-remote 5

echo "===== 本地清理 ====="
cd "$LOCAL_CKPT_DIR"
mapfile -t local_periodic < <(find . -maxdepth 1 -type d -name "step_*" ! -name "*_final" | sort)
lp=${#local_periodic[@]}
if (( lp > KEEP_LOCAL )); then
    for d in "${local_periodic[@]:0:$((lp - KEEP_LOCAL))}"; do
        echo "  本地删除: $d"
        rm -rf "$d"
    done
fi
echo "本地剩余:"
ls
echo "===== 完成 ====="
