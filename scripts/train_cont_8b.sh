#!/usr/bin/env bash
# 续训 v1.1: 4B → 8B（在 4B checkpoint 基础上训练新的 4B 数据）
#
# 关键设计:
#   - 权重: 从 4B checkpoint (NAS smollm_blend/step_0162761_final) 加载
#   - 数据: 全新 4B (smollm_blend2)，非重叠
#   - schedule: WSD (warmup 2000 → 稳定 → 最后10% 衰减)，峰值 LR 3e-4
#   - 本地保留: 最多 2 个周期 ckpt（训练时 --keep-last-checkpoints 2）
#   - NAS 保留: 最多 5 个 ckpt（同步后远端清理）
#
# 用法: bash scripts/train_cont_8b.sh
# 建议在 tmux 或 nohup 中运行: nohup bash scripts/train_cont_8b.sh > train_cont.log 2>&1 &

set -euo pipefail
export http_proxy=http://127.0.0.1:7890/
export https_proxy=http://127.0.0.1:7890/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tinymixtral

cd "$(dirname "$0")/.."

EXPERIMENT="v1.1_8b"
LOCAL_CKPT_DIR="checkpoints/$EXPERIMENT"      # 训练输出（新 ckpt 都写这里）
LOCAL_START_DIR="checkpoints/start_4b"        # 起始 4B ckpt（只读起点，与输出分离）
NAS_BASE="/mnt/Jetpack/v2_checkpoints"
NAS_CKPT_DIR="$NAS_BASE/$EXPERIMENT"
SRC_CKPT="/mnt/Jetpack/v2_checkpoints/smollm_blend/step_0162761_final"

KEEP_LOCAL=2        # 本地保留周期 ckpt 数
KEEP_REMOTE=5       # NAS 保留 ckpt 数（含 final）
SAVE_EVERY_MIN=120
LOG_EVERY=100

echo "===== 前置检查 ====="

if ! mountpoint -q /mnt/Jetpack; then
    echo "ERROR: /mnt/Jetpack 未挂载" >&2; exit 1
fi
if [[ ! -d data/pretrain/smollm_blend2 ]]; then
    echo "ERROR: 新数据 data/pretrain/smollm_blend2 不存在，先跑 prepare_data_round2.sh" >&2; exit 1
fi
nshards=$(ls data/pretrain/smollm_blend2/train_*.pt 2>/dev/null | wc -l)
if (( nshards < 41 )); then
    echo "ERROR: smollm_blend2 数据不完整: $nshards/41 shards，先等 prepare_data_round2.sh 完成" >&2; exit 1
fi
if ps aux | grep -E "train.py|resume.py|prepare_data" | grep -v grep >/dev/null; then
    echo "ERROR: 已有训练/数据准备进程在运行" >&2; exit 1
fi

echo "===== [1/5] 拉取 4B checkpoint 到本地起点目录 ====="
mkdir -p "$LOCAL_START_DIR" "$LOCAL_CKPT_DIR"
# 起始 ckpt 放独立目录，与训练输出分离，避免混淆
rsync -a --delete "$SRC_CKPT/" "$LOCAL_START_DIR/$(basename "$SRC_CKPT")/"
ls "$LOCAL_START_DIR/"

echo "===== [2/5] 续训 4B 新数据 (WSD, peak LR 3e-4) ====="
echo "实验: $EXPERIMENT"
echo "目标: ~4B 新 token / 162,761 步 / WSD warmup=2000"
echo "本地保留: $KEEP_LOCAL 周期 ckpt | NAS保留: $KEEP_REMOTE"

python scripts/resume.py \
  --checkpoint-dir "$LOCAL_START_DIR" \
  --output-dir "$LOCAL_CKPT_DIR" \
  --cache-dir data/pretrain/smollm_blend2 \
  --max-tokens 4000000000 \
  --schedule wsd --lr 3e-4 --warmup-steps 2000 \
  --batch-size 24 --bf16-optim \
  --save-every-min "$SAVE_EVERY_MIN" \
  --log-every "$LOG_EVERY" \
  --keep-last-checkpoints "$KEEP_LOCAL"

echo "===== [3/5] 同步到 NAS ====="
bash scripts/sync_nas.sh --experiment "$EXPERIMENT" --checkpoint-dir "$LOCAL_CKPT_DIR" --keep-remote "$KEEP_REMOTE"

echo "===== [4/5] NAS 保留清理（最多 $KEEP_REMOTE 个）====="
# 远端清理: 保留最近 KEEP_REMOTE 个 step_* (含 final)，删除更早的
# 按字母序 (step_XXXXXXX 零填充，字母序=数值序)，final 排最后
cd "$NAS_CKPT_DIR"
mapfile -t all_ckpts < <(find . -maxdepth 1 -type d -name "step_*" | sort)
total=${#all_ckpts[@]}
echo "NAS 当前: $total 个 ckpt (保留 $KEEP_REMOTE)"
if (( total > KEEP_REMOTE )); then
    to_delete=("${all_ckpts[@]:0:$((total - KEEP_REMOTE))}")
    for d in "${to_delete[@]}"; do
        echo "  NAS 删除: $d"
        rm -rf "$d"
    done
else
    echo "  无需清理"
fi

echo "===== [5/5] 本地清理（输出目录最多保留 $KEEP_LOCAL 个 ckpt）====="
cd "$LOCAL_CKPT_DIR"
# 本地输出目录: 保留最近 KEEP_LOCAL 个周期 ckpt + final（若 NAS 已备份则删 final）
mapfile -t local_periodic < <(find . -maxdepth 1 -type d -name "step_*" ! -name "*_final" | sort)
lp=${#local_periodic[@]}
if (( lp > KEEP_LOCAL )); then
    for d in "${local_periodic[@]:0:$((lp - KEEP_LOCAL))}"; do
        echo "  本地删除: $d"
        rm -rf "$d"
    done
fi
# final: NAS 已备份则删本地 final（本地最多 KEEP_LOCAL 个）
local_final=$(find . -maxdepth 1 -type d -name "*_final" | sort | tail -1)
if [[ -n "$local_final" ]]; then
    base=$(basename "$local_final")
    if [[ -d "$NAS_CKPT_DIR/$base" ]]; then
        echo "  NAS 已备份 $base，删除本地 final"
        rm -rf "$local_final"
    else
        echo "  WARN: NAS 无 $base，保留本地 final 作为安全备份"
    fi
fi
echo "本地输出目录剩余:"
ls
echo "起始 ckpt (保留):"
ls "$LOCAL_START_DIR"

echo ""
echo "===== 全部完成 ====="
echo "本地: $LOCAL_CKPT_DIR"
echo "NAS:  $NAS_CKPT_DIR"
df -h /mnt/Jetpack | tail -1
