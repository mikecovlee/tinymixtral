#!/usr/bin/env bash
# 续训 v1.1_8b: 8B → 12B（在 8B checkpoint 基础上训练新的 4B 数据）
#
# 关键设计:
#   - 权重: 从 8B checkpoint (NAS v1.1_8b/step_0162761_final) 加载
#   - 数据: 全新 4B (smollm_blend3)，非重叠
#   - schedule: WSD (warmup 2000 → 稳定 → 最后10% 衰减)，峰值 LR 3e-4
#   - 本地保留: 最多 2 个周期 ckpt
#   - NAS 保留: 最多 5 个 ckpt
#
# 用法: bash scripts/train_cont_12b.sh
# 建议: nohup bash scripts/train_cont_12b.sh > train_cont_12b.log 2>&1 &

set -euo pipefail
export http_proxy=http://127.0.0.1:7890/
export https_proxy=http://127.0.0.1:7890/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tinymixtral

cd "$(dirname "$0")/.."

EXPERIMENT="v1.1_12b"
LOCAL_CKPT_DIR="checkpoints/$EXPERIMENT"      # 训练输出（新 ckpt 都写这里）
LOCAL_START_DIR="checkpoints/start_8b"        # 起始 8B ckpt（只读起点，与输出分离）
NAS_BASE="/mnt/Jetpack/v2_checkpoints"
NAS_CKPT_DIR="$NAS_BASE/$EXPERIMENT"
SRC_CKPT="/mnt/Jetpack/v2_checkpoints/v1.1_8b/step_0162761_final"

KEEP_LOCAL=2        # 本地保留周期 ckpt 数
KEEP_REMOTE=5       # NAS 保留 ckpt 数（含 final）
SAVE_EVERY_MIN=120
LOG_EVERY=100

echo "===== 前置检查 ====="

if ! mountpoint -q /mnt/Jetpack; then
    echo "ERROR: /mnt/Jetpack 未挂载" >&2; exit 1
fi
if [[ ! -d data/pretrain/smollm_blend3 ]]; then
    echo "ERROR: 新数据 data/pretrain/smollm_blend3 不存在，先跑 prepare_data_round3.sh" >&2; exit 1
fi
nshards=$(ls data/pretrain/smollm_blend3/train_*.pt 2>/dev/null | wc -l)
if (( nshards < 1 )); then
    echo "ERROR: smollm_blend3 数据为空" >&2; exit 1
fi
echo "注意: smollm_blend3 有 $nshards/41 shards（FineWeb-Edu sample-10BT 可能已接近耗尽）"
if ps aux | grep -E "train.py|resume.py|prepare_data" | grep -v grep >/dev/null; then
    echo "ERROR: 已有训练/数据准备进程在运行" >&2; exit 1
fi

echo "===== [1/5] 拉取 8B checkpoint 到本地起点目录 ====="
mkdir -p "$LOCAL_START_DIR" "$LOCAL_CKPT_DIR"
rsync -a --delete "$SRC_CKPT/" "$LOCAL_START_DIR/$(basename "$SRC_CKPT")/"
ls "$LOCAL_START_DIR/"

echo "===== [2/5] 续训 4B 新数据 (WSD, peak LR 3e-4) ====="
echo "实验: $EXPERIMENT"
echo "目标: ~4B 新 token / 最大 162,601 步 / WSD warmup=2000"
echo "本地保留: $KEEP_LOCAL 周期 ckpt | NAS保留: $KEEP_REMOTE"

python scripts/resume.py \
  --checkpoint-dir "$LOCAL_START_DIR" \
  --output-dir "$LOCAL_CKPT_DIR" \
  --cache-dir data/pretrain/smollm_blend3 \
  --max-tokens 3996082176 \
  --schedule wsd --lr 3e-4 --warmup-steps 2000 \
  --batch-size 24 --bf16-optim \
  --save-every-min "$SAVE_EVERY_MIN" \
  --log-every "$LOG_EVERY" \
  --keep-last-checkpoints "$KEEP_LOCAL"

echo "===== [3/5] 同步到 NAS ====="
bash scripts/sync_nas.sh --experiment "$EXPERIMENT" --checkpoint-dir "$LOCAL_CKPT_DIR" --keep-remote "$KEEP_REMOTE"

echo "===== [4/5] NAS 保留清理（最多 $KEEP_REMOTE 个）====="
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
mapfile -t local_periodic < <(find . -maxdepth 1 -type d -name "step_*" ! -name "*_final" | sort)
lp=${#local_periodic[@]}
if (( lp > KEEP_LOCAL )); then
    for d in "${local_periodic[@]:0:$((lp - KEEP_LOCAL))}"; do
        echo "  本地删除: $d"
        rm -rf "$d"
    done
fi
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
