#!/usr/bin/env bash
# 将本地 checkpoints 同步到 NAS（经 /mnt/Jetpack SMB 挂载）
# 用法:
#   bash scripts/sync_nas.sh --experiment <name> [--checkpoint-dir <dir>] [--keep-remote <N>]
# 默认同步 <repo>/checkpoints/ 到 /mnt/Jetpack/v2_checkpoints/<experiment>/
# NAS 默认保留最近 5 个 checkpoint（含 final），更早的自动清理
#
# 示例:
#   bash scripts/sync_nas.sh --experiment v1.1_8b
#   bash scripts/sync_nas.sh --experiment v1.1_8b --keep-remote 5

set -euo pipefail

NAS_BASE="/mnt/Jetpack/v2_checkpoints"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)/checkpoints"
EXPERIMENT=""
KEEP_REMOTE=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment)      EXPERIMENT="$2";   shift 2;;
        --checkpoint-dir)  LOCAL_ROOT="$2";   shift 2;;
        --keep-remote)     KEEP_REMOTE="$2";  shift 2;;
        *) echo "Unknown option: $1" >&2; exit 1;;
    esac
done

if [[ -z "$EXPERIMENT" ]]; then
    echo "ERROR: --experiment <name> 必填" >&2
    exit 1
fi

if ! mountpoint -q /mnt/Jetpack; then
    echo "ERROR: /mnt/Jetpack 未挂载，请先挂载 SMB" >&2
    exit 1
fi

if [[ ! -d "$LOCAL_ROOT" ]]; then
    echo "ERROR: 本地 checkpoint 目录不存在: $LOCAL_ROOT" >&2
    exit 1
fi

DEST="$NAS_BASE/$EXPERIMENT"
mkdir -p "$DEST"

echo "同步 $LOCAL_ROOT → $DEST"
rsync -a --progress \
    --include="step_*/" \
    --include="step_*/pytorch_model.bin" \
    --include="step_*/config.json" \
    --include="step_*/training_state.pt" \
    --exclude="*" \
    "$LOCAL_ROOT/" "$DEST/"

# NAS 保留清理: 保留最近 KEEP_REMOTE 个 step_* (含 final)
echo "NAS 保留清理: 最多 $KEEP_REMOTE 个"
cd "$DEST"
mapfile -t all_ckpts < <(find . -maxdepth 1 -type d -name "step_*" | sort)
total=${#all_ckpts[@]}
if (( total > KEEP_REMOTE )); then
    to_delete=("${all_ckpts[@]:0:$((total - KEEP_REMOTE))}")
    for d in "${to_delete[@]}"; do
        echo "  NAS 删除: $d"
        rm -rf "$d"
    done
else
    echo "  无需清理 ($total/$KEEP_REMOTE)"
fi

echo "同步完成: $(date '+%Y-%m-%d %H:%M:%S')"
