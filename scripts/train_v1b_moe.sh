#!/usr/bin/env bash
# 1B MoE 预训练 (1024/16L/8E/intermediate=2816, ~1182M)
# 数据: data/pretrain/smollm_blend (原始 4B)
# batch=16 (实测显存 14.5GB, 余量充足), WSD schedule
# 用法: bash scripts/train_v1b_moe.sh
# 建议 nohup 后台运行

set -euo pipefail
export http_proxy=http://127.0.0.1:7890/
export https_proxy=http://127.0.0.1:7890/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tinymixtral

cd "$(dirname "$0")/.."

EXPERIMENT="v1b_moe"
OUTPUT_DIR="checkpoints/$EXPERIMENT"
NAS_BASE="/mnt/Jetpack/v2_checkpoints"
NAS_DIR="$NAS_BASE/$EXPERIMENT"
CONFIG="configs/model/v1b_moe.json"
KEEP_LOCAL=2
KEEP_REMOTE=5
SAVE_EVERY_MIN=120
LOG_EVERY=100

echo "===== 前置检查 ====="
if ps aux | grep -E "train.py|resume.py|prepare_data" | grep -v grep >/dev/null; then
    echo "ERROR: 已有训练/数据进程在运行" >&2; exit 1
fi
if [[ ! -d data/pretrain/smollm_blend ]] || (( $(ls data/pretrain/smollm_blend/train_*.pt 2>/dev/null | wc -l) < 41 )); then
    echo "ERROR: 数据不完整" >&2; exit 1
fi
if ! mountpoint -q /mnt/Jetpack; then
    echo "ERROR: NAS 未挂载" >&2; exit 1
fi

echo "===== 启动训练 (1B MoE, batch=16, WSD, 4B tokens) ====="
echo "配置: $CONFIG ($(python3 -c "from model.config import TinyMixtralConfig; c=TinyMixtralConfig.from_json_file('$CONFIG'); print(f'{c.hidden_size}d/{c.num_hidden_layers}L/{c.num_local_experts}E intermediate={c.expert_intermediate_size}')"))"
echo "输出: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"
python scripts/train.py \
  --config "$CONFIG" \
  --cache-dir data/pretrain/smollm_blend \
  --output-dir "$OUTPUT_DIR" \
  --batch-size 16 \
  --max-tokens 3996090368 \
  --lr 7e-4 --schedule wsd --warmup-steps 2000 \
  --bf16-optim \
  --save-every-min "$SAVE_EVERY_MIN" \
  --log-every "$LOG_EVERY" \
  --keep-last-checkpoints "$KEEP_LOCAL"

echo "===== 训练完成，同步 NAS ====="
bash scripts/sync_nas.sh --experiment "$EXPERIMENT" --checkpoint-dir "$OUTPUT_DIR" --keep-remote "$KEEP_REMOTE"

echo "===== 本地清理 ====="
cd "$OUTPUT_DIR"
mapfile -t periodic < <(find . -maxdepth 1 -type d -name "step_*" ! -name "*_final" | sort)
lp=${#periodic[@]}
if (( lp > KEEP_LOCAL )); then
    for d in "${periodic[@]:0:$((lp - KEEP_LOCAL))}"; do
        echo "  删除: $d"; rm -rf "$d"
    done
fi
final=$(find . -maxdepth 1 -type d -name "*_final" | tail -1)
if [[ -n "$final" ]] && [[ -d "$NAS_DIR/$(basename "$final")" ]]; then
    echo "  NAS 已备份 $(basename "$final")，删除本地"; rm -rf "$final"
fi
echo "本地剩余:"; ls
echo "===== 完成 ====="
