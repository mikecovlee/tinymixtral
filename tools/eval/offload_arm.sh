#!/usr/bin/env bash
# usage: offload_arm.sh <arm-name> <tag> [checkpoint-dir]
# publishes a checkpoint locally, scp's it to the work machine, then prints the
# tmux command that runs the full offloaded eval (gen + rubric + lm-eval).
set -euo pipefail
ARM=$1
TAG=$2
CKPT=${3:-}
cd /home/mikecovlee/work/tinymixtral
PY=/home/mikecovlee/miniconda3/envs/tinymixtral-cudev/bin/python
SNAP=$(ls -d /home/mikecovlee/.cache/huggingface/hub/models--mikecovlee--tinymixtral/snapshots/*/ | head -1)
if [ -z "$CKPT" ]; then
  CKPT=$(ls -d checkpoints/sft_${TAG}/step_*_final 2>/dev/null | tail -1)
fi
if [ -z "$CKPT" ]; then
  ALT=$(echo "$TAG" | sed -E 's/^sftv([0-9]+)v([0-9]+)$/sft_v\1_v\2/')
  CKPT=$(ls -d checkpoints/${ALT}/step_*_final 2>/dev/null | tail -1)
fi
if [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then
  echo "ERROR: checkpoint not found: '$CKPT'"; exit 1
fi
echo "checkpoint = $CKPT"
echo "tokenizer  = $SNAP"
$PY scripts/publish_hf.py --checkpoint "$CKPT" --output "publish/$ARM" --tokenizer "$SNAP"
ls -la "publish/$ARM"
if ls "publish/$ARM"/model.safetensors >/dev/null 2>&1; then
  echo "ERROR: publish/$ARM contains model.safetensors (whitelist regression)"; exit 1
fi
REMOTE="C:/Users/mikecovlee/tinymixtral-improve/publish/$ARM"
timeout 90 ssh mikecovlee@10.31.0.14 "Remove-Item -Recurse -Force '$REMOTE' -ErrorAction SilentlyContinue" 2>/dev/null || true
scp -q -r "publish/$ARM" mikecovlee@10.31.0.14:'C:/Users/mikecovlee/tinymixtral-improve/publish/'
echo "scp done"
echo
echo "now run on the work machine:"
echo "  C:\\Software\\Shell\\bin\\tmux.exe new-session -d -s $TAG \"powershell -NoProfile -ExecutionPolicy Bypass -File C:\\Users\\mikecovlee\\tinymixtral-improve\\scripts\\run_offload_arm.ps1 -Arm $ARM -Tag $TAG\""
