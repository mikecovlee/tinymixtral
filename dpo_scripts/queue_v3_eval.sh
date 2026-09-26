#!/usr/bin/env bash
set -u
cd /home/mikecovlee/work/tinymixtral
LOG=dpo_local2/sftv3_queue.log
echo "=== V3EVAL WAITER START $(date +%T) ==="
while ! grep -q "SFTV3_DONE" "$LOG" 2>/dev/null; do sleep 60; done
echo "=== SFTV3_DONE seen $(date +%T), publishing + shipping V3 ==="
CKPT=$(ls -d checkpoints/sft_v2_v3/step_*_final 2>/dev/null | tail -1)
if [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then
  echo "ERROR: V3 final checkpoint not found under checkpoints/sft_v2_v3"; exit 1
fi
echo "ckpt = $CKPT"
if ! bash dpo_scripts/offload_arm.sh imp-sft-v2-v3 sftv2v3 "$CKPT"; then
  echo "ERROR: offload_arm failed; not launching remote eval"; exit 1
fi
echo "=== launching remote tmux sftv2v3 ==="
ssh mikecovlee@10.31.0.14 'C:\Software\Shell\bin\tmux.exe new-session -d -s sftv2v3 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mikecovlee\tinymixtral-improve\scripts\run_offload_arm.ps1 -Arm imp-sft-v2-v3 -Tag sftv2v3"'
echo "=== V3EVAL LAUNCHED $(date +%T) ==="
