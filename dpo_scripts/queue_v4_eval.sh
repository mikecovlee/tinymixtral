#!/usr/bin/env bash
set -u
cd /home/mikecovlee/work/tinymixtral
LOG=dpo_local2/sftv4_outer.log
echo "=== V4EVAL WAITER START $(date +%T) ==="
while ! grep -q "SFTV4_DONE" "$LOG" 2>/dev/null; do sleep 60; done
CKPT=$(ls -d checkpoints/sft_v4/step_*_final 2>/dev/null | tail -1)
if [ -z "$CKPT" ]; then echo "ERROR: no final ckpt in checkpoints/sft_v4"; exit 1; fi
echo "=== SFTV4_DONE seen $(date +%T), ckpt=$CKPT, publishing + shipping V4 ==="
if ! bash dpo_scripts/offload_arm.sh imp-sft-v2-v4 sftv4 "$CKPT"; then
  echo "ERROR: offload_arm.sh failed"; exit 1
fi
echo "=== launching remote tmux sftv4 ==="
ssh mikecovlee@10.31.0.14 'C:\Software\Shell\bin\tmux.exe new-session -d -s sftv4 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mikecovlee\tinymixtral-improve\scripts\run_offload_arm.ps1 -Arm imp-sft-v2-v4 -Tag sftv4"'
echo "=== V4EVAL LAUNCHED $(date +%T) ==="
