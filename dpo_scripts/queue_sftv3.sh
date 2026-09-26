#!/usr/bin/env bash
set -u
cd /home/mikecovlee/work/tinymixtral
echo "=== SFTV3 WAITER START $(date +%T) ==="
while ! grep -q SFTV2_DONE dpo_local2/sftv2_outer.log 2>/dev/null; do sleep 60; done
echo "=== SFTV2_DONE seen, starting V3 at $(date +%T) ==="
bash dpo_scripts/run_sftv3.sh
echo "=== SFTV3 WAITER WRAP $(date +%T) ==="
