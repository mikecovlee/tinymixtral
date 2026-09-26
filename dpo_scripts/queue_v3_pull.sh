#!/usr/bin/env bash
set -u
cd /home/mikecovlee/work/tinymixtral
RUSER=mikecovlee@10.31.0.14
RREPO='C:/Users/mikecovlee/tinymixtral-improve'
FILTER='grep -a -v -E "post-quantum|store now|upgraded|openssh.com"'
sshq() { timeout 50 ssh "$RUSER" "$1" 2>/dev/null | eval "$FILTER" | tr -d '\r\000'; }

echo "=== V3EVAL PULL WAITER START $(date +%T) ==="
while ! grep -q "SFTV3_DONE" dpo_local2/sftv3_queue.log 2>/dev/null; do sleep 60; done
echo "SFTV3_DONE seen $(date +%T), waiting for remote V3 offload eval"
while :; do
  OUT=$(sshq "Select-String -Path '$RREPO/logs/offload_sftv2v3.log' -Pattern 'OFFLOAD_ARM_DONE' | Select-Object -Last 1 | ForEach-Object { \$_.Line }")
  if [ -n "$OUT" ]; then echo "DONE: $OUT"; break; fi
  sleep 120
done

for task in harness ifeval gsm8k; do
  RJSON=$(sshq "Get-ChildItem -Path '$RREPO/evals/$task/imp-sft-v2-v3' -Recurse -Filter results_*.json | Sort-Object LastWriteTime | Select-Object -Last 1 | ForEach-Object { \$_.FullName }" | tr '\\' '/')
  echo "$task -> $RJSON"
  if [ -n "$RJSON" ]; then
    scp "$RUSER:$RJSON" "data/dpo/${task}_sftv2v3.json" >/dev/null 2>&1 && echo "pulled data/dpo/${task}_sftv2v3.json"
  fi
done

RRUB="$RREPO/data/dpo/rubric2v2_sftv2v3.jsonl"
scp "$RUSER:$RRUB" data/dpo/rubric2v2_sftv2v3.jsonl >/dev/null 2>&1 && echo "pulled data/dpo/rubric2v2_sftv2v3.jsonl"
echo "=== V3EVAL PULLED $(date +%T) ==="
