#!/usr/bin/env bash
set -u
cd /home/mikecovlee/work/tinymixtral
RUSER=mikecovlee@10.31.0.14
RREPO='C:/Users/mikecovlee/tinymixtral-improve'
FILTER='grep -a -v -E "post-quantum|store now|upgraded|openssh.com"'
sshq() { timeout 50 ssh "$RUSER" "$1" 2>/dev/null | eval "$FILTER" | tr -d '\r\000'; }

echo "=== V2EVAL WAITER START $(date +%T) ==="
while :; do
  OUT=$(sshq "Select-String -Path '$RREPO/logs/offload_sftv2v2.log' -Pattern 'OFFLOAD_ARM_DONE' | Select-Object -Last 1 | ForEach-Object { \$_.Line }")
  if [ -n "$OUT" ]; then echo "DONE: $OUT"; break; fi
  sleep 120
done

for task in ifeval gsm8k; do
  RJSON=$(sshq "Get-ChildItem -Path '$RREPO/evals/$task/imp-sft-v2-v2' -Recurse -Filter results_*.json | Sort-Object LastWriteTime | Select-Object -Last 1 | ForEach-Object { \$_.FullName }" | tr '\\' '/')
  echo "$task -> $RJSON"
  if [ -n "$RJSON" ]; then
    scp "$RUSER:$RJSON" "data/dpo/${task}_sftv2v2.json" >/dev/null 2>&1 && echo "pulled data/dpo/${task}_sftv2v2.json"
  fi
done
echo "=== V2EVAL PULLED $(date +%T) ==="
