#!/usr/bin/env bash
set -u
cd /home/mikecovlee/work/tinymixtral
RUSER=mikecovlee@10.31.0.14
RREPO='C:/Users/mikecovlee/tinymixtral-improve'
FILTER='grep -a -v -E "post-quantum|store now|upgraded|openssh.com"'
echo "=== TOPUP WATCHER START $(date +%T) ==="
scp -q dpo_scripts/probe_once.ps1 "$RUSER:$RREPO/scripts/probe_once.ps1"
while :; do
  timeout 240 ssh "$RUSER" "powershell -NoProfile -ExecutionPolicy Bypass -File $RREPO/scripts/probe_once.ps1" >/dev/null 2>&1
  ST=$(timeout 60 ssh "$RUSER" "Get-Content '$RREPO/logs/probe_status.log' -Tail 3 -ErrorAction SilentlyContinue" 2>/dev/null | eval "$FILTER" | tr -d '\r\000')
  echo "[$(date +%T)] ${ST}"
  if echo "$ST" | grep -q "HTTP 200"; then
    timeout 60 ssh "$RUSER" "Move-Item -Force '$RREPO/logs/v4rub.log' '$RREPO/logs/v4rub.log.aborted' -ErrorAction SilentlyContinue" >/dev/null 2>&1
    timeout 60 ssh "$RUSER" 'C:\Software\Shell\bin\tmux.exe new-session -d -s v4rub "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mikecovlee\tinymixtral-improve\scripts\run_v4rubfix.ps1"' >/dev/null 2>&1
    echo "=== TOPUP RESUME LAUNCHED $(date +%T) ==="
    break
  fi
  sleep 300
done
