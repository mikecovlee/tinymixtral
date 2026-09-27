#!/usr/bin/env bash
set -u
cd /home/mikecovlee/work/tinymixtral
RUSER=mikecovlee@10.31.0.14
RREPO='C:/Users/mikecovlee/tinymixtral-improve'
FILTER='grep -a -v -E "post-quantum|store now|upgraded|openssh.com"'
PY=/home/mikecovlee/miniconda3/envs/tinymixtral-cudev/bin/python
echo "=== V4RUBFIX WAITER START $(date +%T) ==="
while :; do
  OUT=$(timeout 50 ssh "$RUSER" 'Get-Content C:\Users\mikecovlee\tinymixtral-improve\logs\v4rub.log -Tail 1' 2>/dev/null | eval "$FILTER" | tr -d '\r\000')
  echo "$OUT" | grep -q "V4RUB_DONE" && break
  sleep 120
done
echo "=== V4RUB_DONE seen $(date +%T), pulling full rubric ==="
scp -q "$RUSER:$RREPO/data/dpo/rubric2v2_sftv4.jsonl" data/dpo/rubric2v2_sftv4.jsonl
echo "local lines: $(wc -l < data/dpo/rubric2v2_sftv4.jsonl)"
"$PY" dpo_scripts/final_table.py --dir data/dpo > data/dpo/final_table.md
"$PY" dpo_scripts/summarize_evals.py --dir data/dpo --detailed > data/dpo/final_evals_detailed.txt
echo "=== V4RUBFIX PULLED + TABLE REFRESHED $(date +%T) ==="
