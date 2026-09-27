#!/usr/bin/env bash
set -u
cd /home/mikecovlee/work/tinymixtral
PY=/home/mikecovlee/miniconda3/envs/tinymixtral-cudev/bin/python
echo "=== V4VERDICT WAITER START $(date +%T) ==="
while :; do
  n=$(ls data/dpo/harness_sftv4.json data/dpo/ifeval_sftv4.json data/dpo/gsm8k_sftv4.json data/dpo/rubric2v2_sftv4.jsonl 2>/dev/null | wc -l)
  [ "$n" -ge 4 ] && break
  sleep 120
done
sleep 30
echo "=== all V4 artifacts present $(date +%T), generating final table ==="
"$PY" dpo_scripts/final_table.py --dir data/dpo | tee data/dpo/final_table.md
"$PY" dpo_scripts/summarize_evals.py --dir data/dpo --detailed | tee data/dpo/final_evals_detailed.txt
echo "=== V4VERDICT READY $(date +%T) ==="
