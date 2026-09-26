#!/usr/bin/env bash
# Held-out generation (win-rate inputs) for additional arms.
set -u
source /home/mikecovlee/miniconda3/etc/profile.d/conda.sh
conda activate tinymixtral-cudev
export USERPROFILE=/home/mikecovlee
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd /home/mikecovlee/work/tinymixtral
PY=/home/mikecovlee/miniconda3/envs/tinymixtral-cudev/bin/python
for pair in "sweepbest:publish/sweep_best" "grpoifeval:publish/imp-ifeval" "grpogsm:publish/imp-gsm"; do
  n="${pair%%:*}"; m="${pair#*:}"
  echo "=== GEN $n START $(date +%T) ==="
  $PY dpo_scripts/dpo_eval_judge.py gen --model "$m" --prompts data/dpo/prompts_heldout.parquet \
      --out "data/dpo/evalv2_$n.jsonl" --batch-size 8 --max-new-tokens 448 > "dpo_local2/gen_$n.log" 2>&1
  echo "=== GEN $n rc=$? $(date +%T) ==="
done
echo GEN_MORE_DONE
