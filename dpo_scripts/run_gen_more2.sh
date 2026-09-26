#!/usr/bin/env bash
# Held-out generation for the two newest arms (base v3.0, imp-base-exam3)
# so they can join the anchored rubric comparison.
set -u
source /home/mikecovlee/miniconda3/etc/profile.d/conda.sh
conda activate tinymixtral-cudev
export USERPROFILE=/home/mikecovlee
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd /home/mikecovlee/work/tinymixtral
PY=/home/mikecovlee/miniconda3/envs/tinymixtral-cudev/bin/python
SNAP=$(ls -d /home/mikecovlee/.cache/huggingface/hub/models--mikecovlee--tinymixtral/snapshots/*/ | head -1)
for pair in "base:$SNAP" "exam3:/home/mikecovlee/work/tinymixtral/publish/imp-base-exam3"; do
  n="${pair%%:*}"; m="${pair#*:}"
  echo "=== GEN2 $n START $(date +%T) ==="
  $PY dpo_scripts/dpo_eval_judge.py gen --model "$m" \
      --prompts data/dpo/prompts_heldout.parquet \
      --out "data/dpo/evalv2_$n.jsonl" --batch-size 8 --max-new-tokens 448 \
      > "dpo_local2/gen2_$n.log" 2>&1
  echo "=== GEN2 $n rc=$? $(date +%T) ==="
done
echo GEN2_DONE
