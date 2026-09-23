#!/usr/bin/env bash
set -u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tinymixtral
cd "$(dirname "$0")/.."

for name in publish publish_posttrain publish_v114b publish_v118b publish_v11pt publish_v11sft; do
  echo "=== RUN $name $(date) ===" >> evals/ifeval_all.log
  lm_eval --model hf --model_args "pretrained=${name}/,trust_remote_code=True,dtype=bfloat16" \
    --tasks ifeval --apply_chat_template --batch_size 8 --device cuda \
    --output_path "evals/ifeval_${name}" >> evals/ifeval_all.log 2>&1
  echo "=== DONE $name exit=$? $(date) ===" >> evals/ifeval_all.log
done
echo "=== ALL COMPLETE $(date) ===" >> evals/ifeval_all.log
