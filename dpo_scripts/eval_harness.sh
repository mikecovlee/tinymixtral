#!/usr/bin/env bash
# 8-task 0-shot harness for one local arm.  usage: eval_harness.sh <NAME> <MODEL_DIR>
set -u
NAME="$1"; DIR="$2"
source /home/mikecovlee/miniconda3/etc/profile.d/conda.sh
conda activate tinymixtral-cudev
export HTTP_PROXY=http://10.31.0.14:7890
export HTTPS_PROXY=http://10.31.0.14:7890
cd /home/mikecovlee/work/tinymixtral
mkdir -p evals/harness "$(dirname "$DIR")"
echo "=== $NAME HARNESS START $(date +%H:%M:%S) ==="
lm_eval --model hf \
  --model_args "pretrained=$DIR,tokenizer=$DIR,trust_remote_code=True,dtype=bfloat16" \
  --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq,lambada_openai \
  --batch_size 16 --device cuda --output_path "evals/harness/$NAME" \
  > "dpo_local2/$NAME.harness.log" 2>&1
echo "=== $NAME HARNESS rc=$? $(date +%H:%M:%S) ==="
