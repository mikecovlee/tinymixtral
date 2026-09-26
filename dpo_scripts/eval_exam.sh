#!/usr/bin/env bash
# Evaluate an exam-SFT arm: 8-task harness + ceval/cmmlu + IFEval + GSM8K.
set -u
NAME="${1:-imp-base-exam}"
DIR="${2:-publish/$NAME}"
source /home/mikecovlee/miniconda3/etc/profile.d/conda.sh
conda activate tinymixtral-cudev
export HTTP_PROXY=http://10.31.0.14:7890
export HTTPS_PROXY=http://10.31.0.14:7890
export NLTK_ALLOW_PROXIED_URLOPEN=1
cd /home/mikecovlee/work/tinymixtral
MA="pretrained=$DIR,tokenizer=$DIR,trust_remote_code=True,dtype=bfloat16"

echo "=== $NAME HARNESS START $(date +%T) ==="
lm_eval --model hf --model_args "$MA" \
  --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq,lambada_openai \
  --batch_size 16 --device cuda --output_path "evals/harness/$NAME" > "dpo_local2/$NAME.harness.log" 2>&1
echo "=== $NAME HARNESS rc=$? $(date +%T) ==="

echo "=== $NAME EXAM START $(date +%T) ==="
lm_eval --model hf --model_args "$MA" \
  --tasks ceval-valid,cmmlu \
  --batch_size 8 --device cuda --output_path "evals/exam/$NAME" > "dpo_local2/$NAME.exam.log" 2>&1
echo "=== $NAME EXAM rc=$? $(date +%T) ==="

echo "=== $NAME IFEVAL START $(date +%T) ==="
lm_eval --model hf --model_args "$MA" \
  --tasks ifeval --apply_chat_template --batch_size 8 --device cuda \
  --output_path "evals/ifeval/$NAME" > "dpo_local2/$NAME.ifeval.log" 2>&1
echo "=== $NAME IFEVAL rc=$? $(date +%T) ==="

echo "=== $NAME GSM8K START $(date +%T) ==="
lm_eval --model hf --model_args "$MA" \
  --tasks gsm8k --batch_size 8 --device cuda \
  --output_path "evals/gsm8k/$NAME" > "dpo_local2/$NAME.gsm8k.log" 2>&1
echo "=== $NAME GSM8K rc=$? $(date +%T) ==="
echo "=== $NAME EXAM_EVAL_DONE ==="
