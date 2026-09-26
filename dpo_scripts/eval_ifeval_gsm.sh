#!/usr/bin/env bash
# IFEval + GSM8K for one local arm (proxy required for the HF datasets).
set -u
NAME="$1"; DIR="$2"
source /home/mikecovlee/miniconda3/etc/profile.d/conda.sh
conda activate tinymixtral-cudev
export HTTP_PROXY=http://10.31.0.14:7890
export HTTPS_PROXY=http://10.31.0.14:7890
export NLTK_ALLOW_PROXIED_URLOPEN=1
cd /home/mikecovlee/work/tinymixtral
MA="pretrained=$DIR,tokenizer=$DIR,trust_remote_code=True,dtype=bfloat16"
echo "=== $NAME IFEVAL START $(date +%T) ==="
lm_eval --model hf --model_args "$MA" --tasks ifeval --apply_chat_template \
  --batch_size 8 --device cuda --output_path "evals/ifeval/$NAME" > "dpo_local2/$NAME.ifeval.log" 2>&1
echo "=== $NAME IFEVAL rc=$? $(date +%T) ==="
lm_eval --model hf --model_args "$MA" --tasks gsm8k \
  --batch_size 8 --device cuda --output_path "evals/gsm8k/$NAME" > "dpo_local2/$NAME.gsm8k.log" 2>&1
echo "=== $NAME GSM8K rc=$? $(date +%T) ==="
echo "=== $NAME IFEVAL_GSM_DONE ==="
