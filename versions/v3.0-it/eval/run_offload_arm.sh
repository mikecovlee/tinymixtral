#!/usr/bin/env bash
# usage: run_offload_arm.sh <arm-name> <tag>
# Linux eval-machine counterpart of run_offload_arm.ps1:
# gen -> rubric -> harness -> ifeval -> gsm8k for one published arm.
# Env: EVAL_PY (default python), PROMPTS_FILE (default
#      versions/v3.0-it/eval_prompts/heldout_prompts_id_1k5.parquet), PROXY_URL (optional).
set -u
ARM=$1
TAG=$2
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
PY="${EVAL_PY:-python}"
PROMPTS="${PROMPTS_FILE:-versions/v3.0-it/eval_prompts/heldout_prompts_id_1k5.parquet}"
if [ -n "${PROXY_URL:-}" ]; then export HTTP_PROXY="$PROXY_URL" HTTPS_PROXY="$PROXY_URL"; fi
mkdir -p logs data/dpo
LOG="logs/offload_$TAG.log"
log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

mdir="publish/$ARM"
[ -d "$mdir" ] || { log "FATAL missing $mdir"; exit 1; }

log "GEN3 $ARM START"
"$PY" versions/v3.0-it/eval/dpo_eval_judge.py gen --model "$mdir" --prompts "$PROMPTS" \
  --out "data/dpo/evalv3b_$TAG.jsonl" --batch-size 8 --max-new-tokens 448 >>"$LOG" 2>&1
log "GEN3 $ARM rc=$?"

log "RUBRIC3 $ARM START"
"$PY" versions/v3.0-it/judge/rubric_judge2.py --responses "data/dpo/evalv3b_$TAG.jsonl" \
  --out "data/dpo/rubric2v2_$TAG.jsonl" --limit 5000 --concurrency 8 >>"$LOG" 2>&1
log "RUBRIC3 $ARM rc=$?"

ma="pretrained=$mdir,tokenizer=$mdir,trust_remote_code=True,dtype=bfloat16"

log "HARNESS $ARM START"
"$PY" -m lm_eval --model hf --model_args "$ma" \
  --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq,lambada_openai \
  --batch_size 16 --device cuda --output_path "evals/harness/$ARM" >>"$LOG" 2>&1
log "HARNESS $ARM rc=$?"

log "IFEVAL $ARM START"
"$PY" -m lm_eval --model hf --model_args "$ma" --tasks ifeval --apply_chat_template \
  --batch_size 8 --device cuda --output_path "evals/ifeval/$ARM" >>"$LOG" 2>&1
log "IFEVAL $ARM rc=$?"

log "GSM8K $ARM START"
"$PY" -m lm_eval --model hf --model_args "$ma" --tasks gsm8k \
  --batch_size 8 --device cuda --output_path "evals/gsm8k/$ARM" >>"$LOG" 2>&1
log "GSM8K $ARM rc=$?"

log "OFFLOAD_ARM_DONE $ARM"
