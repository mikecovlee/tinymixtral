#!/usr/bin/env bash
# usage: offload_arm.sh <arm-name> <tag> [checkpoint-dir]
# Publishes a checkpoint locally, scp's it to the eval machine, then prints the
# tmux command that runs the full offloaded eval (gen + rubric + lm-eval).
# Env: PUBLISH_PY (python with torch+transformers; default `python` on PATH),
#      TOKENIZER_SNAP (default latest local models--mikecovlee--tinymixtral
#      snapshot), EVAL_USER + EVAL_HOST + EVAL_REPO (required; EVAL_REPO is
#      the repo path on the eval machine, forward slashes).
set -euo pipefail
ARM=$1
TAG=$2
CKPT=${3:-}
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
PY="${PUBLISH_PY:-python}"
: "${EVAL_USER:?set EVAL_USER (ssh user on the eval machine)}"
: "${EVAL_HOST:?set EVAL_HOST (eval machine host/IP)}"
: "${EVAL_REPO:?set EVAL_REPO (repo path on the eval machine)}"
if [ -z "${TOKENIZER_SNAP:-}" ]; then
  TOKENIZER_SNAP=$(ls -d "$HOME"/.cache/huggingface/hub/models--mikecovlee--tinymixtral/snapshots/*/ | head -1)
fi
if [ -z "$CKPT" ]; then
  CKPT=$(ls -d checkpoints/sft_${TAG}/step_*_final 2>/dev/null | tail -1)
fi
if [ -z "$CKPT" ]; then
  ALT=$(echo "$TAG" | sed -E 's/^sftv([0-9]+)v([0-9]+)$/sft_v\1_v\2/')
  CKPT=$(ls -d checkpoints/${ALT}/step_*_final 2>/dev/null | tail -1)
fi
if [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then
  echo "ERROR: checkpoint not found: '$CKPT'"; exit 1
fi
echo "checkpoint = $CKPT"
echo "tokenizer  = $TOKENIZER_SNAP"
"$PY" scripts/publish_hf.py --checkpoint "$CKPT" --output "publish/$ARM" --tokenizer "$TOKENIZER_SNAP"
ls -la "publish/$ARM"
if ls "publish/$ARM"/model.safetensors >/dev/null 2>&1; then
  echo "ERROR: publish/$ARM contains model.safetensors (whitelist regression)"; exit 1
fi
REMOTE="$EVAL_REPO/publish/$ARM"
timeout 90 ssh "$EVAL_USER@$EVAL_HOST" "Remove-Item -Recurse -Force '$REMOTE' -ErrorAction SilentlyContinue" 2>/dev/null || true
scp -q -r "publish/$ARM" "$EVAL_USER@$EVAL_HOST:'$EVAL_REPO/publish/'"
echo "scp done"
echo
echo "now run on the eval machine:"
echo "  ${TMUX_EXE:-C:\\Software\\Shell\\bin\\tmux.exe} new-session -d -s $TAG \"powershell -NoProfile -ExecutionPolicy Bypass -File $EVAL_REPO\\versions\\v3.0-it\\eval\\run_offload_arm.ps1 -Arm $ARM -Tag $TAG\""
