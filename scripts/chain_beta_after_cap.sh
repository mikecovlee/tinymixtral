#!/usr/bin/env bash
# Wait for the capacity_factor chain (PID 2090250) to finish, then run the beta_max scan.
set -uo pipefail
WAIT_PID="${1:-2090250}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
echo "[chain-beta] waiting for cap chain PID $WAIT_PID ..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[chain-beta] cap chain gone at $(date '+%H:%M:%S'), starting beta sweep"
bash "$REPO/scripts/run_beta_sweep.sh"
