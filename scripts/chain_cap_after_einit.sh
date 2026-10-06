#!/usr/bin/env bash
set -uo pipefail
# Wait for the einit sweep driver to finish, then run the capacity_factor scan.
#
# This replaces chain_linear_after_einit.sh: the matched-budget linear proxy moved to the idle
# A5000 at 10.31.0.14 (see run_remote_proxy.ps1), so the local slot is repurposed for the
# capacity_factor scan instead of duplicating linear.
#
# Sequential (not concurrent) so neither run's throughput measurement is distorted and both
# stay inside the 200W power envelope.
WAIT_PID="${1:-1967408}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo "[chain] waiting for einit driver PID $WAIT_PID ..."
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done
echo "[chain] einit driver exited at $(date -Is)"

echo "[chain] starting capacity_factor scan at $(date -Is)"
bash "$REPO/scripts/run_cap_sweep.sh"
echo "[chain] capacity_factor scan finished at $(date -Is) rc=$?"
