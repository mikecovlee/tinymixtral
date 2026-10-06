#!/usr/bin/env bash
set -uo pipefail
# Wait for the einit sweep driver to finish, then run the matched-budget linear proxy.
# Sequential (not concurrent) so neither run's throughput measurement is distorted and
# both stay inside the 200W power envelope.
WAIT_PID="${1:-1967408}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo "[chain] waiting for einit driver PID $WAIT_PID ..."
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done
echo "[chain] einit driver exited at $(date -Is)"

echo "[chain] starting linear proxy at $(date -Is)"
bash "$REPO/scripts/run_linear_proxy.sh"
echo "[chain] linear proxy finished at $(date -Is) rc=$?"
