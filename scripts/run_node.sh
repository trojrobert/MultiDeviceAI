#!/usr/bin/env bash
#
# Start this device as a node in the MultiDeviceLLM cluster.
# Other devices on the same network auto-discover it. API + dashboard on :52415.
#
# Usage: ./scripts/run_node.sh [--no-worker]
set -euo pipefail

EXO_DIR="${MDLLM_EXO_DIR:-$(cd "$(dirname "$0")/.." && pwd)/vendor/exo}"

if [ ! -d "${EXO_DIR}" ]; then
  echo "exo not found at ${EXO_DIR}. Run ./scripts/bootstrap_mac.sh first." >&2
  exit 1
fi

cd "${EXO_DIR}"
echo "Starting exo node. Dashboard: http://localhost:52415  (Ctrl-C to leave)"
exec uv run exo "$@"
