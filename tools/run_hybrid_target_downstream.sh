#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
exec python "$SCRIPT_DIR/run_hybrid_target_downstream.py" "$@"
