#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
exec python "$SCRIPT_DIR/run_semantic_state_probe.py" "$@"
