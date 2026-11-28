#!/usr/bin/env bash
# Extended experiment, with outputs distinct from the block-3 run.
exec bash "$(dirname -- "$0")/run_b3_skip.sh" --skip-placement-suite "$@"
