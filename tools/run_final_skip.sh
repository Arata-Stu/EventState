#!/usr/bin/env bash
# Final-only and four-stage skip experiments.
exec bash "$(dirname -- "$0")/run_b3_skip.sh" --final-skip-suite "$@"
