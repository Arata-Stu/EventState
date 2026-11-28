#!/usr/bin/env bash
# Stage 1 and frozen Semantic evaluation for the block-3 or extended residual experiments.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
export PATH="$PWD/env/bin:$PATH"

pipeline=b3_skip
date_tag=20261126
stage1_args=(--b3-skip)
semantic_args=(--conditions E1_b3_skip)
monitor=(python3 -u -m tools.b3_skip_progress)
if [[ ${1:-} == --skip-placement-suite ]]; then
    shift
    pipeline=skip_placement
    date_tag=20261128
    stage1_args=(--skip-placement-suite)
    semantic_args=(--skip-placement-suite)
    monitor+=(--skip-placement-suite)
fi
if [[ ${1:-} == --final-skip-suite ]]; then
    shift
    pipeline=final_skip
    date_tag=20261128
    stage1_args=(--final-skip-suite)
    semantic_args=(--final-skip-suite)
    monitor+=(--final-skip-suite)
fi
mode=run
case "${1:-}" in
    --follow|--watch|--status) mode=${1#--}; shift ;;
    -h|--help)
        echo "Usage: bash tools/{run_b3_skip,run_skip_placements}.sh [--follow|--watch|--status] [0,1,2]"
        echo "  --follow  start detached on the Linux server and show its log"
        echo "  --watch   show progress of an existing run without starting jobs"
        echo "  --status  print progress once"
        exit 0 ;;
esac
if (( $# > 1 )); then
    echo "Expected a single comma-separated GPU list" >&2
    exit 2
fi
gpus=${1:-0,1,2}
case "$mode" in
    watch)  exec "${monitor[@]}" --gpus "$gpus" --watch ;;
    status) exec "${monitor[@]}" --gpus "$gpus" ;;
    follow) exec "${monitor[@]}" --gpus "$gpus" --follow ;;
esac

"${monitor[@]}" --watch --interval 300 --parent-pid "$$" --gpus "$gpus" &
monitor_pid=$!
stop_monitor() {
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
}
trap stop_monitor EXIT
for stage in smoke full; do
    root="outputs/stage1_${pipeline}_${stage}_${date_tag}"
    if [[ -f "$root/_SUCCESS" ]]; then
        echo "Completed Stage 1 $stage: $root"
        continue
    fi
    env/bin/python -u tools/run_stage1_common.py "${stage1_args[@]}" --stage "$stage" \
        --gpus "$gpus" --seed 0 --output-root "$root" \
        --statistics-dir outputs/stage1_joined_baseline_full_20261107 --execute
done
env/bin/python -u tools/run_placement_semantic.py "${semantic_args[@]}" \
    --gpus "$gpus" --execute
