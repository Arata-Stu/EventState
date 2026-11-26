#!/usr/bin/env bash
# Compare the residual block-3 model with the existing frozen Semantic controls.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."

gpus=${1:-0,1,2}
for stage in smoke full; do
    root="outputs/stage1_b3_skip_${stage}_20261126"
    if [[ -f "$root/_SUCCESS" ]]; then
        echo "Completed Stage 1 $stage: $root"
        continue
    fi
    env/bin/python -u tools/run_stage1_common.py --b3-skip --stage "$stage" \
        --gpus "$gpus" --seed 0 --output-root "$root" \
        --statistics-dir outputs/stage1_joined_baseline_full_20261107 --execute
done
env/bin/python -u tools/run_placement_semantic.py --conditions E1_b3_skip \
    --gpus "$gpus" --execute
