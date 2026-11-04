#!/usr/bin/env bash
# Server only. New root, synthetic preflight -> normalization -> GPU smoke -> full.
set -euo pipefail
if [[ $# != 1 ]]; then
  echo 'Usage: bash tools/start_stage1_common.sh NEW_RUN_ROOT' >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run_root="$1"
if [[ -e "$run_root" ]]; then
  echo "Choose a new run root: $run_root" >&2
  exit 2
fi
mkdir -p "$run_root/statistics"
run_root="$(cd "$run_root" && pwd)"
data_root="${DATA_ROOT:-/home/iASL/Arata_repo/dataset}"
teacher="${TEACHER:-/home/iASL/Arata_repo/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth}"
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/test_ddp_training.py tests/test_stage1_common_stdlib.py \
  tests/test_stream_sampling.py tests/test_training_runtime.py \
  tests/test_dense_protocol_stdlib.py tests/test_stage1_common_configs.py 2>&1 | tee "$run_root/preflight.log"
python tools/compute_dense_normalization.py --common-stage1 dsec \
  --event-cache-dir "$data_root/DSEC_cache/events/gep_rgb" \
  --output "$run_root/statistics/dsec_common.json" 2>&1 | tee "$run_root/statistics/dsec.log"
python tools/compute_dense_normalization.py --common-stage1 m3ed \
  --event-cache-dir "$data_root/m3ed_cache/half_dagr" \
  --output "$run_root/statistics/m3ed_common.json" 2>&1 | tee "$run_root/statistics/m3ed.log"
for stage in smoke full; do
  python tools/run_stage1_common.py --stage "$stage" --phase all \
    --data-root "$data_root" --teacher-checkpoint "$teacher" \
    --statistics-dir "$run_root/statistics" --output-root "$run_root/$stage" \
    --gpus "${GPUS:-0,1,2}" --execute --skip-preflight \
    2>&1 | tee "$run_root/${stage}_launcher.log"
  test -f "$run_root/$stage/_SUCCESS"
done
