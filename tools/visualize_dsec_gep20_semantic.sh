#!/usr/bin/env bash
# Baseline z+h distillation at both windows; matched h readout by default.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
data="${DATA_ROOT:-/home/iASL/Arata_repo/dataset}"
feature="${FEATURE:-h}"
sequence="${SEQUENCE:-zurich_city_07_a}"
output="${OUTPUT_DIR:-outputs/gep20_semantic_video_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
python tools/visualize_dsec_semantic_windows.py \
  --backbone-50 outputs/hybrid_targets_full_20261001_231617/dsec_z_h/checkpoints/step_00100000.pt \
  --head-50 "outputs/hybrid_target_downstream_20261004/dsec_z_h/semantic/$feature/seed_0/best.pt" \
  --backbone-20 outputs/dsec_gep20_full_20261019_235316/baseline/checkpoints/step_00100000.pt \
  --head-20 "outputs/dsec_gep20_downstream_20261021/dsec_gep20_baseline/semantic/$feature/seed_0/best.pt" \
  --dataset-root "$data/DSEC" --event-cache-dir "$data/DSEC_cache/events/gep_rgb" \
  --teacher-checkpoint /home/iASL/Arata_repo/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --feature "$feature" --sequence "$sequence" --output-dir "$output" "$@"
