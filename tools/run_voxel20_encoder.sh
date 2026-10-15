#!/usr/bin/env bash
# GPU server only. Raw events and online frozen RGB teacher; no dense caches.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "$SCRIPT_DIR/.."
STAGE=${1:-smoke}
STATS=${2:?Usage: bash tools/run_voxel20_encoder.sh smoke\|full /absolute/stats.json [320|160]}
RESOLUTION=${3:-320}
case "$RESOLUTION" in
  320|160) ;;
  *) echo "resolution must be 320 or 160" >&2; exit 2 ;;
esac
MODE=${4:-linear}
SUFFIX=""
case "$MODE" in
  linear) ;;
  none) SUFFIX="_histogram" ;;
  *) echo "mode must be linear or none" >&2; exit 2 ;;
esac
GPU=${GPU:-0}
DATA_ROOT=${DATA_ROOT:-/home/iASL/Arata_repo/dataset/DSEC}
TEACHER=${TEACHER:-/home/iASL/Arata_repo/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth}
case "$STAGE" in
  smoke) STEPS=100; WARMUP=10; SAVE=100 ;;
  full) STEPS=100000; WARMUP=1000; SAVE=10000 ;;
  *) echo 'stage must be smoke or full' >&2; exit 2 ;;
esac
[[ -f "$STATS" && -f "$TEACHER" && -d "$DATA_ROOT" ]] || { echo 'Missing stats/teacher/dataset' >&2; exit 2; }
OUTPUT="outputs/voxel20_encoder_${RESOLUTION}${SUFFIX}_${STAGE}_$(date +%Y%m%d_%H%M%S)"
[[ ! -e "$OUTPUT" ]] || { echo 'Output exists' >&2; exit 2; }
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_histogram_encoder.py tests/test_voxel20_encoder.py tests/test_fixed_normalization.py tests/test_models.py tests/test_transforms.py
mkdir -p "$OUTPUT"
cp "$STATS" "$OUTPUT/normalization.json"
git rev-parse HEAD > "$OUTPUT/revision.txt"
git diff HEAD > "$OUTPUT/source.patch"
echo "[voxel20] outputs: $OUTPUT"
CUDA_VISIBLE_DEVICES="$GPU" python train.py \
  dataset=dsec_det_train41_voxel20 model=no_memory "experiment=voxel20_encoder_${RESOLUTION}${SUFFIX}" \
  "dataset.root=$DATA_ROOT" "teacher.checkpoint=$TEACHER" \
  "dataset.representation.fixed_normalization_file=$STATS" \
  "training.max_steps=$STEPS" "scheduler.warmup_steps=$WARMUP" \
  "training.checkpoint_every=$SAVE" "hydra.run.dir=$OUTPUT" \
  2>&1 | tee "$OUTPUT/train.log"
printf -v CHECKPOINT 'step_%08d.pt' "$STEPS"
python tools/export_event_encoder.py --checkpoint "$OUTPUT/checkpoints/$CHECKPOINT" \
  --output "$OUTPUT/event_encoder.pt"
