#!/usr/bin/env bash

# Export final E0/E2/E4 maps and warp them to the official DSEC-Det geometry.

set -euo pipefail

RUN_DIR=""
EVENT_CACHE_DIR=""
DATASET_ROOT=""
OUTPUT_ROOT=""
TEACHER_CHECKPOINT=""

usage() {
  cat <<'EOF'
Usage:
  bash tools/prepare_frozen_dsec_detection_e0_e2_e4.sh \
    --run-dir PATH --event-cache-dir PATH --dataset-root PATH --output-root PATH \
    [--teacher-checkpoint PATH]

GPU 0/1/2 export E0/E2/E4 concurrently. Train and validation roles are
processed, then warped into OUTPUT_ROOT/dsec_det/{E0,E2,E4}.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-dir) RUN_DIR=${2:?}; shift 2 ;;
    --event-cache-dir) EVENT_CACHE_DIR=${2:?}; shift 2 ;;
    --dataset-root) DATASET_ROOT=${2:?}; shift 2 ;;
    --output-root) OUTPUT_ROOT=${2:?}; shift 2 ;;
    --teacher-checkpoint) TEACHER_CHECKPOINT=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[ -d "$RUN_DIR" ] || fail "run directory not found: $RUN_DIR"
[ -d "$EVENT_CACHE_DIR" ] || fail "event cache not found: $EVENT_CACHE_DIR"
[ -d "$DATASET_ROOT" ] || fail "dataset root not found: $DATASET_ROOT"
[ -n "$OUTPUT_ROOT" ] || fail "--output-root is required"
if [ -n "$TEACHER_CHECKPOINT" ]; then
  [ -f "$TEACHER_CHECKPOINT" ] || fail "teacher checkpoint not found: $TEACHER_CHECKPOINT"
fi
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env"

RUN_DIR=$(cd "$RUN_DIR" && pwd -P)
EVENT_CACHE_DIR=$(cd "$EVENT_CACHE_DIR" && pwd -P)
DATASET_ROOT=$(cd "$DATASET_ROOT" && pwd -P)
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT=$(cd "$OUTPUT_ROOT" && pwd -P)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
mkdir -p "$OUTPUT_ROOT/logs"
cd "$PROJECT_ROOT"

LABELS=(E0 E2 E4)
RUN_NAMES=(
  e0_gep_like_dinov3
  e2_dual_distill_lstm
  e4_dual_distill_lstm_event_dropout
)
PIDS=()

prepare_model() {
  local gpu=$1
  local label=$2
  local run_name=$3
  local checkpoint="$RUN_DIR/$run_name/checkpoints/step_00100000.pt"
  local rectified="$OUTPUT_ROOT/rectified/$label"
  local benchmark="$OUTPUT_ROOT/dsec_det/$label"
  [ -f "$checkpoint" ] || fail "final checkpoint not found: $checkpoint"

  for role in train val; do
    local teacher_args=()
    if [ -n "$TEACHER_CHECKPOINT" ]; then
      teacher_args=(--teacher-checkpoint "$TEACHER_CHECKPOINT")
    fi
    CUDA_VISIBLE_DEVICES="$gpu" python tools/cache_dsec_detection_features.py \
      --checkpoint "$checkpoint" \
      --event-cache-dir "$EVENT_CACHE_DIR" \
      --output-dir "$rectified" \
      --role "$role" \
      --features z h \
      --state-policy continuous \
      --device cuda \
      "${teacher_args[@]}"

    CUDA_VISIBLE_DEVICES="$gpu" python tools/prepare_dsec_detection_benchmark_features.py \
      --input-dir "$rectified" \
      --output-dir "$benchmark" \
      --dataset-root "$DATASET_ROOT" \
      --role "$role" \
      --features z h \
      --device cuda
  done
}

for gpu in 0 1 2; do
  prepare_model "$gpu" "${LABELS[$gpu]}" "${RUN_NAMES[$gpu]}" \
    >"$OUTPUT_ROOT/logs/${LABELS[$gpu]}.log" 2>&1 &
  PIDS+=("$!")
  printf '[frozen-cache] %s gpu=%s pid=%s\n' "${LABELS[$gpu]}" "$gpu" "$!"
done

STATUS=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || STATUS=1
done
[ "$STATUS" -eq 0 ] || fail "one or more feature exports failed; inspect $OUTPUT_ROOT/logs"
printf '[frozen-cache] complete: %s/dsec_det\n' "$OUTPUT_ROOT"
