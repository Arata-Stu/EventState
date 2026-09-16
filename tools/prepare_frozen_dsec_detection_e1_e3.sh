#!/usr/bin/env bash

# Export final E1/E3 hidden-state maps and warp them to DSEC-Det geometry.

set -euo pipefail

RUN_DIR=""
EVENT_CACHE_DIR=""
DATASET_ROOT=""
OUTPUT_ROOT=""
TEACHER_CHECKPOINT=""
ROLES="train,val"
GPU_E1=0
GPU_E3=1

usage() {
  cat <<'EOF'
Usage:
  bash tools/prepare_frozen_dsec_detection_e1_e3.sh \
    --run-dir PATH --event-cache-dir PATH --dataset-root PATH --output-root PATH \
    [--teacher-checkpoint PATH] [--roles train,val|test|train,val,test] \
    [--gpu-e1 N] [--gpu-e3 N]

Only the LSTM hidden feature h is stored. E1 and E3 run concurrently and
existing completed sequence caches are reused by the underlying exporters.
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
    --roles) ROLES=${2:?}; shift 2 ;;
    --gpu-e1) GPU_E1=${2:?}; shift 2 ;;
    --gpu-e3) GPU_E3=${2:?}; shift 2 ;;
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

IFS=',' read -r -a ROLE_VALUES <<< "$ROLES"
for role in "${ROLE_VALUES[@]}"; do
  case "$role" in
    train|val|test) ;;
    *) fail "roles must be a comma-separated subset of train,val,test" ;;
  esac
done

prepare_model() {
  local gpu=$1
  local label=$2
  local run_name=$3
  local checkpoint="$RUN_DIR/$run_name/checkpoints/step_00100000.pt"
  local rectified="$OUTPUT_ROOT/rectified/$label"
  local benchmark="$OUTPUT_ROOT/dsec_det/$label"
  local teacher_args=()
  [ -f "$checkpoint" ] || fail "final checkpoint not found: $checkpoint"
  if [ -n "$TEACHER_CHECKPOINT" ]; then
    teacher_args=(--teacher-checkpoint "$TEACHER_CHECKPOINT")
  fi

  for role in "${ROLE_VALUES[@]}"; do
    printf '[frozen-cache] %s role=%s export h\n' "$label" "$role"
    CUDA_VISIBLE_DEVICES="$gpu" python tools/cache_dsec_detection_features.py \
      --checkpoint "$checkpoint" \
      --event-cache-dir "$EVENT_CACHE_DIR" \
      --output-dir "$rectified" \
      --role "$role" \
      --features h \
      --state-policy continuous \
      --device cuda \
      "${teacher_args[@]}"

    printf '[frozen-cache] %s role=%s warp h\n' "$label" "$role"
    CUDA_VISIBLE_DEVICES="$gpu" python tools/prepare_dsec_detection_benchmark_features.py \
      --input-dir "$rectified" \
      --output-dir "$benchmark" \
      --dataset-root "$DATASET_ROOT" \
      --role "$role" \
      --features h \
      --device cuda
  done
}

prepare_model "$GPU_E1" E1 e1_h_distill_lstm \
  >"$OUTPUT_ROOT/logs/E1.log" 2>&1 &
PID_E1=$!
printf '[frozen-cache] E1 gpu=%s pid=%s\n' "$GPU_E1" "$PID_E1"

prepare_model "$GPU_E3" E3 e3_h_distill_lstm_event_dropout \
  >"$OUTPUT_ROOT/logs/E3.log" 2>&1 &
PID_E3=$!
printf '[frozen-cache] E3 gpu=%s pid=%s\n' "$GPU_E3" "$PID_E3"

STATUS=0
wait "$PID_E1" || STATUS=1
wait "$PID_E3" || STATUS=1
[ "$STATUS" -eq 0 ] || fail "one or more feature exports failed; inspect $OUTPUT_ROOT/logs"
printf '[frozen-cache] complete: %s/dsec_det\n' "$OUTPUT_ROOT"
