#!/usr/bin/env bash

# Train the missing E1/E3 ablations on the official DSEC-Detection train split.

set -euo pipefail

ROOT=""
EVENT_CACHE_DIR=""
TEACHER_CACHE_DIR=""
CHECKPOINT=""
OUTPUT_ROOT=""
BATCH_SIZE=8
MAX_STEPS=100000
GPU_E1=0
GPU_E3=1

usage() {
  cat <<'EOF'
Usage:
  bash tools/run_v100_e1_e3.sh \
    --root PATH \
    --event-cache-dir PATH \
    --teacher-cache-dir PATH \
    --checkpoint PATH \
    [--output-root PATH] [--batch-size 8] [--max-steps 100000] \
    [--gpu-e1 0] [--gpu-e3 1]

GPU assignment by default:
  GPU 0: E1, h-only DINOv3 distillation
  GPU 1: E3, E1 plus temporal event dropout

Both runs use dataset=dsec_det_train41, a one-layer LSTM, and no representation
validation. The final comparison checkpoint is step_00100000.pt. Reusing the
same --output-root resumes each incomplete run from its latest step checkpoint.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) ROOT=${2:?}; shift 2 ;;
    --event-cache-dir) EVENT_CACHE_DIR=${2:?}; shift 2 ;;
    --teacher-cache-dir) TEACHER_CACHE_DIR=${2:?}; shift 2 ;;
    --checkpoint) CHECKPOINT=${2:?}; shift 2 ;;
    --output-root) OUTPUT_ROOT=${2:?}; shift 2 ;;
    --batch-size) BATCH_SIZE=${2:?}; shift 2 ;;
    --max-steps) MAX_STEPS=${2:?}; shift 2 ;;
    --gpu-e1) GPU_E1=${2:?}; shift 2 ;;
    --gpu-e3) GPU_E3=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[ -d "$ROOT" ] || fail "DSEC root not found: $ROOT"
[ -d "$EVENT_CACHE_DIR" ] || fail "event cache not found: $EVENT_CACHE_DIR"
[ -d "$TEACHER_CACHE_DIR" ] || fail "teacher cache not found: $TEACHER_CACHE_DIR"
[ -f "$CHECKPOINT" ] || fail "DINOv3 checkpoint not found: $CHECKPOINT"
for value in "$BATCH_SIZE" "$MAX_STEPS" "$GPU_E1" "$GPU_E3"; do
  case "$value" in *[!0-9]*|'') fail "numeric arguments must be non-negative integers" ;; esac
done
[ "$BATCH_SIZE" -gt 0 ] || fail "--batch-size must be positive"
[ "$MAX_STEPS" -gt 1000 ] || fail "--max-steps must exceed the default warmup"
[ "$GPU_E1" != "$GPU_E3" ] || fail "E1 and E3 must use different GPUs"
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
if [ -z "$OUTPUT_ROOT" ]; then
  OUTPUT_ROOT="$PROJECT_ROOT/outputs/v100_dsec_det_e1_e3_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_ROOT/logs"
OUTPUT_ROOT=$(cd "$OUTPUT_ROOT" && pwd -P)
cd "$PROJECT_ROOT"

PIDS=()
NAMES=()

launch() {
  local gpu=$1
  local experiment=$2
  local name=$3
  local run_dir="$OUTPUT_ROOT/$name"
  local checkpoint_dir="$run_dir/checkpoints"
  local final_checkpoint="$checkpoint_dir/step_$(printf '%08d' "$MAX_STEPS").pt"
  local log_path="$OUTPUT_ROOT/logs/$name.log"
  local resume_args=()

  if [ -f "$final_checkpoint" ]; then
    printf '[e1-e3-pretraining] complete checkpoint exists; skip %s: %s\n' \
      "$name" "$final_checkpoint"
    return
  fi
  if [ -d "$checkpoint_dir" ]; then
    local latest
    latest=$(find "$checkpoint_dir" -maxdepth 1 -type f -name 'step_*.pt' | sort | tail -n 1)
    if [ -n "$latest" ]; then
      resume_args+=(training.resume="$latest")
      printf '[e1-e3-pretraining] GPU %s: resume %s from %s\n' \
        "$gpu" "$name" "$latest"
    fi
  fi
  if [ "${#resume_args[@]}" -eq 0 ]; then
    printf '[e1-e3-pretraining] GPU %s: start %s\n' "$gpu" "$name"
  fi

  CUDA_VISIBLE_DEVICES="$gpu" python train.py \
    dataset=dsec_det_train41 \
    model=lstm \
    experiment="$experiment" \
    dataset.root="$ROOT" \
    dataset.event_cache_dir="$EVENT_CACHE_DIR" \
    teacher.cache_dir="$TEACHER_CACHE_DIR" \
    teacher.checkpoint="$CHECKPOINT" \
    training.batch_size="$BATCH_SIZE" \
    training.max_steps="$MAX_STEPS" \
    training.validation_enabled=false \
    hydra.run.dir="$run_dir" \
    "${resume_args[@]}" \
    >>"$log_path" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
}

terminate_children() {
  local pid
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

launch "$GPU_E1" h_distill_lstm e1_h_distill_lstm
launch "$GPU_E3" h_distill_lstm_event_dropout e3_h_distill_lstm_event_dropout

STATUS=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    printf '[e1-e3-pretraining] complete: %s\n' "${NAMES[$index]}"
  else
    printf '[e1-e3-pretraining] failed: %s (see %s/logs/%s.log)\n' \
      "${NAMES[$index]}" "$OUTPUT_ROOT" "${NAMES[$index]}" >&2
    STATUS=1
  fi
done

trap - INT TERM
printf '[e1-e3-pretraining] outputs: %s\n' "$OUTPUT_ROOT"
exit "$STATUS"
