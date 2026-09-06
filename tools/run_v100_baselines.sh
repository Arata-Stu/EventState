#!/usr/bin/env bash

# Run the benchmark-clean E0/E1/E2 comparison concurrently on three V100 GPUs.

set -euo pipefail

ROOT=""
EVENT_CACHE_DIR=""
TEACHER_CACHE_DIR=""
CHECKPOINT=""
OUTPUT_ROOT=""
BATCH_SIZE=8
MAX_STEPS=100000

usage() {
  cat <<'EOF'
Usage:
  bash tools/run_v100_baselines.sh \
    --root PATH \
    --event-cache-dir PATH \
    --teacher-cache-dir PATH \
    --checkpoint PATH \
    [--output-root PATH] [--batch-size N] [--max-steps N]

GPU assignment:
  GPU 0: E0, direct z-to-DINOv3 baseline
  GPU 1: E1, LSTM h-to-DINOv3
  GPU 2: E2, joint z/h-to-DINOv3

The launcher uses the fixed dataset=dsec_benchmark_clean protocol and writes
one log and one Hydra run directory per experiment. Re-running without an
explicit training.resume starts new runs; it never overwrites checkpoints.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root)
      [ "$#" -ge 2 ] || fail "--root requires a value"
      ROOT=$2
      shift 2
      ;;
    --event-cache-dir)
      [ "$#" -ge 2 ] || fail "--event-cache-dir requires a value"
      EVENT_CACHE_DIR=$2
      shift 2
      ;;
    --teacher-cache-dir)
      [ "$#" -ge 2 ] || fail "--teacher-cache-dir requires a value"
      TEACHER_CACHE_DIR=$2
      shift 2
      ;;
    --checkpoint)
      [ "$#" -ge 2 ] || fail "--checkpoint requires a value"
      CHECKPOINT=$2
      shift 2
      ;;
    --output-root)
      [ "$#" -ge 2 ] || fail "--output-root requires a value"
      OUTPUT_ROOT=$2
      shift 2
      ;;
    --batch-size)
      [ "$#" -ge 2 ] || fail "--batch-size requires a value"
      BATCH_SIZE=$2
      shift 2
      ;;
    --max-steps)
      [ "$#" -ge 2 ] || fail "--max-steps requires a value"
      MAX_STEPS=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[ -n "$ROOT" ] || fail "--root is required"
[ -n "$EVENT_CACHE_DIR" ] || fail "--event-cache-dir is required"
[ -n "$TEACHER_CACHE_DIR" ] || fail "--teacher-cache-dir is required"
[ -n "$CHECKPOINT" ] || fail "--checkpoint is required"
[ -d "$ROOT" ] || fail "DSEC root not found: $ROOT"
[ -d "$EVENT_CACHE_DIR" ] || fail "event cache not found: $EVENT_CACHE_DIR"
[ -d "$TEACHER_CACHE_DIR" ] || fail "teacher cache not found: $TEACHER_CACHE_DIR"
[ -f "$CHECKPOINT" ] || fail "DINOv3 checkpoint not found: $CHECKPOINT"
case "$BATCH_SIZE" in *[!0-9]*|'') fail "--batch-size must be a positive integer" ;; esac
case "$MAX_STEPS" in *[!0-9]*|'') fail "--max-steps must be a positive integer" ;; esac
[ "$BATCH_SIZE" -gt 0 ] || fail "--batch-size must be positive"
[ "$MAX_STEPS" -gt 1000 ] || fail "--max-steps must exceed the default 1000 warmup steps"
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env first"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
if [ -z "$OUTPUT_ROOT" ]; then
  OUTPUT_ROOT="$PROJECT_ROOT/outputs/v100_baselines_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_ROOT/logs"
cd "$PROJECT_ROOT"

PIDS=()
NAMES=()

launch() {
  local gpu=$1
  local model=$2
  local experiment=$3
  local name=$4
  local run_dir="$OUTPUT_ROOT/$name"
  local log_path="$OUTPUT_ROOT/logs/$name.log"

  printf '[v100-baselines] GPU %s: starting %s\n' "$gpu" "$name"
  CUDA_VISIBLE_DEVICES=$gpu python train.py \
    dataset=dsec_benchmark_clean \
    model="$model" \
    experiment="$experiment" \
    dataset.root="$ROOT" \
    dataset.event_cache_dir="$EVENT_CACHE_DIR" \
    teacher.cache_dir="$TEACHER_CACHE_DIR" \
    teacher.checkpoint="$CHECKPOINT" \
    training.batch_size="$BATCH_SIZE" \
    training.max_steps="$MAX_STEPS" \
    hydra.run.dir="$run_dir" \
    >"$log_path" 2>&1 &
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

launch 0 no_memory gep_baseline e0_gep_like_dinov3
launch 1 lstm h_distill_lstm e1_h_distill_lstm
launch 2 lstm h_distill_lstm_zloss e2_dual_distill_lstm

STATUS=0
INDEX=0
for PID in "${PIDS[@]}"; do
  if wait "$PID"; then
    printf '[v100-baselines] complete: %s\n' "${NAMES[$INDEX]}"
  else
    printf '[v100-baselines] failed: %s (see %s/logs/%s.log)\n' \
      "${NAMES[$INDEX]}" "$OUTPUT_ROOT" "${NAMES[$INDEX]}" >&2
    STATUS=1
  fi
  INDEX=$((INDEX + 1))
done

trap - INT TERM
printf '[v100-baselines] outputs: %s\n' "$OUTPUT_ROOT"
exit "$STATUS"
