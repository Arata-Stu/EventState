#!/usr/bin/env bash

# Train M3ED E3/E4 event-drop variants concurrently on two GPUs.

set -euo pipefail

ROOT=""
PREPARED_ROOT=""
TEACHER_CACHE_DIR=""
CHECKPOINT=""
EVENT_STATISTICS=""
OUTPUT_ROOT=""
BATCH_SIZE=4
MAX_STEPS=100000
SEQUENCE_LENGTH=16

usage() {
  printf '%s\n' \
    'Usage:' \
    '  bash tools/run_m3ed_event_dropout.sh \' \
    '    --root PATH \' \
    '    --prepared-root PATH \' \
    '    --teacher-cache-dir PATH \' \
    '    --checkpoint PATH \' \
    '    --event-statistics PATH \' \
    '    [--output-root PATH] [--batch-size N] [--max-steps N] \' \
    '    [--sequence-length N]' \
    '' \
    'GPU 0: E3, h distillation with temporal event dropout' \
    'GPU 1: E4, joint z/h distillation with dropout-frame z masking'
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) ROOT=${2:?}; shift 2 ;;
    --prepared-root) PREPARED_ROOT=${2:?}; shift 2 ;;
    --teacher-cache-dir) TEACHER_CACHE_DIR=${2:?}; shift 2 ;;
    --checkpoint) CHECKPOINT=${2:?}; shift 2 ;;
    --event-statistics) EVENT_STATISTICS=${2:?}; shift 2 ;;
    --output-root) OUTPUT_ROOT=${2:?}; shift 2 ;;
    --batch-size) BATCH_SIZE=${2:?}; shift 2 ;;
    --max-steps) MAX_STEPS=${2:?}; shift 2 ;;
    --sequence-length) SEQUENCE_LENGTH=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[ -d "$ROOT" ] || fail "M3ED root not found: $ROOT"
[ -d "$PREPARED_ROOT" ] || fail "prepared M3ED root not found: $PREPARED_ROOT"
[ -d "$TEACHER_CACHE_DIR" ] || fail "teacher cache not found: $TEACHER_CACHE_DIR"
[ -f "$CHECKPOINT" ] || fail "DINOv3 checkpoint not found: $CHECKPOINT"
[ -f "$EVENT_STATISTICS" ] || fail "event statistics not found: $EVENT_STATISTICS"
for value in "$BATCH_SIZE" "$MAX_STEPS" "$SEQUENCE_LENGTH"; do
  case "$value" in *[!0-9]*|'') fail "numeric options must be positive integers" ;; esac
  [ "$value" -gt 0 ] || fail "numeric options must be positive"
done
[ "$MAX_STEPS" -gt 1000 ] || fail "--max-steps must exceed the default warmup"
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env first"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
if [ -z "$OUTPUT_ROOT" ]; then
  OUTPUT_ROOT="$PROJECT_ROOT/outputs/m3ed_event_dropout_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_ROOT/logs"
cd "$PROJECT_ROOT"

NORMALIZE_MEAN=$(python -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["normalize_mean"]))' "$EVENT_STATISTICS")
NORMALIZE_STD=$(python -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["normalize_std"]))' "$EVENT_STATISTICS")

PIDS=()
NAMES=()

launch() {
  local gpu=$1
  local experiment=$2
  local name=$3
  local run_dir="$OUTPUT_ROOT/$name"
  local log_path="$OUTPUT_ROOT/logs/$name.log"

  printf '[m3ed-pretraining] GPU %s: starting %s\n' "$gpu" "$name"
  CUDA_VISIBLE_DEVICES=$gpu python train.py \
    dataset=m3ed_half_dagr \
    model=lstm \
    experiment="$experiment" \
    dataset.root="$ROOT" \
    dataset.prepared_root="$PREPARED_ROOT" \
    dataset.sequence_length="$SEQUENCE_LENGTH" \
    dataset.representation.normalize_mean="$NORMALIZE_MEAN" \
    dataset.representation.normalize_std="$NORMALIZE_STD" \
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

launch 0 h_distill_lstm_event_dropout e3_h_distill_lstm_event_dropout
launch 1 h_distill_lstm_zloss_event_dropout e4_dual_distill_lstm_event_dropout

STATUS=0
INDEX=0
for PID in "${PIDS[@]}"; do
  if wait "$PID"; then
    printf '[m3ed-pretraining] complete: %s\n' "${NAMES[$INDEX]}"
  else
    printf '[m3ed-pretraining] failed: %s (see logs)\n' "${NAMES[$INDEX]}" >&2
    STATUS=1
  fi
  INDEX=$((INDEX + 1))
done

trap - INT TERM
printf '[m3ed-pretraining] outputs: %s\n' "$OUTPUT_ROOT"
exit "$STATUS"
