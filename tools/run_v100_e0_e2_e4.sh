#!/usr/bin/env bash

# Train the prioritized E0/E2/E4 comparison concurrently on three V100 GPUs.

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
  bash tools/run_v100_e0_e2_e4.sh \
    --root PATH \
    --event-cache-dir PATH \
    --teacher-cache-dir PATH \
    --checkpoint PATH \
    [--output-root PATH] [--batch-size N] [--max-steps N]

GPU assignment:
  GPU 0: E0, direct z-to-DINOv3 baseline
  GPU 1: E2, joint z/h-to-DINOv3
  GPU 2: E4, E2 plus temporal event dropout and masked z loss

All runs use dataset=dsec_det_train41: every canonical DSEC-Detection train
sequence receives gradient updates. Validation and best-checkpoint selection
are disabled; use checkpoints/step_00100000.pt for the default 100000-step run.
Official DSEC-Detection val/test sequences are never loaded.

E1 is intentionally deferred and is not launched by this script.
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
    -h|--help) usage; exit 0 ;;
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
[ "$MAX_STEPS" -gt 1000 ] || fail "--max-steps must exceed the default warmup"
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env first"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
if [ -z "$OUTPUT_ROOT" ]; then
  OUTPUT_ROOT="$PROJECT_ROOT/outputs/v100_dsec_det_e0_e2_e4_$(date +%Y%m%d_%H%M%S)"
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

  printf '[dsec-det-pretraining] GPU %s: starting %s\n' "$gpu" "$name"
  CUDA_VISIBLE_DEVICES=$gpu python train.py \
    dataset=dsec_det_train41 \
    model="$model" \
    experiment="$experiment" \
    dataset.root="$ROOT" \
    dataset.event_cache_dir="$EVENT_CACHE_DIR" \
    teacher.cache_dir="$TEACHER_CACHE_DIR" \
    teacher.checkpoint="$CHECKPOINT" \
    training.batch_size="$BATCH_SIZE" \
    training.max_steps="$MAX_STEPS" \
    training.validation_enabled=false \
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
launch 1 lstm h_distill_lstm_zloss e2_dual_distill_lstm
launch 2 lstm h_distill_lstm_zloss_event_dropout e4_dual_distill_lstm_event_dropout

STATUS=0
INDEX=0
for PID in "${PIDS[@]}"; do
  if wait "$PID"; then
    printf '[dsec-det-pretraining] complete: %s\n' "${NAMES[$INDEX]}"
  else
    printf '[dsec-det-pretraining] failed: %s (see %s/logs/%s.log)\n' \
      "${NAMES[$INDEX]}" "$OUTPUT_ROOT" "${NAMES[$INDEX]}" >&2
    STATUS=1
  fi
  INDEX=$((INDEX + 1))
done

trap - INT TERM
printf '[dsec-det-pretraining] outputs: %s\n' "$OUTPUT_ROOT"
exit "$STATUS"
