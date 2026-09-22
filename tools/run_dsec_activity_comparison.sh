#!/usr/bin/env bash
# Three matched final-fit experiments. Run on the GPU training host.
set -euo pipefail

ROOT=""
EVENT_CACHE_DIR=""
TEACHER_CACHE_DIR=""
CHECKPOINT=""
OUTPUT_ROOT=""
GPU_LIST="0,1,2"
STAGE="smoke"
BATCH_SIZE=8
NUM_WORKERS=4
SEED=0
DRY_RUN=0
RUN_TESTS=1

usage() {
  cat <<'EOF'
Usage:
  bash tools/run_dsec_activity_comparison.sh \
    --root PATH --event-cache-dir PATH --teacher-cache-dir PATH \
    --checkpoint PATH [--gpus 0,1,2] [--stage smoke|full] \
    [--output-root PATH] [--batch-size 8] [--num-workers 4] [--seed 0] \
    [--dry-run] [--skip-tests]

GPU order: baseline E2 / activity-only cosine+MSE / ScaleEvent CrossGram.
All use the same LSTM, seed, batch, optimizer and 41 training sequences.
Smoke: 100 steps, warmup 10. Full: 100000 steps, warmup 1000.
No pretraining validation or best.pt selection. Full starts from DINO again;
do not resume a smoke checkpoint as a full comparison.

By default, numerical tests must pass before any GPU run starts.
Tests and training use only EventState code; no ScaleEvent checkout is needed.
--dry-run prints commands without running Python, tests, or training.
--skip-tests is for a host where the same revision has already passed tests.
Use an empty output directory; existing runs are never overwritten.
EOF
}

fail() { printf 'error: %s\n' "$*" >&2; exit 1; }
while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) ROOT=${2:?}; shift 2 ;;
    --event-cache-dir) EVENT_CACHE_DIR=${2:?}; shift 2 ;;
    --teacher-cache-dir) TEACHER_CACHE_DIR=${2:?}; shift 2 ;;
    --checkpoint) CHECKPOINT=${2:?}; shift 2 ;;
    --output-root) OUTPUT_ROOT=${2:?}; shift 2 ;;
    --gpus) GPU_LIST=${2:?}; shift 2 ;;
    --stage) STAGE=${2:?}; shift 2 ;;
    --batch-size) BATCH_SIZE=${2:?}; shift 2 ;;
    --num-workers) NUM_WORKERS=${2:?}; shift 2 ;;
    --seed) SEED=${2:?}; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-tests) RUN_TESTS=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

case "$STAGE" in
  smoke) MAX_STEPS=100; WARMUP_STEPS=10; LOG_EVERY=1; CHECKPOINT_EVERY=100 ;;
  full) MAX_STEPS=100000; WARMUP_STEPS=1000; LOG_EVERY=20; CHECKPOINT_EVERY=1000 ;;
  *) fail "--stage must be smoke or full" ;;
esac
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
[ "${#GPUS[@]}" -eq 3 ] || fail "--gpus must specify three distinct numeric GPU IDs"
for value in "$BATCH_SIZE" "$NUM_WORKERS" "$SEED" "${GPUS[@]}"; do
  case "$value" in *[!0-9]*|'') fail "numeric arguments must be non-negative integers" ;; esac
done
[ "$BATCH_SIZE" -gt 0 ] || fail "--batch-size must be positive"
[ "${GPUS[0]}" != "${GPUS[1]}" ] && [ "${GPUS[0]}" != "${GPUS[2]}" ] && \
  [ "${GPUS[1]}" != "${GPUS[2]}" ] || fail "GPU IDs must be distinct"
[ -d "$ROOT" ] || fail "DSEC root not found: $ROOT"
[ -d "$EVENT_CACHE_DIR" ] || fail "event cache not found: $EVENT_CACHE_DIR"
[ -d "$TEACHER_CACHE_DIR" ] || fail "teacher cache not found: $TEACHER_CACHE_DIR"
[ -f "$CHECKPOINT" ] || fail "DINO checkpoint not found: $CHECKPOINT"
# Resolve before changing working directory, including paths supplied relatively.
ROOT=$(cd "$ROOT" && pwd -P)
EVENT_CACHE_DIR=$(cd "$EVENT_CACHE_DIR" && pwd -P)
TEACHER_CACHE_DIR=$(cd "$TEACHER_CACHE_DIR" && pwd -P)
CHECKPOINT=$(cd "$(dirname "$CHECKPOINT")" && pwd -P)/$(basename "$CHECKPOINT")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
if [ -z "$OUTPUT_ROOT" ]; then
  OUTPUT_ROOT="$PROJECT_ROOT/outputs/dsec_activity_${STAGE}_$(date +%Y%m%d_%H%M%S)"
elif [[ "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="$PWD/$OUTPUT_ROOT"
fi
[ ! -e "$OUTPUT_ROOT" ] || fail "output path exists; use a new directory: $OUTPUT_ROOT"
cd "$PROJECT_ROOT"

EXPERIMENTS=(h_distill_lstm_zloss activity_dual scale_event_dual)
NAMES=(baseline_e2 activity_only scale_event_full)
PIDS=()
terminate_children() {
  local pid
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap 'terminate_children; exit 130' INT
trap 'terminate_children; exit 143' TERM

if [ "$DRY_RUN" -eq 0 ]; then
  command -v python >/dev/null 2>&1 || fail "activate the EventState training environment"
  mkdir -p "$OUTPUT_ROOT/logs"
  printf 'stage=%s\ngpus=%s\nseed=%s\nbatch_size=%s\nmax_steps=%s\n' \
    "$STAGE" "$GPU_LIST" "$SEED" "$BATCH_SIZE" "$MAX_STEPS" > "$OUTPUT_ROOT/launch.txt"
  git rev-parse HEAD >> "$OUTPUT_ROOT/launch.txt"
  git diff --stat >> "$OUTPUT_ROOT/launch.txt"
  if [ "$RUN_TESTS" -eq 1 ]; then
    printf '[activity] Running preflight tests; log: %s/logs/preflight.log\n' "$OUTPUT_ROOT"
    if ! (
      export CUDA_VISIBLE_DEVICES=""
      python -c 'import cv2, torch; print("OpenCV", cv2.__version__, "torch", torch.__version__)' || exit 1
      python -m pytest -q tests/test_scale_event.py tests/test_activity_losses.py \
        tests/test_split_guard_stdlib.py tests/test_transforms.py tests/test_dsec.py \
        tests/test_training_runtime.py tests/test_detection.py tests/test_segmentation.py
    ) > "$OUTPUT_ROOT/logs/preflight.log" 2>&1; then
      fail "preflight failed; see $OUTPUT_ROOT/logs/preflight.log (training not started)"
    fi
  fi
fi

for index in 0 1 2; do
  run_dir="$OUTPUT_ROOT/${NAMES[$index]}"
  command_args=(python train.py dataset=dsec_det_train41 model=lstm
    "experiment=${EXPERIMENTS[$index]}" "seed=$SEED"
    "dataset.root=$ROOT" "dataset.event_cache_dir=$EVENT_CACHE_DIR"
    "teacher.cache_dir=$TEACHER_CACHE_DIR" "teacher.checkpoint=$CHECKPOINT"
    "training.batch_size=$BATCH_SIZE" "training.num_workers=$NUM_WORKERS"
    "training.max_steps=$MAX_STEPS" "scheduler.warmup_steps=$WARMUP_STEPS"
    "training.log_every=$LOG_EVERY" "training.checkpoint_every=$CHECKPOINT_EVERY"
    training.gradient_accumulation=1 training.validation_enabled=false
    training.resume=null "hydra.run.dir=$run_dir")
  printf '[activity] GPU %s: %s\n' "${GPUS[$index]}" "${NAMES[$index]}"
  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPUS[$index]}"
    printf '%q ' "${command_args[@]}"
    printf '\n'
  else
    CUDA_VISIBLE_DEVICES="${GPUS[$index]}" "${command_args[@]}" \
      > "$OUTPUT_ROOT/logs/${NAMES[$index]}.log" 2>&1 &
    PIDS+=("$!")
  fi
done
if [ "$DRY_RUN" -eq 1 ]; then exit 0; fi

STATUS=0
for index in 0 1 2; do
  if wait "${PIDS[$index]}"; then
    printf '[activity] complete: %s\n' "${NAMES[$index]}"
  else
    printf '[activity] failed: %s; inspect its log\n' "${NAMES[$index]}" >&2
    STATUS=1
  fi
done
trap - INT TERM
printf '[activity] outputs: %s\n' "$OUTPUT_ROOT"
exit "$STATUS"
