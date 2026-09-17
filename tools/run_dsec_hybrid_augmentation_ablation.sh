#!/usr/bin/env bash

# Compare cached/online hybrid training and sequence-consistent spatial
# augmentation on three GPUs. The existing E1 run supplies the random baseline.

set -euo pipefail

ROOT=""
EVENT_CACHE_DIR=""
TEACHER_CACHE_DIR=""
CHECKPOINT=""
OUTPUT_ROOT=""
MAX_STEPS=100000
WARMUP_STEPS=1000
RANDOM_BATCH_SIZE=4
STREAM_BATCH_SIZE=4

usage() {
  cat <<'EOF'
Usage:
  bash tools/run_dsec_hybrid_augmentation_ablation.sh \
    --root PATH \
    --event-cache-dir PATH \
    --teacher-cache-dir PATH \
    --checkpoint PATH \
    [--output-root PATH] [--max-steps 100000] [--warmup-steps 1000] \
    [--random-batch-size 4] [--stream-batch-size 4]

GPU assignment:
  GPU 0: 1:1 hybrid, augmentation off, cached teacher
  GPU 1: 1:1 hybrid, augmentation off, online teacher
  GPU 2: 1:1 hybrid, augmentation on, online teacher

The extra no-augmentation online run controls for the teacher delivery mode:
augmentation itself must not be compared only against a cached-teacher run.
Crop/flip parameters are fixed for an entire stream sequence. All runs use the
41-sequence DSEC-Detection training manifest and the one-layer h-distillation
LSTM objective. Compare GPU 0 with the existing E1 random/no-augmentation result
to measure the sampling change.
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
    --max-steps) MAX_STEPS=${2:?}; shift 2 ;;
    --warmup-steps) WARMUP_STEPS=${2:?}; shift 2 ;;
    --random-batch-size) RANDOM_BATCH_SIZE=${2:?}; shift 2 ;;
    --stream-batch-size) STREAM_BATCH_SIZE=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[ -d "$ROOT" ] || fail "DSEC root not found: $ROOT"
[ -d "$EVENT_CACHE_DIR" ] || fail "event cache not found: $EVENT_CACHE_DIR"
[ -d "$TEACHER_CACHE_DIR" ] || fail "teacher cache not found: $TEACHER_CACHE_DIR"
[ -f "$CHECKPOINT" ] || fail "DINOv3 checkpoint not found: $CHECKPOINT"
for value in "$MAX_STEPS" "$WARMUP_STEPS" "$RANDOM_BATCH_SIZE" "$STREAM_BATCH_SIZE"; do
  case "$value" in *[!0-9]*|'') fail "step and batch values must be integers" ;; esac
done
[ "$MAX_STEPS" -gt 0 ] || fail "--max-steps must be positive"
[ "$RANDOM_BATCH_SIZE" -gt 0 ] || fail "--random-batch-size must be positive"
[ "$STREAM_BATCH_SIZE" -gt 0 ] || fail "--stream-batch-size must be positive"
[ "$WARMUP_STEPS" -lt "$MAX_STEPS" ] || fail "--warmup-steps must be below --max-steps"
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
if [ -z "$OUTPUT_ROOT" ]; then
  OUTPUT_ROOT="$PROJECT_ROOT/outputs/dsec_hybrid_aug_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_ROOT/logs"
OUTPUT_ROOT=$(cd "$OUTPUT_ROOT" && pwd -P)
cd "$PROJECT_ROOT"

PIDS=()
NAMES=()

launch() {
  local gpu=$1
  local name=$2
  shift 2
  local run_dir="$OUTPUT_ROOT/$name"
  local checkpoint_dir="$run_dir/checkpoints"
  local final_checkpoint="$checkpoint_dir/step_$(printf '%08d' "$MAX_STEPS").pt"
  local resume_args=()

  if [ -f "$final_checkpoint" ]; then
    printf '[hybrid-ablation] complete checkpoint exists; skip %s\n' "$name"
    return
  fi
  if [ -d "$checkpoint_dir" ]; then
    local latest
    latest=$(find "$checkpoint_dir" -maxdepth 1 -type f -name 'step_*.pt' | sort | tail -n 1)
    if [ -n "$latest" ]; then
      resume_args+=(training.resume="$latest")
    fi
  fi

  printf '[hybrid-ablation] GPU %s: start %s\n' "$gpu" "$name"
  CUDA_VISIBLE_DEVICES="$gpu" python train.py \
    dataset=dsec_det_train41 \
    model=lstm \
    experiment=h_distill_lstm_hybrid \
    dataset.root="$ROOT" \
    dataset.event_cache_dir="$EVENT_CACHE_DIR" \
    teacher.checkpoint="$CHECKPOINT" \
    training.max_steps="$MAX_STEPS" \
    scheduler.warmup_steps="$WARMUP_STEPS" \
    training.validation_enabled=false \
    training.sampling.random_batch_size="$RANDOM_BATCH_SIZE" \
    training.sampling.stream_batch_size="$STREAM_BATCH_SIZE" \
    hydra.run.dir="$run_dir" \
    "$@" \
    "${resume_args[@]}" \
    >"$OUTPUT_ROOT/logs/$name.log" 2>&1 &
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

launch 0 hybrid_noaug_cache \
  training.sampling.mode=mixed \
  teacher.cache_features=true \
  teacher.cache_dir="$TEACHER_CACHE_DIR" \
  dataset.augmentation.enabled=false

launch 1 hybrid_noaug_online \
  training.sampling.mode=mixed \
  teacher.cache_features=false \
  teacher.cache_dir=null \
  dataset.augmentation.enabled=false

launch 2 hybrid_aug_online \
  training.sampling.mode=mixed \
  teacher.cache_features=false \
  teacher.cache_dir=null \
  dataset.augmentation.enabled=true

STATUS=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    printf '[hybrid-ablation] complete: %s\n' "${NAMES[$index]}"
  else
    printf '[hybrid-ablation] failed: %s (see logs)\n' "${NAMES[$index]}" >&2
    STATUS=1
  fi
done

trap - INT TERM
printf '[hybrid-ablation] outputs: %s\n' "$OUTPUT_ROOT"
exit "$STATUS"
