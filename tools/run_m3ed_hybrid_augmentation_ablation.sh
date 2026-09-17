#!/usr/bin/env bash

# Controlled M3ED random/stream and augmentation ablation. GPU 2 runs the two
# online-teacher controls sequentially while GPUs 0 and 1 run cached conditions.

set -euo pipefail

ROOT=""
PREPARED_ROOT=""
TEACHER_CACHE_DIR=""
CHECKPOINT=""
EVENT_STATISTICS=""
OUTPUT_ROOT=""
MAX_STEPS=100000
WARMUP_STEPS=1000
SEQUENCE_LENGTH=16
RANDOM_BATCH_SIZE=2
STREAM_BATCH_SIZE=2

usage() {
  cat <<'EOF'
Usage:
  bash tools/run_m3ed_hybrid_augmentation_ablation.sh \
    --root PATH \
    --prepared-root PATH \
    --teacher-cache-dir PATH \
    --checkpoint PATH \
    --event-statistics PATH \
    [--output-root PATH] [--max-steps 100000] [--warmup-steps 1000] \
    [--sequence-length 16] [--random-batch-size 2] [--stream-batch-size 2]

Runs:
  GPU 0: random BPTT, no augmentation, cached teacher
  GPU 1: hybrid BPTT/TBPTT, no augmentation, cached teacher
  GPU 2: hybrid BPTT/TBPTT, no augmentation, online teacher
         then hybrid BPTT/TBPTT, augmentation, online teacher

The first pair isolates the sampling schedule. The second pair isolates spatial
augmentation without confounding cached and online teacher features.
EOF
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
    --max-steps) MAX_STEPS=${2:?}; shift 2 ;;
    --warmup-steps) WARMUP_STEPS=${2:?}; shift 2 ;;
    --sequence-length) SEQUENCE_LENGTH=${2:?}; shift 2 ;;
    --random-batch-size) RANDOM_BATCH_SIZE=${2:?}; shift 2 ;;
    --stream-batch-size) STREAM_BATCH_SIZE=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[ -d "$ROOT" ] || fail "M3ED root not found: $ROOT"
[ -d "$PREPARED_ROOT" ] || fail "prepared M3ED root not found: $PREPARED_ROOT"
[ -d "$TEACHER_CACHE_DIR" ] || fail "teacher cache not found: $TEACHER_CACHE_DIR"
[ -f "$CHECKPOINT" ] || fail "DINOv3 checkpoint not found: $CHECKPOINT"
[ -f "$EVENT_STATISTICS" ] || fail "event statistics not found: $EVENT_STATISTICS"
for value in \
  "$MAX_STEPS" "$WARMUP_STEPS" "$SEQUENCE_LENGTH" \
  "$RANDOM_BATCH_SIZE" "$STREAM_BATCH_SIZE"; do
  case "$value" in *[!0-9]*|'') fail "step, length, and batch values must be integers" ;; esac
done
[ "$MAX_STEPS" -gt 0 ] || fail "--max-steps must be positive"
[ "$WARMUP_STEPS" -lt "$MAX_STEPS" ] || fail "--warmup-steps must be below --max-steps"
[ "$SEQUENCE_LENGTH" -gt 0 ] || fail "--sequence-length must be positive"
[ "$RANDOM_BATCH_SIZE" -gt 0 ] || fail "--random-batch-size must be positive"
[ "$STREAM_BATCH_SIZE" -gt 0 ] || fail "--stream-batch-size must be positive"
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
if [ -z "$OUTPUT_ROOT" ]; then
  OUTPUT_ROOT="$PROJECT_ROOT/outputs/m3ed_hybrid_aug_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_ROOT/logs"
OUTPUT_ROOT=$(cd "$OUTPUT_ROOT" && pwd -P)
cd "$PROJECT_ROOT"

NORMALIZE_MEAN=$(python -c \
  'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["normalize_mean"]))' \
  "$EVENT_STATISTICS")
NORMALIZE_STD=$(python -c \
  'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["normalize_std"]))' \
  "$EVENT_STATISTICS")

run_one() {
  local gpu=$1
  local name=$2
  local experiment=$3
  shift 3
  local run_dir="$OUTPUT_ROOT/$name"
  local checkpoint_dir="$run_dir/checkpoints"
  local final_checkpoint="$checkpoint_dir/step_$(printf '%08d' "$MAX_STEPS").pt"
  local resume_args=()

  if [ -f "$final_checkpoint" ]; then
    printf '[m3ed-hybrid] complete checkpoint exists; skip %s\n' "$name"
    return
  fi
  if [ -d "$checkpoint_dir" ]; then
    local latest
    latest=$(find "$checkpoint_dir" -maxdepth 1 -type f -name 'step_*.pt' | sort | tail -n 1)
    if [ -n "$latest" ]; then
      resume_args+=(training.resume="$latest")
      printf '[m3ed-hybrid] GPU %s: resume %s\n' "$gpu" "$name"
    fi
  fi

  printf '[m3ed-hybrid] GPU %s: start %s\n' "$gpu" "$name"
  CUDA_VISIBLE_DEVICES="$gpu" python train.py \
    dataset=m3ed_half_dagr \
    model=lstm \
    experiment="$experiment" \
    dataset.root="$ROOT" \
    dataset.prepared_root="$PREPARED_ROOT" \
    dataset.sequence_length="$SEQUENCE_LENGTH" \
    dataset.representation.normalize_mean="$NORMALIZE_MEAN" \
    dataset.representation.normalize_std="$NORMALIZE_STD" \
    teacher.checkpoint="$CHECKPOINT" \
    training.max_steps="$MAX_STEPS" \
    scheduler.warmup_steps="$WARMUP_STEPS" \
    training.sampling.random_batch_size="$RANDOM_BATCH_SIZE" \
    training.sampling.stream_batch_size="$STREAM_BATCH_SIZE" \
    hydra.run.dir="$run_dir" \
    "$@" \
    "${resume_args[@]}" \
    >>"$OUTPUT_ROOT/logs/$name.log" 2>&1
}

worker_random_cached() {
  run_one 0 random_noaug_cache h_distill_lstm \
    training.sampling.mode=random \
    training.sampling.random_batch_size=$((RANDOM_BATCH_SIZE + STREAM_BATCH_SIZE)) \
    teacher.cache_features=true \
    teacher.cache_dir="$TEACHER_CACHE_DIR" \
    dataset.augmentation.enabled=false
}

worker_hybrid_cached() {
  run_one 1 hybrid_noaug_cache h_distill_lstm_hybrid \
    training.sampling.mode=mixed \
    teacher.cache_features=true \
    teacher.cache_dir="$TEACHER_CACHE_DIR" \
    dataset.augmentation.enabled=false
}

worker_online_pair() {
  run_one 2 hybrid_noaug_online h_distill_lstm_hybrid \
    training.sampling.mode=mixed \
    teacher.cache_features=false \
    teacher.cache_dir=null \
    dataset.augmentation.enabled=false
  run_one 2 hybrid_aug_online h_distill_lstm_hybrid \
    training.sampling.mode=mixed \
    teacher.cache_features=false \
    teacher.cache_dir=null \
    dataset.augmentation.enabled=true
}

PIDS=()
NAMES=("random cached" "hybrid cached" "online pair")
worker_random_cached & PIDS+=("$!")
worker_hybrid_cached & PIDS+=("$!")
worker_online_pair & PIDS+=("$!")

terminate_children() {
  local pid
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

STATUS=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    printf '[m3ed-hybrid] complete: %s\n' "${NAMES[$index]}"
  else
    printf '[m3ed-hybrid] failed: %s (see logs)\n' "${NAMES[$index]}" >&2
    STATUS=1
  fi
done

trap - INT TERM
printf '[m3ed-hybrid] outputs: %s\n' "$OUTPUT_ROOT"
exit "$STATUS"
