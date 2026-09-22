#!/usr/bin/env bash

# Cache sparse DSEC-Semantic maps, train a frozen head, and evaluate official test.

set -euo pipefail

CHECKPOINT=""
EVENT_CACHE_DIR=""
LABELS_ROOT=""
FEATURE_CACHE_DIR=""
OUTPUT_DIR=""
TEACHER_CHECKPOINT=""
FEATURE="h"
GPU=0
BATCH_SIZE=8
EPOCHS=50
NUM_WORKERS=4
SEED=0
HEAD_WIDTH=192
HEAD_TYPE=linear
LOSS=ce
ACTIVITY_ACTIVE_WEIGHT=1.0
ACTIVITY_INACTIVE_WEIGHT=1.0

usage() {
  cat <<'EOF'
Usage:
  bash tools/run_dsec_semantic_frozen.sh \
    --checkpoint PATH --event-cache-dir PATH --labels-root PATH \
    --feature-cache-dir PATH --output-dir PATH \
    [--feature z|h|concat] [--teacher-checkpoint PATH] [--gpu N] \
    [--batch-size N] [--epochs N] [--num-workers N] [--seed N] \
    [--head-type linear|nonlinear|gep_patch] [--loss ce|ce-dice] \
    [--activity-active-weight FLOAT] [--activity-inactive-weight FLOAT]

The EventState checkpoint stays frozen. The script updates recurrent state on
every frame, saves maps only at semantic-label frames, trains the segmentation
head on the fixed 6/2 development split, and reports the untouched 3-sequence
official test split. Completed cache sequences are skipped safely on restart.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --checkpoint) CHECKPOINT=${2:?}; shift 2 ;;
    --event-cache-dir) EVENT_CACHE_DIR=${2:?}; shift 2 ;;
    --labels-root) LABELS_ROOT=${2:?}; shift 2 ;;
    --feature-cache-dir) FEATURE_CACHE_DIR=${2:?}; shift 2 ;;
    --output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
    --teacher-checkpoint) TEACHER_CHECKPOINT=${2:?}; shift 2 ;;
    --feature) FEATURE=${2:?}; shift 2 ;;
    --gpu) GPU=${2:?}; shift 2 ;;
    --batch-size) BATCH_SIZE=${2:?}; shift 2 ;;
    --epochs) EPOCHS=${2:?}; shift 2 ;;
    --num-workers) NUM_WORKERS=${2:?}; shift 2 ;;
    --seed) SEED=${2:?}; shift 2 ;;
    --head-width) HEAD_WIDTH=${2:?}; shift 2 ;;
    --head-type) HEAD_TYPE=${2:?}; shift 2 ;;
    --loss) LOSS=${2:?}; shift 2 ;;
    --activity-active-weight) ACTIVITY_ACTIVE_WEIGHT=${2:?}; shift 2 ;;
    --activity-inactive-weight) ACTIVITY_INACTIVE_WEIGHT=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

case "$FEATURE" in
  z|h|concat) ;;
  *) fail "--feature must be z, h, or concat" ;;
esac
case "$HEAD_TYPE" in
  linear|nonlinear|gep_patch) ;;
  *) fail "--head-type must be linear, nonlinear, or gep_patch" ;;
esac
case "$LOSS" in
  ce|ce-dice) ;;
  *) fail "--loss must be ce or ce-dice" ;;
esac

[ -f "$CHECKPOINT" ] || fail "checkpoint not found: $CHECKPOINT"
[ -d "$EVENT_CACHE_DIR" ] || fail "event cache not found: $EVENT_CACHE_DIR"
[ -d "$LABELS_ROOT" ] || fail "semantic labels not found: $LABELS_ROOT"
[ -n "$FEATURE_CACHE_DIR" ] || fail "--feature-cache-dir is required"
[ -n "$OUTPUT_DIR" ] || fail "--output-dir is required"
if [ -n "$TEACHER_CHECKPOINT" ]; then
  [ -f "$TEACHER_CHECKPOINT" ] || fail "teacher checkpoint not found: $TEACHER_CHECKPOINT"
fi
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env"

CHECKPOINT=$(cd "$(dirname "$CHECKPOINT")" && pwd -P)/$(basename "$CHECKPOINT")
EVENT_CACHE_DIR=$(cd "$EVENT_CACHE_DIR" && pwd -P)
LABELS_ROOT=$(cd "$LABELS_ROOT" && pwd -P)
mkdir -p \
  "$FEATURE_CACHE_DIR" \
  "$OUTPUT_DIR/logs" \
  "$OUTPUT_DIR/development" \
  "$OUTPUT_DIR/final"
FEATURE_CACHE_DIR=$(cd "$FEATURE_CACHE_DIR" && pwd -P)
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd -P)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
cd "$PROJECT_ROOT"

if [ "$FEATURE" = concat ]; then
  FEATURES=(z h)
else
  FEATURES=("$FEATURE")
fi
TEACHER_ARGS=()
if [ -n "$TEACHER_CHECKPOINT" ]; then
  TEACHER_CHECKPOINT=$(cd "$(dirname "$TEACHER_CHECKPOINT")" && pwd -P)/$(basename "$TEACHER_CHECKPOINT")
  TEACHER_ARGS=(--teacher-checkpoint "$TEACHER_CHECKPOINT")
fi

ACTIVITY_CACHE_ARGS=()
if [ "$ACTIVITY_ACTIVE_WEIGHT" != "1.0" ] || [ "$ACTIVITY_INACTIVE_WEIGHT" != "1.0" ]; then
  ACTIVITY_CACHE_ARGS=(--include-activity)
fi

for ROLE in train val test; do
  printf '[dsec-semantic] cache role=%s feature=%s gpu=%s\n' "$ROLE" "$FEATURE" "$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" python tools/cache_dsec_semantic_features.py \
    --checkpoint "$CHECKPOINT" \
    --event-cache-dir "$EVENT_CACHE_DIR" \
    --labels-root "$LABELS_ROOT" \
    --output-dir "$FEATURE_CACHE_DIR" \
    --role "$ROLE" \
    --features "${FEATURES[@]}" \
    "${ACTIVITY_CACHE_ARGS[@]}" \
    --state-policy continuous \
    --device cuda \
    "${TEACHER_ARGS[@]}" \
    >"$OUTPUT_DIR/logs/cache_${ROLE}.log" 2>&1
done

DEVELOPMENT_RESUME=()
if [ -f "$OUTPUT_DIR/development/last.pt" ]; then
  DEVELOPMENT_RESUME=(--resume "$OUTPUT_DIR/development/last.pt")
  printf '[dsec-semantic] resuming development head from %s\n' \
    "$OUTPUT_DIR/development/last.pt"
fi

CUDA_VISIBLE_DEVICES="$GPU" python tools/train_dsec_semantic.py \
  --feature-cache-dir "$FEATURE_CACHE_DIR" \
  --labels-root "$LABELS_ROOT" \
  --output-dir "$OUTPUT_DIR/development" \
  --protocol development \
  --feature "$FEATURE" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --num-workers "$NUM_WORKERS" \
  --head-width "$HEAD_WIDTH" \
  --head-type "$HEAD_TYPE" \
  --loss "$LOSS" \
  --activity-active-weight "$ACTIVITY_ACTIVE_WEIGHT" \
  --activity-inactive-weight "$ACTIVITY_INACTIVE_WEIGHT" \
  --seed "$SEED" \
  --precision fp16 \
  --device cuda \
  "${DEVELOPMENT_RESUME[@]}" \
  >"$OUTPUT_DIR/logs/train.log" 2>&1

[ -f "$OUTPUT_DIR/development/best.pt" ] || fail "best development head was not produced"
FINAL_RESUME=()
if [ -f "$OUTPUT_DIR/final/last.pt" ]; then
  FINAL_RESUME=(--resume "$OUTPUT_DIR/final/last.pt")
  printf '[dsec-semantic] resuming official final head from %s\n' \
    "$OUTPUT_DIR/final/last.pt"
fi

CUDA_VISIBLE_DEVICES="$GPU" python tools/train_dsec_semantic.py \
  --feature-cache-dir "$FEATURE_CACHE_DIR" \
  --labels-root "$LABELS_ROOT" \
  --output-dir "$OUTPUT_DIR/final" \
  --protocol official-final \
  --selected-epochs-from "$OUTPUT_DIR/development/best.pt" \
  --feature "$FEATURE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --head-width "$HEAD_WIDTH" \
  --head-type "$HEAD_TYPE" \
  --loss "$LOSS" \
  --activity-active-weight "$ACTIVITY_ACTIVE_WEIGHT" \
  --activity-inactive-weight "$ACTIVITY_INACTIVE_WEIGHT" \
  --seed "$SEED" \
  --precision fp16 \
  --device cuda \
  "${FINAL_RESUME[@]}" \
  >"$OUTPUT_DIR/logs/final_fit.log" 2>&1

[ -f "$OUTPUT_DIR/final/last.pt" ] || fail "official final semantic head was not produced"
CUDA_VISIBLE_DEVICES="$GPU" python tools/evaluate_dsec_semantic.py \
  --checkpoint "$OUTPUT_DIR/final/last.pt" \
  --feature-cache-dir "$FEATURE_CACHE_DIR" \
  --labels-root "$LABELS_ROOT" \
  --role test \
  --feature "$FEATURE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --precision fp16 \
  --device cuda \
  --output "$OUTPUT_DIR/test_metrics.json" \
  >"$OUTPUT_DIR/logs/test.log" 2>&1

printf '[dsec-semantic] complete: %s/test_metrics.json\n' "$OUTPUT_DIR"
