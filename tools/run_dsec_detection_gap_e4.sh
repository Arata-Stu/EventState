#!/usr/bin/env bash

# Compare frozen and temporal-head-fine-tuned E4 under deterministic event gaps.

set -euo pipefail

RUN_DIR=""
FROZEN_DETECTION_DIR=""
TEMPORAL_DETECTION_DIR=""
EVENT_CACHE_DIR=""
LABELS_ROOT=""
DATASET_ROOT=""
TEACHER_CHECKPOINT=""
OUTPUT_DIR="outputs/dsec_detection_gap_e4"
GPU="2"
SEED="0"
DROP_START="32"
DROP_STRIDE="64"
GAPS="0,1,2,4,8"
NUM_WORKERS="2"
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage:
  bash tools/run_dsec_detection_gap_e4.sh \
    --run-dir PATH \
    --frozen-detection-dir PATH \
    --temporal-detection-dir PATH \
    --event-cache-dir PATH --labels-root PATH --dataset-root PATH \
    [--teacher-checkpoint PATH] [--output-dir PATH] [--gpu 2] \
    [--seed 0] [--gaps 0,1,2,4,8] [--drop-start 32] [--drop-stride 64] \
    [--num-workers 2] [--overwrite]

Runs online inference from the event cache. It does not create feature caches.
Existing non-empty JSON results are skipped unless --overwrite is supplied.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-dir) RUN_DIR=${2:?}; shift 2 ;;
    --frozen-detection-dir) FROZEN_DETECTION_DIR=${2:?}; shift 2 ;;
    --temporal-detection-dir) TEMPORAL_DETECTION_DIR=${2:?}; shift 2 ;;
    --event-cache-dir) EVENT_CACHE_DIR=${2:?}; shift 2 ;;
    --labels-root) LABELS_ROOT=${2:?}; shift 2 ;;
    --dataset-root) DATASET_ROOT=${2:?}; shift 2 ;;
    --teacher-checkpoint) TEACHER_CHECKPOINT=${2:?}; shift 2 ;;
    --output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
    --gpu) GPU=${2:?}; shift 2 ;;
    --seed) SEED=${2:?}; shift 2 ;;
    --gaps) GAPS=${2:?}; shift 2 ;;
    --drop-start) DROP_START=${2:?}; shift 2 ;;
    --drop-stride) DROP_STRIDE=${2:?}; shift 2 ;;
    --num-workers) NUM_WORKERS=${2:?}; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[ -d "$RUN_DIR" ] || fail "pretraining run directory not found: $RUN_DIR"
[ -d "$FROZEN_DETECTION_DIR" ] || fail "frozen detection directory not found: $FROZEN_DETECTION_DIR"
[ -d "$TEMPORAL_DETECTION_DIR" ] || fail "temporal detection directory not found: $TEMPORAL_DETECTION_DIR"
[ -d "$EVENT_CACHE_DIR" ] || fail "event cache not found: $EVENT_CACHE_DIR"
[ -d "$LABELS_ROOT" ] || fail "labels root not found: $LABELS_ROOT"
[ -d "$DATASET_ROOT" ] || fail "dataset root not found: $DATASET_ROOT"
if [ -n "$TEACHER_CHECKPOINT" ]; then
  [ -f "$TEACHER_CHECKPOINT" ] || fail "teacher checkpoint not found: $TEACHER_CHECKPOINT"
fi
case "$GPU" in *[!0-9]*|'') fail "--gpu must be a non-negative integer" ;; esac
case "$SEED" in *[!0-9]*|'') fail "--seed must be a non-negative integer" ;; esac
case "$DROP_START" in *[!0-9]*|'') fail "--drop-start must be non-negative" ;; esac
case "$DROP_STRIDE" in *[!0-9]*|'') fail "--drop-stride must be positive" ;; esac
case "$NUM_WORKERS" in *[!0-9]*|'') fail "--num-workers must be non-negative" ;; esac
[ "$DROP_STRIDE" -gt 0 ] || fail "--drop-stride must be positive"
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env"
python -c 'from pycocotools.coco import COCO' >/dev/null 2>&1 || \
  fail "pycocotools missing; run: uv sync --active --extra detection"

RUN_DIR=$(cd "$RUN_DIR" && pwd -P)
FROZEN_DETECTION_DIR=$(cd "$FROZEN_DETECTION_DIR" && pwd -P)
TEMPORAL_DETECTION_DIR=$(cd "$TEMPORAL_DETECTION_DIR" && pwd -P)
EVENT_CACHE_DIR=$(cd "$EVENT_CACHE_DIR" && pwd -P)
LABELS_ROOT=$(cd "$LABELS_ROOT" && pwd -P)
DATASET_ROOT=$(cd "$DATASET_ROOT" && pwd -P)
if [ -n "$TEACHER_CHECKPOINT" ]; then
  TEACHER_CHECKPOINT=$(cd "$(dirname "$TEACHER_CHECKPOINT")" && pwd -P)/$(basename "$TEACHER_CHECKPOINT")
fi
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd -P)
mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/frozen_E4" "$OUTPUT_DIR/temporal_E4"
cd "$PROJECT_ROOT"

STUDENT="$RUN_DIR/e4_dual_distill_lstm_event_dropout/checkpoints/step_00100000.pt"
FROZEN_HEAD="$FROZEN_DETECTION_DIR/E4/seed_$SEED/best.pt"
TEMPORAL="$TEMPORAL_DETECTION_DIR/finetune/E4/seed_$SEED/best.pt"
[ -f "$STUDENT" ] || fail "E4 pretraining checkpoint not found: $STUDENT"
[ -f "$FROZEN_HEAD" ] || fail "frozen E4 detector not found: $FROZEN_HEAD"
[ -f "$TEMPORAL" ] || fail "temporal E4 detector not found: $TEMPORAL"

IFS=',' read -r -a GAP_VALUES <<< "$GAPS"
for GAP in "${GAP_VALUES[@]}"; do
  case "$GAP" in *[!0-9]*|'') fail "invalid gap value: $GAP" ;; esac
  [ "$GAP" -le "$DROP_STRIDE" ] || fail "gap $GAP exceeds stride $DROP_STRIDE"
  for SOURCE in frozen_E4 temporal_E4; do
    RESULT="$OUTPUT_DIR/$SOURCE/gap${GAP}.json"
    LOG="$OUTPUT_DIR/logs/${SOURCE}_gap${GAP}.log"
    if [ -s "$RESULT" ] && [ "$OVERWRITE" -eq 0 ]; then
      printf '[gap-eval] skip existing source=%s gap=%s\n' "$SOURCE" "$GAP"
      continue
    fi
    COMMON=(
      --event-cache-dir "$EVENT_CACHE_DIR"
      --labels-root "$LABELS_ROOT"
      --dataset-root "$DATASET_ROOT"
      --role test
      --drop-length "$GAP"
      --drop-start "$DROP_START"
      --drop-stride "$DROP_STRIDE"
      --num-workers "$NUM_WORKERS"
      --device cuda
      --precision fp16
      --output "$RESULT"
    )
    printf '[gap-eval] source=%s gap=%s gpu=%s\n' "$SOURCE" "$GAP" "$GPU"
    if [ "$SOURCE" = "frozen_E4" ]; then
      TEACHER_ARGS=()
      if [ -n "$TEACHER_CHECKPOINT" ]; then
        TEACHER_ARGS=(--teacher-checkpoint "$TEACHER_CHECKPOINT")
      fi
      CUDA_VISIBLE_DEVICES="$GPU" nice -n 10 python \
        tools/evaluate_dsec_detection_gap.py \
        --mode frozen \
        --checkpoint "$FROZEN_HEAD" \
        --student-checkpoint "$STUDENT" \
        --feature h \
        "${TEACHER_ARGS[@]}" \
        "${COMMON[@]}" >"$LOG" 2>&1
    else
      CUDA_VISIBLE_DEVICES="$GPU" nice -n 10 python \
        tools/evaluate_dsec_detection_gap.py \
        --mode end-to-end \
        --checkpoint "$TEMPORAL" \
        "${COMMON[@]}" >"$LOG" 2>&1
    fi
  done
done

python tools/summarize_dsec_detection_gap.py \
  --input-dir "$OUTPUT_DIR" \
  --output "$OUTPUT_DIR/summary.csv"
printf '[gap-eval] complete: %s\n' "$OUTPUT_DIR"
