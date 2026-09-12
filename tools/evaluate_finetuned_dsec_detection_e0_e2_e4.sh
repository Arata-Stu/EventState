#!/usr/bin/env bash

# Evaluate seed-0 fine-tuned E0/E2/E4 best checkpoints sequentially on one GPU.

set -euo pipefail

DETECTION_DIR=""
EVENT_CACHE_DIR=""
LABELS_ROOT=""
DATASET_ROOT=""
GPU="2"
SEED="0"
NUM_WORKERS="2"
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage:
  bash tools/evaluate_finetuned_dsec_detection_e0_e2_e4.sh \
    --detection-dir PATH --event-cache-dir PATH --labels-root PATH \
    --dataset-root PATH [--gpu 2] [--seed 0] [--num-workers 2] [--overwrite]

Evaluates finetune/E0, E2, and E4 best.pt sequentially on the official test
split. Results are written beside each checkpoint as test_metrics.json.
Existing complete result files are skipped unless --overwrite is supplied.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --detection-dir) DETECTION_DIR=${2:?}; shift 2 ;;
    --event-cache-dir) EVENT_CACHE_DIR=${2:?}; shift 2 ;;
    --labels-root) LABELS_ROOT=${2:?}; shift 2 ;;
    --dataset-root) DATASET_ROOT=${2:?}; shift 2 ;;
    --gpu) GPU=${2:?}; shift 2 ;;
    --seed) SEED=${2:?}; shift 2 ;;
    --num-workers) NUM_WORKERS=${2:?}; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[ -d "$DETECTION_DIR" ] || fail "detection directory not found: $DETECTION_DIR"
[ -d "$EVENT_CACHE_DIR" ] || fail "event cache not found: $EVENT_CACHE_DIR"
[ -d "$LABELS_ROOT" ] || fail "labels root not found: $LABELS_ROOT"
[ -d "$DATASET_ROOT" ] || fail "dataset root not found: $DATASET_ROOT"
case "$GPU" in *[!0-9]*|'') fail "--gpu must be a non-negative integer" ;; esac
case "$SEED" in *[!0-9]*|'') fail "--seed must be a non-negative integer" ;; esac
case "$NUM_WORKERS" in *[!0-9]*|'') fail "--num-workers must be non-negative" ;; esac
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env"
python -c 'from pycocotools.coco import COCO' >/dev/null 2>&1 || \
  fail "pycocotools missing; run: uv sync --active --extra detection"

DETECTION_DIR=$(cd "$DETECTION_DIR" && pwd -P)
EVENT_CACHE_DIR=$(cd "$EVENT_CACHE_DIR" && pwd -P)
LABELS_ROOT=$(cd "$LABELS_ROOT" && pwd -P)
DATASET_ROOT=$(cd "$DATASET_ROOT" && pwd -P)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
LOG_DIR="$DETECTION_DIR/logs/seed_$SEED"
mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

for label in E0 E2 E4; do
  run_dir="$DETECTION_DIR/finetune/$label/seed_$SEED"
  checkpoint="$run_dir/best.pt"
  output="$run_dir/test_metrics.json"
  log="$LOG_DIR/test_${label}.log"
  [ -f "$checkpoint" ] || fail "fine-tune best checkpoint not found: $checkpoint"
  if [ -s "$output" ] && [ "$OVERWRITE" -eq 0 ]; then
    printf '[fine-tune-test] skip existing: %s\n' "$output"
    continue
  fi
  printf '[fine-tune-test] %s gpu=%s checkpoint=%s\n' "$label" "$GPU" "$checkpoint"
  CUDA_VISIBLE_DEVICES="$GPU" nice -n 10 python \
    tools/evaluate_dsec_detection_end_to_end.py \
    --checkpoint "$checkpoint" \
    --event-cache-dir "$EVENT_CACHE_DIR" \
    --labels-root "$LABELS_ROOT" \
    --dataset-root "$DATASET_ROOT" \
    --role test \
    --device cuda \
    --precision fp16 \
    --num-workers "$NUM_WORKERS" \
    --output "$output" \
    >"$log" 2>&1
  printf '[fine-tune-test] complete: %s\n' "$label"
done

printf '[fine-tune-test] all complete: %s/finetune\n' "$DETECTION_DIR"
