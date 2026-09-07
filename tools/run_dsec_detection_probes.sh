#!/usr/bin/env bash

# Train z, h, and concat frozen probes concurrently on three GPUs.

set -euo pipefail

FEATURE_CACHE_DIR=""
LABELS_ROOT=""
DATASET_ROOT=""
OUTPUT_DIR=""
BATCH_SIZE=16
EPOCHS=50
PROTOCOL=probe

usage() {
  cat <<'EOF'
Usage:
  bash tools/run_dsec_detection_probes.sh \
    --feature-cache-dir PATH --labels-root PATH --dataset-root PATH --output-dir PATH \
    [--batch-size N] [--epochs N] [--protocol probe|dsec-det]

GPU 0 trains z, GPU 1 trains h, and GPU 2 trains concat[z,h].
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --feature-cache-dir) FEATURE_CACHE_DIR=${2:?}; shift 2 ;;
    --labels-root) LABELS_ROOT=${2:?}; shift 2 ;;
    --dataset-root) DATASET_ROOT=${2:?}; shift 2 ;;
    --output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
    --batch-size) BATCH_SIZE=${2:?}; shift 2 ;;
    --epochs) EPOCHS=${2:?}; shift 2 ;;
    --protocol) PROTOCOL=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

case "$PROTOCOL" in
  probe|dsec-det) ;;
  *) fail "--protocol must be probe or dsec-det" ;;
esac

[ -d "$FEATURE_CACHE_DIR" ] || fail "feature cache not found: $FEATURE_CACHE_DIR"
[ -d "$LABELS_ROOT" ] || fail "labels root not found: $LABELS_ROOT"
[ -d "$DATASET_ROOT" ] || fail "dataset root not found: $DATASET_ROOT"
[ -n "$OUTPUT_DIR" ] || fail "--output-dir is required"
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env"

FEATURE_CACHE_DIR=$(cd "$FEATURE_CACHE_DIR" && pwd -P)
LABELS_ROOT=$(cd "$LABELS_ROOT" && pwd -P)
DATASET_ROOT=$(cd "$DATASET_ROOT" && pwd -P)
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd -P)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
mkdir -p "$OUTPUT_DIR/logs"
cd "$PROJECT_ROOT"

FEATURES=(z h concat)
PIDS=()
for GPU in 0 1 2; do
  FEATURE=${FEATURES[$GPU]}
  CUDA_VISIBLE_DEVICES=$GPU python tools/train_dsec_detection.py \
    --feature-cache-dir "$FEATURE_CACHE_DIR" \
    --labels-root "$LABELS_ROOT" \
    --dataset-root "$DATASET_ROOT" \
    --feature "$FEATURE" \
    --protocol "$PROTOCOL" \
    --output-dir "$OUTPUT_DIR/$FEATURE" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --device cuda \
    > "$OUTPUT_DIR/logs/${FEATURE}.log" 2>&1 &
  PIDS+=("$!")
  printf '[dsec-detection] feature=%s gpu=%s pid=%s\n' "$FEATURE" "$GPU" "$!"
done

STATUS=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || STATUS=1
done
[ "$STATUS" -eq 0 ] || fail "one or more DSEC-Detection probes failed; inspect logs"
printf '[dsec-detection] all probes complete: %s\n' "$OUTPUT_DIR"
