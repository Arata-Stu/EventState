#!/usr/bin/env bash

# Main frozen-representation comparison: E0-z, E2-h, and E4-h.

set -euo pipefail

FEATURE_ROOT=""
LABELS_ROOT=""
DATASET_ROOT=""
OUTPUT_DIR=""
BATCH_SIZE=16
EPOCHS=50
SEEDS="0"

usage() {
  cat <<'EOF'
Usage:
  bash tools/run_frozen_dsec_detection_e0_e2_e4.sh \
    --feature-root PATH --labels-root PATH --dataset-root PATH --output-dir PATH \
    [--batch-size N] [--epochs N] [--seeds 0,1,2]

FEATURE_ROOT must contain E0, E2, and E4 caches in prewarped DSEC-Det
coordinates. Each seed runs E0-z, E2-h, and E4-h concurrently on GPUs 0/1/2.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --feature-root) FEATURE_ROOT=${2:?}; shift 2 ;;
    --labels-root) LABELS_ROOT=${2:?}; shift 2 ;;
    --dataset-root) DATASET_ROOT=${2:?}; shift 2 ;;
    --output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
    --batch-size) BATCH_SIZE=${2:?}; shift 2 ;;
    --epochs) EPOCHS=${2:?}; shift 2 ;;
    --seeds) SEEDS=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[ -d "$FEATURE_ROOT/E0" ] || fail "E0 feature cache not found: $FEATURE_ROOT/E0"
[ -d "$FEATURE_ROOT/E2" ] || fail "E2 feature cache not found: $FEATURE_ROOT/E2"
[ -d "$FEATURE_ROOT/E4" ] || fail "E4 feature cache not found: $FEATURE_ROOT/E4"
[ -d "$LABELS_ROOT" ] || fail "labels root not found: $LABELS_ROOT"
[ -d "$DATASET_ROOT" ] || fail "dataset root not found: $DATASET_ROOT"
[ -n "$OUTPUT_DIR" ] || fail "--output-dir is required"
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env"

FEATURE_ROOT=$(cd "$FEATURE_ROOT" && pwd -P)
LABELS_ROOT=$(cd "$LABELS_ROOT" && pwd -P)
DATASET_ROOT=$(cd "$DATASET_ROOT" && pwd -P)
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd -P)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
cd "$PROJECT_ROOT"

IFS=',' read -r -a SEED_VALUES <<< "$SEEDS"
LABELS=(E0 E2 E4)
FEATURES=(z h h)

for seed in "${SEED_VALUES[@]}"; do
  case "$seed" in *[!0-9]*|'') fail "seeds must be comma-separated non-negative integers" ;; esac
  PIDS=()
  for gpu in 0 1 2; do
    label=${LABELS[$gpu]}
    feature=${FEATURES[$gpu]}
    run_output="$OUTPUT_DIR/$label/seed_$seed"
    log_dir="$OUTPUT_DIR/logs/seed_$seed"
    mkdir -p "$log_dir"
    CUDA_VISIBLE_DEVICES="$gpu" python tools/train_dsec_detection.py \
      --feature-cache-dir "$FEATURE_ROOT/$label" \
      --labels-root "$LABELS_ROOT" \
      --dataset-root "$DATASET_ROOT" \
      --feature "$feature" \
      --protocol dsec-det \
      --output-dir "$run_output" \
      --batch-size "$BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --seed "$seed" \
      --device cuda \
      >"$log_dir/$label.log" 2>&1 &
    PIDS+=("$!")
    printf '[frozen-detection] seed=%s %s-%s gpu=%s pid=%s\n' \
      "$seed" "$label" "$feature" "$gpu" "$!"
  done
  STATUS=0
  for pid in "${PIDS[@]}"; do
    wait "$pid" || STATUS=1
  done
  [ "$STATUS" -eq 0 ] || fail "seed $seed failed; inspect $OUTPUT_DIR/logs/seed_$seed"
done

printf '[frozen-detection] complete: %s\n' "$OUTPUT_DIR"
