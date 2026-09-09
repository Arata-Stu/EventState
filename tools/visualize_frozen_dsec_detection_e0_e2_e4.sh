#!/usr/bin/env bash

# Render official DSEC-Det predictions and shared-PCA maps for E0/E2/E4.

set -euo pipefail

DET_DIR=""
FEATURE_ROOT=""
EVENT_CACHE_DIR=""
LABELS_ROOT=""
DATASET_ROOT=""
OUTPUT_DIR=""
ROLE="test"
SEED=0
GPU=0
FPS=20
SCORE_THRESHOLD=0.25
DETECTION_BACKGROUND="rgb"
MAX_FRAMES=""
SEQUENCES=()

usage() {
  cat <<'EOF'
Usage:
  bash tools/visualize_frozen_dsec_detection_e0_e2_e4.sh \
    --det-dir PATH --feature-root PATH --event-cache-dir PATH \
    --labels-root PATH --dataset-root PATH --output-dir PATH \
    [--role val|test] [--seed 0] [--gpu 0] [--fps 20] \
    [--score-threshold 0.25] [--detection-background rgb|event] \
    [--max-frames N] \
    [--sequences SEQUENCE ...]

Without --sequences, all official sequences in the selected role are rendered.
Each MP4 shows E0/E2/E4 detections above their shared-PCA feature maps.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --det-dir) DET_DIR=${2:?}; shift 2 ;;
    --feature-root) FEATURE_ROOT=${2:?}; shift 2 ;;
    --event-cache-dir) EVENT_CACHE_DIR=${2:?}; shift 2 ;;
    --labels-root) LABELS_ROOT=${2:?}; shift 2 ;;
    --dataset-root) DATASET_ROOT=${2:?}; shift 2 ;;
    --output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
    --role) ROLE=${2:?}; shift 2 ;;
    --seed) SEED=${2:?}; shift 2 ;;
    --gpu) GPU=${2:?}; shift 2 ;;
    --fps) FPS=${2:?}; shift 2 ;;
    --score-threshold) SCORE_THRESHOLD=${2:?}; shift 2 ;;
    --detection-background) DETECTION_BACKGROUND=${2:?}; shift 2 ;;
    --max-frames) MAX_FRAMES=${2:?}; shift 2 ;;
    --sequences)
      shift
      while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
        SEQUENCES+=("$1")
        shift
      done
      ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

case "$ROLE" in val|test) ;; *) fail "--role must be val or test" ;; esac
case "$DETECTION_BACKGROUND" in rgb|event) ;; *) fail "invalid detection background" ;; esac
case "$SEED" in *[!0-9]*|'') fail "--seed must be a non-negative integer" ;; esac
case "$GPU" in *[!0-9]*|'') fail "--gpu must be a non-negative integer" ;; esac
[ -d "$DET_DIR" ] || fail "detector output directory not found: $DET_DIR"
[ -d "$FEATURE_ROOT/E0" ] || fail "E0 feature cache not found: $FEATURE_ROOT/E0"
[ -d "$FEATURE_ROOT/E2" ] || fail "E2 feature cache not found: $FEATURE_ROOT/E2"
[ -d "$FEATURE_ROOT/E4" ] || fail "E4 feature cache not found: $FEATURE_ROOT/E4"
[ -d "$EVENT_CACHE_DIR" ] || fail "event cache not found: $EVENT_CACHE_DIR"
[ -d "$LABELS_ROOT" ] || fail "labels root not found: $LABELS_ROOT"
[ -d "$DATASET_ROOT" ] || fail "dataset root not found: $DATASET_ROOT"
[ -n "$OUTPUT_DIR" ] || fail "--output-dir is required"

for label in E0 E2 E4; do
  [ -f "$DET_DIR/$label/seed_$SEED/best.pt" ] || \
    fail "detector checkpoint not found: $DET_DIR/$label/seed_$SEED/best.pt"
done

if [ "${#SEQUENCES[@]}" -eq 0 ]; then
  if [ "$ROLE" = "val" ]; then
    SEQUENCES=(
      zurich_city_16_a zurich_city_17_a zurich_city_18_a
      zurich_city_19_a zurich_city_20_a zurich_city_21_a
    )
  else
    SEQUENCES=(
      thun_01_a thun_01_b thun_02_a interlaken_00_a interlaken_00_b
      interlaken_01_a zurich_city_12_a zurich_city_13_a zurich_city_13_b
      zurich_city_14_a zurich_city_14_b zurich_city_14_c zurich_city_15_a
    )
  fi
fi

DET_DIR=$(cd "$DET_DIR" && pwd -P)
FEATURE_ROOT=$(cd "$FEATURE_ROOT" && pwd -P)
EVENT_CACHE_DIR=$(cd "$EVENT_CACHE_DIR" && pwd -P)
LABELS_ROOT=$(cd "$LABELS_ROOT" && pwd -P)
DATASET_ROOT=$(cd "$DATASET_ROOT" && pwd -P)
mkdir -p "$OUTPUT_DIR/logs"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd -P)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
cd "$PROJECT_ROOT"

command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env"
python -c 'import imageio_ffmpeg' >/dev/null 2>&1 || \
  fail "visualization dependencies missing; run uv sync --active --extra visualize --extra detection"

for sequence in "${SEQUENCES[@]}"; do
  printf '[detection-visualization] rendering %s\n' "$sequence"
  extra_args=()
  if [ -n "$MAX_FRAMES" ]; then
    extra_args=(--max-frames "$MAX_FRAMES")
  fi
  CUDA_VISIBLE_DEVICES="$GPU" python tools/visualize_dsec_detection_sequence.py \
    --source "E0:$DET_DIR/E0/seed_$SEED/best.pt:$FEATURE_ROOT/E0:z" \
    --source "E2:$DET_DIR/E2/seed_$SEED/best.pt:$FEATURE_ROOT/E2:h" \
    --source "E4:$DET_DIR/E4/seed_$SEED/best.pt:$FEATURE_ROOT/E4:h" \
    --event-cache-dir "$EVENT_CACHE_DIR" \
    --labels-root "$LABELS_ROOT" \
    --dataset-root "$DATASET_ROOT" \
    --role "$ROLE" \
    --sequence "$sequence" \
    --output "$OUTPUT_DIR/$sequence.mp4" \
    --score-threshold "$SCORE_THRESHOLD" \
    --detection-background "$DETECTION_BACKGROUND" \
    --fps "$FPS" \
    --device cuda \
    "${extra_args[@]}" \
    >"$OUTPUT_DIR/logs/$sequence.log" 2>&1
  printf '[detection-visualization] complete: %s\n' "$sequence"
done

printf '[detection-visualization] outputs: %s\n' "$OUTPUT_DIR"
