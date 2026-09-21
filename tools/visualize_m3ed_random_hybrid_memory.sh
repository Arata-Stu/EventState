#!/usr/bin/env bash

# Compare Random training with Hybrid TBPTT and isolate cross-clip state carry.

set -euo pipefail

RANDOM_CHECKPOINT=""
HYBRID_CHECKPOINT=""
PREPARED_ROOT=""
TEACHER_CACHE_DIR=""
TEACHER_CHECKPOINT=""
SEQUENCE="car_urban_day_ucity_small_loop"
DEVICE="cuda"
FPS="30"
MAX_FRAMES=""
OUTPUT_DIR=""
OVERWRITE=0

usage() {
  printf '%s\n' \
    'Usage:' \
    '  bash tools/visualize_m3ed_random_hybrid_memory.sh \' \
    '    --random-checkpoint PATH --hybrid-checkpoint PATH \' \
    '    --prepared-root PATH --teacher-cache-dir PATH \' \
    '    [--teacher-checkpoint PATH] [--sequence NAME] \' \
    '    [--device DEVICE] [--fps N] [--max-frames N] \' \
    '    --output-dir PATH [--overwrite]' \
    '' \
    'Panels:' \
    '  Random h, continuous state' \
    '  Hybrid h, continuous state' \
    '  Hybrid h, state reset at every clip boundary' \
    'All feature panels and the DINOv3 teacher use one shared PCA.'
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --random-checkpoint) RANDOM_CHECKPOINT=${2:?}; shift 2 ;;
    --hybrid-checkpoint) HYBRID_CHECKPOINT=${2:?}; shift 2 ;;
    --prepared-root) PREPARED_ROOT=${2:?}; shift 2 ;;
    --teacher-cache-dir) TEACHER_CACHE_DIR=${2:?}; shift 2 ;;
    --teacher-checkpoint) TEACHER_CHECKPOINT=${2:?}; shift 2 ;;
    --sequence) SEQUENCE=${2:?}; shift 2 ;;
    --device) DEVICE=${2:?}; shift 2 ;;
    --fps) FPS=${2:?}; shift 2 ;;
    --max-frames) MAX_FRAMES=${2:?}; shift 2 ;;
    --output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[ -f "$RANDOM_CHECKPOINT" ] || fail "random checkpoint not found: $RANDOM_CHECKPOINT"
[ -f "$HYBRID_CHECKPOINT" ] || fail "hybrid checkpoint not found: $HYBRID_CHECKPOINT"
[ -d "$PREPARED_ROOT" ] || fail "prepared M3ED root not found: $PREPARED_ROOT"
[ -d "$TEACHER_CACHE_DIR" ] || fail "teacher cache not found: $TEACHER_CACHE_DIR"
[ -n "$OUTPUT_DIR" ] || fail "--output-dir is required"
if [ -n "$TEACHER_CHECKPOINT" ]; then
  [ -f "$TEACHER_CHECKPOINT" ] || fail "teacher checkpoint not found: $TEACHER_CHECKPOINT"
fi
if [ -n "$MAX_FRAMES" ]; then
  case "$MAX_FRAMES" in *[!0-9]*|'') fail "--max-frames must be a positive integer" ;; esac
  [ "$MAX_FRAMES" -gt 0 ] || fail "--max-frames must be positive"
fi
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
RANDOM_CHECKPOINT=$(cd "$(dirname "$RANDOM_CHECKPOINT")" && pwd -P)/$(basename "$RANDOM_CHECKPOINT")
HYBRID_CHECKPOINT=$(cd "$(dirname "$HYBRID_CHECKPOINT")" && pwd -P)/$(basename "$HYBRID_CHECKPOINT")
PREPARED_ROOT=$(cd "$PREPARED_ROOT" && pwd -P)
TEACHER_CACHE_DIR=$(cd "$TEACHER_CACHE_DIR" && pwd -P)
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd -P)
cd "$PROJECT_ROOT"

COMMON_ARGS=(
  --root "$PREPARED_ROOT"
  --event-cache-dir "$PREPARED_ROOT"
  --teacher-cache-dir "$TEACHER_CACHE_DIR"
  --sequence "$SEQUENCE"
  --device "$DEVICE"
  --feature Ph
)
if [ -n "$TEACHER_CHECKPOINT" ]; then
  COMMON_ARGS+=(--teacher-checkpoint "$TEACHER_CHECKPOINT")
fi
if [ -n "$MAX_FRAMES" ]; then
  COMMON_ARGS+=(--max-frames "$MAX_FRAMES")
fi
OVERWRITE_ARGS=()
if [ "$OVERWRITE" -eq 1 ]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

printf '[m3ed-memory] Random h; continuous state\n'
python tools/export_feature_sequence.py \
  --checkpoint "$RANDOM_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR/random_continuous" \
  --context-dir "$OUTPUT_DIR/context" \
  --state-policy continuous \
  "${COMMON_ARGS[@]}" \
  "${OVERWRITE_ARGS[@]}"

printf '[m3ed-memory] Hybrid h; continuous state\n'
python tools/export_feature_sequence.py \
  --checkpoint "$HYBRID_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR/hybrid_continuous" \
  --state-policy continuous \
  "${COMMON_ARGS[@]}" \
  "${OVERWRITE_ARGS[@]}"

printf '[m3ed-memory] Hybrid h; clip-reset state\n'
python tools/export_feature_sequence.py \
  --checkpoint "$HYBRID_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR/hybrid_clip_reset" \
  --state-policy clip \
  "${COMMON_ARGS[@]}" \
  "${OVERWRITE_ARGS[@]}"

RENDER_ARGS=(
  --context-dir "$OUTPUT_DIR/context"
  --source "Random h (continuous):$OUTPUT_DIR/random_continuous:Ph"
  --source "Hybrid h (continuous):$OUTPUT_DIR/hybrid_continuous:Ph"
  --source "Hybrid h (clip reset):$OUTPUT_DIR/hybrid_clip_reset:Ph"
  --fps "$FPS"
  --output "$OUTPUT_DIR/memory_comparison.mp4"
)
if [ -n "$MAX_FRAMES" ]; then
  RENDER_ARGS+=(--max-frames "$MAX_FRAMES")
fi
python tools/render_feature_sequence.py "${RENDER_ARGS[@]}"

printf '[m3ed-memory] video: %s\n' "$OUTPUT_DIR/memory_comparison.mp4"
printf '[m3ed-memory] frame metrics: %s\n' "$OUTPUT_DIR/memory_comparison.csv"
printf '[m3ed-memory] summary: %s\n' "$OUTPUT_DIR/memory_comparison.json"
