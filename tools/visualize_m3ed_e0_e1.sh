#!/usr/bin/env bash

# Compare M3ED E0/E1 features over one complete prepared sequence.

set -euo pipefail

RUN_DIR=""
PREPARED_ROOT=""
TEACHER_CACHE_DIR=""
TEACHER_CHECKPOINT=""
SEQUENCE="car_urban_day_ucity_small_loop"
E0_CHECKPOINT=""
E1_CHECKPOINT=""
DEVICE="cuda"
FPS="20"
MAX_FRAMES=""
OUTPUT_DIR=""
OVERWRITE=0

usage() {
  printf '%s\n' \
    'Usage:' \
    '  bash tools/visualize_m3ed_e0_e1.sh \' \
    '    --run-dir PATH \' \
    '    --prepared-root PATH \' \
    '    --teacher-cache-dir PATH \' \
    '    [--teacher-checkpoint PATH] [--sequence NAME] \' \
    '    [--e0-checkpoint PATH] [--e1-checkpoint PATH] \' \
    '    [--device DEVICE] [--fps N] [--max-frames N] \' \
    '    [--output-dir PATH] [--overwrite]' \
    '' \
    'If explicit checkpoints are omitted, best.pt or the latest step_*.pt is used.' \
    'Raw M3ED is not required; --prepared-root supplies aligned RGB and events.' \
    'The video compares events, RGB, E0 Pz, E1 Ph, and the DINOv3 teacher.'
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-dir) RUN_DIR=${2:?}; shift 2 ;;
    --prepared-root) PREPARED_ROOT=${2:?}; shift 2 ;;
    --teacher-cache-dir) TEACHER_CACHE_DIR=${2:?}; shift 2 ;;
    --teacher-checkpoint) TEACHER_CHECKPOINT=${2:?}; shift 2 ;;
    --sequence) SEQUENCE=${2:?}; shift 2 ;;
    --e0-checkpoint) E0_CHECKPOINT=${2:?}; shift 2 ;;
    --e1-checkpoint) E1_CHECKPOINT=${2:?}; shift 2 ;;
    --device) DEVICE=${2:?}; shift 2 ;;
    --fps) FPS=${2:?}; shift 2 ;;
    --max-frames) MAX_FRAMES=${2:?}; shift 2 ;;
    --output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[ -n "$RUN_DIR" ] || fail "--run-dir is required"
[ -d "$RUN_DIR" ] || fail "run directory not found: $RUN_DIR"
[ -d "$PREPARED_ROOT" ] || fail "prepared M3ED root not found: $PREPARED_ROOT"
[ -d "$TEACHER_CACHE_DIR" ] || fail "teacher cache not found: $TEACHER_CACHE_DIR"
if [ -n "$TEACHER_CHECKPOINT" ]; then
  [ -f "$TEACHER_CHECKPOINT" ] || fail "teacher checkpoint not found: $TEACHER_CHECKPOINT"
fi
if [ -n "$MAX_FRAMES" ]; then
  case "$MAX_FRAMES" in *[!0-9]*|'') fail "--max-frames must be a positive integer" ;; esac
  [ "$MAX_FRAMES" -gt 0 ] || fail "--max-frames must be positive"
fi
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env first"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
RUN_DIR=$(cd "$RUN_DIR" && pwd -P)
PREPARED_ROOT=$(cd "$PREPARED_ROOT" && pwd -P)
TEACHER_CACHE_DIR=$(cd "$TEACHER_CACHE_DIR" && pwd -P)

resolve_checkpoint() {
  local experiment_dir=$1
  local best="$experiment_dir/checkpoints/best.pt"
  local candidate
  local latest=""
  if [ -f "$best" ]; then
    printf '%s\n' "$best"
    return
  fi
  for candidate in "$experiment_dir"/checkpoints/step_*.pt; do
    if [ -f "$candidate" ]; then
      latest=$candidate
    fi
  done
  [ -n "$latest" ] || fail "checkpoint not found under: $experiment_dir/checkpoints"
  printf '%s\n' "$latest"
}

if [ -z "$E0_CHECKPOINT" ]; then
  E0_CHECKPOINT=$(resolve_checkpoint "$RUN_DIR/e0_no_memory")
fi
if [ -z "$E1_CHECKPOINT" ]; then
  E1_CHECKPOINT=$(resolve_checkpoint "$RUN_DIR/e1_lstm")
fi
[ -f "$E0_CHECKPOINT" ] || fail "E0 checkpoint not found: $E0_CHECKPOINT"
[ -f "$E1_CHECKPOINT" ] || fail "E1 checkpoint not found: $E1_CHECKPOINT"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$RUN_DIR/feature_visualization/${SEQUENCE}_e0_e1"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd -P)
cd "$PROJECT_ROOT"

COMMON_ARGS=(
  # M3EDSequenceDataset reads prepared_root; root only satisfies the shared
  # dataset contract, so raw M3ED is intentionally unnecessary here.
  --root "$PREPARED_ROOT"
  --event-cache-dir "$PREPARED_ROOT"
  --teacher-cache-dir "$TEACHER_CACHE_DIR"
  --sequence "$SEQUENCE"
  --state-policy continuous
  --device "$DEVICE"
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

printf '[m3ed-visualization] exporting E0 Pz\n'
python tools/export_feature_sequence.py \
  --checkpoint "$E0_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR/E0" \
  --context-dir "$OUTPUT_DIR/context" \
  --feature Pz \
  "${COMMON_ARGS[@]}" \
  "${OVERWRITE_ARGS[@]}"

printf '[m3ed-visualization] exporting E1 Ph\n'
python tools/export_feature_sequence.py \
  --checkpoint "$E1_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR/E1" \
  --feature Ph \
  "${COMMON_ARGS[@]}" \
  "${OVERWRITE_ARGS[@]}"

RENDER_ARGS=(
  --context-dir "$OUTPUT_DIR/context"
  --source "E0 Pz:$OUTPUT_DIR/E0:Pz"
  --source "E1 Ph:$OUTPUT_DIR/E1:Ph"
  --fps "$FPS"
  --output "$OUTPUT_DIR/alignment.mp4"
)
if [ -n "$MAX_FRAMES" ]; then
  RENDER_ARGS+=(--max-frames "$MAX_FRAMES")
fi
python tools/render_feature_sequence.py "${RENDER_ARGS[@]}"

printf '[m3ed-visualization] video: %s\n' "$OUTPUT_DIR/alignment.mp4"
printf '[m3ed-visualization] frame metrics: %s\n' "$OUTPUT_DIR/alignment.csv"
printf '[m3ed-visualization] summary: %s\n' "$OUTPUT_DIR/alignment.json"
