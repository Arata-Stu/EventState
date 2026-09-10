#!/usr/bin/env bash

# Compare M3ED E3/E4 features over one complete sequence, including natural stops.

set -euo pipefail

RUN_DIR=""
ROOT=""
PREPARED_ROOT=""
TEACHER_CACHE_DIR=""
TEACHER_CHECKPOINT=""
SEQUENCE="car_urban_day_ucity_small_loop"
E3_CHECKPOINT=""
E4_CHECKPOINT=""
DEVICE="cuda"
FPS="20"
OUTPUT_DIR=""
OVERWRITE=0

usage() {
  printf '%s\n' \
    'Usage:' \
    '  bash tools/visualize_m3ed_e3_e4.sh \\' \
    '    --run-dir PATH \\' \
    '    --root PATH \\' \
    '    --prepared-root PATH \\' \
    '    --teacher-cache-dir PATH \\' \
    '    --teacher-checkpoint PATH \\' \
    '    [--sequence NAME] [--e3-checkpoint PATH] [--e4-checkpoint PATH] \' \
    '    [--device DEVICE] [--fps N] [--output-dir PATH] [--overwrite]' \
    '' \
    'If explicit checkpoints are omitted, best.pt or the latest step_*.pt is used.' \
    'The output compares E3 Ph, E4 Ph, E4 Pz, DINO teacher, RGB, and events.'
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-dir) RUN_DIR=${2:?}; shift 2 ;;
    --root) ROOT=${2:?}; shift 2 ;;
    --prepared-root) PREPARED_ROOT=${2:?}; shift 2 ;;
    --teacher-cache-dir) TEACHER_CACHE_DIR=${2:?}; shift 2 ;;
    --teacher-checkpoint) TEACHER_CHECKPOINT=${2:?}; shift 2 ;;
    --sequence) SEQUENCE=${2:?}; shift 2 ;;
    --e3-checkpoint) E3_CHECKPOINT=${2:?}; shift 2 ;;
    --e4-checkpoint) E4_CHECKPOINT=${2:?}; shift 2 ;;
    --device) DEVICE=${2:?}; shift 2 ;;
    --fps) FPS=${2:?}; shift 2 ;;
    --output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[ -n "$RUN_DIR" ] || fail "--run-dir is required"
[ -d "$RUN_DIR" ] || fail "run directory not found: $RUN_DIR"
[ -d "$ROOT" ] || fail "M3ED root not found: $ROOT"
[ -d "$PREPARED_ROOT" ] || fail "prepared M3ED root not found: $PREPARED_ROOT"
[ -d "$TEACHER_CACHE_DIR" ] || fail "teacher cache not found: $TEACHER_CACHE_DIR"
[ -f "$TEACHER_CHECKPOINT" ] || fail "teacher checkpoint not found: $TEACHER_CHECKPOINT"
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env first"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
RUN_DIR=$(cd "$RUN_DIR" && pwd -P)

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

if [ -z "$E3_CHECKPOINT" ]; then
  E3_CHECKPOINT=$(resolve_checkpoint "$RUN_DIR/e3_h_distill_lstm_event_dropout")
fi
if [ -z "$E4_CHECKPOINT" ]; then
  E4_CHECKPOINT=$(resolve_checkpoint "$RUN_DIR/e4_dual_distill_lstm_event_dropout")
fi
[ -f "$E3_CHECKPOINT" ] || fail "E3 checkpoint not found: $E3_CHECKPOINT"
[ -f "$E4_CHECKPOINT" ] || fail "E4 checkpoint not found: $E4_CHECKPOINT"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$RUN_DIR/feature_visualization/${SEQUENCE}_e3_e4"
fi
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

COMMON_ARGS=(
  --root "$ROOT"
  --event-cache-dir "$PREPARED_ROOT"
  --teacher-cache-dir "$TEACHER_CACHE_DIR"
  --teacher-checkpoint "$TEACHER_CHECKPOINT"
  --sequence "$SEQUENCE"
  --state-policy continuous
  --device "$DEVICE"
)
OVERWRITE_ARGS=()
if [ "$OVERWRITE" -eq 1 ]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

printf '[m3ed-visualization] exporting E3 Ph\n'
python tools/export_feature_sequence.py \
  --checkpoint "$E3_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR/E3" \
  --context-dir "$OUTPUT_DIR/context" \
  --feature Ph \
  "${COMMON_ARGS[@]}" \
  "${OVERWRITE_ARGS[@]}"

printf '[m3ed-visualization] exporting E4 Ph/Pz\n'
python tools/export_feature_sequence.py \
  --checkpoint "$E4_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR/E4" \
  --feature Ph \
  --feature Pz \
  "${COMMON_ARGS[@]}" \
  "${OVERWRITE_ARGS[@]}"

python tools/render_feature_sequence.py \
  --context-dir "$OUTPUT_DIR/context" \
  --source "E3 Ph:$OUTPUT_DIR/E3:Ph" \
  --source "E4 Ph:$OUTPUT_DIR/E4:Ph" \
  --source "E4 Pz:$OUTPUT_DIR/E4:Pz" \
  --fps "$FPS" \
  --output "$OUTPUT_DIR/alignment.mp4"

printf '[m3ed-visualization] video: %s\n' "$OUTPUT_DIR/alignment.mp4"
printf '[m3ed-visualization] frame metrics: %s\n' "$OUTPUT_DIR/alignment.csv"
printf '[m3ed-visualization] summary: %s\n' "$OUTPUT_DIR/alignment.json"
