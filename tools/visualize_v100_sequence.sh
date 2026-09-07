#!/usr/bin/env bash

# Export E0/E1/E2 over one complete sequence with continuous state and render MP4.

set -euo pipefail

RUN_DIR=""
ROOT=""
EVENT_CACHE_DIR=""
TEACHER_CACHE_DIR=""
TEACHER_CHECKPOINT=""
SEQUENCE="interlaken_00_c"
DEVICE="cuda"
FPS="20"
OUTPUT_DIR=""
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/visualize_v100_sequence.sh \
    --run-dir PATH \
    --root PATH \
    --event-cache-dir PATH \
    --teacher-cache-dir PATH \
    --teacher-checkpoint PATH \
    [--sequence NAME] [--device DEVICE] [--fps N] [--output-dir PATH] [--overwrite]

E0/E1/E2 are evaluated sequentially on one GPU. Recurrent state is preserved
across every non-overlapping validation clip and reset only at sequence start.
Existing valid clip artifacts are preserved, so an interrupted export can be rerun.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-dir) RUN_DIR=${2:?}; shift 2 ;;
    --root) ROOT=${2:?}; shift 2 ;;
    --event-cache-dir) EVENT_CACHE_DIR=${2:?}; shift 2 ;;
    --teacher-cache-dir) TEACHER_CACHE_DIR=${2:?}; shift 2 ;;
    --teacher-checkpoint) TEACHER_CHECKPOINT=${2:?}; shift 2 ;;
    --sequence) SEQUENCE=${2:?}; shift 2 ;;
    --device) DEVICE=${2:?}; shift 2 ;;
    --fps) FPS=${2:?}; shift 2 ;;
    --output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[ -n "$RUN_DIR" ] || fail "--run-dir is required"
[ -n "$ROOT" ] || fail "--root is required"
[ -n "$EVENT_CACHE_DIR" ] || fail "--event-cache-dir is required"
[ -n "$TEACHER_CACHE_DIR" ] || fail "--teacher-cache-dir is required"
[ -n "$TEACHER_CHECKPOINT" ] || fail "--teacher-checkpoint is required"
[ -d "$RUN_DIR" ] || fail "run directory not found: $RUN_DIR"
[ -d "$ROOT" ] || fail "DSEC root not found: $ROOT"
[ -d "$EVENT_CACHE_DIR" ] || fail "event cache not found: $EVENT_CACHE_DIR"
[ -d "$TEACHER_CACHE_DIR" ] || fail "teacher cache not found: $TEACHER_CACHE_DIR"
[ -f "$TEACHER_CHECKPOINT" ] || fail "teacher checkpoint not found: $TEACHER_CHECKPOINT"
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env first"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
RUN_DIR=$(cd "$RUN_DIR" && pwd -P)
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$RUN_DIR/feature_visualization/${SEQUENCE}_sequence"
fi
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

NAMES=(e0_gep_like_dinov3 e1_h_distill_lstm e2_dual_distill_lstm)
LABELS=(E0 E1 E2)
CONTEXT_DIR="$OUTPUT_DIR/context"

for INDEX in 0 1 2; do
  NAME=${NAMES[$INDEX]}
  LABEL=${LABELS[$INDEX]}
  CHECKPOINT="$RUN_DIR/$NAME/checkpoints/best.pt"
  FEATURE_DIR="$OUTPUT_DIR/${LABEL}"
  [ -f "$CHECKPOINT" ] || fail "best checkpoint not found: $CHECKPOINT"
  EXTRA_ARGS=()
  if [ "$INDEX" -eq 0 ]; then
    EXTRA_ARGS+=(--context-dir "$CONTEXT_DIR")
  fi
  if [ "$OVERWRITE" -eq 1 ]; then
    EXTRA_ARGS+=(--overwrite)
  fi
  printf '[sequence-visualization] exporting %s\n' "$LABEL"
  python tools/export_feature_sequence.py \
    --checkpoint "$CHECKPOINT" \
    --root "$ROOT" \
    --event-cache-dir "$EVENT_CACHE_DIR" \
    --teacher-cache-dir "$TEACHER_CACHE_DIR" \
    --teacher-checkpoint "$TEACHER_CHECKPOINT" \
    --sequence "$SEQUENCE" \
    --device "$DEVICE" \
    --output-dir "$FEATURE_DIR" \
    "${EXTRA_ARGS[@]}"
done

python tools/render_feature_sequence.py \
  --context-dir "$CONTEXT_DIR" \
  --source "E0 Pz:$OUTPUT_DIR/E0:Pz" \
  --source "E1 Ph:$OUTPUT_DIR/E1:Ph" \
  --source "E2 Pz:$OUTPUT_DIR/E2:Pz" \
  --source "E2 Ph:$OUTPUT_DIR/E2:Ph" \
  --fps "$FPS" \
  --output "$OUTPUT_DIR/alignment.mp4"

printf '[sequence-visualization] outputs: %s\n' "$OUTPUT_DIR"
