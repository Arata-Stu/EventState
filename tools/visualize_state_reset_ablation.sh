#!/usr/bin/env bash

# Compare continuous, clip-reset, and frame-reset recurrent inference for E1/E2.

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
  CUDA_VISIBLE_DEVICES=0 bash tools/visualize_state_reset_ablation.sh \
    --run-dir PATH \
    --root PATH \
    --event-cache-dir PATH \
    --teacher-cache-dir PATH \
    --teacher-checkpoint PATH \
    [--sequence NAME] [--device DEVICE] [--fps N] [--output-dir PATH] [--overwrite]

Each model is run under three inference policies:
  continuous  state survives across the complete sequence
  clip        state resets at each non-overlapping validation clip
  frame       state resets before every frame

Inference runs sequentially on one GPU. Existing valid artifacts are reused.
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
  OUTPUT_DIR="$RUN_DIR/feature_visualization/${SEQUENCE}_state_reset"
fi
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

MODELS=(E1 E2)
NAMES=(e1_h_distill_lstm e2_dual_distill_lstm)
POLICIES=(continuous clip frame)
CONTEXT_DIR="$OUTPUT_DIR/context"

for MODEL_INDEX in 0 1; do
  MODEL=${MODELS[$MODEL_INDEX]}
  NAME=${NAMES[$MODEL_INDEX]}
  CHECKPOINT="$RUN_DIR/$NAME/checkpoints/best.pt"
  [ -f "$CHECKPOINT" ] || fail "best checkpoint not found: $CHECKPOINT"
  for POLICY in "${POLICIES[@]}"; do
    FEATURE_DIR="$OUTPUT_DIR/${MODEL}_${POLICY}"
    EXTRA_ARGS=()
    if [ "$MODEL_INDEX" -eq 0 ] && [ "$POLICY" = "continuous" ]; then
      EXTRA_ARGS+=(--context-dir "$CONTEXT_DIR")
    fi
    if [ "$OVERWRITE" -eq 1 ]; then
      EXTRA_ARGS+=(--overwrite)
    fi
    printf '[state-reset] exporting %s policy=%s\n' "$MODEL" "$POLICY"
    python tools/export_feature_sequence.py \
      --checkpoint "$CHECKPOINT" \
      --root "$ROOT" \
      --event-cache-dir "$EVENT_CACHE_DIR" \
      --teacher-cache-dir "$TEACHER_CACHE_DIR" \
      --teacher-checkpoint "$TEACHER_CHECKPOINT" \
      --sequence "$SEQUENCE" \
      --state-policy "$POLICY" \
      --device "$DEVICE" \
      --output-dir "$FEATURE_DIR" \
      "${EXTRA_ARGS[@]}"
  done
done

for MODEL in "${MODELS[@]}"; do
  python tools/render_feature_sequence.py \
    --context-dir "$CONTEXT_DIR" \
    --source "continuous:$OUTPUT_DIR/${MODEL}_continuous:Ph" \
    --source "clip reset:$OUTPUT_DIR/${MODEL}_clip:Ph" \
    --source "frame reset:$OUTPUT_DIR/${MODEL}_frame:Ph" \
    --fps "$FPS" \
    --output "$OUTPUT_DIR/${MODEL}_state_reset.mp4"
done

printf '[state-reset] outputs: %s\n' "$OUTPUT_DIR"
