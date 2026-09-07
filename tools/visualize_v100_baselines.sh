#!/usr/bin/env bash

# Export one common validation clip from E0/E1/E2 best checkpoints and compare it.

set -euo pipefail

RUN_DIR=""
ROOT=""
EVENT_CACHE_DIR=""
TEACHER_CACHE_DIR=""
TEACHER_CHECKPOINT=""
SEQUENCE="interlaken_00_c"
CLIP_INDEX=0
DEVICE="cuda"
OUTPUT_DIR=""

usage() {
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/visualize_v100_baselines.sh \
    --run-dir PATH \
    --root PATH \
    --event-cache-dir PATH \
    --teacher-cache-dir PATH \
    --teacher-checkpoint PATH \
    [--sequence NAME] [--clip-index N] [--device DEVICE] [--output-dir PATH]

The selected run must contain the E0/E1/E2 directories produced by
tools/run_v100_baselines.sh. Run this after training so visualization inference
does not compete with a training process for GPU memory.
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
    --clip-index) CLIP_INDEX=${2:?}; shift 2 ;;
    --device) DEVICE=${2:?}; shift 2 ;;
    --output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
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
case "$CLIP_INDEX" in *[!0-9]*|'') fail "--clip-index must be non-negative" ;; esac
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env first"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
RUN_DIR=$(cd "$RUN_DIR" && pwd -P)
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$RUN_DIR/feature_visualization/${SEQUENCE}_clip${CLIP_INDEX}"
fi
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

NAMES=(e0_gep_like_dinov3 e1_h_distill_lstm e2_dual_distill_lstm)
LABELS=(E0 E1 E2)
ARTIFACTS=()

for INDEX in 0 1 2; do
  NAME=${NAMES[$INDEX]}
  CHECKPOINT="$RUN_DIR/$NAME/checkpoints/best.pt"
  ARTIFACT="$OUTPUT_DIR/e${INDEX}.pt"
  [ -f "$CHECKPOINT" ] || fail "best checkpoint not found: $CHECKPOINT"
  printf '[feature-visualization] exporting %s\n' "${LABELS[$INDEX]}"
  python tools/export_features.py \
    --checkpoint "$CHECKPOINT" \
    --root "$ROOT" \
    --event-cache-dir "$EVENT_CACHE_DIR" \
    --teacher-cache-dir "$TEACHER_CACHE_DIR" \
    --teacher-checkpoint "$TEACHER_CHECKPOINT" \
    --sequence "$SEQUENCE" \
    --clip-index "$CLIP_INDEX" \
    --device "$DEVICE" \
    --output "$ARTIFACT"
  ARTIFACTS+=("$ARTIFACT")
done

python tools/visualize_features.py \
  "${ARTIFACTS[@]}" \
  --labels "${LABELS[@]}" \
  --output "$OUTPUT_DIR/projected_alignment.png"

printf '[feature-visualization] outputs: %s\n' "$OUTPUT_DIR"
