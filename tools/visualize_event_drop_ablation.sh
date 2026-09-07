#!/usr/bin/env bash

# Test whether recurrent state preserves teacher features through missing events.

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
MODEL_SELECTION="E2"
DROP_START=32
DROP_STRIDE=64
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/visualize_event_drop_ablation.sh \
    --run-dir PATH \
    --root PATH \
    --event-cache-dir PATH \
    --teacher-cache-dir PATH \
    --teacher-checkpoint PATH \
    [--sequence NAME] [--model E1|E2|both] [--drop-start N] [--drop-stride N] \
    [--device DEVICE] [--fps N] [--output-dir PATH] [--overwrite]

The default E2 run compares continuous Ph, frame-reset Ph, and continuous Pz
under repeated 0/1/2/4/8-frame event gaps. Use --model both to include E1.
Every condition runs sequentially on one GPU and can resume artifact export.
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
    --model) MODEL_SELECTION=${2:?}; shift 2 ;;
    --drop-start) DROP_START=${2:?}; shift 2 ;;
    --drop-stride) DROP_STRIDE=${2:?}; shift 2 ;;
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
case "$MODEL_SELECTION" in E1|E2|both) ;; *) fail "--model must be E1, E2, or both" ;; esac
case "$DROP_START" in *[!0-9]*|'') fail "--drop-start must be non-negative" ;; esac
case "$DROP_STRIDE" in *[!0-9]*|'') fail "--drop-stride must be positive" ;; esac
[ "$DROP_STRIDE" -gt 0 ] || fail "--drop-stride must be positive"
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env first"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
RUN_DIR=$(cd "$RUN_DIR" && pwd -P)
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$RUN_DIR/feature_visualization/${SEQUENCE}_event_drop"
fi
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

if [ "$MODEL_SELECTION" = "both" ]; then
  MODELS=(E1 E2)
else
  MODELS=("$MODEL_SELECTION")
fi
GAPS=(0 1 2 4 8)
POLICIES=(continuous frame)
SUMMARY_ARGS=()
CONTEXT_DIR="$OUTPUT_DIR/context"

for GAP in "${GAPS[@]}"; do
  [ "$GAP" -le "$DROP_STRIDE" ] || fail "gap $GAP exceeds --drop-stride"
  for MODEL_INDEX in "${!MODELS[@]}"; do
    MODEL=${MODELS[$MODEL_INDEX]}
    case "$MODEL" in
      E1) NAME=e1_h_distill_lstm ;;
      E2) NAME=e2_dual_distill_lstm ;;
    esac
    CHECKPOINT="$RUN_DIR/$NAME/checkpoints/best.pt"
    [ -f "$CHECKPOINT" ] || fail "best checkpoint not found: $CHECKPOINT"
    for POLICY in "${POLICIES[@]}"; do
      FEATURE_DIR="$OUTPUT_DIR/${MODEL}_gap${GAP}_${POLICY}"
      EXTRA_ARGS=()
      if [ "$GAP" -eq 0 ] && [ "$MODEL_INDEX" -eq 0 ] && [ "$POLICY" = "continuous" ]; then
        EXTRA_ARGS+=(--context-dir "$CONTEXT_DIR")
      fi
      if [ "$POLICY" = "frame" ] || [ "$MODEL" = "E1" ]; then
        EXTRA_ARGS+=(--feature Ph)
      fi
      if [ "$OVERWRITE" -eq 1 ]; then
        EXTRA_ARGS+=(--overwrite)
      fi
      printf '[event-drop] model=%s gap=%s policy=%s\n' "$MODEL" "$GAP" "$POLICY"
      python tools/export_feature_sequence.py \
        --checkpoint "$CHECKPOINT" \
        --root "$ROOT" \
        --event-cache-dir "$EVENT_CACHE_DIR" \
        --teacher-cache-dir "$TEACHER_CACHE_DIR" \
        --teacher-checkpoint "$TEACHER_CHECKPOINT" \
        --sequence "$SEQUENCE" \
        --state-policy "$POLICY" \
        --drop-length "$GAP" \
        --drop-start "$DROP_START" \
        --drop-stride "$DROP_STRIDE" \
        --device "$DEVICE" \
        --output-dir "$FEATURE_DIR" \
        "${EXTRA_ARGS[@]}"
    done

    RENDER_ARGS=(
      --context-dir "$CONTEXT_DIR"
      --source "continuous Ph:$OUTPUT_DIR/${MODEL}_gap${GAP}_continuous:Ph"
      --source "frame reset Ph:$OUTPUT_DIR/${MODEL}_gap${GAP}_frame:Ph"
    )
    if [ "$MODEL" = "E2" ]; then
      RENDER_ARGS+=(
        --source "current Pz:$OUTPUT_DIR/${MODEL}_gap${GAP}_continuous:Pz"
      )
    fi
    VIDEO="$OUTPUT_DIR/${MODEL}_gap${GAP}.mp4"
    python tools/render_feature_sequence.py \
      "${RENDER_ARGS[@]}" \
      --fps "$FPS" \
      --output "$VIDEO"
    SUMMARY_ARGS+=(--result "$MODEL=${VIDEO%.mp4}.json")
  done
done

python tools/summarize_event_drop.py \
  "${SUMMARY_ARGS[@]}" \
  --output "$OUTPUT_DIR/event_drop_summary.csv"

printf '[event-drop] outputs: %s\n' "$OUTPUT_DIR"
