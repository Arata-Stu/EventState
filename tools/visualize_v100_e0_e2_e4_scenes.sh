#!/usr/bin/env bash

# Render final E0/E2/E4 representations over multiple unseen DSEC sequences.

set -euo pipefail

RUN_DIR=""
ROOT=""
EVENT_CACHE_DIR=""
TEACHER_CACHE_DIR=""
TEACHER_CHECKPOINT=""
OUTPUT_DIR=""
FPS=20
STEP=100000
GPUS="0,1,2"
OVERWRITE=0
RENDER_ONLY=0
SEQUENCES="thun_01_a,thun_01_b,interlaken_00_a,interlaken_00_b,interlaken_01_a"
SEQUENCES="$SEQUENCES,zurich_city_12_a,zurich_city_13_a,zurich_city_13_b"
SEQUENCES="$SEQUENCES,zurich_city_14_a,zurich_city_14_b,zurich_city_14_c,zurich_city_15_a"

usage() {
  cat <<'EOF'
Usage:
  bash tools/visualize_v100_e0_e2_e4_scenes.sh \
    --run-dir PATH --root PATH --event-cache-dir PATH \
    --teacher-cache-dir PATH --teacher-checkpoint PATH \
    [--output-dir PATH] [--sequences a,b,c] [--gpus 0,1,2] \
    [--step 100000] [--fps 20] [--overwrite] [--render-only]

The default set contains all 12 original DSEC test sequences and therefore was
not used by dsec_det_train41 pretraining. GPU 0/1/2 load E0/E2/E4 once and each
streams every scene. Each video compares E0 Pz, E2 Ph/Pz, and E4 Ph/Pz.
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
    --output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
    --sequences) SEQUENCES=${2:?}; shift 2 ;;
    --gpus) GPUS=${2:?}; shift 2 ;;
    --step) STEP=${2:?}; shift 2 ;;
    --fps) FPS=${2:?}; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --render-only) RENDER_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[ -d "$RUN_DIR" ] || fail "run directory not found: $RUN_DIR"
[ -d "$ROOT" ] || fail "DSEC root not found: $ROOT"
[ -d "$EVENT_CACHE_DIR" ] || fail "event cache not found: $EVENT_CACHE_DIR"
[ -d "$TEACHER_CACHE_DIR" ] || fail "teacher cache not found: $TEACHER_CACHE_DIR"
[ -f "$TEACHER_CHECKPOINT" ] || fail "teacher checkpoint not found: $TEACHER_CHECKPOINT"
case "$STEP" in *[!0-9]*|'') fail "--step must be a positive integer" ;; esac
[ "$STEP" -gt 0 ] || fail "--step must be positive"
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env"
python -c 'import imageio_ffmpeg, matplotlib' >/dev/null 2>&1 || \
  fail "visualization dependencies are missing; run: uv sync --active --extra prepare --extra detection --extra visualize"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
RUN_DIR=$(cd "$RUN_DIR" && pwd -P)
ROOT=$(cd "$ROOT" && pwd -P)
EVENT_CACHE_DIR=$(cd "$EVENT_CACHE_DIR" && pwd -P)
TEACHER_CACHE_DIR=$(cd "$TEACHER_CACHE_DIR" && pwd -P)
TEACHER_DIRECTORY=$(cd "$(dirname "$TEACHER_CHECKPOINT")" && pwd -P)
TEACHER_CHECKPOINT="$TEACHER_DIRECTORY/$(basename "$TEACHER_CHECKPOINT")"
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$RUN_DIR/feature_visualization/test_scenes_step${STEP}"
fi
mkdir -p "$OUTPUT_DIR/logs"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd -P)
cd "$PROJECT_ROOT"

printf -v STEP_PADDED '%08d' "$STEP"
E0_CHECKPOINT="$RUN_DIR/e0_gep_like_dinov3/checkpoints/step_${STEP_PADDED}.pt"
E2_CHECKPOINT="$RUN_DIR/e2_dual_distill_lstm/checkpoints/step_${STEP_PADDED}.pt"
E4_CHECKPOINT="$RUN_DIR/e4_dual_distill_lstm_event_dropout/checkpoints/step_${STEP_PADDED}.pt"
for checkpoint in "$E0_CHECKPOINT" "$E2_CHECKPOINT" "$E4_CHECKPOINT"; do
  if [ "$RENDER_ONLY" -eq 0 ]; then
    [ -f "$checkpoint" ] || fail "checkpoint not found: $checkpoint"
  fi
done

IFS=',' read -r -a GPU_VALUES <<< "$GPUS"
IFS=',' read -r -a SEQUENCE_VALUES <<< "$SEQUENCES"
[ "${#GPU_VALUES[@]}" -eq 3 ] || fail "--gpus must list exactly three devices"
[ "${#SEQUENCE_VALUES[@]}" -gt 0 ] || fail "--sequences is empty"
for gpu in "${GPU_VALUES[@]}"; do
  case "$gpu" in *[!0-9]*|'') fail "--gpus must contain non-negative integers" ;; esac
done
for sequence in "${SEQUENCE_VALUES[@]}"; do
  [ -n "$sequence" ] || fail "--sequences contains an empty name"
  [ -d "$ROOT/test_images/$sequence" ] || fail "test sequence missing: $sequence"
  [ -d "$EVENT_CACHE_DIR/$sequence" ] || fail "event cache missing: $sequence"
  [ -d "$TEACHER_CACHE_DIR/$sequence" ] || fail "teacher cache missing: $sequence"
done

render_sequence() {
  local sequence=$1
  local destination="$OUTPUT_DIR/$sequence"
  local context="$destination/context"
  mkdir -p "$destination"
  python tools/render_feature_sequence.py \
    --context-dir "$context" \
    --source "E0 Pz:$destination/E0:Pz" \
    --source "E2 Ph:$destination/E2:Ph" \
    --source "E2 Pz:$destination/E2:Pz" \
    --source "E4 Ph:$destination/E4:Ph" \
    --source "E4 Pz:$destination/E4:Pz" \
    --fps "$FPS" --output "$destination/alignment.mp4"
}

STATUS=0
if [ "$RENDER_ONLY" -eq 0 ]; then
  OVERWRITE_ARGS=()
  if [ "$OVERWRITE" -eq 1 ]; then OVERWRITE_ARGS+=(--overwrite); fi
  COMMON_EXPORT_ARGS=(
    --root "$ROOT"
    --event-cache-dir "$EVENT_CACHE_DIR"
    --teacher-cache-dir "$TEACHER_CACHE_DIR"
    --teacher-checkpoint "$TEACHER_CHECKPOINT"
    --split test
    --sequences "${SEQUENCE_VALUES[@]}"
    --output-root "$OUTPUT_DIR"
    --device cuda
  )

  CUDA_VISIBLE_DEVICES="${GPU_VALUES[0]}" python tools/export_feature_sequences.py \
    "${COMMON_EXPORT_ARGS[@]}" --checkpoint "$E0_CHECKPOINT" --label E0 \
    --feature Pz --write-context "${OVERWRITE_ARGS[@]}" \
    >"$OUTPUT_DIR/logs/export_E0.log" 2>&1 &
  PIDS=("$!")
  printf '[multi-scene] GPU %s: exporting E0 pid=%s\n' "${GPU_VALUES[0]}" "$!"
  CUDA_VISIBLE_DEVICES="${GPU_VALUES[1]}" python tools/export_feature_sequences.py \
    "${COMMON_EXPORT_ARGS[@]}" --checkpoint "$E2_CHECKPOINT" --label E2 \
    --feature Pz --feature Ph "${OVERWRITE_ARGS[@]}" \
    >"$OUTPUT_DIR/logs/export_E2.log" 2>&1 &
  PIDS+=("$!")
  printf '[multi-scene] GPU %s: exporting E2 pid=%s\n' "${GPU_VALUES[1]}" "$!"
  CUDA_VISIBLE_DEVICES="${GPU_VALUES[2]}" python tools/export_feature_sequences.py \
    "${COMMON_EXPORT_ARGS[@]}" --checkpoint "$E4_CHECKPOINT" --label E4 \
    --feature Pz --feature Ph "${OVERWRITE_ARGS[@]}" \
    >"$OUTPUT_DIR/logs/export_E4.log" 2>&1 &
  PIDS+=("$!")
  printf '[multi-scene] GPU %s: exporting E4 pid=%s\n' "${GPU_VALUES[2]}" "$!"
  LABELS=(E0 E2 E4)
  for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
      printf '[multi-scene] export complete: %s\n' "${LABELS[$index]}"
    else
      printf '[multi-scene] export failed: %s\n' "${LABELS[$index]}" >&2
      STATUS=1
    fi
  done
  [ "$STATUS" -eq 0 ] || fail "feature export failed: $OUTPUT_DIR/logs"
else
  printf '[multi-scene] render-only: reusing exported feature artifacts\n'
fi

sequence_index=0
while [ "$sequence_index" -lt "${#SEQUENCE_VALUES[@]}" ]; do
  PIDS=()
  NAMES=()
  for render_index in 0 1 2; do
    [ "$sequence_index" -lt "${#SEQUENCE_VALUES[@]}" ] || break
    sequence=${SEQUENCE_VALUES[$sequence_index]}
    render_sequence "$sequence" >"$OUTPUT_DIR/logs/render_$sequence.log" 2>&1 &
    PIDS+=("$!")
    NAMES+=("$sequence")
    printf '[multi-scene] rendering sequence=%s pid=%s\n' "$sequence" "$!"
    sequence_index=$((sequence_index + 1))
  done
  for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
      printf '[multi-scene] complete: %s\n' "${NAMES[$index]}"
    else
      printf '[multi-scene] failed: %s (see logs)\n' "${NAMES[$index]}" >&2
      STATUS=1
    fi
  done
  [ "$STATUS" -eq 0 ] || break
done

[ "$STATUS" -eq 0 ] || fail "one or more scene visualizations failed: $OUTPUT_DIR/logs"
python tools/summarize_sequence_visualizations.py \
  "$OUTPUT_DIR" --output "$OUTPUT_DIR/summary.csv"
printf '[multi-scene] outputs: %s\n' "$OUTPUT_DIR"
