#!/usr/bin/env bash

# Compare the current seed-0 fine-tuned E0/E2/E4 best checkpoints on one sequence.

set -euo pipefail

DETECTION_DIR=""
EVENT_CACHE_DIR=""
LABELS_ROOT=""
DATASET_ROOT=""
OUTPUT_DIR=""
ROLE="test"
GPU="0"
FRAME_MODE="all"
MAX_FRAMES=""
START_FRAME="0"
SCORE_THRESHOLD="0.25"
FPS="20"
DETECTION_BACKGROUND="rgb"
SEQUENCES=()

usage() {
  cat <<'EOF'
Usage:
  bash tools/visualize_finetuned_dsec_detection_e0_e2_e4.sh \
    --detection-dir PATH --event-cache-dir PATH --labels-root PATH \
    --dataset-root PATH --output-dir PATH \
    [--role test] [--gpu 0] [--frame-mode all|evaluated] \
    [--start-frame N] [--max-frames N] [--score-threshold X] [--fps X] \
    [--detection-background rgb|event] [--sequences SEQUENCE ...]

The three best.pt files are loaded from:
  DETECTION_DIR/finetune/{E0,E2,E4}/seed_0/best.pt

Only one model is resident on the selected GPU at a time. All checkpoints are
snapshotted in CPU memory before inference so an active trainer cannot change
the weights midway through this visualization run.
Without --sequences, all official sequences in the selected role are rendered.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --detection-dir) DETECTION_DIR=${2:?}; shift 2 ;;
    --event-cache-dir) EVENT_CACHE_DIR=${2:?}; shift 2 ;;
    --labels-root) LABELS_ROOT=${2:?}; shift 2 ;;
    --dataset-root) DATASET_ROOT=${2:?}; shift 2 ;;
    --output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
    --sequence) SEQUENCES+=("${2:?}"); shift 2 ;;
    --role) ROLE=${2:?}; shift 2 ;;
    --gpu) GPU=${2:?}; shift 2 ;;
    --frame-mode) FRAME_MODE=${2:?}; shift 2 ;;
    --start-frame) START_FRAME=${2:?}; shift 2 ;;
    --max-frames) MAX_FRAMES=${2:?}; shift 2 ;;
    --score-threshold) SCORE_THRESHOLD=${2:?}; shift 2 ;;
    --fps) FPS=${2:?}; shift 2 ;;
    --detection-background) DETECTION_BACKGROUND=${2:?}; shift 2 ;;
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

[ -d "$DETECTION_DIR" ] || fail "detection directory not found: $DETECTION_DIR"
[ -d "$EVENT_CACHE_DIR" ] || fail "event cache not found: $EVENT_CACHE_DIR"
[ -d "$LABELS_ROOT" ] || fail "labels root not found: $LABELS_ROOT"
[ -d "$DATASET_ROOT" ] || fail "dataset root not found: $DATASET_ROOT"
[ -n "$OUTPUT_DIR" ] || fail "--output-dir is required"
case "$ROLE" in train|val|test) ;; *) fail "invalid role: $ROLE" ;; esac
case "$FRAME_MODE" in all|evaluated) ;; *) fail "invalid frame mode: $FRAME_MODE" ;; esac
case "$DETECTION_BACKGROUND" in rgb|event) ;; *) fail "invalid detection background" ;; esac
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env"

if [ "${#SEQUENCES[@]}" -eq 0 ]; then
  if [ "$ROLE" = "val" ]; then
    SEQUENCES=(
      zurich_city_16_a zurich_city_17_a zurich_city_18_a
      zurich_city_19_a zurich_city_20_a zurich_city_21_a
    )
  elif [ "$ROLE" = "test" ]; then
    SEQUENCES=(
      thun_01_a thun_01_b thun_02_a interlaken_00_a interlaken_00_b
      interlaken_01_a zurich_city_12_a zurich_city_13_a zurich_city_13_b
      zurich_city_14_a zurich_city_14_b zurich_city_14_c zurich_city_15_a
    )
  else
    fail "--sequences is required for role=train"
  fi
fi

DETECTION_DIR=$(cd "$DETECTION_DIR" && pwd -P)
EVENT_CACHE_DIR=$(cd "$EVENT_CACHE_DIR" && pwd -P)
LABELS_ROOT=$(cd "$LABELS_ROOT" && pwd -P)
DATASET_ROOT=$(cd "$DATASET_ROOT" && pwd -P)
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd -P)
mkdir -p "$OUTPUT_DIR/logs"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
cd "$PROJECT_ROOT"

E0="$DETECTION_DIR/finetune/E0/seed_0/best.pt"
E2="$DETECTION_DIR/finetune/E2/seed_0/best.pt"
E4="$DETECTION_DIR/finetune/E4/seed_0/best.pt"
for checkpoint in "$E0" "$E2" "$E4"; do
  [ -f "$checkpoint" ] || fail "fine-tune best checkpoint not found: $checkpoint"
done
python -c 'import imageio_ffmpeg' >/dev/null 2>&1 || \
  fail "visualization dependencies missing; run uv sync --active --extra visualize --extra detection"

extra_args=()
if [ -n "$MAX_FRAMES" ]; then
  extra_args+=(--max-frames "$MAX_FRAMES")
fi

for sequence in "${SEQUENCES[@]}"; do
  output="$OUTPUT_DIR/${sequence}.mp4"
  printf '[finetune-visualization] sequence=%s gpu=%s output=%s\n' \
    "$sequence" "$GPU" "$output"
  CUDA_VISIBLE_DEVICES="$GPU" nice -n 10 python \
    tools/visualize_finetuned_dsec_detection_sequence.py \
    --source "E0:$E0:z" \
    --source "E2:$E2:h" \
    --source "E4:$E4:h" \
    --event-cache-dir "$EVENT_CACHE_DIR" \
    --labels-root "$LABELS_ROOT" \
    --dataset-root "$DATASET_ROOT" \
    --role "$ROLE" \
    --sequence "$sequence" \
    --frame-mode "$FRAME_MODE" \
    --output "$output" \
    --device cuda \
    --precision fp16 \
    --score-threshold "$SCORE_THRESHOLD" \
    --detection-background "$DETECTION_BACKGROUND" \
    --fps "$FPS" \
    --start-frame "$START_FRAME" \
    "${extra_args[@]}" \
    >"$OUTPUT_DIR/logs/$sequence.log" 2>&1
  printf '[finetune-visualization] complete: %s\n' "$output"
done

printf '[finetune-visualization] outputs: %s\n' "$OUTPUT_DIR"
