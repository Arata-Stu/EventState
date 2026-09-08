#!/usr/bin/env bash

# Fine-tune E0/E2/E4, then train matched non-recurrent/recurrent scratch baselines.

set -euo pipefail

RUN_DIR=""
EVENT_CACHE_DIR=""
LABELS_ROOT=""
DATASET_ROOT=""
OUTPUT_DIR=""
BATCH_SIZE=4
EPOCHS=50
VALIDATE_EVERY=5
SEEDS="0"

usage() {
  cat <<'EOF'
Usage:
  bash tools/run_dsec_detection_finetune_scratch.sh \
    --run-dir PATH --event-cache-dir PATH --labels-root PATH \
    --dataset-root PATH --output-dir PATH \
    [--batch-size N] [--epochs N] [--validate-every N] [--seeds 0,1,2]

For each seed, GPUs 0/1/2 first fine-tune E0-z, E2-h, and E4-h. Then GPU 0
trains the E0-shaped non-recurrent scratch model and GPU 1 trains the E2-shaped
one-layer-LSTM scratch model. All use the same official split and YOLOX recipe.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-dir) RUN_DIR=${2:?}; shift 2 ;;
    --event-cache-dir) EVENT_CACHE_DIR=${2:?}; shift 2 ;;
    --labels-root) LABELS_ROOT=${2:?}; shift 2 ;;
    --dataset-root) DATASET_ROOT=${2:?}; shift 2 ;;
    --output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
    --batch-size) BATCH_SIZE=${2:?}; shift 2 ;;
    --epochs) EPOCHS=${2:?}; shift 2 ;;
    --validate-every) VALIDATE_EVERY=${2:?}; shift 2 ;;
    --seeds) SEEDS=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[ -d "$RUN_DIR" ] || fail "pretraining run not found: $RUN_DIR"
[ -d "$EVENT_CACHE_DIR" ] || fail "event cache not found: $EVENT_CACHE_DIR"
[ -d "$LABELS_ROOT" ] || fail "labels root not found: $LABELS_ROOT"
[ -d "$DATASET_ROOT" ] || fail "dataset root not found: $DATASET_ROOT"
[ -n "$OUTPUT_DIR" ] || fail "--output-dir is required"
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env"

RUN_DIR=$(cd "$RUN_DIR" && pwd -P)
EVENT_CACHE_DIR=$(cd "$EVENT_CACHE_DIR" && pwd -P)
LABELS_ROOT=$(cd "$LABELS_ROOT" && pwd -P)
DATASET_ROOT=$(cd "$DATASET_ROOT" && pwd -P)
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd -P)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
cd "$PROJECT_ROOT"

E0_CHECKPOINT="$RUN_DIR/e0_gep_like_dinov3/checkpoints/step_00100000.pt"
E2_CHECKPOINT="$RUN_DIR/e2_dual_distill_lstm/checkpoints/step_00100000.pt"
E4_CHECKPOINT="$RUN_DIR/e4_dual_distill_lstm_event_dropout/checkpoints/step_00100000.pt"
for checkpoint in "$E0_CHECKPOINT" "$E2_CHECKPOINT" "$E4_CHECKPOINT"; do
  [ -f "$checkpoint" ] || fail "final checkpoint not found: $checkpoint"
done

train_one() {
  local gpu=$1
  local mode=$2
  local label=$3
  local feature=$4
  local checkpoint=$5
  local seed=$6
  local log_dir="$OUTPUT_DIR/logs/seed_$seed"
  mkdir -p "$log_dir"
  CUDA_VISIBLE_DEVICES="$gpu" python tools/train_dsec_detection_end_to_end.py \
    --mode "$mode" \
    --reference-checkpoint "$checkpoint" \
    --event-cache-dir "$EVENT_CACHE_DIR" \
    --labels-root "$LABELS_ROOT" \
    --dataset-root "$DATASET_ROOT" \
    --feature "$feature" \
    --output-dir "$OUTPUT_DIR/$mode/$label/seed_$seed" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --validate-every "$VALIDATE_EVERY" \
    --seed "$seed" \
    --device cuda \
    >"$log_dir/${mode}_${label}.log" 2>&1
}

IFS=',' read -r -a SEED_VALUES <<< "$SEEDS"
for seed in "${SEED_VALUES[@]}"; do
  case "$seed" in *[!0-9]*|'') fail "seeds must be comma-separated non-negative integers" ;; esac

  PIDS=()
  train_one 0 finetune E0 z "$E0_CHECKPOINT" "$seed" & PIDS+=("$!")
  train_one 1 finetune E2 h "$E2_CHECKPOINT" "$seed" & PIDS+=("$!")
  train_one 2 finetune E4 h "$E4_CHECKPOINT" "$seed" & PIDS+=("$!")
  printf '[detection] seed=%s fine-tuning E0/E2/E4\n' "$seed"
  STATUS=0
  for pid in "${PIDS[@]}"; do wait "$pid" || STATUS=1; done
  [ "$STATUS" -eq 0 ] || fail "fine-tuning failed; inspect $OUTPUT_DIR/logs/seed_$seed"

  PIDS=()
  train_one 0 scratch no_memory z "$E0_CHECKPOINT" "$seed" & PIDS+=("$!")
  train_one 1 scratch lstm h "$E2_CHECKPOINT" "$seed" & PIDS+=("$!")
  printf '[detection] seed=%s scratch no_memory/lstm\n' "$seed"
  STATUS=0
  for pid in "${PIDS[@]}"; do wait "$pid" || STATUS=1; done
  [ "$STATUS" -eq 0 ] || fail "scratch training failed; inspect $OUTPUT_DIR/logs/seed_$seed"
done

printf '[detection] complete: %s\n' "$OUTPUT_DIR"
