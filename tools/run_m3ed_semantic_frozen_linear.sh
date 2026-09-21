#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash tools/run_m3ed_semantic_frozen_linear.sh \
    --prepared-root PATH --target-cache-dir PATH --teacher-cache-dir PATH \
    --teacher-checkpoint PATH --feature-cache-root PATH --output-root PATH \
    --model LABEL:CHECKPOINT:FEATURE:GPU [--model ...]

Each model is cached and trained independently on its assigned GPU. FEATURE is
z, h, or concat. Validation uses car_urban_day_ucity_small_loop.
EOF
}

PREPARED_ROOT=""
TARGET_CACHE_DIR=""
TEACHER_CACHE_DIR=""
TEACHER_CHECKPOINT=""
FEATURE_CACHE_ROOT=""
OUTPUT_ROOT=""
EPOCHS=50
BATCH_SIZE=8
NUM_WORKERS=4
VALIDATE_EVERY=5
MODELS=()

while (($#)); do
  case "$1" in
    --prepared-root) PREPARED_ROOT="$2"; shift 2 ;;
    --target-cache-dir) TARGET_CACHE_DIR="$2"; shift 2 ;;
    --teacher-cache-dir) TEACHER_CACHE_DIR="$2"; shift 2 ;;
    --teacher-checkpoint) TEACHER_CHECKPOINT="$2"; shift 2 ;;
    --feature-cache-root) FEATURE_CACHE_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --model) MODELS+=("$2"); shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --validate-every) VALIDATE_EVERY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for VALUE in \
  "$PREPARED_ROOT" "$TARGET_CACHE_DIR" "$TEACHER_CACHE_DIR" \
  "$TEACHER_CHECKPOINT" "$FEATURE_CACHE_ROOT" "$OUTPUT_ROOT"; do
  [[ -n "$VALUE" ]] || { usage >&2; exit 2; }
done
((${#MODELS[@]} > 0)) || { echo "At least one --model is required" >&2; exit 2; }

mkdir -p "$FEATURE_CACHE_ROOT" "$OUTPUT_ROOT/logs"

run_model() {
  local spec="$1"
  local label checkpoint feature gpu extra
  IFS=: read -r label checkpoint feature gpu extra <<<"$spec"
  [[ -n "$label" && -n "$checkpoint" && -n "$feature" && -n "$gpu" && -z "${extra:-}" ]] || {
    echo "Invalid --model specification: $spec" >&2
    return 2
  }
  [[ "$feature" == "z" || "$feature" == "h" || "$feature" == "concat" ]] || {
    echo "Invalid feature in --model: $spec" >&2
    return 2
  }
  [[ -f "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; return 1; }

  local cache_dir="$FEATURE_CACHE_ROOT/$label"
  local run_dir="$OUTPUT_ROOT/$label/seed_0"
  mkdir -p "$cache_dir" "$run_dir"

  local cache_features=("$feature")
  if [[ "$feature" == "concat" ]]; then
    cache_features=(z h)
  fi
  for role in train validation; do
    echo "[m3ed-semantic] label=$label role=$role gpu=$gpu cache"
    CUDA_VISIBLE_DEVICES="$gpu" python tools/cache_m3ed_semantic_features.py \
      --checkpoint "$checkpoint" \
      --prepared-root "$PREPARED_ROOT" \
      --teacher-cache-dir "$TEACHER_CACHE_DIR" \
      --target-cache-dir "$TARGET_CACHE_DIR" \
      --teacher-checkpoint "$TEACHER_CHECKPOINT" \
      --output-dir "$cache_dir" \
      --role "$role" \
      --features "${cache_features[@]}" \
      --device cuda
  done

  echo "[m3ed-semantic] label=$label gpu=$gpu train linear+CE"
  CUDA_VISIBLE_DEVICES="$gpu" python tools/train_m3ed_semantic.py \
    --feature-cache-dir "$cache_dir" \
    --target-cache-dir "$TARGET_CACHE_DIR" \
    --output-dir "$run_dir" \
    --feature "$feature" \
    --head-type linear \
    --loss ce \
    --epochs "$EPOCHS" \
    --validate-every "$VALIDATE_EVERY" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --precision fp16 \
    --seed 0 \
    --device cuda
  echo "[m3ed-semantic] complete label=$label result=$run_dir/validation_metrics.json"
}

pids=()
labels=()
for spec in "${MODELS[@]}"; do
  label="${spec%%:*}"
  run_model "$spec" >"$OUTPUT_ROOT/logs/$label.log" 2>&1 &
  pids+=("$!")
  labels+=("$label")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "[m3ed-semantic] failed label=${labels[$index]}; inspect $OUTPUT_ROOT/logs/${labels[$index]}.log" >&2
    failed=1
  fi
done
((failed == 0)) || exit 1
echo "[m3ed-semantic] all complete: $OUTPUT_ROOT"
