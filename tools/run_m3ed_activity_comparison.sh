#!/usr/bin/env bash
# Three matched final-fit experiments. Run on the GPU training host.
set -euo pipefail

ROOT=""
EVENT_STATISTICS=""
EVENT_CACHE_DIR=""
TEACHER_CACHE_DIR=""
CHECKPOINT=""
OUTPUT_ROOT=""
GPU_LIST="0,1,2"
STAGE="smoke"
BATCH_SIZE=4
NUM_WORKERS=4
SEED=0
DRY_RUN=0
RUN_TESTS=1
SUITE="m3ed"
SAMPLING="random"

usage() {
  cat <<'EOF'
Usage: bash tools/run_m3ed_activity_comparison.sh
  --prepared-root PATH --teacher-cache-dir PATH --checkpoint PATH
  [--event-statistics PATH] [--gpus 0,1,2] [--stage smoke|full]
  [--output-root PATH] [--batch-size 4] [--num-workers 4] [--seed 0]
  [--sampling random|hybrid]
  [--dry-run] [--skip-tests]

GPU order: baseline_e2 / activity_only (hard) / active_z_h_soft (alpha=0.5).
All: GEP RGB, clip 16, total batch 4, no augmentation/dropout.
Random (default): independent clips. Hybrid: half random, half stream TBPTT,
equal loss weights, state carried across stream clips and detached at boundaries.
Hybrid requires an even total batch size and starts from DINO, not a Random run.
Established train 4 / validation 1; no test input. Validation every 1000 steps.
Smoke: 100 steps; full: 100000, newly initialized (not resumed from smoke).
Normalization defaults to prepared-root/event_statistics.json, from train 4 only.
Existing prepared and teacher caches are read-only. Raw recordings not required.
Dry-run validates paths/statistics with stdlib Python, but runs no ML code.
EOF
}

fail() { printf 'error: %s\n' "$*" >&2; exit 1; }
while [ "$#" -gt 0 ]; do
  case "$1" in
    --prepared-root) ROOT=${2:?}; EVENT_CACHE_DIR=$ROOT; shift 2 ;;
    --event-statistics) EVENT_STATISTICS=${2:?}; shift 2 ;;
    --teacher-cache-dir) TEACHER_CACHE_DIR=${2:?}; shift 2 ;;
    --checkpoint) CHECKPOINT=${2:?}; shift 2 ;;
    --output-root) OUTPUT_ROOT=${2:?}; shift 2 ;;
    --gpus) GPU_LIST=${2:?}; shift 2 ;;
    --stage) STAGE=${2:?}; shift 2 ;;
    --batch-size) BATCH_SIZE=${2:?}; shift 2 ;;
    --num-workers) NUM_WORKERS=${2:?}; shift 2 ;;
    --seed) SEED=${2:?}; shift 2 ;;
    --sampling) SAMPLING=${2:?}; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-tests) RUN_TESTS=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

EXPERIMENTS=(h_distill_lstm_zloss activity_dual activity_z_h_soft)
NAMES=(baseline_e2 activity_only active_z_h_soft)
PREFIX=m3ed_activity

case "$STAGE" in
  smoke) MAX_STEPS=100; WARMUP_STEPS=10; LOG_EVERY=1; CHECKPOINT_EVERY=100 ;;
  full) MAX_STEPS=100000; WARMUP_STEPS=1000; LOG_EVERY=20; CHECKPOINT_EVERY=1000 ;;
  *) fail "--stage must be smoke or full" ;;
esac
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
[ "${#GPUS[@]}" -eq 3 ] || fail "--gpus must specify three distinct numeric GPU IDs"
for value in "$BATCH_SIZE" "$NUM_WORKERS" "$SEED" "${GPUS[@]}"; do
  case "$value" in *[!0-9]*|'') fail "numeric arguments must be non-negative integers" ;; esac
done
[ "$BATCH_SIZE" -gt 0 ] || fail "--batch-size must be positive"
SAMPLING_ARGS=(training.sampling.mode=random "training.sampling.random_batch_size=$BATCH_SIZE")
case "$SAMPLING" in
  random) ;;
  hybrid)
    [ $((BATCH_SIZE % 2)) -eq 0 ] || fail "hybrid requires an even --batch-size (random + stream)"
    HALF_BATCH=$((BATCH_SIZE / 2))
    SAMPLING_ARGS=(training.sampling.mode=mixed
      "training.sampling.random_batch_size=$HALF_BATCH"
      "training.sampling.stream_batch_size=$HALF_BATCH"
      training.sampling.random_weight=1.0 training.sampling.stream_weight=1.0
      training.sampling.shuffle_sequences=true)
    PREFIX=m3ed_activity_hybrid ;;
  *) fail "--sampling must be random or hybrid" ;;
esac
[ "${GPUS[0]}" != "${GPUS[1]}" ] && [ "${GPUS[0]}" != "${GPUS[2]}" ] && \
  [ "${GPUS[1]}" != "${GPUS[2]}" ] || fail "GPU IDs must be distinct"
[ -d "$ROOT" ] || fail "prepared M3ED root not found: $ROOT"
[ -d "$EVENT_CACHE_DIR" ] || fail "event cache not found: $EVENT_CACHE_DIR"
[ -d "$TEACHER_CACHE_DIR" ] || fail "teacher cache not found: $TEACHER_CACHE_DIR"
[ -f "$CHECKPOINT" ] || fail "DINO checkpoint not found: $CHECKPOINT"
# Resolve before changing working directory, including paths supplied relatively.
ROOT=$(cd "$ROOT" && pwd -P)
EVENT_CACHE_DIR=$(cd "$EVENT_CACHE_DIR" && pwd -P)
TEACHER_CACHE_DIR=$(cd "$TEACHER_CACHE_DIR" && pwd -P)
CHECKPOINT=$(cd "$(dirname "$CHECKPOINT")" && pwd -P)/$(basename "$CHECKPOINT")
EVENT_STATISTICS=${EVENT_STATISTICS:-"$ROOT/event_statistics.json"}
[ -f "$EVENT_STATISTICS" ] || fail "normalization statistics not found: $EVENT_STATISTICS"
EVENT_STATISTICS=$(cd "$(dirname "$EVENT_STATISTICS")" && pwd -P)/$(basename "$EVENT_STATISTICS")
command -v python >/dev/null 2>&1 || fail "activate the EventState env"
NORMALIZATION=$(python - "$EVENT_STATISTICS" <<'PY_STATS'
import json, math, sys
with open(sys.argv[1]) as handle:
    stats = json.load(handle)
train = {"car_urban_day_city_hall", "car_urban_day_horse",
         "car_urban_day_penno_big_loop", "car_urban_day_penno_small_loop"}
sequences = stats.get("sequences", [])
if set(sequences) != train or len(sequences) != 4:
    raise SystemExit("Normalization must be computed from the established four training sequences only; check event_statistics.json provenance")
if stats.get("representation") != "gep_rgb":
    raise SystemExit("Normalization statistics must be for GEP RGB")
mean, std = stats["normalize_mean"], stats["normalize_std"]
if len(mean) != 3 or len(std) != 3 or not all(math.isfinite(x) for x in mean + std) or min(std) <= 0:
    raise SystemExit("Invalid GEP normalization coefficients")
print(json.dumps(mean, separators=(",", ":")), json.dumps(std, separators=(",", ":")))
PY_STATS
) || fail "normalization validation failed (training not started)"
read -r NORMALIZE_MEAN NORMALIZE_STD <<< "$NORMALIZATION"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
if [ -z "$OUTPUT_ROOT" ]; then
  OUTPUT_ROOT="$PROJECT_ROOT/outputs/${PREFIX}_${STAGE}_$(date +%Y%m%d_%H%M%S)"
elif [[ "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="$PWD/$OUTPUT_ROOT"
fi
[ ! -e "$OUTPUT_ROOT" ] || fail "output path exists; use a new directory: $OUTPUT_ROOT"
cd "$PROJECT_ROOT"

printf '[activity] suite: %s; outputs: %s\n' "$SUITE" "$OUTPUT_ROOT"
PIDS=()
terminate_children() {
  local pid
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap 'terminate_children; exit 130' INT
trap 'terminate_children; exit 143' TERM

if [ "$DRY_RUN" -eq 0 ]; then
  command -v python >/dev/null 2>&1 || fail "activate the EventState training environment"
  mkdir -p "$OUTPUT_ROOT/logs"
  printf 'stage=%s\ngpus=%s\nseed=%s\nbatch_size=%s\nmax_steps=%s\n' \
    "$STAGE" "$GPU_LIST" "$SEED" "$BATCH_SIZE" "$MAX_STEPS" > "$OUTPUT_ROOT/launch.txt"
  cp "$EVENT_STATISTICS" "$OUTPUT_ROOT/event_statistics.json"
  git rev-parse HEAD >> "$OUTPUT_ROOT/launch.txt"
  printf 'suite=%s\n' "$SUITE" >> "$OUTPUT_ROOT/launch.txt"
  printf 'sampling=%s\n' "$SAMPLING" >> "$OUTPUT_ROOT/launch.txt"
  git diff --stat >> "$OUTPUT_ROOT/launch.txt"
  if [ "$RUN_TESTS" -eq 1 ]; then
    printf '[activity] Running preflight tests; log: %s/logs/preflight.log\n' "$OUTPUT_ROOT"
    if ! (
      export CUDA_VISIBLE_DEVICES=""
      python -c 'import cv2, torch; print("OpenCV", cv2.__version__, "torch", torch.__version__)' || exit 1
      python -m pytest -q tests/test_scale_event.py tests/test_activity_losses.py \
        tests/test_activity_h_relaxation.py tests/test_m3ed.py tests/test_m3ed_activity.py \
        tests/test_transforms.py tests/test_training_runtime.py tests/test_stream_sampling.py
    ) > "$OUTPUT_ROOT/logs/preflight.log" 2>&1; then
      fail "preflight failed; see $OUTPUT_ROOT/logs/preflight.log (training not started)"
    fi
  fi
fi

for index in 0 1 2; do
  run_dir="$OUTPUT_ROOT/${NAMES[$index]}"
  command_args=(python train.py dataset=m3ed_half_dagr model=lstm
    "experiment=${EXPERIMENTS[$index]}" "seed=$SEED"
    "dataset.root=$ROOT" "dataset.prepared_root=$ROOT"
    +dataset.pretraining_protocol=m3ed_activity dataset.sequence_length=16
    "dataset.representation.normalize_mean=$NORMALIZE_MEAN"
    "dataset.representation.normalize_std=$NORMALIZE_STD"
    dataset.augmentation.enabled=false teacher.cache_features=true
    "${SAMPLING_ARGS[@]}"
    training.event_dropout.enabled=false
    "teacher.cache_dir=$TEACHER_CACHE_DIR" "teacher.checkpoint=$CHECKPOINT"
    "training.batch_size=$BATCH_SIZE" "training.num_workers=$NUM_WORKERS"
    "training.max_steps=$MAX_STEPS" "scheduler.warmup_steps=$WARMUP_STEPS"
    "training.log_every=$LOG_EVERY" "training.checkpoint_every=$CHECKPOINT_EVERY"
    training.gradient_accumulation=1 training.validation_enabled=true
    training.resume=null "hydra.run.dir=$run_dir")
  printf '[activity] GPU %s: %s\n' "${GPUS[$index]}" "${NAMES[$index]}"
  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPUS[$index]}"
    printf '%q ' "${command_args[@]}"
    printf '\n'
  else
    CUDA_VISIBLE_DEVICES="${GPUS[$index]}" "${command_args[@]}" \
      > "$OUTPUT_ROOT/logs/${NAMES[$index]}.log" 2>&1 &
    PIDS+=("$!")
  fi
done
if [ "$DRY_RUN" -eq 1 ]; then exit 0; fi

STATUS=0
for index in 0 1 2; do
  if wait "${PIDS[$index]}"; then
    printf '[activity] complete: %s\n' "${NAMES[$index]}"
  else
    printf '[activity] failed: %s; inspect its log\n' "${NAMES[$index]}" >&2
    STATUS=1
  fi
done
trap - INT TERM
printf '[activity] outputs: %s\n' "$OUTPUT_ROOT"
exit "$STATUS"
