#!/usr/bin/env bash
# Six frozen backbones, three GPUs, one job per GPU. Run on the training host.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "$SCRIPT_DIR/.."
RANDOM_ROOT=outputs/m3ed_activity_full_20260927_192110
HYBRID_ROOT=outputs/m3ed_activity_hybrid_full_20260929_131030
DATA=/home/iASL/Arata_repo/dataset/m3ed_cache
TEACHER=/home/iASL/Arata_repo/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
CACHE_ROOT="$DATA/semantic_features/activity_random_hybrid_20260930"
OUTPUT_ROOT=outputs/m3ed_semantic_activity_random_hybrid_20260930
DRY_RUN=0
RESUME_CACHE=0
while (($#)); do
  case "$1" in
    --random-root) RANDOM_ROOT=${2:?}; shift 2 ;;
    --hybrid-root) HYBRID_ROOT=${2:?}; shift 2 ;;
    --cache-root) CACHE_ROOT=${2:?}; shift 2 ;;
    --output-root) OUTPUT_ROOT=${2:?}; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --resume-cache) RESUME_CACHE=1; shift ;;
    -h|--help)
      echo 'Usage: bash tools/run_m3ed_semantic_activity_comparison.sh [--random-root PATH] [--hybrid-root PATH] [--cache-root PATH] [--output-root PATH] [--resume-cache] [--dry-run]'
      echo '--resume-cache reuses partial feature files and appends logs; requires no existing head runs.'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
NAMES=(baseline_e2 activity_only active_z_h_soft)
run() {
  if ((DRY_RUN)); then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
if ((!DRY_RUN)); then
  # Fail before starting any GPU job if a final checkpoint is missing.
  for root in "$RANDOM_ROOT" "$HYBRID_ROOT"; do
    for name in "${NAMES[@]}"; do
      ckpt="$root/$name/checkpoints/step_00100000.pt"
      [[ -f "$ckpt" ]] || { echo "Missing final checkpoint: $ckpt" >&2; exit 1; }
    done
  done
  for path in "$DATA/half_dagr" "$DATA/dinov3_vits16_640x352" "$DATA/m3ed_downstream"; do
    [[ -d "$path" ]] || { echo "Missing data: $path" >&2; exit 1; }
  done
  [[ -f "$TEACHER" ]] || { echo "Missing teacher: $TEACHER" >&2; exit 1; }
  if ((RESUME_CACHE)); then
    for mode in random hybrid; do
      [[ ! -e "$OUTPUT_ROOT/$mode" ]] || {
        echo 'Head output exists. Use a new --output-root with --resume-cache to preserve it.' >&2; exit 1;
      }
    done
  else
    [[ ! -e "$OUTPUT_ROOT" && ! -e "$CACHE_ROOT" ]] || {
      echo 'Existing paths: use --resume-cache for an interrupted extraction, or new paths.' >&2; exit 1;
    }
  fi
  mkdir -p "$OUTPUT_ROOT/logs" "$CACHE_ROOT"
fi
worker() {
  local mode=$1 root=$2 gpu=$3 name=${NAMES[$3]}
  local cache="$CACHE_ROOT/$mode/$name"
  local checkpoint="$root/$name/checkpoints/step_00100000.pt"
  local role feature
  for role in train validation; do
    run env CUDA_VISIBLE_DEVICES="$gpu" python tools/cache_m3ed_semantic_features.py \
      --checkpoint "$checkpoint" --prepared-root "$DATA/half_dagr" \
      --teacher-cache-dir "$DATA/dinov3_vits16_640x352" \
      --teacher-checkpoint "$TEACHER" --target-cache-dir "$DATA/m3ed_downstream" \
      --output-dir "$cache" --role "$role" --features z h --device cuda || return $?
  done
  for feature in h z concat; do
    run env CUDA_VISIBLE_DEVICES="$gpu" python tools/train_m3ed_semantic.py \
      --feature-cache-dir "$cache" --target-cache-dir "$DATA/m3ed_downstream" \
      --output-dir "$OUTPUT_ROOT/$mode/$name/$feature/seed_0" \
      --feature "$feature" --head-type linear --loss ce --epochs 50 \
      --validate-every 5 --batch-size 8 --num-workers 4 --precision fp16 \
      --seed 0 --device cuda || return $?
  done
}
pids=()
terminate() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap 'terminate; exit 130' INT
trap 'terminate; exit 143' TERM
for mode in random hybrid; do
  root=$RANDOM_ROOT
  [[ "$mode" == random ]] || root=$HYBRID_ROOT
  pids=()
  for gpu in 0 1 2; do
    name=${NAMES[$gpu]}
    echo "[m3ed-semantic] $mode GPU $gpu: $name (h/z/concat)"
    if ((DRY_RUN)); then
      worker "$mode" "$root" "$gpu"
    else
      worker "$mode" "$root" "$gpu" >>"$OUTPUT_ROOT/logs/${mode}_${name}.log" 2>&1 &
      pids+=("$!")
    fi
  done
  failed=0
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "[m3ed-semantic] complete: $mode ${NAMES[$index]}"
    else
      echo "[m3ed-semantic] failed: $mode ${NAMES[$index]}; inspect logs" >&2
      failed=1
    fi
  done
  ((failed == 0)) || exit 1
  pids=()
done
echo "[m3ed-semantic] outputs: $OUTPUT_ROOT"
