#!/usr/bin/env bash
# Server-only runbook: clean pretraining, public target indexing, frozen Depth probes.
set -euo pipefail

if [[ $# != 2 || ! "$1" =~ ^(pretrain|targets|depth)$ ]]; then
  echo "Usage: bash tools/run_dense_f3_pipeline.sh {pretrain|targets|depth} RUN_ROOT" >&2
  exit 2
fi
stage="$1"
run_root="$2"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p "$run_root"
run_root="$(cd "$run_root" && pwd)"
data_root="${DATA_ROOT:-/home/iASL/Arata_repo/dataset}"
teacher="${TEACHER:-/home/iASL/Arata_repo/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth}"
cache_root="${CACHE_ROOT:-$run_root/features}"
mkdir -p "$run_root/logs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
read -r -a datasets <<< "${DATASETS:-dsec m3ed}"
read -r -a conditions <<< "${CONDITIONS:-baseline alpha1}"

logged() {
  local name="$1"
  shift
  "$@" 2>&1 | tee -a "$run_root/logs/$name.log"
}

for ds in "${datasets[@]}"; do
  case "$ds" in
    dsec)
      data_dir="$data_root/DSEC"
      event_dir="$data_root/DSEC_cache/events/gep_rgb"
      teacher_dir="$data_root/DSEC_cache/dinov3_vits16"
      stats="$data_root/DSEC_cache/dsec_f3_gep_stats.json"
      target_root="${DSEC_TARGET_ROOT:-$data_root/DSEC/train_disparity}"
      calibration_root="${DSEC_CALIBRATION_ROOT:-$data_root/DSEC/train_calibration}"
      ;;
    m3ed)
      data_dir="$data_root/m3ed_cache/half_dagr"
      event_dir="$data_dir"
      teacher_dir="$data_root/m3ed_cache/dinov3_vits16_640x352"
      stats="$data_root/m3ed_cache/m3ed_f3_gep_stats.json"
      target_root="${M3ED_TARGET_ROOT:-$data_root/m3ed_cache/m3ed_downstream}"
      ;;
    *) echo "Unknown dataset: $ds" >&2; exit 2 ;;
  esac

  if [[ "$stage" == targets ]]; then
    extra=()
    if [[ "$ds" == dsec ]]; then
      extra=(--calibration-root "$calibration_root")
    fi
    for role in train val; do
      logged "${ds}_targets_${role}" python tools/prepare_dense_targets.py \
        --protocol "${ds}_f3" --role "$role" --task depth \
        --target-root "$target_root" "${extra[@]}" \
        --output "$run_root/targets/${ds}_${role}.json"
    done
    continue
  fi

  for condition in "${conditions[@]}"; do
    case "$condition" in
      baseline|alpha1) ;;
      *) echo "Unknown condition: $condition" >&2; exit 2 ;;
    esac
    name="${ds}_${condition}"
    pretrain="$run_root/pretrain/$name"
    checkpoint="$pretrain/checkpoints/step_00100000.pt"
    if [[ "$stage" == pretrain ]]; then
      logged "${name}_pretrain" python tools/run_dense_pretrain.py \
        --protocol "${ds}_f3" --statistics "$stats" \
        --data-root "$data_dir" --event-cache-dir "$event_dir" \
        --teacher-cache-dir "$teacher_dir" --teacher-checkpoint "$teacher" \
        --condition "$condition" --stage full --seed 0 \
        --output-dir "$pretrain" --execute
      test -s "$checkpoint"
      ls -lh "$checkpoint"
      continue
    fi

    test -s "$checkpoint"
    cache="$cache_root/$name"
    for spec in train:continuous val:continuous val:frame; do
      role="${spec%:*}"
      policy="${spec#*:}"
      logged "${name}_cache_${role}_${policy}" python tools/cache_dense_features.py \
        --checkpoint "$checkpoint" --target-index "$run_root/targets/${ds}_${role}.json" \
        --statistics-provenance "$stats" --event-cache-dir "$event_dir" \
        --teacher-checkpoint "$teacher" --state-policy "$policy" \
        --output-dir "$cache/${role}_${policy}"
    done
    for feature in z h concat; do
      head="$run_root/heads/$name/$feature"
      logged "${name}_${feature}_train" python tools/train_dense_probe.py \
        --train-cache "$cache/train_continuous" --val-cache "$cache/val_continuous" \
        --feature "$feature" --epochs 50 --batch-size 4 --seed 0 --output-dir "$head"
      for policy in continuous frame; do
        logged "${name}_${feature}_${policy}" python tools/evaluate_dense_probe.py \
          --checkpoint "$head/best.pt" --cache "$cache/val_${policy}" \
          --output "$head/${policy}.json"
      done
      logged "${name}_${feature}_state_delta" python tools/compare_dense_probes.py \
        --baseline "$head/frame.json" --candidate "$head/continuous.json" \
        --mode state --output "$head/state_delta.json"
    done
  done

  # Model comparison is available once both conditions have been evaluated.
  if [[ "$stage" == depth && -f "$run_root/heads/${ds}_baseline/concat/continuous.json" \
        && -f "$run_root/heads/${ds}_alpha1/concat/continuous.json" ]]; then
    for feature in z h concat; do
      for policy in continuous frame; do
        logged "${ds}_${feature}_${policy}_model_delta" python tools/compare_dense_probes.py \
          --baseline "$run_root/heads/${ds}_baseline/$feature/${policy}.json" \
          --candidate "$run_root/heads/${ds}_alpha1/$feature/${policy}.json" \
          --mode model --output "$run_root/comparison/${ds}_${feature}_${policy}.json"
      done
    done
  fi
done
echo "Completed stage=$stage run_root=$run_root"
