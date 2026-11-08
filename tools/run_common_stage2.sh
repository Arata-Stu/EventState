#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
stage=${1:?semantic-check|semantic|flow-check|flow-cache|flow-train|flow-export}
data=${DATA_ROOT:-/home/iASL/Arata_repo/dataset}
run=${STAGE1_ROOT:-outputs/stage1_joined_ddp_full_20261105}
out=${OUTPUT_ROOT:-outputs/stage2_joined_e1}
cache=${CACHE_ROOT:-$data/downstream_scratch/stage2_joined_e1}
case "$stage" in
semantic|semantic-check)
  opts=()
  [[ "$stage" != semantic-check ]] || opts+=(--preflight)
  [[ -z "${EXTRA_TARGET_ROOT:-}" ]] || opts+=(--extra-target-root "$EXTRA_TARGET_ROOT")
  python tools/run_common_m3ed_semantic.py --checkpoint "${M3ED_CHECKPOINT:-$run/m3ed_E1_w3/checkpoints/step_00033333.pt}" \
    --prepared-root "${PREPARED_ROOT:-$data/m3ed_cache/half_dagr}" --target-root "${TARGET_ROOT:-$data/m3ed_cache/m3ed_downstream}" \
    --cache "$cache/m3ed_semantic" --output "$out/m3ed_semantic" "${opts[@]}"
  ;;
flow-check|flow-cache|flow-train|flow-export)
  feature=${FEATURE:-h}; window=${DSEC_WINDOW_MS:-rgb-interval}
  ckpt=${DSEC_CHECKPOINT:-$run/dsec_E1_w3/checkpoints/step_00033333.pt}
  fc="$cache/flow_${feature}_${window}"; fd="$out/flow_${feature}_${window}"
  case "$stage" in
  flow-check|flow-cache)
    op=cache; [[ "$stage" != flow-check ]] || op=preflight
    python tools/run_common_dsec_flow.py "$op" --dsec "$data/DSEC" --checkpoint "$ckpt" \
      --flow-root "${FLOW_GT_ROOT:-$data/DSEC/train_optical_flow}" --feature "$feature" --window-ms "$window" --cache "$fc" ;;
  flow-train) python tools/run_common_dsec_flow.py train --cache "$fc" --output "$fd" ;;
  flow-export)
    mkdir -p "$out"
    ts="$out/test_forward_optical_flow_timestamps.zip"
    if [[ ! -f "$ts" ]]; then
      curl -fLsS https://download.ifi.uzh.ch/rpg/DSEC/test_forward_optical_flow_timestamps.zip -o "$ts.partial"
      mv "$ts.partial" "$ts"
    fi
    python tools/run_common_dsec_flow.py export --dsec "$data/DSEC" --checkpoint "$ckpt" --head "$fd/best.pt" \
      --test-timestamps "$ts" --output "$fd/test" ;;
  esac ;;
*) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
