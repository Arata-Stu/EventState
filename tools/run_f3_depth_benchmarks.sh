#!/usr/bin/env bash
# Run on the training server; keep hidden-test predictions and GT-free manifests separate.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run=${RUN_ROOT:-outputs/dense_f3_full_20261027_172342}
data=${DATA_ROOT:-/home/iASL/Arata_repo/dataset}
out=${BENCH_ROOT:-outputs/f3_official_benchmarks_20261107}
stage=${1:?choose m3ed-audit, m3ed, dsec-val, dsec-preflight, dsec, or dsec-check}
mkdir -p "$out"
case "$stage" in
m3ed|m3ed-audit)
  : "${M3ED_RAW_ROOT:?set to directory containing raw *_data.h5 and *_depth_gt.h5 for both M3ED val sequences}"
  repo=${F3_REPO:-/home/iASL/Arata_repo/fast-feature-fields}
  args=(--f3-repo "$repo" --raw-root "$M3ED_RAW_ROOT")
  if [[ "$stage" == m3ed-audit ]]; then
    python tools/eval_m3ed_f3_native.py "${args[@]}" --audit-only --output "$out/m3ed_reference"
  else
    cond=${CONDITION:-alpha1}
    feature=${FEATURE:-z}
    python tools/eval_m3ed_f3_native.py "${args[@]}" --reference-support "$out/m3ed_reference/support.json" \
      --run "$run" --condition "$cond" --feature "$feature" --output "$out/m3ed_${cond}_${feature}"
  fi
  ;;
dsec-val)
  window=${DSEC_WINDOW_MS:-rgb-interval}
  python tools/evaluate_dsec_disparity_window.py --run "$run" --dsec "$data/DSEC" --window-ms "$window" \
    --output "$out/dsec_val/window_${window}.json"
  ;;
dsec|dsec-preflight|dsec-check)
  csv="$out/official_csv"
  if [[ ! -d "$csv" ]]; then
    work=$(mktemp -d "$out/csv.XXXXXX")
    curl -fsSL https://download.ifi.uzh.ch/rpg/DSEC/test_disparity_timestamps.zip -o "$work/timestamps.zip"
    (cd "$work" && echo '403f3ad5d28ae2ddc3737c54039b9a737af210236fb084c8cde1de33cb6dd609  timestamps.zip' | sha256sum -c -)
    unzip -q "$work/timestamps.zip" -d "$work/csv"
    mv -T "$work/csv" "$csv"
  fi
  if [[ "$stage" == dsec-check ]]; then
    : "${DSEC_CHECKER:?set to official scripts/check_disparity_submission.py from uzh-rpg/DSEC}"
    scratch=$(mktemp -d)
    trap 'rm -rf -- "$scratch"' EXIT
    unzip -q "$out/dsec/submission.zip" -d "$scratch/png"
    # Official checker uses numpy.loadtxt(delimiter=', '), unsupported since numpy 1.23.
    uv run --isolated --no-project --python 3.10 --with numpy==1.22.4 --with Pillow \
      python "$DSEC_CHECKER" "$scratch/png" "$csv"
  else
    opts=(--window-ms "${DSEC_WINDOW_MS:-rgb-interval}")
    if [[ "$stage" == dsec-preflight ]]; then opts+=(--preflight); fi
    python tools/export_dsec_depth_submission.py --run "$run" --dsec "$data/DSEC" \
      --timestamps "$csv" --output "$out/dsec" "${opts[@]}"
  fi
  ;;
*) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
