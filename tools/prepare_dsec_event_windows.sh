#!/usr/bin/env bash

# Build resumable GEP caches from causal tail fractions of each RGB interval.

set -euo pipefail

ROOT=""
OUTPUT_ROOT=""
SPLIT="all"
FRACTIONS="0.5,0.25,0.125"
SEQUENCES=()
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage:
  bash tools/prepare_dsec_event_windows.sh \
    --root PATH \
    --output-root PATH \
    [--split train|test|all] \
    [--fractions 0.5,0.25,0.125] \
    [--sequences NAME ...] [--overwrite]

Each cache uses only the final fraction of every RGB interval and is written to
<output-root>/gep_rgb_tail_<fraction>. Existing valid frames are preserved, so
rerunning the same command resumes an interrupted preparation.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) ROOT=${2:?}; shift 2 ;;
    --output-root) OUTPUT_ROOT=${2:?}; shift 2 ;;
    --split) SPLIT=${2:?}; shift 2 ;;
    --fractions) FRACTIONS=${2:?}; shift 2 ;;
    --sequences)
      shift
      while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
        SEQUENCES+=("$1")
        shift
      done
      ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[ -n "$ROOT" ] || fail "--root is required"
[ -n "$OUTPUT_ROOT" ] || fail "--output-root is required"
[ -d "$ROOT" ] || fail "DSEC root not found: $ROOT"
case "$SPLIT" in train|test|all) ;; *) fail "--split must be train, test, or all" ;; esac
[ -n "$FRACTIONS" ] || fail "--fractions must not be empty"
if [ "$SPLIT" = "all" ] && [ "${#SEQUENCES[@]}" -gt 0 ]; then
  fail "--sequences cannot be combined with --split all"
fi
command -v python >/dev/null 2>&1 || fail "python not found; activate the EventState env first"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
mkdir -p "$OUTPUT_ROOT"
cd "$PROJECT_ROOT"

IFS=',' read -r -a FRACTION_VALUES <<< "$FRACTIONS"
for FRACTION in "${FRACTION_VALUES[@]}"; do
  [ -n "$FRACTION" ] || fail "--fractions contains an empty value"
  case "$FRACTION" in
    *[!0-9.eE+-]*) fail "invalid fraction: $FRACTION" ;;
  esac
  LABEL=${FRACTION//./p}
  LABEL=${LABEL//+/plus}
  LABEL=${LABEL//-/minus}
  CACHE_DIR="$OUTPUT_ROOT/gep_rgb_tail_$LABEL"
  ARGS=(
    --root "$ROOT"
    --split "$SPLIT"
    --event-cache-dir "$CACHE_DIR"
    --representation gep_rgb
    --event-window-fraction "$FRACTION"
  )
  if [ "${#SEQUENCES[@]}" -gt 0 ]; then
    ARGS+=(--sequences "${SEQUENCES[@]}")
  fi
  if [ "$OVERWRITE" -eq 1 ]; then
    ARGS+=(--overwrite)
  fi
  printf '[event-window] fraction=%s cache=%s\n' "$FRACTION" "$CACHE_DIR"
  python tools/prepare_dsec.py "${ARGS[@]}"
done

printf '[event-window] complete: %s\n' "$OUTPUT_ROOT"
