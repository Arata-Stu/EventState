#!/usr/bin/env bash

# Download the DSEC archives needed by EventState.  The script is intentionally
# dependency-light and remains compatible with the Bash 3.2 shipped by macOS.

set -euo pipefail

readonly DSEC_BASE_URL="https://download.ifi.uzh.ch/rpg/DSEC"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SPLIT_MANIFEST="$SCRIPT_DIR/manifests/dsec_det_official_split.yaml"

ROOT=""
DATASET="dsec"
SPLIT=""
COMPONENTS_CSV="events,images,calibration"
DOWNLOAD_ONLY=0
KEEP_ARCHIVES=0
REPAIR_EXISTING=0
ASSUME_YES=0
DRY_RUN=0
LOCK_DIR=""

usage() {
  cat <<'EOF'
Usage:
  bash tools/download_dsec.sh --root PATH --split train|test|all [options]

Options:
  --dataset dsec|dsec-det-extra|both
      Download the original DSEC archives (default), the seven raw sequences
      added by DSEC-Detection, or both.  Extra sequences are kept under
      PATH/dsec_det_extra and are never merged into the main dataset.
  --components LIST
      Comma-separated subset of events,images,calibration (default: all three).
  --download-only
      Download complete ZIP files but do not extract them.
  --keep-archives
      Retain ZIP files after verified extraction.  Interrupted .part files are
      always retained so the next invocation can continue them.
  --repair-existing
      Permit extraction over an incomplete target that was not created by this
      script.  Existing files are not deleted; matching archive paths may be
      replaced by unzip.
  --yes
      Skip the large-download confirmation prompt.
  --dry-run
      Print URLs and destinations without changing the filesystem.
  -h, --help
      Show this help.

Examples:
  bash tools/download_dsec.sh --root /data/DSEC --split train
  bash tools/download_dsec.sh --root /data/DSEC --split all --yes
  bash tools/download_dsec.sh --root /data/DSEC --split all \
    --dataset dsec-det-extra --yes
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

log() {
  printf '[dsec-download] %s\n' "$*"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root)
      [ "$#" -ge 2 ] || fail "--root requires a value"
      ROOT=$2
      shift 2
      ;;
    --dataset)
      [ "$#" -ge 2 ] || fail "--dataset requires a value"
      DATASET=$2
      shift 2
      ;;
    --split)
      [ "$#" -ge 2 ] || fail "--split requires a value"
      SPLIT=$2
      shift 2
      ;;
    --components)
      [ "$#" -ge 2 ] || fail "--components requires a value"
      COMPONENTS_CSV=$2
      shift 2
      ;;
    --download-only)
      DOWNLOAD_ONLY=1
      shift
      ;;
    --keep-archives)
      KEEP_ARCHIVES=1
      shift
      ;;
    --repair-existing)
      REPAIR_EXISTING=1
      shift
      ;;
    --yes)
      ASSUME_YES=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

[ -n "$ROOT" ] || fail "--root is required"
case "$DATASET" in
  dsec|dsec-det-extra|both) ;;
  *) fail "--dataset must be dsec, dsec-det-extra, or both" ;;
esac
case "$SPLIT" in
  train|test|all) ;;
  *) fail "--split must be train, test, or all" ;;
esac
[ -n "$COMPONENTS_CSV" ] || fail "--components cannot be empty"

OLD_IFS=$IFS
IFS=,
read -r -a COMPONENTS <<< "$COMPONENTS_CSV"
IFS=$OLD_IFS
[ "${#COMPONENTS[@]}" -gt 0 ] || fail "no components selected"
for component in "${COMPONENTS[@]}"; do
  case "$component" in
    events|images|calibration) ;;
    *) fail "unsupported component: $component" ;;
  esac
done

if [ "$SPLIT" = "all" ]; then
  SPLITS=(train test)
else
  SPLITS=("$SPLIT")
fi

if [ "$DATASET" = "both" ]; then
  DATASETS=(dsec dsec-det-extra)
else
  DATASETS=("$DATASET")
fi

case "$ROOT" in
  /) fail "refusing to use / as the dataset root" ;;
esac
if [ -n "${HOME:-}" ] && [ "$ROOT" = "$HOME" ]; then
  fail "refusing to use the home directory itself as the dataset root"
fi

print_plan() {
  log "root: $ROOT"
  log "dataset: $DATASET; split: $SPLIT; components: $COMPONENTS_CSV"
  if [ "$DATASET" = "dsec" ] || [ "$DATASET" = "both" ]; then
    log "original DSEC events+images are about 341 GB (train) and 70 GB (test) compressed"
  fi
  if [ "$DATASET" = "dsec-det-extra" ] || [ "$DATASET" = "both" ]; then
    log "DSEC-Detection extra raw data are about 81.9 GB in total"
    log "extra data stay quarantined at $ROOT/dsec_det_extra"
  fi
  log "extraction requires substantial additional free space"
  log "terms and citations: https://dsec.ifi.uzh.ch/dsec-datasets/download/"
}

artifact_url() {
  local dataset=$1
  local split=$2
  local component=$3
  if [ "$dataset" = "dsec" ]; then
    printf '%s/%s_coarse/%s_%s.zip\n' \
      "$DSEC_BASE_URL" "$split" "$split" "$component"
  else
    printf '%s/%s_object_detection_coarse/%s_%s.zip\n' \
      "$DSEC_BASE_URL" "$split" "$split" "$component"
  fi
}

artifact_destination() {
  local dataset=$1
  local split=$2
  local component=$3
  if [ "$dataset" = "dsec" ]; then
    printf '%s/%s_%s\n' "$ROOT" "$split" "$component"
  else
    printf '%s/dsec_det_extra\n' "$ROOT"
  fi
}

print_artifacts() {
  local dataset split component url destination
  for dataset in "${DATASETS[@]}"; do
    for split in "${SPLITS[@]}"; do
      for component in "${COMPONENTS[@]}"; do
        url=$(artifact_url "$dataset" "$split" "$component")
        destination=$(artifact_destination "$dataset" "$split" "$component")
        printf '  %s\n    -> %s\n' "$url" "$destination"
      done
    done
  done
}

print_plan
print_artifacts
if [ "$DRY_RUN" -eq 1 ]; then
  exit 0
fi

[ -f "$SPLIT_MANIFEST" ] || fail "split manifest not found: $SPLIT_MANIFEST"
for command_name in curl unzip zipinfo find grep sed awk wc date hostname cmp sort; do
  require_command "$command_name"
done

mkdir -p "$ROOT"
ROOT=$(cd "$ROOT" && pwd -P)
[ "$ROOT" != "/" ] || fail "refusing to use / as the dataset root"
if [ -n "${HOME:-}" ] && [ -d "$HOME" ]; then
  PHYSICAL_HOME=$(cd "$HOME" && pwd -P)
  [ "$ROOT" != "$PHYSICAL_HOME" ] || \
    fail "refusing to use the home directory itself as the dataset root"
fi
case "$ROOT/" in
  */reference_repo/*)
    fail "dataset root must not be inside the disposable reference_repo directory"
    ;;
esac
readonly STATE_ROOT="$ROOT/.event_state-download"
readonly ARCHIVE_ROOT="$STATE_ROOT/archives"
readonly MARKER_ROOT="$STATE_ROOT/state"
LOCK_DIR="$STATE_ROOT/lock"
mkdir -p "$ARCHIVE_ROOT" "$MARKER_ROOT"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  fail "another download appears active; if it crashed, remove $LOCK_DIR and retry"
fi
printf 'pid=%s\nhost=%s\n' "$$" "$(hostname)" > "$LOCK_DIR/owner"

release_lock() {
  if [ -n "$LOCK_DIR" ] && [ -d "$LOCK_DIR" ]; then
    rm -f "$LOCK_DIR/owner"
    rmdir "$LOCK_DIR" 2>/dev/null || true
  fi
}
trap release_lock EXIT
trap 'exit 130' INT TERM HUP

if [ "$ASSUME_YES" -ne 1 ]; then
  if [ ! -t 0 ]; then
    fail "confirmation requires a terminal; pass --yes for non-interactive use"
  fi
  printf 'Continue with these official downloads? [y/N] '
  read -r answer
  case "$answer" in
    y|Y|yes|YES) ;;
    *) log "cancelled"; exit 0 ;;
  esac
fi

manifest_section() {
  local section=$1
  awk -v wanted="$section" '
    /^[[:alnum:]_-]+:$/ {
      current = $1
      sub(/:$/, "", current)
      next
    }
    current == wanted && /^[[:space:]]+-[[:space:]]+/ { print $2 }
  ' "$SPLIT_MANIFEST"
}

base_sequences() {
  case "$1" in
    train)
      manifest_section train
      ;;
    test)
      manifest_section test | awk '$0 != "thun_02_a"'
      ;;
    *)
      return 1
      ;;
  esac
}

sequence_component_complete() {
  local sequence_dir=$1
  local component=$2
  local first_image
  case "$component" in
    events)
      [ -s "$sequence_dir/events/left/events.h5" ] && \
        [ -s "$sequence_dir/events/left/rectify_map.h5" ]
      ;;
    images)
      [ -s "$sequence_dir/images/timestamps.txt" ] || return 1
      [ -d "$sequence_dir/images/left/rectified" ] || return 1
      first_image=$(find "$sequence_dir/images/left/rectified" \
        -type f -name '*.png' -print -quit)
      [ -n "$first_image" ]
      ;;
    calibration)
      [ -s "$sequence_dir/calibration/cam_to_cam.yaml" ]
      ;;
    *)
      return 1
      ;;
  esac
}

validate_base_layout() {
  local split=$1
  local component=$2
  local destination=$3
  local expected_names actual_names sequence_dir
  [ -d "$destination" ] || return 1
  expected_names=$(base_sequences "$split" | LC_ALL=C sort)
  [ -n "$expected_names" ] || return 1
  actual_names=$(
    for sequence_dir in "$destination"/*; do
      [ -d "$sequence_dir" ] || continue
      printf '%s\n' "${sequence_dir##*/}"
    done | LC_ALL=C sort
  )
  [ "$actual_names" = "$expected_names" ] || return 1
  for sequence_dir in "$destination"/*; do
    [ -d "$sequence_dir" ] || continue
    sequence_component_complete "$sequence_dir" "$component" || return 1
  done
}

extra_sequences() {
  case "$1" in
    train)
      printf '%s\n' \
        zurich_city_16_a \
        zurich_city_17_a \
        zurich_city_18_a \
        zurich_city_19_a \
        zurich_city_20_a \
        zurich_city_21_a
      ;;
    test)
      printf '%s\n' thun_02_a
      ;;
    *)
      return 1
      ;;
  esac
}

validate_extra_layout() {
  local split=$1
  local component=$2
  local destination=$3
  local sequence sequence_dir
  [ -d "$destination/$split" ] || return 1
  while IFS= read -r sequence; do
    sequence_dir="$destination/$split/$sequence"
    sequence_component_complete "$sequence_dir" "$component" || return 1
  done < <(extra_sequences "$split")
}

layout_valid() {
  local dataset=$1
  local split=$2
  local component=$3
  local destination=$4
  if [ "$dataset" = "dsec" ]; then
    validate_base_layout "$split" "$component" "$destination"
  else
    validate_extra_layout "$split" "$component" "$destination"
  fi
}

artifact_has_partial_output() {
  local dataset=$1
  local split=$2
  local component=$3
  local destination=$4
  local first sequence sequence_dir
  if [ "$dataset" = "dsec" ]; then
    [ -d "$destination" ] || return 1
    first=$(find "$destination" -mindepth 1 -maxdepth 1 -print -quit)
    [ -n "$first" ]
    return
  fi
  while IFS= read -r sequence; do
    sequence_dir="$destination/$split/$sequence"
    case "$component" in
      events) [ -e "$sequence_dir/events" ] && return 0 ;;
      images) [ -e "$sequence_dir/images" ] && return 0 ;;
      calibration) [ -e "$sequence_dir/calibration" ] && return 0 ;;
    esac
  done < <(extra_sequences "$split")
  return 1
}

receipt_matches() {
  local receipt=$1
  local url=$2
  [ -f "$receipt" ] && grep -Fqx "url=$url" "$receipt"
}

write_receipt() {
  local path=$1
  local url=$2
  local archive_bytes=$3
  local status=$4
  local temporary="${path}.tmp.$$"
  {
    printf 'schema=event_state_dsec_download_v1\n'
    printf 'url=%s\n' "$url"
    printf 'archive_bytes=%s\n' "$archive_bytes"
    printf 'status=%s\n' "$status"
    printf 'updated_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } > "$temporary"
  mv -f "$temporary" "$path"
}

header_value() {
  local headers=$1
  local name=$2
  awk -v wanted="$name" '
    {
      line = $0
      sub(/\r$/, "", line)
      separator = index(line, ":")
      if (separator == 0) {
        next
      }
      key = tolower(substr(line, 1, separator - 1))
      if (key == wanted) {
        value = substr(line, separator + 1)
        sub(/^[[:space:]]+/, "", value)
      }
    }
    END { print value }
  ' "$headers"
}

fetch_remote_identity() {
  local url=$1
  local output=$2
  local headers="${output}.headers.$$"
  local temporary="${output}.tmp.$$"
  local content_length etag last_modified accept_ranges
  if ! curl \
    --fail \
    --location \
    --silent \
    --show-error \
    --head \
    --retry 10 \
    --retry-delay 5 \
    --connect-timeout 30 \
    --max-time 120 \
    "$url" > "$headers"; then
    rm -f "$headers"
    fail "could not read remote metadata: $url"
  fi
  content_length=$(header_value "$headers" content-length)
  etag=$(header_value "$headers" etag)
  last_modified=$(header_value "$headers" last-modified)
  accept_ranges=$(header_value "$headers" accept-ranges)
  rm -f "$headers"
  [ -n "$content_length" ] || fail "server did not provide Content-Length: $url"
  case "$content_length" in
    *[!0-9]*) fail "invalid remote Content-Length for $url: $content_length" ;;
  esac
  [ "$accept_ranges" = "bytes" ] || \
    fail "server does not currently advertise byte-range resume for $url"
  {
    printf 'url=%s\n' "$url"
    printf 'content_length=%s\n' "$content_length"
    printf 'etag=%s\n' "$etag"
    printf 'last_modified=%s\n' "$last_modified"
  } > "$temporary"
  mv -f "$temporary" "$output"
}

download_archive() {
  local url=$1
  local archive=$2
  local partial="${archive}.part"
  local remote_identity="${archive}.remote"
  local current_identity="${remote_identity}.current.$$"
  local curl_status expected_bytes actual_bytes
  if [ -f "$archive" ]; then
    log "using completed archive: $archive"
    return
  fi
  mkdir -p "$(dirname "$archive")"
  fetch_remote_identity "$url" "$current_identity"
  if [ -f "$partial" ]; then
    if [ ! -f "$remote_identity" ]; then
      rm -f "$current_identity"
      fail "partial file has no remote identity receipt: $partial"
    fi
    if ! cmp -s "$remote_identity" "$current_identity"; then
      rm -f "$current_identity"
      fail "remote archive changed; move the stale .part aside before retrying: $partial"
    fi
  fi
  mv -f "$current_identity" "$remote_identity"
  log "downloading (resume enabled): $url"
  set +e
  curl \
    --fail \
    --location \
    --continue-at - \
    --retry 10 \
    --retry-delay 5 \
    --connect-timeout 30 \
    --output "$partial" \
    "$url"
  curl_status=$?
  set -e
  if [ "$curl_status" -ne 0 ]; then
    # curl reports a range error when a .part file is already exactly complete.
    # A full CRC pass is costly, so it is only used for this recovery path.
    if [ -s "$partial" ] && unzip -tq "$partial" >/dev/null 2>&1; then
      log "the partial file is already a complete ZIP; accepting it"
    else
      fail "download interrupted (curl $curl_status); rerun to continue $partial"
    fi
  fi
  expected_bytes=$(sed -n 's/^content_length=//p' "$remote_identity")
  actual_bytes=$(wc -c < "$partial" | tr -d '[:space:]')
  if [ "$actual_bytes" != "$expected_bytes" ]; then
    fail "download size mismatch for $partial: expected $expected_bytes, got $actual_bytes"
  fi
  mv -f "$partial" "$archive"
}

quarantine_archive() {
  local archive=$1
  local suffix=$2
  local quarantined="${archive}.${suffix}.$(date -u '+%Y%m%dT%H%M%SZ').$$"
  mv -f "$archive" "$quarantined"
  log "quarantined unusable archive: $quarantined"
}

preflight_archive() {
  local archive=$1
  local key=$2
  local member_list="$MARKER_ROOT/${key}.members.$$"
  local unsafe_member unsafe_link
  if ! zipinfo -1 "$archive" > "$member_list"; then
    rm -f "$member_list"
    quarantine_archive "$archive" corrupt
    fail "cannot read ZIP directory: $archive"
  fi
  unsafe_member=$(grep -E '(^/|(^|/)\.\.(/|$)|\\|^[[:alpha:]]:)' \
    "$member_list" | sed -n '1p' || true)
  rm -f "$member_list"
  if [ -n "$unsafe_member" ]; then
    quarantine_archive "$archive" rejected
    fail "unsafe ZIP member: $unsafe_member"
  fi
  unsafe_link=$(zipinfo -l "$archive" | awk '$1 ~ /^l/ {print $NF; exit}' || true)
  if [ -n "$unsafe_link" ]; then
    quarantine_archive "$archive" rejected
    fail "ZIP symlink entries are not accepted: $unsafe_link"
  fi
}

process_artifact() {
  local dataset=$1
  local split=$2
  local component=$3
  local family key url destination archive_dir archive complete in_progress
  local bytes owned=0
  if [ "$dataset" = "dsec" ]; then
    family=base
  else
    family=dsec_det_extra
  fi
  key="${family}_${split}_${component}"
  url=$(artifact_url "$dataset" "$split" "$component")
  destination=$(artifact_destination "$dataset" "$split" "$component")
  archive_dir="$ARCHIVE_ROOT/$family"
  archive="$archive_dir/${split}_${component}.zip"
  complete="$MARKER_ROOT/${key}.complete"
  in_progress="$MARKER_ROOT/${key}.extracting"

  if [ "$DOWNLOAD_ONLY" -eq 1 ]; then
    download_archive "$url" "$archive"
    preflight_archive "$archive" "$key"
    log "download complete: $archive"
    return
  fi

  if [ -f "$complete" ]; then
    receipt_matches "$complete" "$url" || \
      fail "completion receipt does not match the official URL: $complete"
    if layout_valid "$dataset" "$split" "$component" "$destination"; then
      log "already complete: $dataset $split $component"
      return
    fi
    log "completion marker exists but files are incomplete; repairing from archive"
    owned=1
  elif layout_valid "$dataset" "$split" "$component" "$destination"; then
    log "adopting an already complete layout: $destination"
    write_receipt "$complete" "$url" unknown adopted-existing
    return
  fi

  if [ -f "$in_progress" ]; then
    receipt_matches "$in_progress" "$url" || \
      fail "in-progress receipt does not match the official URL: $in_progress"
    owned=1
    log "resuming an interrupted extraction: $dataset $split $component"
  fi
  if [ "$owned" -eq 0 ] && \
    artifact_has_partial_output "$dataset" "$split" "$component" "$destination" && \
    [ "$REPAIR_EXISTING" -ne 1 ]; then
    fail "untracked partial output at $destination; inspect it, then pass --repair-existing"
  fi

  download_archive "$url" "$archive"
  preflight_archive "$archive" "$key"
  bytes=$(wc -c < "$archive" | tr -d '[:space:]')
  write_receipt "$in_progress" "$url" "$bytes" extracting
  [ ! -L "$destination" ] || fail "refusing to extract through a symlink: $destination"
  mkdir -p "$destination"
  log "extracting: $archive"
  if ! unzip -oq "$archive" -d "$destination"; then
    if unzip -tq "$archive" >/dev/null 2>&1; then
      fail "extraction failed although ZIP CRC is valid; check disk space and rerun"
    fi
    quarantine_archive "$archive" corrupt
    fail "ZIP CRC verification failed; rerun to download a fresh archive"
  fi
  if ! layout_valid "$dataset" "$split" "$component" "$destination"; then
    fail "archive extracted, but the expected $dataset layout was not found at $destination"
  fi
  write_receipt "$complete" "$url" "$bytes" complete
  rm -f "$in_progress"
  log "verified: $dataset $split $component"
  if [ "$KEEP_ARCHIVES" -ne 1 ]; then
    rm -f "$archive"
    log "removed verified archive (use --keep-archives to retain it)"
  fi
}

for dataset in "${DATASETS[@]}"; do
  for split in "${SPLITS[@]}"; do
    for component in "${COMPONENTS[@]}"; do
      process_artifact "$dataset" "$split" "$component"
    done
  done
done

log "all requested artifacts are ready"
if [ "$DATASET" = "dsec-det-extra" ] || [ "$DATASET" = "both" ]; then
  log "DSEC-Detection extras remain isolated under $ROOT/dsec_det_extra"
  log "do not expose them to standard inductive pretraining/validation"
fi
