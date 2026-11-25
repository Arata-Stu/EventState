"""Compare four recurrent placements with the archived single-placement E1 Depth probe."""
import json
import math
from pathlib import Path

from event_state.evaluation.placement_features import CONDITIONS, FOUR_STAGE


RECORDS = Path(__file__).resolve().parents[1] / "docs/evaluation_records"
ACTIVITY_REFERENCE = RECORDS / "20261115_semantic_depth_activity_metrics.json"
DISPARITY_REFERENCE = RECORDS / "20261115_common_depth_summary.json"
HASHES = ("samples_sha256", "samples_targets_sha256", "activity_assignments_sha256",
          "validity_sha256", "validity_support_sha256")
SUPPORT = ("status", "frames", "valid_frames", "valid_pixels", "sequence_frames", "sequences",
           "evaluated_frame_fraction", "event_rate_hz_min", "event_rate_hz_max")


def _mae_pair(baseline, four_stage, unit):
    if any(type(x) not in (float, int) or not math.isfinite(x) or x < 0 for x in (baseline, four_stage)):
        raise ValueError("Depth MAEs must be finite, nonnegative numbers")
    return {"E1": baseline, FOUR_STAGE: four_stage, "delta_" + unit: four_stage - baseline}


def baseline_record(dataset):
    records = json.loads(ACTIVITY_REFERENCE.read_text())["results"]
    reference, = [row for row in records if row["task"] == "depth" and row["condition"] == f"{dataset}_E1"]
    return reference


def attach_comparison(job, result):
    """Return a new result with MAE deltas in native units (four placements minus one)."""
    dataset = job["dataset"]
    if dataset not in ("dsec", "m3ed") or job["name"] != f"{dataset}_{FOUR_STAGE}":
        raise ValueError("Expected a four-placement Depth job on DSEC or M3ED")
    reference = baseline_record(dataset)
    payload = json.loads(Path(job["activity_report"]).read_text())
    activity = payload["activity_evaluation"]
    if (payload["task"] != "depth" or payload["role"] != "val" or payload["protocol"]["dataset"] != dataset
            or payload["feature"] != "stages" or payload["state_policy"] != "continuous"
            or activity["condition_label"] != job["label"] or result["condition"] != job["label"]
            or activity["partition"] != reference["partition"]
            or activity["depth_frame_metric_protocol"] != reference["depth_frame_metric_protocol"]
            or activity["groups"].keys() != reference["groups"].keys()
            or activity["groups"] != result["activity_groups"]):
        raise ValueError(f"Unmatched Depth condition, evaluation split, or metric protocol: {job['name']}")
    hashes = {key: activity[key] for key in HASHES}
    hashes["manifest_sha256"] = activity["activity_manifest"]["manifest_sha256"]
    for key, value in hashes.items():
        if value != reference[key] or (key in ("samples_targets_sha256", "validity_sha256") and payload[key] != value):
            raise ValueError(f"Depth evaluation differs from the archived E1 baseline: {key}")
    groups = {}
    for name, baseline in reference["groups"].items():
        current = activity["groups"][name]
        if any(current[key] != baseline[key] for key in SUPPORT):
            raise ValueError(f"Different evaluation support for {dataset} {name}")
        groups[name] = {
            "pooled_mae_m": _mae_pair(baseline["metrics"]["mae"], current["metrics"]["mae"], "m"),
            "frame_mean_mae_m": _mae_pair(baseline["metrics_frame_mean_f3"]["mae"],
                                          current["metrics_frame_mean_f3"]["mae"], "m"),
        }
    comparison = dict(conditions=CONDITIONS, reference=ACTIVITY_REFERENCE.name,
                      matched_hashes=hashes, groups=groups)
    if dataset == "dsec" and "mae_px" in result:
        baseline = json.loads(DISPARITY_REFERENCE.read_text())["dsec_E1"]["mae_px"]
        comparison["disparity_reference"] = DISPARITY_REFERENCE.name
        comparison["disparity_mae_px"] = _mae_pair(baseline, result["mae_px"], "px")
    return dict(result, comparison_to_E1=comparison)
