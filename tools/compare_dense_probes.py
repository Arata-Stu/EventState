#!/usr/bin/env python3
"""Compare identical public depth samples, optionally enforcing paired state intervention."""
import argparse
import json
from pathlib import Path


def compare(baseline, candidate, mode="state"):
    if mode not in {"state", "model", "readout"}:
        raise ValueError("Unknown comparison mode")
    keys = ("protocol_sha256", "role", "task", "minimum", "maximum", "scale_alignment",
            "samples_targets_sha256", "validity_sha256", "frames", "geometry", "low_activity_threshold", "low_activity_metric")
    if mode == "readout":
        keys += ("backbone_sha256", "cache_metadata_sha256", "normalization_audit")
        if baseline["feature"] != "z" or candidate["feature"] not in {"h", "concat"}:
            raise ValueError("Readout comparison requires z versus h/concat")
        if baseline["state_policy"] != "continuous" or candidate["state_policy"] != "continuous":
            raise ValueError("Natural readout comparison requires continuous state")
    else:
        keys += ("feature",)
    if mode == "state":
        keys += ("head_sha256", "backbone_sha256")
        if {baseline["state_policy"], candidate["state_policy"]} != {"frame", "continuous"}:
            raise ValueError("State intervention must pair continuous and frame-reset")
    else:
        keys += ("state_policy",)
    for key in keys:
        if key not in baseline or key not in candidate or baseline[key] != candidate[key]:
            raise ValueError(f"Unpaired regression results: {key}")
    if mode == "state" and baseline["feature"] == "z":
        if not baseline.get("z_features_sha256") or baseline["z_features_sha256"] != candidate.get("z_features_sha256"):
            raise ValueError("z changed during the state intervention")
    if baseline["metrics"].keys() != candidate["metrics"].keys():
        raise ValueError("Metric groups differ")
    result = {}
    for group, values in baseline["metrics"].items():
        other = candidate["metrics"][group]
        if any(values[k] != other[k] for k in ("valid_pixels", "frames")):
            raise ValueError("Metric support differs")
        result[group] = {k: None if v is None or other[k] is None else other[k] - v
                         for k, v in values.items() if k not in {"valid_pixels", "frames"}}
    return dict(mode=mode, delta="candidate minus baseline; negative MAE/RMSE/AbsRel is improvement",
                baseline_policy=baseline["state_policy"], candidate_policy=candidate["state_policy"],
                baseline_feature=baseline["feature"], candidate_feature=candidate["feature"],
                interpretation=("Same-backbone readout comparison with separately trained heads; "
                                "not an independently trained no-LSTM control") if mode == "readout" else None,
                groups={g: {"baseline": baseline["metrics"][g], "candidate": candidate["metrics"][g]}
                        for g in result}, differences=result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--mode", choices=("state", "model", "readout"), default="state")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    value = compare(json.loads(a.baseline.read_text()), json.loads(a.candidate.read_text()), a.mode)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    if a.mode == "readout":
        print("Natural continuous readout: candidate - z; negative MAE is improvement")
        print("group                       z MAE  candidate MAE      delta     frames       pixels")
        for group, pair in value["groups"].items():
            b, c = pair["baseline"], pair["candidate"]
            def fmt(v):
                return "N/A" if v is None else f"{v:.5f}"
            print(f"{group:26} {fmt(b['mae']):>9} {fmt(c['mae']):>14} "
                  f"{fmt(value['differences'][group]['mae']):>10} "
                  f"{b['frames']:10,d} {b['valid_pixels']:12,d}")
    else:
        print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
