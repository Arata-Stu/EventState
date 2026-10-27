#!/usr/bin/env python3
"""Compute GEP normalization only from the published dense-probe train sequences."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from event_state.dense.protocol import digest, load_protocol, sequences_for, sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True)
    p.add_argument("--event-cache-dir", type=Path)
    p.add_argument("--raw-event-root", type=Path)
    p.add_argument("--window-us", type=int)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    protocol = load_protocol(a.protocol)
    if (a.event_cache_dir is None) == (a.raw_event_root is None):
        raise ValueError("Choose event-cache-dir or raw-event-root")
    if a.raw_event_root is not None and (protocol["dataset"] != "dsec" or not a.window_us or a.window_us <= 0):
        raise ValueError("Raw statistics require DSEC and a positive fixed --window-us")
    if a.event_cache_dir is not None and a.window_us is not None:
        raise ValueError("Fixed-window statistics must not use legacy caches")
    import torch
    from omegaconf import OmegaConf
    from event_state.dense.source import input_geometry, prepared_frames
    cfg = OmegaConf.create(dict(dataset=dict(name=protocol["dataset"], sensor_height=480 if protocol["dataset"] == "dsec" else 360,
                                             sensor_width=640, event_window_us=a.window_us,
                                             representation=dict(type="gep_rgb", channels=3, percentile=90))))
    total, frame_count = 0, 0
    sums, squares = torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    sources, rates = {}, []
    for sequence in sequences_for(protocol, "train"):
        raw = None
        if a.raw_event_root:
            from event_state.data.dsec_feature_source import RawGEPFeatureSource
            raw = RawGEPFeatureSource(a.raw_event_root, sequence, cfg)
            sources[sequence] = raw.metadata
            frames = (raw.load(path) for path in raw.paths)
        else:
            sources[sequence] = sha256_file(a.event_cache_dir / sequence / "metadata.json")
            frames = prepared_frames(a.event_cache_dir, sequence, cfg, protocol["dataset"])
        try:
            for payload in frames:
                x = input_geometry(payload["events"], protocol["dataset"]).double()
                sums += x.sum((1, 2))
                squares += x.square().sum((1, 2))
                total += x.shape[1] * x.shape[2]
                frame_count += 1
                window = int(payload.get("event_window_us") or payload["timestamp"] - payload["previous_timestamp"])
                rates.append(payload["event_count"] * 1e6 / window)
        finally:
            if raw:
                raw.close()
        print(f"statistics: {sequence} complete", flush=True)
    if not total:
        raise ValueError("No training pixels")
    mean = sums / total
    std = (squares / total - mean.square()).clamp_min(0).sqrt()
    if not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid fixed normalization")
    result = dict(protocol_sha256=digest(protocol), sequences=list(sequences_for(protocol, "train")),
                  source="compute_dense_normalization: all train GEP pixels, including background",
                  sources=sources, normalize_mean=mean.tolist(), normalize_std=std.tolist(),
                  pixel_count=total, frame_count=frame_count,
                  event_window_us=a.window_us,
                  event_rate_p10_hz=sorted(rates)[int(.1 * (len(rates) - 1))],
                  event_rate_quantile_method="lower order statistic at floor(0.1*(N-1)), train only",
                  geometry="DSEC full480x640; M3ED central352x640")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
