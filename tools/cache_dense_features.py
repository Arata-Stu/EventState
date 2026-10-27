#!/usr/bin/env python3
"""Extract frozen z/h for published depth protocols with strict split audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from event_state.dense.protocol import (audit_pretraining, digest, load_protocol,
                                        load_statistics_provenance, sequences_for, sha256_file)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--target-index", type=Path, required=True)
    p.add_argument("--statistics-provenance", type=Path, required=True)
    p.add_argument("--event-cache-dir", type=Path)
    p.add_argument("--raw-event-root", type=Path)
    p.add_argument("--teacher-checkpoint", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--state-policy", choices=("continuous", "frame"), default="continuous")
    p.add_argument("--low-activity-threshold", type=float, default=0.1)
    p.add_argument("--low-activity-metric", choices=("event_rate_hz", "activity_fraction"), default="event_rate_hz")
    p.add_argument("--low-event-rate-threshold", type=float,
                   help="Predeclared training-derived rate; default is matching-window train p10 from statistics")
    p.add_argument("--device", default="cuda")
    p.add_argument("--audit-only", action="store_true")
    a = p.parse_args()
    if not 0 <= a.low_activity_threshold <= 1:
        raise ValueError("Activity threshold must be in [0,1]")
    index = json.loads(a.target_index.read_text())
    protocol = load_protocol(index["protocol"]["name"])
    if digest(protocol) != index["protocol_sha256"] or index["protocol"] != protocol:
        raise ValueError("Target index does not match the published protocol")
    sequences = sequences_for(protocol, index["role"])
    from event_state.training import load_checkpoint_config_metadata
    metadata = load_checkpoint_config_metadata(a.checkpoint)
    config = metadata.config
    statistics = load_statistics_provenance(a.statistics_provenance, config)
    audit = audit_pretraining(config, protocol, statistics_sequences=statistics.get("sequences"))
    audit["statistics_provenance_sha256"] = sha256_file(a.statistics_provenance)
    print(json.dumps(audit, indent=2), flush=True)
    if a.audit_only:
        return
    if a.output_dir.exists():
        raise FileExistsError("Use a new cache directory; existing completed/partial artifacts are preserved")
    if (a.event_cache_dir is None) == (a.raw_event_root is None):
        raise ValueError("Choose exactly one event-cache-dir or raw-event-root")
    if protocol["dataset"] == "m3ed" and a.raw_event_root is not None:
        raise ValueError("M3ED dense probes use the existing DAGR prepared cache")
    import torch
    from omegaconf import OmegaConf
    from event_state.training import build_model, load_checkpoint
    from event_state.data.activity import scale_event_activation
    from event_state.data.dsec_feature_source import RawGEPFeatureSource, validate_source
    from event_state.dense.source import input_geometry, prepared_frames
    from event_state.models.tokens import tokens_to_map

    cfg = OmegaConf.create(config)
    if a.low_activity_metric == "event_rate_hz":
        if a.low_event_rate_threshold is None:
            if statistics.get("event_window_us") != OmegaConf.select(cfg, "dataset.event_window_us"):
                raise ValueError("Event-rate threshold statistics use a different accumulation window")
            rate_threshold = statistics.get("event_rate_p10_hz")
            if rate_threshold is None:
                raise ValueError("Train event-rate p10 is missing; recompute train-only statistics")
        else:
            rate_threshold = a.low_event_rate_threshold
        import math
        if not math.isfinite(rate_threshold) or rate_threshold < 0:
            raise ValueError("Event-rate threshold must be finite and nonnegative")
        low_threshold = float(rate_threshold)
    else:
        low_threshold = a.low_activity_threshold
    if cfg.dataset.representation.type != "gep_rgb":
        raise ValueError("Dense activity probes currently require GEP RGB")
    if protocol["dataset"] == "dsec":
        validate_source(cfg, a.event_cache_dir, a.raw_event_root)
    if a.teacher_checkpoint:
        cfg.teacher.checkpoint = str(a.teacher_checkpoint.resolve())
        cfg.model.event_encoder.checkpoint = str(a.teacher_checkpoint.resolve())
    device = torch.device(a.device)
    model = build_model(cfg, device=device)
    load_checkpoint(a.checkpoint, model=model, device=device, restore_rng=False)
    model.eval().requires_grad_(False)
    checkpoint_hash = sha256_file(a.checkpoint)
    identity = dict(format_version=1, protocol_sha256=digest(protocol),
                    checkpoint_sha256=checkpoint_hash, target_index_sha256=sha256_file(a.target_index),
                    state_policy=a.state_policy, low_activity_threshold=low_threshold,
                    low_activity_metric=a.low_activity_metric,
                    normalization_audit=audit, dataset=protocol["dataset"], task=index["task"],
                    role=index["role"], labeled=index["labeled"],
                    geometry="DSEC full rectified 480x640; M3ED half-scale rectified crop rows4:356")
    a.output_dir.mkdir(parents=True)
    (a.output_dir / "run_identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    records, skipped, source_identity = [], [], {}
    with torch.inference_mode():
        for sequence in sequences:
            requested = {r["timestamp"]: r for r in index["records"] if r["sequence"] == sequence}
            if not requested or len(requested) != sum(r["sequence"] == sequence for r in index["records"]):
                raise ValueError("Empty or duplicate target sequence timeline")
            raw = None
            if a.raw_event_root:
                raw = RawGEPFeatureSource(a.raw_event_root, sequence, cfg)
                frames = (raw.load(path) for path in raw.paths)
                source_identity[sequence] = raw.metadata
            else:
                frames = prepared_frames(a.event_cache_dir, sequence, cfg, protocol["dataset"])
                source_path = a.event_cache_dir / sequence / "metadata.json"
                source_identity[sequence] = dict(metadata_sha256=sha256_file(source_path), path=str(source_path.resolve()))
            from event_state.dense.temporal import LowActivityClock
            clock = LowActivityClock()
            state, first_previous = None, None
            destination = a.output_dir / sequence
            destination.mkdir()
            try:
                for frame in frames:
                    stamp, previous = int(frame["timestamp"]), int(frame["previous_timestamp"])
                    if first_previous is None:
                        first_previous = previous
                    events = input_geometry(frame["events"].float(), protocol["dataset"])
                    height, width = events.shape[-2:]
                    if height % 16 or width % 16:
                        raise ValueError("Input geometry is not divisible by patch stride16")
                    activity = scale_event_activation((events * 255).round().byte().permute(1, 2, 0).numpy(),
                                                      height=height, width=width)
                    fraction = float(activity.mean())
                    window = int(frame.get("event_window_us") or stamp - previous)
                    event_rate = int(frame["event_count"]) * 1e6 / window
                    low = (fraction if a.low_activity_metric == "activity_fraction" else event_rate) <= low_threshold
                    low_duration = clock.update(previous, stamp, low)
                    mean = events.new_tensor(list(cfg.dataset.representation.normalize_mean))[:, None, None]
                    std = events.new_tensor(list(cfg.dataset.representation.normalize_std))[:, None, None]
                    if not torch.isfinite(std).all() or (std <= 0).any():
                        raise ValueError("Invalid checkpoint normalization")
                    normalized = ((events - mean) / std).to(device)[None, None]
                    outputs = model(normalized, state=None if a.state_policy == "frame" else state)
                    state = outputs.get("state")
                    if stamp not in requested:
                        continue  # All intervening frames still update the state and activity duration.
                    target = requested.pop(stamp)
                    maps = {key: tokens_to_map(outputs[key][0, 0], (height // 16, width // 16)).cpu().half()
                            for key in ("z", "h")}
                    if any(not torch.isfinite(value).all() for value in maps.values()):
                        raise ValueError("Nonfinite frozen features")
                    payload = dict(sequence=sequence, timestamp=stamp, checkpoint_sha256=checkpoint_hash,
                                   protocol_sha256=digest(protocol), state_policy=a.state_policy,
                                   features=maps, size=[height, width], activity_fraction=fraction,
                                   low_activity=low, low_duration_us=low_duration,
                                   event_count=int(frame["event_count"]),
                                   event_rate_hz=event_rate)
                    path = destination / f"{stamp}.pt"
                    torch.save(payload, path)
                    records.append({**target, "feature_path": str(path.resolve())})
            finally:
                if raw:
                    raw.close()
            # The first RGB frame has no causal accumulation interval. Only it may be omitted.
            for stamp, target in requested.items():
                if first_previous is not None and stamp == first_previous and index["labeled"]:
                    skipped.append({**target, "reason": "first frame has no causal event interval"})
                else:
                    raise ValueError(f"No exact event timestamp for target {sequence}:{stamp}; do not use future features")
            print(f"{sequence}: features complete", flush=True)
    result = {**identity, "protocol": protocol, "records": records, "skipped": skipped,
              "source_identity": source_identity, "target_source_identity": index["source_identity"]}
    (a.output_dir / "metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    (a.output_dir / "_SUCCESS").write_text("complete\n")


if __name__ == "__main__":
    main()
