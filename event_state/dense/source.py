"""Validated unnormalized GEP frames, processed from the sequence start."""
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf


def prepared_frames(root, sequence, config, dataset):
    from event_state.data.dsec import validate_event_cache_payload, event_representation_metadata
    from event_state.data.cache_metadata import canonical_json_sha256
    from event_state.training.data import build_event_representation

    base = Path(root) / sequence
    meta = json.loads((base / "metadata.json").read_text())
    # Legacy M3ED writes final metadata after preparation, without a _SUCCESS file.
    if dataset == "dsec" and not (base / "_SUCCESS").is_file():
        raise ValueError(f"Incomplete prepared events: {base}")
    if meta.get("sequence_name") != sequence or meta.get("dataset", "").lower() != dataset:
        raise ValueError("Prepared event sequence/dataset mismatch")
    expected_space = "rectified_event" if dataset == "dsec" else "rectified_left_event"
    if meta.get("coordinate_space") != expected_space:
        raise ValueError("Depth targets require matching rectified event coordinates")
    if dataset == "m3ed" and (meta.get("event_size") != [360, 640]
                              or meta.get("downsampling") != "dagr_stateful_signed_event_downsampling"):
        raise ValueError("M3ED probes require the existing half-scale DAGR preparation")
    if meta.get("representation", {}).get("type") != "gep_rgb":
        raise ValueError("Initial dense probes support rectified GEP checkpoints only")
    if meta["representation"].get("percentile") != float(config.dataset.representation.percentile):
        raise ValueError("Checkpoint/cache GEP percentile differs")
    if OmegaConf.select(config, "dataset.event_window_us") is not None:
        raise ValueError("Fixed-window DSEC checkpoint must read raw events, never legacy caches")
    fraction = float(OmegaConf.select(config, "dataset.event_window_fraction", default=1))
    if float(meta.get("event_window_fraction", 1)) != fraction:
        raise ValueError("Checkpoint/cache accumulation windows differ")
    folder = base if dataset == "dsec" else base / "events"
    paths = sorted((p for p in folder.glob("*.pt") if p.stem.isdecimal()), key=lambda p: int(p.stem))
    expected_count = int(meta["frame_count"]) - (1 if dataset == "m3ed" else 0)
    if len(paths) != expected_count or not paths:
        raise ValueError("Prepared event frame count mismatch")
    representation = build_event_representation(config.dataset)
    last, last_index = None, 0
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("sequence_name") != sequence:
            raise ValueError("Event payload belongs to a different sequence")
        if payload.get("frame_index") != last_index + 1:
            raise ValueError("Missing or reordered prepared frame index")
        last_index = payload["frame_index"]
        if dataset == "dsec":
            events, _ = validate_event_cache_payload(
                payload, cache_path=path, frame_index=int(payload["frame_index"]),
                timestamp=int(path.stem), previous_timestamp=int(payload["previous_timestamp"]),
                sequence_name=sequence, split=meta["split"], height=480, width=640,
                representation=event_representation_metadata(representation), rectified=True,
                input_fingerprint_digest=meta["input_fingerprint"]["digest"],
                manifest_digest=canonical_json_sha256(meta), event_window_fraction=fraction,
                cache_dtype=meta.get("cache_dtype", "float32"))
        else:
            i = int(payload["frame_index"])
            stamps = meta["timestamps"]
            if (i <= 0 or i >= len(stamps) or stamps[i] != int(path.stem)
                    or payload["timestamp"] != stamps[i]
                    or payload["previous_timestamp"] != stamps[i - 1]
                    or payload.get("representation") != "gep_rgb"):
                raise ValueError("M3ED event payload timeline/representation mismatch")
            events = payload["events"].float()
        if last is not None and payload["previous_timestamp"] != last:
            raise ValueError("Missing/gapped event intervals would invalidate recurrent state")
        last = payload["timestamp"]
        if not isinstance(payload.get("event_count"), int) or payload["event_count"] < 0:
            raise ValueError("Missing event count provenance")
        if events.shape != (3, 480 if dataset == "dsec" else 360, 640):
            raise ValueError("Unexpected native event geometry")
        if not torch.isfinite(events).all() or ((events < 0) | (events > 1)).any():
            raise ValueError("GEP frames must be finite unnormalized values in [0,1]")
        yield {**payload, "events": events}


def input_geometry(events, dataset):
    # Full DSEC event view. M3ED retains the existing half-scale central 352 rows.
    return events if dataset == "dsec" else events[:, 4:356, :]
