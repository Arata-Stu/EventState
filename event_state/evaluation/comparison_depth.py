"""Evaluated common Depth checkpoints and exact-time public validation targets.

Heavy inference dependencies are imported only when a backend is constructed.
The target index and condition lookup stay usable by read-only planners.
"""
from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path

from event_state.dense.protocol import digest, load_protocol, sha256_file


FRAME_SIZES = {"dsec": (480, 640), "m3ed": (352, 640)}
MIN_DEPTH_M = 0.1
MAX_DEPTH_M = 80.0


def _dataset(value):
    if value not in FRAME_SIZES:
        raise ValueError("Depth comparison dataset must be dsec or m3ed")
    return value


def _resolve(root, path):
    value = Path(path).expanduser()
    return (value if value.is_absolute() else root / value).resolve()


class DepthBackend:
    """One frozen, already evaluated four-stage Depth model; never train a head."""

    def __init__(self, dataset, condition, project_root, depth_run,
                 teacher_checkpoint=None, device="cuda"):
        from tools import run_common_depth
        from event_state.dense.benchmark_inference import DepthPredictor

        self.dataset = _dataset(dataset)
        self.condition = condition
        self.project_root = Path(project_root).expanduser().resolve()
        self.depth_run = _resolve(self.project_root, depth_run)
        matches = [spec for spec in run_common_depth.CONDITIONS if spec[0] == condition]
        if len(matches) != 1:
            raise ValueError(f"Unknown common Depth condition: {condition}")
        specification = matches[0]
        self.label = specification[1]
        # Reuse the authoritative condition/step lookup while permitting a
        # caller's checkout root, rather than the module's imported checkout.
        relative = run_common_depth.checkpoint(dataset, specification).relative_to(run_common_depth.ROOT)
        backbone = self.project_root / relative
        teacher = _resolve(self.project_root, teacher_checkpoint) if teacher_checkpoint else None
        name = f"{dataset}_{condition}"
        self.predictor = DepthPredictor(self.depth_run, name, "stages", device,
                                        teacher_weights=teacher, backbone=backbone)
        self.cfg = self.predictor.cfg
        prepared = self.cfg.dataset.get("event_cache_dir") or self.cfg.dataset.get("prepared_root")
        if not prepared:
            raise ValueError("Depth checkpoint has no prepared event source")
        self.prepared_root = _resolve(self.project_root, prepared)
        if not self.prepared_root.is_dir():
            raise FileNotFoundError(self.prepared_root)
        self.frame_size = FRAME_SIZES[dataset]
        self.identity = dict(self.predictor.identity, condition=condition, condition_label=self.label,
                             backbone_path=str(backbone),
                             head_path=str(self.depth_run / "heads" / name / "stages" / "best.pt"),
                             prepared_root=str(self.prepared_root), frame_size=list(self.frame_size),
                             state_context="continuous from sequence start",
                             pretraining_audit=self.predictor.pretraining_audit)

    def reset(self):
        self.predictor.state = None

    def step(self, frame_dict, predict=True):
        from event_state.dense.source import input_geometry

        events = input_geometry(frame_dict["events"].float(), self.dataset)
        if tuple(events.shape) != (3, *self.frame_size):
            raise ValueError(f"Unexpected Depth event geometry: {tuple(events.shape)}")
        prediction = self.predictor.step(events, predict=bool(predict))
        if not predict:
            return None
        if prediction is None:
            raise RuntimeError("Depth predictor returned no requested frame")
        return prediction.detach().cpu().numpy()


class DepthTargets:
    """Public-val GT lookup without interpolation or borrowing nearby labels."""

    def __init__(self, run, dataset, sequence):
        self.dataset = _dataset(dataset)
        self.sequence = sequence
        self.frame_size = FRAME_SIZES[dataset]
        self.index_path = Path(run).expanduser().resolve() / "targets" / f"{dataset}_val.json"
        index = json.loads(self.index_path.read_text())
        protocol = load_protocol(dataset + "_f3")
        if (index.get("protocol") != protocol or index.get("protocol_sha256") != digest(protocol)
                or index.get("dataset") != dataset or index.get("task") != "depth"
                or index.get("role") != "val" or index.get("labeled") is not True):
            raise ValueError("Depth comparison requires the matching labeled public-val depth index")
        allowed = set(protocol["splits"]["val"])
        if sequence not in allowed:
            raise ValueError(f"Not a public Depth validation sequence: {sequence}")
        records = index.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("Empty Depth target index")
        identities = []
        for record in records:
            stamp = record.get("timestamp")
            if (record.get("sequence") not in allowed or isinstance(stamp, bool)
                    or not isinstance(stamp, Integral) or stamp < 0):
                raise ValueError("Invalid Depth target sequence or timestamp")
            identities.append((record["sequence"], int(stamp)))
        if len(set(identities)) != len(identities) or {s for s, _ in identities} != allowed:
            raise ValueError("Duplicate or incomplete public-val target membership")
        self.records = {int(r["timestamp"]): r for r in records if r["sequence"] == sequence}
        sources = index.get("source_identity")
        if not isinstance(sources, dict):
            raise ValueError("Depth target index lacks source provenance")
        selected_sources = {path: value for path, value in sources.items()
                            if sequence in Path(path).parts}
        if not selected_sources:
            raise ValueError("Depth target index lacks this sequence's source provenance")
        for path, expected in selected_sources.items():
            if sha256_file(path) != expected:
                raise ValueError(f"Depth target provenance changed: {path}")
        self.identity = dict(index_path=str(self.index_path), index_sha256=sha256_file(self.index_path),
                             protocol_sha256=digest(protocol), dataset=dataset, sequence=sequence,
                             role="val", labeled=True, source_identity=selected_sources,
                             minimum_exclusive_m=MIN_DEPTH_M, maximum_inclusive_m=MAX_DEPTH_M,
                             timestamp_policy="exact match; unavailable labels remain unavailable")

    def get(self, timestamp):
        if isinstance(timestamp, bool) or not isinstance(timestamp, Integral):
            raise ValueError("Depth timestamp must be an integer in microseconds")
        record = self.records.get(int(timestamp))
        if record is None:
            return None
        from event_state.dense.targets import read_target
        import numpy as np

        if "target_sha256" in record and sha256_file(record["target_path"]) != record["target_sha256"]:
            raise ValueError(f"Depth target changed since indexing: {record['target_path']}")
        depth, valid = read_target(record, "depth")
        if depth.shape != self.frame_size or valid.shape != self.frame_size:
            raise ValueError("Depth GT geometry differs from the evaluated input coordinates")
        valid = np.asarray(valid, dtype=bool).copy()
        valid &= np.isfinite(depth) & (depth > MIN_DEPTH_M) & (depth <= MAX_DEPTH_M)
        return depth, valid
