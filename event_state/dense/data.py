"""Frozen maps plus official sparse regression targets, with provenance checks."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .protocol import audit_pretraining, digest, load_protocol, sequences_for, sha256_file
from .targets import read_target


class DenseFeatureDataset(Dataset):
    def __init__(self, root, feature, *, require_labels=True, minimum=0.1, maximum=80.0):
        self.root = Path(root)
        if not (self.root / "_SUCCESS").is_file():
            raise ValueError(f"Dense feature extraction incomplete: {self.root}")
        self.meta = json.loads((self.root / "metadata.json").read_text())
        self.protocol = load_protocol(self.meta["protocol"]["name"])
        if self.meta["protocol"] != self.protocol or self.meta["protocol_sha256"] != digest(self.protocol):
            raise ValueError("Frozen cache protocol differs from the published protocol")
        audit = self.meta["normalization_audit"]
        if audit.get("protocol_sha256") != digest(self.protocol) or audit.get("status") != "declared_inputs_disjoint":
            raise ValueError("Frozen cache has no successful pretraining leakage audit")
        audit_pretraining(dict(dataset=dict(name=self.protocol["dataset"],
                         train_sequences=audit["pretrain_train"], val_sequences=audit["pretrain_validation_used"]),
                         training=dict(validation_enabled=bool(audit["pretrain_validation_used"]))),
                          self.protocol, statistics_sequences=audit["statistics_sequences"])
        names = set(sequences_for(self.protocol, self.meta["role"], labels=require_labels))
        if feature not in {"z", "h", "concat"} or not 0 <= minimum < maximum:
            raise ValueError("Invalid feature or regression range")
        self.minimum, self.maximum, self.feature = minimum, maximum, feature
        self.records = self.meta["records"]
        identities = [(r["sequence"], r["timestamp"]) for r in self.records]
        if not identities or len(set(identities)) != len(identities):
            raise ValueError("Empty or duplicated regression samples")
        if {r["sequence"] for r in self.records} != names:
            raise ValueError("Cached sample membership differs from the complete published split")
        for path, expected in self.meta["target_source_identity"].items():
            if sha256_file(path) != expected:
                raise ValueError(f"Target provenance changed since indexing: {path}")
        self.require_labels = require_labels
        self.size = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        payload = torch.load(record["feature_path"], map_location="cpu", weights_only=True)
        for key, expected in (("sequence", record["sequence"]), ("timestamp", record["timestamp"]),
                              ("checkpoint_sha256", self.meta["checkpoint_sha256"]),
                              ("protocol_sha256", self.meta["protocol_sha256"]),
                              ("state_policy", self.meta["state_policy"])):
            if payload.get(key) != expected:
                raise ValueError(f"Frozen feature identity mismatch: {record['feature_path']}: {key}")
        maps = payload["features"]
        x = torch.cat([maps["z"], maps["h"]], 0) if self.feature == "concat" else maps[self.feature]
        if x.ndim != 3 or not torch.isfinite(x).all():
            raise ValueError("Invalid dense frozen map")
        size = tuple(payload["size"])
        if tuple(x.shape[-2:]) != (size[0] // 16, size[1] // 16):
            raise ValueError("Frozen map/target pixel geometry mismatch")
        if self.size is not None and size != self.size:
            raise ValueError("Mixed target geometries")
        self.size = size
        if self.require_labels:
            if "target_sha256" in record and sha256_file(record["target_path"]) != record["target_sha256"]:
                raise ValueError("DSEC target PNG changed since indexing")
            target, valid = read_target(record, self.meta["task"])
            if target.shape != size:
                raise ValueError(f"Target shape {target.shape} != frozen geometry {size}")
            valid &= (target > self.minimum) & (target <= self.maximum)
        else:
            target = np.zeros(size, dtype=np.float32)
            valid = np.zeros(size, dtype=bool)
        return dict(feature=x.float(), target=torch.from_numpy(target), valid=torch.from_numpy(valid),
                    sequence=record["sequence"], timestamp=record["timestamp"],
                    output_index=record["output_index"], low_activity=payload["low_activity"],
                    low_duration_us=payload["low_duration_us"], event_rate_hz=payload["event_rate_hz"],
                    activity_fraction=payload["activity_fraction"])
