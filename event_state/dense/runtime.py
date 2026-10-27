"""Shared training/evaluation provenance and atomic checkpoints."""
import hashlib

import torch

from .evaluation import DenseEvaluator


def validate_pair(train, val):
    for key in ("protocol_sha256", "checkpoint_sha256", "task", "dataset", "geometry",
                "low_activity_threshold", "low_activity_metric", "normalization_audit"):
        if train.meta[key] != val.meta[key]:
            raise ValueError(f"Train/evaluation provenance mismatch: {key}")
    if {r["sequence"] for r in train.records} & {r["sequence"] for r in val.records}:
        raise ValueError("Train/evaluation sequence overlap")


def atomic_save(payload, path):
    # Reuse the tested writer: its suffix keeps the archive basename nonempty.
    # A path like '.dense-XXXX' loses everything when PyTorch strips its extension.
    from event_state.training.checkpoint import atomic_torch_save

    atomic_torch_save(payload, path)


@torch.inference_mode()
def evaluate(model, dataset, device, *, include_frames=True):
    model.eval()
    evaluator = DenseEvaluator(dataset.meta["task"])
    identity, validity, z_digest = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    frames = []
    for index in range(len(dataset)):
        sample = dataset[index]
        prediction = model(sample["feature"][None].to(device), sample["target"].shape)[0].cpu().numpy()
        target, mask = sample["target"].numpy(), sample["valid"].numpy()
        evaluator.update(prediction, target, mask, low_activity=sample["low_activity"],
                         low_duration_us=sample["low_duration_us"])
        identity.update(f"{sample['sequence']}:{sample['timestamp']}\n".encode())
        identity.update(target.tobytes())
        validity.update(mask.tobytes())
        if dataset.feature == "z":
            z_digest.update(sample["feature"].numpy().tobytes())
        if include_frames:
            frames.append({k: sample[k] for k in ("sequence", "timestamp", "low_activity", "low_duration_us",
                                                 "activity_fraction", "event_rate_hz")})
    return dict(metrics=evaluator.compute(), samples_targets_sha256=identity.hexdigest(),
                validity_sha256=validity.hexdigest(), frames=frames,
                z_features_sha256=z_digest.hexdigest() if dataset.feature == "z" else None)
