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
def evaluate(model, dataset, device, *, include_frames=True, activity_index=None,
             condition_label=None, include_frame_metrics=False):
    model.eval()
    evaluator = DenseEvaluator(dataset.meta["task"], frame_metrics=include_frame_metrics)
    activity = None
    if activity_index is not None:
        from .activity_evaluation import DenseActivityEvaluator, validate_dense_activity_sources

        if not dataset.require_labels:
            raise ValueError("Activity metrics require public ground-truth labels")
        validate_dense_activity_sources(activity_index, dataset.meta)
        activity = DenseActivityEvaluator(
            activity_index, dataset=dataset.meta["dataset"], task=dataset.meta["task"],
            expected_keys=((record["sequence"], record["timestamp"]) for record in dataset.records),
            condition_label=condition_label, frame_metrics=include_frame_metrics,
        )
    identity, validity, z_digest = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    frames = []
    for index in range(len(dataset)):
        sample = dataset[index]
        prediction = model(sample["feature"][None].to(device), sample["target"].shape)[0].cpu().numpy()
        target, mask = sample["target"].numpy(), sample["valid"].numpy()
        frame_options = {}
        if include_frame_metrics:
            from .depth_frame_metrics import compute_frame_metrics

            frame_options["frame_result"] = compute_frame_metrics(prediction, target, mask)
        evaluator.update(prediction, target, mask, low_activity=sample["low_activity"],
                         low_duration_us=sample["low_duration_us"], **frame_options)
        if activity is not None:
            activity.update(prediction, target, mask, sequence=sample["sequence"],
                            timestamp=sample["timestamp"], event_rate_hz=sample["event_rate_hz"],
                            **frame_options)
        identity.update(f"{sample['sequence']}:{sample['timestamp']}\n".encode())
        identity.update(target.tobytes())
        validity.update(mask.tobytes())
        if dataset.feature == "z":
            z_digest.update(sample["feature"].numpy().tobytes())
        if include_frames:
            frames.append({k: sample[k] for k in ("sequence", "timestamp", "low_activity", "low_duration_us",
                                                 "activity_fraction", "event_rate_hz")})
    result = dict(metrics=evaluator.compute(), samples_targets_sha256=identity.hexdigest(),
                  validity_sha256=validity.hexdigest(), frames=frames,
                  z_features_sha256=z_digest.hexdigest() if dataset.feature == "z" else None)
    if include_frame_metrics:
        from .depth_frame_metrics import F3_FRAME_PROTOCOL

        result["depth_frame_metric_protocol"] = F3_FRAME_PROTOCOL
        result["metrics_frame_mean_f3"] = evaluator.compute_frame_mean()
    if activity is not None:
        result["activity_evaluation"] = activity.compute()
        result["activity_evaluation"].update(samples_targets_sha256=identity.hexdigest(),
                                             validity_sha256=validity.hexdigest())
    return result
