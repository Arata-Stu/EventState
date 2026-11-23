"""Decode the two evaluated Semantic heads from the raw, continuous block-12 features."""
from pathlib import Path

from event_state.activity_quantiles import canonical_sha256
from event_state.dense.protocol import sha256_file
from event_state.evaluation.comparison_semantic import SemanticTargets, _read, _resolve, _safe_load
from event_state.evaluation.placement_features import CONDITIONS, FOUR_STAGE
from tools.run_placement_semantic import COMPARISON_HASHES, baseline_results
from tools.run_semantic_depth_activity import ROOT, semantic_run


def _receipt(root, dataset):
    return root / "outputs/stage2_placement_semantic_20261118" / f"{dataset}_{FOUR_STAGE}" / "verified.json"


def semantic_head_paths(dataset, project_root):
    """Resolve the selected heads without importing PyTorch or scanning failed attempts."""
    if dataset not in ("dsec", "m3ed"):
        raise ValueError("Expected dsec or m3ed")
    root = Path(project_root).resolve()
    receipt = _receipt(root, dataset)
    attempt = _read(receipt)["attempt"]
    if not attempt.startswith("run-") or Path(attempt).name != attempt:
        raise ValueError(f"Invalid Semantic attempt in {receipt}")
    name = "final/last.pt" if dataset == "dsec" else "h/best.pt"
    return {"E1": root / semantic_run(dataset, "E1").relative_to(ROOT) / name,
            FOUR_STAGE: receipt.parent / attempt / name}


class PlacementSemanticOutputs:
    def __init__(self, dataset, project_root, pair, sequence):
        import torch
        from event_state.segmentation import (DSEC_SEMANTIC_11_CLASSES, EventStateSegmentationHead,
                                             load_dsec_semantic_split, load_m3ed_semantic_split)

        root = Path(project_root).resolve()
        baseline = baseline_results()[dataset]
        role = "test" if dataset == "dsec" else "validation"
        manifest = ROOT / "tools/manifests" / ("dsec_semantic_split.yaml" if dataset == "dsec"
                                             else "m3ed_semantic_common_v1.yaml")
        split = load_dsec_semantic_split(manifest) if dataset == "dsec" else load_m3ed_semantic_split(manifest)
        evaluated = split.sequences(role)
        train = split.official_train if dataset == "dsec" else split.train
        if sequence not in evaluated or pair.dataset != dataset:
            raise ValueError(f"Wrong dataset or Semantic evaluation sequence: {dataset}/{sequence}")
        self.sequence, self.device = sequence, pair.device
        self.frame_size = pair.frame_size
        self.grid_size = tuple(size // 16 for size in pair.input_size)
        self.heads, self.identity, self.use_amp = {}, {}, {}
        paths = semantic_head_paths(dataset, root)
        receipt = _read(_receipt(root, dataset))
        run = paths[FOUR_STAGE].parents[1]
        report = run / ("activity_test.json" if dataset == "dsec" else "h/activity_groups.json")
        placement = _read(report)
        activity = placement["activity_evaluation"]
        if (receipt["dataset"] != dataset or receipt["condition"] != FOUR_STAGE or receipt["role"] != role
                or receipt["backbone_sha256"] != pair.identity[FOUR_STAGE]["backbone_sha256"]
                or receipt["head_sha256"] != activity["head_sha256"]
                or placement["role"] != role or placement["feature"] != "h"
                or any(activity[k] != baseline[k] for k in (*COMPARISON_HASHES, "state_policy_by_sequence"))
                or activity["groups"]["all"]["sequence_frames"] != baseline["groups"]["all"]["sequence_frames"]):
            raise ValueError(f"Different Semantic evaluation samples, readout or model: {report}")
        self.expected_gt_frames = baseline["groups"]["all"]["sequence_frames"][sequence]
        split_sha = sha256_file(manifest)
        for key, path in paths.items():
            expected = baseline if key == "E1" else activity
            head_sha = sha256_file(path)
            if head_sha != expected["head_sha256"] or expected["state_policy_by_sequence"][sequence] != "continuous":
                raise ValueError(f"Semantic head or state policy differs from the evaluated condition: {path}")
            payload = _safe_load(path, torch)
            cfg = payload["config"]
            if (cfg["feature"] != "h" or cfg["head_type"] != "linear" or cfg["in_channels"] != 384
                    or cfg["num_classes"] != 11 or cfg.get("ignore_index", 255) != 255
                    or tuple(cfg["label_size"]) != self.frame_size or tuple(cfg["classes"]) != DSEC_SEMANTIC_11_CLASSES
                    or sha256_file(_resolve(cfg["split_manifest"], root)) != split_sha
                    or tuple(cfg["training_sequences"]) != train
                    or (dataset == "dsec" and cfg["protocol"] != "official-final")
                    or (dataset == "m3ed" and tuple(cfg["validation_sequences"]) != evaluated)
                    or cfg["precision"] not in ("fp16", "fp32")):
                raise ValueError(f"Unexpected frozen Semantic readout or split: {path}")
            head = EventStateSegmentationHead(in_channels=cfg["in_channels"], num_classes=cfg["num_classes"],
                width=cfg["head_width"], head_type=cfg["head_type"], output_size=self.frame_size)
            head.load_state_dict(payload["model"], strict=True)
            self.heads[key] = head.to(self.device).eval().requires_grad_(False)
            self.use_amp[key] = self.device.type == "cuda" and cfg["precision"] == "fp16"
            self.identity[key] = dict(head=str(path), head_sha256=head_sha, split_sha256=split_sha, role=role,
                condition=key, condition_label=CONDITIONS[key], feature="h", block=12, classes=list(cfg["classes"]),
                ignore_index=255, feature_rounding="fp16", head_precision=cfg["precision"],
                label_source="DSEC-11" if dataset == "dsec" else "M3ED InternImage pseudo-labels",
                samples_sha256=expected["samples_sha256"], expected_gt_frames=self.expected_gt_frames)
            if key == "E1":
                self.targets = SemanticTargets(dataset, cfg, pair.cfg, root, pair.prepared_root, self.frame_size, evaluated)
        # Both reports refer to exactly the same ordered (sequence, timestamp) keys.
        # Recheck against on-disk labels even when the training feature caches were removed.
        try:
            stamps = {name: self.targets.labeled_timestamps(name) for name in evaluated}
            keys = sorted((name, t) for name, times in stamps.items() for t in times)
            if canonical_sha256(keys) != baseline["samples_sha256"]:
                raise ValueError(f"Current {dataset} Semantic GT timestamps differ from the evaluated set")
            self.labeled = stamps[sequence]
        except BaseException:
            self.targets.close()
            raise

    def predict(self, maps):
        import torch

        predictions = {}
        with torch.inference_mode():
            for key, head in self.heads.items():
                raw = maps[key][12]
                if tuple(raw.shape) != (*self.grid_size, 384):
                    raise ValueError(f"Invalid Semantic block-12 geometry for {key}: {tuple(raw.shape)}")
                features = raw.permute(2, 0, 1).contiguous().half().float()[None]
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp[key]):
                    logits = head(features)
                if not torch.isfinite(logits).all():
                    raise ValueError(f"Nonfinite Semantic logits for {key}")
                predictions[key] = logits.argmax(1)[0].to(torch.uint8).cpu().numpy()
        return predictions

    def target(self, timestamp):
        if timestamp not in self.labeled:
            return None
        label = self.targets.get(self.sequence, timestamp)
        if label is None:
            raise ValueError(f"Semantic GT coverage changed at {self.sequence}/{timestamp}")
        return label

    def close(self):
        self.targets.close()
