"""Stream aligned intermediate features from the two frozen Semantic placement models."""
import json
import math
from pathlib import Path

from event_state.dense.common_protocol import audit_common
from event_state.dense.protocol import sha256_file
from event_state.evaluation.activity_scenes import eligible_sequences
from tools.placement_conditions import placement_label


BLOCKS = (3, 6, 9, 12)
FOUR_STAGE = "E1_b3_6_9_12"
CONDITIONS = {
    "E1": "E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM（12層後）",
    FOUR_STAGE: FOUR_STAGE + " " + placement_label(FOUR_STAGE),
}


def checkpoint_paths(dataset, project_root):
    if dataset not in ("dsec", "m3ed"):
        raise ValueError("Expected dsec or m3ed")
    output = Path(project_root).resolve() / "outputs"
    roots = {"E1": "stage1_joined_baseline_full_20261107", FOUR_STAGE: "stage1_placement_full_20261118"}
    return {key: output / root / f"{dataset}_{key}_w1/checkpoints/step_00099999.pt" for key, root in roots.items()}


def _input_contract(raw, dataset, condition, step, project_root):
    ds, training = raw["dataset"], raw["training"]
    temporal, loss = raw["model"]["temporal"], raw["loss"]
    stages = condition == FOUR_STAGE
    expected_size = (448 if dataset == "dsec" else 352, 640)
    if (step != 99999 or training["max_steps"] != step or training.get("distributed_world_size") != 1
            or raw["seed"] != 0 or ds["name"] != dataset
            or temporal["type"] != ("four_stage_lstm" if stages else "lstm")
            or temporal.get("after_blocks") != (list(BLOCKS) if stages else None)
            or any(temporal.get(name) != 384 for name in ("input_dim", "hidden_dim", "output_dim"))
            or temporal.get("num_layers") != 1
            or not loss["h_distill"]["enabled"] or loss["h_distill"]["cosine_weight"] != 1
            or loss["h_distill"]["mse_weight"] != 1 or loss["z_objective"]["type"] != "none"
            or loss["z_objective"]["weight"] != 0):
        raise ValueError(f"Wrong Stage 1 checkpoint for {dataset} {condition}")
    audit_common(raw)
    rep = ds["representation"]
    if (rep["type"] != "gep_rgb" or ds.get("event_window", "rgb_interval") != "rgb_interval"
            or ds.get("event_window_us") is not None or ds.get("event_window_fraction", 1) != 1
            or (ds["input_height"], ds["input_width"]) != expected_size):
        raise ValueError(f"Expected common Semantic RGB-interval GEP geometry for {dataset}")
    mean, std = list(rep["normalize_mean"]), list(rep["normalize_std"])
    if len(mean) != 3 or len(std) != 3 or not all(math.isfinite(x) for x in mean + std) or min(std) <= 0:
        raise ValueError("Invalid event normalization")
    prepared = ds.get("prepared_root") if dataset == "m3ed" else None
    return dict(prepared_root=str((project_root / (prepared or ds["event_cache_dir"])).resolve()),
                rgb_root=str((project_root / ds["root"]).resolve()), input_size=list(expected_size),
                percentile=rep["percentile"], normalize_mean=mean, normalize_std=std)


def _check_semantic_provenance(root, dataset, condition, digest):
    """Cross-check retained Semantic records without requiring heads or feature caches to run."""
    if condition == FOUR_STAGE:
        receipt = root / "outputs/stage2_placement_semantic_20261118" / f"{dataset}_{condition}" / "verified.json"
        if not receipt.is_file():
            return []
        row = json.loads(receipt.read_text(encoding="utf-8"))
        if (row["dataset"] != dataset or row["condition"] != condition or row["after_blocks"] != list(BLOCKS)
                or row["seed"] != 0 or row["global_step"] != 99999 or row["backbone_sha256"] != digest):
            raise ValueError(f"Placement checkpoint differs from completed Semantic evaluation: {receipt}")
        return [str(receipt)]

    from tools.run_semantic_depth_activity import ROOT, semantic_run
    head_root = root / semantic_run(dataset, condition).relative_to(ROOT)
    config = head_root / ("final" if dataset == "dsec" else "h") / "config.json"
    if not config.is_file():
        return []
    cache = root / json.loads(config.read_text(encoding="utf-8"))["feature_cache_dir"]
    metadata = sorted(cache.glob("*/metadata.json"))
    for path in metadata:
        if json.loads(path.read_text(encoding="utf-8"))["checkpoint_sha256"] != digest:
            raise ValueError(f"E1 checkpoint differs from the Semantic feature cache: {path}")
    return [str(p) for p in metadata]


class PlacementFeaturePair:
    def __init__(self, dataset, project_root, device="cuda", *, teacher_checkpoint=None):
        import torch
        from omegaconf import OmegaConf
        from event_state.training import build_model, load_checkpoint, load_checkpoint_config_metadata

        root = Path(project_root).expanduser().resolve()
        paths = checkpoint_paths(dataset, root)
        self.dataset = dataset
        self.device = torch.device(device)
        self.frame_size = (440 if dataset == "dsec" else 352, 640)
        self.models, self.identity = {}, {}
        shared_input = None
        for condition, path in paths.items():
            meta = load_checkpoint_config_metadata(path)
            contract = _input_contract(meta.config, dataset, condition, meta.global_step, root)
            if shared_input is not None and contract != shared_input:
                raise ValueError("Models use different Semantic input data, geometry or normalization")
            shared_input = contract
            digest = sha256_file(path)
            evidence = _check_semantic_provenance(root, dataset, condition, digest)
            cfg = OmegaConf.create(meta.config)
            if teacher_checkpoint is None:
                # The Stage 1 state dict contains the entire encoder, loaded strictly below.
                OmegaConf.update(cfg, "model.event_encoder.pretrained_init", False)
                OmegaConf.update(cfg, "model.event_encoder.checkpoint", None, force_add=True)
                OmegaConf.update(cfg, "teacher.checkpoint", None, force_add=True)
            else:
                teacher = (root / teacher_checkpoint).resolve(strict=True)
                OmegaConf.update(cfg, "model.event_encoder.checkpoint", str(teacher), force_add=True)
                OmegaConf.update(cfg, "teacher.checkpoint", str(teacher), force_add=True)
            model = build_model(cfg, device=self.device)
            load_checkpoint(path, model=model, device=self.device, restore_rng=False)
            self.models[condition] = model.eval().requires_grad_(False)
            self.identity[condition] = dict(condition_label=CONDITIONS[condition], dataset=dataset,
                backbone=str(path), backbone_sha256=digest, global_step=meta.global_step, seed=0,
                after_blocks=list(BLOCKS) if condition == FOUR_STAGE else [12], state_policy="continuous",
                semantic_provenance=evidence, input=contract,
                inference_precision="float32", patch_stride=16)
            if condition == "E1":
                self.cfg = cfg
        self.prepared_root = Path(shared_input["prepared_root"])
        self.input_size = tuple(shared_input["input_size"])
        self.mean = torch.tensor(shared_input["normalize_mean"], device=self.device).view(3, 1, 1)
        self.std = torch.tensor(shared_input["normalize_std"], device=self.device).view(3, 1, 1)

    def frames(self, sequence):
        """Start both models with fresh state and advance them once on every prepared input."""
        import torch
        from event_state.dense.source import input_geometry, prepared_frames

        permitted, _ = eligible_sequences(self.dataset, "semantic")
        if sequence not in permitted:
            raise ValueError(f"Not a {self.dataset} Semantic evaluation recording: {sequence}")
        states = dict.fromkeys(CONDITIONS)
        gh, gw = (size // 16 for size in self.input_size)
        if self.dataset == "dsec":
            from tools.cache_dsec_semantic_features import _semantic_transform
        for frame in prepared_frames(self.prepared_root, sequence, self.cfg, self.dataset):
            events = frame["events"]
            with torch.inference_mode():
                if self.dataset == "dsec":
                    value = _semantic_transform(events, self.cfg).to(self.device)
                else:
                    crop = input_geometry(events, self.dataset).to(self.device)
                    value = ((crop - self.mean) / self.std)[None]
                maps = {}
                for condition, model in self.models.items():
                    levels, states[condition] = model.forward_depth_stages(value[:, None], states[condition])
                    if len(levels) != len(BLOCKS) or any(x.shape != (1, 1, gh * gw, 384) for x in levels):
                        raise ValueError(f"Wrong ViT feature geometry for {condition} at {sequence}/{frame['timestamp']}")
                    maps[condition] = {
                        block: x[0, 0].reshape(gh, gw, 384).float() for block, x in zip(BLOCKS, levels)
                    }
            yield frame, maps
