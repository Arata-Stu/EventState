"""Dependency-free experiment configuration: preserve audited data/split settings."""
from copy import deepcopy

from event_state.causal_rgb import validate_context


def make_plan(base, teacher, *, output, stage, batch_size=None):
    cfg = deepcopy(base)
    if cfg["teacher"]["type"] != "dinov3_vits16":
        raise ValueError("Start from an audited DINOv3 configuration, not another video run")
    if cfg["model"]["event_encoder"]["architecture"] != "dinov3_vits16":
        raise ValueError("This experiment holds the student at ViT-S/16")
    if cfg["model"]["temporal"]["type"] != "lstm":
        raise ValueError("This experiment requires the existing LSTM")
    dataset = cfg["dataset"]
    if dataset["name"] not in {"dsec", "m3ed"} or not dataset.get("train_sequences"):
        raise ValueError("Explicit audited DSEC/M3ED training sequences are required")
    if cfg["training"].get("validation_enabled", True):
        raise ValueError("Use a final-fit base config with validation disabled (no hidden test selection)")
    mode = teacher.get("input_mode", "video")
    validate_context(mode, int(teacher["context_frames"]), teacher.get("sample_fps"))
    student = cfg["model"]["event_encoder"]
    for field in ("checkpoint", "backend", "repository", "source"):
        student.setdefault(field, cfg["teacher"].get(field))
        if not student.get(field):
            raise ValueError(f"Missing explicit student DINO {field}")
    cfg["teacher"] = deepcopy(teacher)
    cfg["model"]["projector"]["output_dim"] = 768
    cfg["model"]["temporal"]["dropout"] = 0.0
    dataset["augmentation"]["enabled"] = False
    dataset["activity_mask"] = False
    cfg["loss"] = {
        "h_distill": {"enabled": True, "cosine_weight": 1.0, "mse_weight": 1.0},
        "z_objective": {"type": "none", "weight": 0.0, "cosine_weight": 1.0, "mse_weight": 1.0},
    }
    training = cfg["training"]
    training["resume"] = None
    training["event_dropout"]["enabled"] = False
    training["sampling"]["mode"] = "mixed"
    if batch_size is not None:
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Hybrid batch size must be a positive even number >=2")
        training["batch_size"] = batch_size
        training["sampling"]["random_batch_size"] = batch_size // 2
        training["sampling"]["stream_batch_size"] = batch_size // 2
    if stage == "smoke":
        training.update(max_steps=10, log_every=1, checkpoint_every=10)
        cfg["scheduler"]["warmup_steps"] = 2
    elif stage == "full":
        training["checkpoint_every"] = 10000
    else:
        raise ValueError("stage must be smoke or full")
    cfg["experiment"] = {"name": f"{dataset['name']}_vjepa21_{mode}_h_only_{stage}"}
    cfg["output"] = {"directory": str(output)}
    cfg["hydra"] = {"run": {"dir": str(output)}, "job": {"chdir": False}}
    cfg.pop("defaults", None)
    return cfg
