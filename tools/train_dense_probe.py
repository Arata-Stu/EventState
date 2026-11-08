#!/usr/bin/env python3
"""Train a frozen depth/disparity probe. Test samples never select weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cache", type=Path, required=True)
    p.add_argument("--val-cache", type=Path)
    p.add_argument("--feature", choices=("z", "h", "concat", "stages"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--minimum", type=float)
    p.add_argument("--maximum", type=float)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.epochs <= 0 or a.batch_size <= 0 or a.width <= 0 or a.learning_rate <= 0:
        raise ValueError("Training hyperparameters must be positive")
    if a.output_dir.exists() and not a.resume:
        raise FileExistsError("Existing head/checkpoint artifacts are preserved; choose a new output-dir")
    import torch
    from torch.utils.data import DataLoader
    from event_state.dense.data import DenseFeatureDataset
    from event_state.dense.head import build_dense_head
    from event_state.dense.runtime import evaluate, atomic_save, validate_pair
    from event_state.dense.protocol import digest, sha256_file

    task = json.loads((a.train_cache / "metadata.json").read_text())["task"]
    minimum = a.minimum if a.minimum is not None else (0.1 if task == "depth" else 0)
    maximum = a.maximum if a.maximum is not None else (80 if task == "depth" else 1000)
    train = DenseFeatureDataset(a.train_cache, a.feature, minimum=minimum, maximum=maximum)
    if train.meta["role"] != "train" or train.meta["state_policy"] != "continuous":
        raise ValueError("Head training requires the continuous training split")
    expects_val = bool(train.protocol["splits"]["val"])
    if expects_val != (a.val_cache is not None):
        raise ValueError("Use the published validation set when defined; otherwise fixed final epoch only")
    val = None
    if a.val_cache:
        val = DenseFeatureDataset(a.val_cache, a.feature, minimum=minimum, maximum=maximum)
        validate_pair(train, val)
        if val.meta["role"] != "val" or val.meta["state_policy"] != "continuous":
            raise ValueError("Only published validation with continuous state may select checkpoints")
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    model = build_dense_head(a.feature, train[0]["feature"].shape[0], a.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate)
    generator = torch.Generator().manual_seed(a.seed)
    loader = DataLoader(train, batch_size=a.batch_size, shuffle=True, generator=generator)
    a.output_dir.mkdir(parents=True, exist_ok=a.resume)
    settings = dict(feature=a.feature, width=a.width, task=task, minimum=minimum, maximum=maximum,
                    epochs=a.epochs, seed=a.seed, learning_rate=a.learning_rate, batch_size=a.batch_size,
                    selection="validation_mae" if val else "fixed_final_epoch",
                    backbone_sha256=train.meta["checkpoint_sha256"],
                    protocol_sha256=digest(train.protocol), protocol=train.protocol,
                    train_cache_metadata_sha256=sha256_file(a.train_cache / "metadata.json"),
                    normalization_audit=train.meta["normalization_audit"],
                    low_activity_threshold=train.meta["low_activity_threshold"],
                    low_activity_metric=train.meta["low_activity_metric"],
                    geometry=train.meta["geometry"], upstream_rgb_data="unknown",
                    in_channels=train[0]["feature"].shape[0], decoder="four_level_dpt" if a.feature == "stages" else "nonrecurrent_two_conv_probe",
                    scale_alignment="none", loss="log_l1" if task == "depth" else "l1")
    best = float("inf")
    start = 0
    if a.resume:
        saved = json.loads((a.output_dir / "config.json").read_text())
        if saved != settings:
            raise ValueError("Resume config/provenance differs from existing head")
        payload = torch.load(a.output_dir / "last.pt", map_location=device, weights_only=False)
        if payload["config"] != settings:
            raise ValueError("Resume checkpoint config mismatch")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start = int(payload["epoch"])
        if not 0 < start <= a.epochs:
            raise ValueError("Invalid resume epoch")
        if (a.output_dir / "best.pt").exists():
            previous = torch.load(a.output_dir / "best.pt", map_location="cpu", weights_only=False)
            best = previous["validation"]["metrics"]["all"]["mae"]
        if payload.get("validation"):
            mae = payload["validation"]["metrics"]["all"]["mae"]
            if mae < best:
                best = mae
                atomic_save(payload, a.output_dir / "best.pt")
        if "loader_rng" in payload:
            generator.set_state(payload["loader_rng"].cpu())
        else:
            # Legacy checkpoints lack RNG. Replay DataLoader iterator seed and
            # RandomSampler consumption without loading any feature tensors.
            for _ in range(start):
                torch.empty((), dtype=torch.int64).random_(generator=generator)
                for _ in loader.batch_sampler:
                    pass
            print("Legacy resume: replayed loader RNG; bitwise equivalence is not guaranteed", flush=True)
        print(f"Resuming after epoch {start}/{a.epochs}: {a.output_dir}", flush=True)
    else:
        (a.output_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    for epoch in range(start, a.epochs):
        model.train()
        loss_sum, batches = 0.0, 0
        for batch in loader:
            x, y, valid = (batch[key].to(device) for key in ("feature", "target", "valid"))
            if not valid.any():
                continue
            prediction = model(x, y.shape[-2:])
            if task == "depth":
                loss = (prediction[valid].log() - y[valid].log()).abs().mean()
            else:
                loss = (prediction[valid] - y[valid]).abs().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite regression loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):
                raise ValueError("Nonfinite head gradients")
            optimizer.step()
            loss_sum += float(loss.detach())
            batches += 1
        if not batches:
            raise ValueError("No valid training depth pixels")
        result = evaluate(model, val, device, include_frames=False) if val is not None else None
        payload = dict(model=model.state_dict(), config=settings, epoch=epoch + 1,
                       optimizer=optimizer.state_dict(), validation=result,
                       loader_rng=generator.get_state())
        atomic_save(payload, a.output_dir / "last.pt")
        if result:
            mae = result["metrics"]["all"]["mae"]
            if mae is None:
                raise ValueError("No valid validation depth pixels")
            if mae < best:
                best = mae
                atomic_save(payload, a.output_dir / "best.pt")
        with (a.output_dir / "train.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(epoch=epoch + 1, loss=loss_sum / batches,
                                         validation=result)) + "\n")
        print(f"epoch {epoch + 1}/{a.epochs}: loss={loss_sum / batches:.6f}", flush=True)
    (a.output_dir / "_SUCCESS").write_text("complete\n")


if __name__ == "__main__":
    main()
