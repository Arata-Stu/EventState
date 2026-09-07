#!/usr/bin/env python3
"""Compare exported EventState representations on one shared DSEC clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from event_state.evaluation import joint_pca_rgb, temporal_feature_stability, token_similarity_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one or more feature artifacts on a common PCA basis"
    )
    parser.add_argument("artifacts", type=Path, nargs="+")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=Path("feature_visualization.png"))
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument("--query-index", type=int, default=None)
    return parser.parse_args()


def safe_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _validate_payload(payload: Any, path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"Feature artifact must be a mapping: {path}")
    required = {
        "experiment",
        "checkpoint_step",
        "sequence_name",
        "clip_index",
        "timestamps",
        "event_counts",
        "grid_size",
        "event_rgb",
        "rgb",
        "z",
        "h",
        "z_projected",
        "h_projected",
        "teacher",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise KeyError(f"Feature artifact lacks {', '.join(missing)}: {path}")
    grid_height, grid_width = (int(value) for value in payload["grid_size"])
    token_count = grid_height * grid_width
    for name in ("z", "h", "z_projected", "h_projected", "teacher"):
        tensor = payload[name]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise ValueError(f"{name} must have shape [T, N, D]: {path}")
        if tensor.shape[1] != token_count:
            raise ValueError(f"{name} token count does not match grid_size: {path}")
    return payload


def _validate_common_clip(payloads: list[dict[str, Any]]) -> None:
    first = payloads[0]
    for payload in payloads[1:]:
        if payload["sequence_name"] != first["sequence_name"]:
            raise ValueError("All artifacts must contain the same sequence")
        if list(payload["grid_size"]) != list(first["grid_size"]):
            raise ValueError("All artifacts must use the same patch grid")
        if not torch.equal(payload["timestamps"], first["timestamps"]):
            raise ValueError("All artifacts must contain the same timestamps")
        if not torch.allclose(
            payload["teacher"].float(), first["teacher"].float(), atol=0, rtol=0
        ):
            raise ValueError("Teacher features differ across artifacts")


def _image(tensor: torch.Tensor, frame_index: int) -> Any:
    value = tensor[frame_index]
    if value.ndim != 3:
        raise ValueError("Input preview must have shape [T, C, H, W]")
    if value.shape[0] == 1:
        return value[0].numpy()
    return value[:3].permute(1, 2, 0).numpy()


def _derived_path(output: Path, label: str) -> Path:
    suffix = output.suffix or ".png"
    return output.with_name(f"{output.stem}_{label}{suffix}")


def _plot_feature_grid(
    plt: Any,
    *,
    features: list[torch.Tensor],
    titles: list[str],
    grid_height: int,
    grid_width: int,
    frame_index: int,
    query_index: int,
    output: Path,
    shared_pca: bool,
) -> None:
    # Training aligns token directions (cosine and normalized MSE), leaving
    # feature magnitudes unconstrained. PCA must therefore see L2-normalized
    # tokens; otherwise projector norm/offset differences dominate the colors.
    pca_inputs = [F.normalize(feature.float(), dim=-1) for feature in features]
    if shared_pca:
        pca_values = joint_pca_rgb(*pca_inputs)
    else:
        pca_values = [joint_pca_rgb(feature)[0] for feature in pca_inputs]
    figure, axes = plt.subplots(
        2,
        len(features),
        figsize=(max(5, 3.1 * len(features)), 6.2),
        squeeze=False,
    )
    for column, (title, pca_value, feature) in enumerate(
        zip(titles, pca_values, features)
    ):
        axes[0, column].imshow(
            pca_value[frame_index].reshape(grid_height, grid_width, 3).numpy()
        )
        axes[0, column].set_title(f"{title}\nnormalized PCA")
        similarity = token_similarity_map(feature[frame_index], query_index)
        axes[1, column].imshow(
            similarity.reshape(grid_height, grid_width).numpy(),
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
        )
        axes[1, column].set_title(f"{title}\nquery cosine")
        for row in range(2):
            axes[row, column].axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_inputs(plt: Any, payload: dict[str, Any], output: Path) -> None:
    events = payload["event_rgb"]
    images = payload["rgb"]
    time_steps = int(events.shape[0])
    figure, axes = plt.subplots(
        2,
        time_steps,
        figsize=(max(8, 2.4 * time_steps), 5.2),
        squeeze=False,
    )
    timestamps = payload["timestamps"].tolist()
    counts = payload["event_counts"].tolist()
    for index in range(time_steps):
        axes[0, index].imshow(_image(events, index))
        axes[0, index].set_title(f"t{index} event\ncount={int(counts[index]):,}")
        axes[1, index].imshow(_image(images, index))
        axes[1, index].set_title(f"RGB\n{int(timestamps[index])}")
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    figure.suptitle(
        f"{payload['sequence_name']} clip {payload['clip_index']}", fontsize=14
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _stability(features: torch.Tensor) -> dict[str, float]:
    return {
        str(lag): float(value)
        for lag, value in temporal_feature_stability(features).items()
    }


def _plot_stability(
    plt: Any,
    payloads: list[dict[str, Any]],
    labels: list[str],
    output: Path,
) -> dict[str, Any]:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), squeeze=False)
    result: dict[str, Any] = {"raw": {}, "projected": {}, "alignment": {}}
    for payload, label in zip(payloads, labels):
        for name in ("z", "h"):
            values = _stability(payload[name])
            result["raw"][f"{label}/{name}"] = values
            axes[0, 0].plot(
                [int(lag) for lag in values],
                list(values.values()),
                marker="o",
                label=f"{label}/{name}",
            )
        objectives = payload.get("active_objectives", {"z": True, "h": True})
        for branch, feature_name in (("z", "z_projected"), ("h", "h_projected")):
            if not bool(objectives.get(branch, True)):
                continue
            values = _stability(payload[feature_name])
            result["projected"][f"{label}/P{branch}"] = values
            axes[0, 1].plot(
                [int(lag) for lag in values],
                list(values.values()),
                marker="o",
                label=f"{label}/P{branch}",
            )
            frame_cosine = F.cosine_similarity(
                payload[feature_name].float(), payload["teacher"].float(), dim=-1
            ).mean(-1)
            result["alignment"][f"{label}/P{branch}"] = {
                "mean": float(frame_cosine.mean()),
                "per_frame": [float(value) for value in frame_cosine],
            }
    for column, title in enumerate(("Raw feature stability", "Teacher-space stability")):
        axes[0, column].set_title(title)
        axes[0, column].set_xlabel("temporal lag")
        axes[0, column].set_ylabel("mean token cosine")
        axes[0, column].set_ylim(-0.05, 1.01)
        axes[0, column].grid(alpha=0.25)
        axes[0, column].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return result


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "Install visualization dependencies with `uv sync --active --extra visualize`"
        ) from error
    if args.labels is not None and len(args.labels) != len(args.artifacts):
        raise ValueError("--labels must provide exactly one label per artifact")
    payloads = [
        _validate_payload(safe_load(path.expanduser()), path) for path in args.artifacts
    ]
    _validate_common_clip(payloads)
    labels = args.labels or [
        f"{payload['experiment']}@{int(payload['checkpoint_step'])}"
        for payload in payloads
    ]
    grid_height, grid_width = (int(value) for value in payloads[0]["grid_size"])
    time_steps = int(payloads[0]["teacher"].shape[0])
    frame_index = args.frame % time_steps
    query_index = args.query_index
    if query_index is None:
        query_index = (grid_height // 2) * grid_width + grid_width // 2
    if not 0 <= query_index < grid_height * grid_width:
        raise IndexError("--query-index is outside the patch grid")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    projected_features: list[torch.Tensor] = []
    projected_titles: list[str] = []
    for payload, label in zip(payloads, labels):
        objectives = payload.get("active_objectives", {"z": True, "h": True})
        if bool(objectives.get("z", True)):
            projected_features.append(payload["z_projected"])
            projected_titles.append(f"{label} Pz")
        if bool(objectives.get("h", True)):
            projected_features.append(payload["h_projected"])
            projected_titles.append(f"{label} Ph")
    projected_features.append(payloads[0]["teacher"])
    projected_titles.append("DINOv3 teacher")
    _plot_feature_grid(
        plt,
        features=projected_features,
        titles=projected_titles,
        grid_height=grid_height,
        grid_width=grid_width,
        frame_index=frame_index,
        query_index=query_index,
        output=args.output,
        shared_pca=True,
    )

    raw_features = [payload[name] for payload in payloads for name in ("z", "h")]
    raw_titles = [f"{label} {name}" for label in labels for name in ("z", "h")]
    raw_output = _derived_path(args.output, "raw")
    _plot_feature_grid(
        plt,
        features=raw_features,
        titles=raw_titles,
        grid_height=grid_height,
        grid_width=grid_width,
        frame_index=frame_index,
        query_index=query_index,
        output=raw_output,
        shared_pca=False,
    )
    inputs_output = _derived_path(args.output, "inputs")
    _plot_inputs(plt, payloads[0], inputs_output)
    stability_output = _derived_path(args.output, "stability")
    metrics = _plot_stability(plt, payloads, labels, stability_output)
    metrics.update(
        {
            "sequence_name": payloads[0]["sequence_name"],
            "clip_index": int(payloads[0]["clip_index"]),
            "timestamps": [int(value) for value in payloads[0]["timestamps"]],
            "event_counts": [int(value) for value in payloads[0]["event_counts"]],
            "frame": int(frame_index),
            "query_index": int(query_index),
        }
    )
    metrics_output = args.output.with_suffix(".json")
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Projected alignment: {args.output}")
    print(f"Raw representations: {raw_output}")
    print(f"Input clip: {inputs_output}")
    print(f"Temporal stability: {stability_output}")
    print(f"Metrics: {metrics_output}")


if __name__ == "__main__":
    main()
