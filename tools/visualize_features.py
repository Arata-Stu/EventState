#!/usr/bin/env python3
"""Visualize z, h, and teacher patch tokens from an evaluation artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from event_state.evaluation import joint_pca_rgb, temporal_feature_stability, token_similarity_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path, help=".pt file containing z, h, and teacher tensors")
    parser.add_argument("--output", type=Path, default=Path("feature_visualization.png"))
    parser.add_argument("--grid-height", type=int, default=28)
    parser.add_argument("--grid-width", type=int, default=40)
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument("--query-index", type=int, default=None)
    return parser.parse_args()


def safe_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "Install visualization dependencies with `pip install -e '.[visualize]'`"
        ) from error

    payload = safe_load(args.artifact)
    names = ("z", "h", "teacher")
    features = [payload[name] for name in names]
    features = [tensor[0] if tensor.ndim == 4 else tensor for tensor in features]
    token_count = args.grid_height * args.grid_width
    if any(tensor.ndim != 3 or tensor.shape[1] != token_count for tensor in features):
        raise ValueError(f"Expected [T, {token_count}, D] tensors for z, h, and teacher")

    pca = joint_pca_rgb(*features)
    frame_index = args.frame % features[0].shape[0]
    query_index = args.query_index
    if query_index is None:
        query_index = (args.grid_height // 2) * args.grid_width + args.grid_width // 2

    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for column, (name, pca_value, feature) in enumerate(zip(names, pca, features)):
        axes[0, column].imshow(
            pca_value[frame_index].reshape(args.grid_height, args.grid_width, 3).numpy()
        )
        axes[0, column].set_title(f"{name} PCA")
        similarity = token_similarity_map(feature[frame_index], query_index)
        axes[1, column].imshow(
            similarity.reshape(args.grid_height, args.grid_width).numpy(),
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
        )
        axes[1, column].set_title(f"{name} cosine query")
        for row in range(2):
            axes[row, column].axis("off")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)

    stability = {
        name: {str(lag): float(value) for lag, value in temporal_feature_stability(tensor).items()}
        for name, tensor in zip(names[:2], features[:2])
    }
    args.output.with_suffix(".json").write_text(
        json.dumps({"temporal_stability": stability}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
