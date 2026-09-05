"""Frozen DINOv3 RGB teacher."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn

from ..dinov3 import (
    DINO_V3_DEFAULT_REPOSITORY,
    DINO_V3_VITS16,
    DINO_V3_VITS16_EMBED_DIM,
    DINO_V3_VITS16_PATCH_SIZE,
    extract_normalized_patch_tokens,
    load_dinov3_backbone,
)
from ..tokens import patch_grid, tokens_to_map


class FrozenDinoV3Teacher(nn.Module):
    """Frozen ViT-S/16 returning normalized dense patch tokens.

    Inputs are RGB float tensors in ``[0, 1]``.  ImageNet normalization is
    applied internally by default.  The teacher remains in evaluation mode
    even if a parent training module receives ``train()``.
    """

    embed_dim = DINO_V3_VITS16_EMBED_DIM
    patch_size = DINO_V3_VITS16_PATCH_SIZE
    in_channels = 3

    def __init__(
        self,
        *,
        checkpoint: str | Path | None = None,
        pretrained: bool = True,
        source: str = "package",
        repository: str | Path = DINO_V3_DEFAULT_REPOSITORY,
        hub_source: str = "github",
        backbone: str | nn.Module = DINO_V3_VITS16,
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
        normalize_inputs: bool = True,
        clone_output: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(backbone, nn.Module):
            model = backbone
        elif isinstance(backbone, str):
            model = load_dinov3_backbone(
                backbone=backbone,
                checkpoint=checkpoint,
                pretrained=pretrained,
                source=source,
                repository=repository,
                hub_source=hub_source,
            )
        else:
            raise TypeError("backbone must be a DINOv3 entry-point name or nn.Module")
        mean = tuple(float(value) for value in image_mean)
        std = tuple(float(value) for value in image_std)
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("image_mean and image_std must each contain three values")
        if any(value <= 0 for value in std):
            raise ValueError("image_std values must be positive")

        self.backbone = model
        self.normalize_inputs = bool(normalize_inputs)
        self.clone_output = bool(clone_output)
        self.register_buffer(
            "image_mean",
            torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenDinoV3Teacher":
        """Keep the frozen teacher in evaluation mode unconditionally."""

        del mode
        super().train(False)
        return self

    def _prepare_images(self, images: Tensor) -> tuple[Tensor, tuple[int, int]]:
        if not isinstance(images, Tensor):
            raise TypeError("images must be a torch.Tensor")
        if images.ndim != 4:
            raise ValueError(
                f"images must have shape [B, 3, H, W], got {tuple(images.shape)}"
            )
        if images.shape[0] <= 0:
            raise ValueError("images batch dimension must be non-empty")
        if images.shape[1] != 3:
            raise ValueError(f"DINOv3 RGB teacher expects 3 channels, got {images.shape[1]}")
        if not images.is_floating_point():
            raise TypeError("images must be floating point tensors scaled to [0, 1]")
        grid_size = patch_grid(images.shape[-2:], self.patch_size, require_divisible=True)
        if self.normalize_inputs:
            mean = self.image_mean.to(dtype=images.dtype)
            std = self.image_std.to(dtype=images.dtype)
            images = (images - mean) / std
        return images, grid_size

    def extract_patch_tokens(self, images: Tensor) -> Tensor:
        images, (grid_height, grid_width) = self._prepare_images(images)
        # inference_mode is faster than no_grad, but its tensors cannot always
        # be saved for a student's backward pass.  Clone after leaving the
        # context to return a normal tensor in online-teacher mode.
        with torch.inference_mode():
            tokens = extract_normalized_patch_tokens(self.backbone, images)
        if self.clone_output:
            tokens = tokens.clone()
        expected_shape = (
            images.shape[0],
            grid_height * grid_width,
            self.embed_dim,
        )
        if tuple(tokens.shape) != expected_shape:
            raise ValueError(f"Teacher returned {tuple(tokens.shape)}, expected {expected_shape}")
        return tokens

    def forward(self, images: Tensor) -> Tensor:
        return self.extract_patch_tokens(images)


FrozenDINOv3Teacher = FrozenDinoV3Teacher


def build_dinov3_teacher(
    *,
    checkpoint: str | Path | None = None,
    pretrained: bool = True,
    source: str = "package",
    repository: str | Path = DINO_V3_DEFAULT_REPOSITORY,
    hub_source: str = "github",
    backbone: str | nn.Module = DINO_V3_VITS16,
    image_mean: Sequence[float] = (0.485, 0.456, 0.406),
    image_std: Sequence[float] = (0.229, 0.224, 0.225),
    normalize_inputs: bool = True,
    clone_output: bool = True,
) -> FrozenDinoV3Teacher:
    return FrozenDinoV3Teacher(
        checkpoint=checkpoint,
        pretrained=pretrained,
        source=source,
        repository=repository,
        hub_source=hub_source,
        backbone=backbone,
        image_mean=image_mean,
        image_std=image_std,
        normalize_inputs=normalize_inputs,
        clone_output=clone_output,
    )


__all__ = [
    "FrozenDINOv3Teacher",
    "FrozenDinoV3Teacher",
    "build_dinov3_teacher",
    "tokens_to_map",
]
