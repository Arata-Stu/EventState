"""Trainable DINOv3 event encoder."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

from .dinov3 import (
    DINO_V3_DEFAULT_REPOSITORY,
    DINO_V3_VITS16,
    DINO_V3_VITS16_EMBED_DIM,
    DINO_V3_VITS16_PATCH_SIZE,
    extract_normalized_patch_tokens,
    load_dinov3_backbone,
)
from .tokens import patch_grid


PatchInitialization = Literal["pretrained", "random", "rgb_mean"]


def _new_patch_projection(old: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if old.groups != 1:
        raise ValueError("DINOv3 patch projection is expected to use groups=1")
    projection = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=1,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    )
    return projection.to(device=old.weight.device, dtype=old.weight.dtype)


def _reset_like_dinov3_patch_embed(projection: nn.Conv2d) -> None:
    fan_in = projection.in_channels * projection.kernel_size[0] * projection.kernel_size[1]
    bound = math.sqrt(1.0 / fan_in)
    nn.init.uniform_(projection.weight, -bound, bound)
    if projection.bias is not None:
        nn.init.uniform_(projection.bias, -bound, bound)


def adapt_patch_embedding(
    backbone: nn.Module,
    *,
    in_channels: int,
    initialization: PatchInitialization | str,
) -> None:
    """Adapt only the RGB patch projection while preserving all other weights."""

    if in_channels <= 0:
        raise ValueError("in_channels must be positive")
    patch_embed = getattr(backbone, "patch_embed", None)
    old = getattr(patch_embed, "proj", None)
    if not isinstance(old, nn.Conv2d):
        raise TypeError("backbone.patch_embed.proj must be nn.Conv2d")
    strategy = str(initialization).lower()
    if strategy not in {"pretrained", "random", "rgb_mean"}:
        raise ValueError("patch_init must be 'pretrained', 'random', or 'rgb_mean'")

    if in_channels == old.in_channels:
        if strategy == "pretrained":
            return
        if strategy == "random":
            _reset_like_dinov3_patch_embed(old)
            return
        if old.in_channels != 3:
            raise ValueError("rgb_mean initialization requires an RGB source projection")
        # For three channels rgb_mean has no useful semantics; rejecting it
        # avoids accidentally discarding the learned channel-specific filters.
        raise ValueError("rgb_mean is only valid when adapting RGB weights to non-RGB input")

    if strategy == "pretrained":
        raise ValueError(
            "patch_init='pretrained' is only valid when in_channels=3; "
            "use 'random' or 'rgb_mean' for a non-RGB event representation"
        )
    if strategy == "rgb_mean" and old.in_channels != 3:
        raise ValueError(
            f"rgb_mean requires a 3-channel source projection, got {old.in_channels} channels"
        )

    projection = _new_patch_projection(old, in_channels)
    _reset_like_dinov3_patch_embed(projection)
    if strategy == "rgb_mean":
        with torch.no_grad():
            mean_filter = old.weight.mean(dim=1, keepdim=True)
            projection.weight.copy_(mean_filter.repeat(1, in_channels, 1, 1))
            if projection.bias is not None and old.bias is not None:
                projection.bias.copy_(old.bias)
    patch_embed.proj = projection
    if hasattr(patch_embed, "in_chans"):
        patch_embed.in_chans = in_channels


class EventEncoder(nn.Module):
    """DINOv3 ViT-S/16 over an event representation.

    The three-channel GEP-compatible baseline preserves the pretrained patch
    projection object and its weights.  Non-RGB representations replace only
    that projection; every transformer parameter remains initialized from the
    RGB checkpoint when ``pretrained=True``.
    """

    embed_dim = DINO_V3_VITS16_EMBED_DIM
    patch_size = DINO_V3_VITS16_PATCH_SIZE

    def __init__(
        self,
        *,
        in_channels: int = 3,
        patch_init: PatchInitialization | str = "pretrained",
        pretrained: bool = True,
        checkpoint: str | Path | None = None,
        source: str = "package",
        repository: str | Path = DINO_V3_DEFAULT_REPOSITORY,
        hub_source: str = "github",
        backbone: str | nn.Module = DINO_V3_VITS16,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
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

        self.backbone = model
        self.in_channels = int(in_channels)
        self.patch_init = str(patch_init).lower()
        adapt_patch_embedding(
            self.backbone,
            in_channels=self.in_channels,
            initialization=self.patch_init,
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(True)

    def _validate_input(self, events: Tensor) -> tuple[int, int]:
        if not isinstance(events, Tensor):
            raise TypeError("events must be a torch.Tensor")
        if events.ndim != 4:
            raise ValueError(
                f"events must have shape [B, C, H, W], got {tuple(events.shape)}"
            )
        if events.shape[0] <= 0:
            raise ValueError("events batch dimension must be non-empty")
        if events.shape[1] != self.in_channels:
            raise ValueError(
                f"EventEncoder expects {self.in_channels} channels, got {events.shape[1]}"
            )
        return patch_grid(events.shape[-2:], self.patch_size, require_divisible=True)

    def forward(self, events: Tensor) -> Tensor:
        grid_height, grid_width = self._validate_input(events)
        tokens = extract_normalized_patch_tokens(self.backbone, events)
        expected_shape = (
            events.shape[0],
            grid_height * grid_width,
            self.embed_dim,
        )
        if tuple(tokens.shape) != expected_shape:
            raise ValueError(
                f"Event encoder returned {tuple(tokens.shape)}, expected {expected_shape}"
            )
        return tokens

    def forward_sequence(self, events: Tensor) -> Tensor:
        """Encode ``[B, T, C, H, W]`` efficiently as ``[B, T, N, D]``."""

        if not isinstance(events, Tensor) or events.ndim != 5:
            shape = tuple(events.shape) if isinstance(events, Tensor) else type(events).__name__
            raise ValueError(f"events must have shape [B, T, C, H, W], got {shape}")
        batch_size, sequence_length, channels, height, width = events.shape
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("events batch and sequence dimensions must be non-empty")
        flat = events.reshape(batch_size * sequence_length, channels, height, width)
        tokens = self(flat)
        return tokens.reshape(batch_size, sequence_length, tokens.shape[1], tokens.shape[2])


DINOv3EventEncoder = EventEncoder


def build_event_encoder(
    *,
    in_channels: int = 3,
    patch_init: PatchInitialization | str = "pretrained",
    pretrained: bool = True,
    checkpoint: str | Path | None = None,
    source: str = "package",
    repository: str | Path = DINO_V3_DEFAULT_REPOSITORY,
    hub_source: str = "github",
    backbone: str | nn.Module = DINO_V3_VITS16,
) -> EventEncoder:
    """Build the Phase 1 event encoder from explicit configuration fields."""

    return EventEncoder(
        in_channels=in_channels,
        patch_init=patch_init,
        pretrained=pretrained,
        checkpoint=checkpoint,
        source=source,
        repository=repository,
        hub_source=hub_source,
        backbone=backbone,
    )


__all__ = [
    "DINOv3EventEncoder",
    "EventEncoder",
    "PatchInitialization",
    "adapt_patch_embedding",
    "build_event_encoder",
]
