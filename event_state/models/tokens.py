"""Dense patch-token shape helpers."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor


def _positive_pair(value: Sequence[int], *, name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two integers")
    pair = (int(value[0]), int(value[1]))
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"{name} values must be positive, got {pair}")
    return pair


def patch_grid(
    image_size: Sequence[int],
    patch_size: int | Sequence[int] = 16,
    *,
    require_divisible: bool = True,
) -> tuple[int, int]:
    """Compute a patch grid and optionally reject silently cropped borders."""

    height, width = _positive_pair(image_size, name="image_size")
    if isinstance(patch_size, int):
        patch_height = patch_width = patch_size
    else:
        patch_height, patch_width = _positive_pair(patch_size, name="patch_size")
    if patch_height <= 0 or patch_width <= 0:
        raise ValueError("patch_size must be positive")
    if require_divisible and (height % patch_height or width % patch_width):
        raise ValueError(
            f"Image size {(height, width)} must be divisible by patch size "
            f"{(patch_height, patch_width)}"
        )
    return height // patch_height, width // patch_width


def tokens_to_map(tokens: Tensor, grid_size: Sequence[int]) -> Tensor:
    """Convert ``[..., N, D]`` tokens to ``[..., D, grid_h, grid_w]``.

    Leading dimensions are retained, so both ``[B, N, D]`` and
    ``[B, T, N, D]`` are supported.
    """

    if not isinstance(tokens, Tensor):
        raise TypeError("tokens must be a torch.Tensor")
    if tokens.ndim < 2:
        raise ValueError("tokens must have shape [..., N, D]")
    grid_height, grid_width = _positive_pair(grid_size, name="grid_size")
    expected_tokens = grid_height * grid_width
    if tokens.shape[-2] != expected_tokens:
        raise ValueError(
            f"Token count {tokens.shape[-2]} does not match grid "
            f"{grid_height}x{grid_width}={expected_tokens}"
        )
    feature_dim = tokens.shape[-1]
    token_map = tokens.reshape(*tokens.shape[:-2], grid_height, grid_width, feature_dim)
    return token_map.movedim(-1, -3).contiguous()


def map_to_tokens(token_map: Tensor) -> Tensor:
    """Convert ``[..., D, grid_h, grid_w]`` back to ``[..., N, D]``."""

    if not isinstance(token_map, Tensor):
        raise TypeError("token_map must be a torch.Tensor")
    if token_map.ndim < 3:
        raise ValueError("token_map must have shape [..., D, grid_h, grid_w]")
    feature_dim, grid_height, grid_width = token_map.shape[-3:]
    return token_map.movedim(-3, -1).reshape(
        *token_map.shape[:-3], grid_height * grid_width, feature_dim
    ).contiguous()


__all__ = ["map_to_tokens", "patch_grid", "tokens_to_map"]
