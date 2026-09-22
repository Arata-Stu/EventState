"""ScaleEvent's released uint8 event-image activation rule (not raw counts)."""

from __future__ import annotations

import numpy as np


def scale_event_activation(image: np.ndarray, *, height: int, width: int) -> np.ndarray:
    """Return [H/16,W/16] bool mask using the reference's two OpenCV resizes.

    Input is an unnormalized white-background uint8 RGB/BGR event image.
    Channel ordering does not matter because all three channels are summed.
    Do not replace this with pooling, float interpolation, or a nonzero test.
    """
    import cv2

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("ScaleEvent activation requires a uint8 three-channel image")
    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise ValueError("ScaleEvent activation requires positive multiples of 16")
    coarse = cv2.resize(image, (width // 4, height // 4), interpolation=cv2.INTER_LINEAR)
    patches = cv2.resize(coarse, (width // 16, height // 16), interpolation=cv2.INTER_LINEAR)
    return patches.sum(axis=-1) < 765
