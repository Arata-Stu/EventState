"""Shared contracts for M3ED downstream target preparation and loading."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


M3ED_DOWNSTREAM_FORMAT_VERSION = 1
M3ED_IGNORE_LABEL = 255

# DSEC-Semantic's 11-class protocol, matching the mapping used by F3.
M3ED_CITYSCAPES_19_TO_DSEC_11 = np.asarray(
    [5, 6, 1, 9, 2, 4, 10, 10, 7, 7, 0, 3, 3, 8, 8, 8, 8, 8, 8],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class TimestampMatches:
    """Nearest source sample for every target timestamp."""

    indices: np.ndarray
    deltas_us: np.ndarray
    valid: np.ndarray


def match_nearest_timestamps(
    source_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    *,
    max_delta_us: int,
) -> TimestampMatches:
    """Match monotonic microsecond timestamps without assuming equal frame rates."""

    source = np.asarray(source_timestamps, dtype=np.int64)
    target = np.asarray(target_timestamps, dtype=np.int64)
    if max_delta_us < 0:
        raise ValueError("max_delta_us must be non-negative")
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("source and target timestamps must be one-dimensional")
    if len(source) == 0:
        return TimestampMatches(
            indices=np.full(len(target), -1, dtype=np.int64),
            deltas_us=np.full(len(target), np.iinfo(np.int64).max, dtype=np.int64),
            valid=np.zeros(len(target), dtype=np.bool_),
        )
    if np.any(np.diff(source) <= 0) or np.any(np.diff(target) <= 0):
        raise ValueError("source and target timestamps must be strictly increasing")

    right = np.searchsorted(source, target, side="left")
    right = np.clip(right, 0, len(source) - 1)
    left = np.clip(right - 1, 0, len(source) - 1)
    left_delta = np.abs(target - source[left])
    right_delta = np.abs(target - source[right])
    choose_right = right_delta < left_delta
    indices = np.where(choose_right, right, left).astype(np.int64)
    signed_deltas = (source[indices] - target).astype(np.int64)
    valid = np.abs(signed_deltas) <= max_delta_us
    indices = np.where(valid, indices, -1).astype(np.int64)
    return TimestampMatches(indices=indices, deltas_us=signed_deltas, valid=valid)


def map_cityscapes_19_to_dsec_11(labels: np.ndarray) -> np.ndarray:
    """Map M3ED InternImage Cityscapes predictions to the DSEC 11-class protocol."""

    source = np.asarray(labels)
    if not np.issubdtype(source.dtype, np.integer):
        raise TypeError("semantic labels must use an integer dtype")
    unexpected = np.unique(
        source[(source != M3ED_IGNORE_LABEL) & ((source < 0) | (source > 18))]
    )
    if len(unexpected):
        raise ValueError(f"M3ED semantic labels outside Cityscapes 0..18/255: {unexpected}")
    result = np.full(source.shape, M3ED_IGNORE_LABEL, dtype=np.uint8)
    valid = (source >= 0) & (source <= 18)
    result[valid] = M3ED_CITYSCAPES_19_TO_DSEC_11[source[valid].astype(np.int64)]
    return result


def so3_log_vector(rotation: np.ndarray) -> np.ndarray:
    """Return the axis-angle logarithm of a 3x3 rotation matrix."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape [3,3]")
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cosine))
    vee = np.asarray(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    if theta < 1e-7:
        return 0.5 * vee
    if np.pi - theta < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eigh((matrix + np.eye(3)) * 0.5)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        axis *= np.sign(np.dot(axis, vee) or 1.0)
        return theta * axis
    return theta * vee / (2.0 * np.sin(theta))


def camera_relative_motion(
    previous_pose: np.ndarray,
    current_pose: np.ndarray,
    *,
    delta_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute M3ED/F3-convention relative motion and linear/angular velocity."""

    previous = np.asarray(previous_pose, dtype=np.float64)
    current = np.asarray(current_pose, dtype=np.float64)
    if previous.shape != (4, 4) or current.shape != (4, 4):
        raise ValueError("poses must have shape [4,4]")
    if not np.isfinite(delta_seconds) or delta_seconds <= 0:
        raise ValueError("delta_seconds must be positive")
    relative = previous @ np.linalg.inv(current)
    linear_velocity = relative[:3, 3] / delta_seconds
    angular_velocity = so3_log_vector(relative[:3, :3]) / delta_seconds
    return relative, linear_velocity, angular_velocity


__all__ = [
    "M3ED_CITYSCAPES_19_TO_DSEC_11",
    "M3ED_DOWNSTREAM_FORMAT_VERSION",
    "M3ED_IGNORE_LABEL",
    "TimestampMatches",
    "camera_relative_motion",
    "map_cityscapes_19_to_dsec_11",
    "match_nearest_timestamps",
    "so3_log_vector",
]
