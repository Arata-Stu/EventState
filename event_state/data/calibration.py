"""Calibration helpers for aligning DSEC RGB images to the event-camera view."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class DSECCameraAlignment:
    """Rotation-only mapping from rectified RGB pixels to rectified event pixels.

    DSEC's two cameras have a non-zero baseline. Without depth there is no exact
    pixel-wise warp between them, so this follows the GEP preprocessing and uses
    the plane-at-infinity (rotation-only) homography.
    """

    width: int
    height: int
    image_to_event_homography: np.ndarray
    event_camera_matrix: np.ndarray
    image_camera_matrix: np.ndarray

    def metadata(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "mapping": "rectified_rgb_to_rectified_event_rotation_only",
            "image_to_event_homography": self.image_to_event_homography.tolist(),
            "event_camera_matrix": self.event_camera_matrix.tolist(),
            "image_camera_matrix": self.image_camera_matrix.tolist(),
        }


def load_dsec_camera_alignment(path: str | Path) -> DSECCameraAlignment:
    """Load ``cam_to_cam.yaml`` and reproduce GEP's RGB-to-event homography."""

    calibration_path = Path(path).expanduser()
    if not calibration_path.is_file():
        raise FileNotFoundError(f"DSEC calibration file not found: {calibration_path}")
    with calibration_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid DSEC calibration document: {calibration_path}")

    try:
        intrinsics = payload["intrinsics"]
        extrinsics = payload["extrinsics"]
        event_camera = _camera_matrix(intrinsics["camRect0"]["camera_matrix"])
        image_camera = _camera_matrix(intrinsics["camRect1"]["camera_matrix"])
        resolution = np.asarray(intrinsics["camRect0"]["resolution"], dtype=np.int64)
        transform_10 = np.asarray(extrinsics["T_10"], dtype=np.float64)
        event_rectification = np.asarray(extrinsics["R_rect0"], dtype=np.float64)
        image_rectification = np.asarray(extrinsics["R_rect1"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Incomplete DSEC calibration file: {calibration_path}") from error

    if resolution.shape != (2,) or np.any(resolution <= 0):
        raise ValueError(f"Invalid camRect0 resolution {resolution!r}: {calibration_path}")
    if transform_10.shape != (4, 4):
        raise ValueError(f"T_10 must have shape (4, 4): {calibration_path}")
    for name, matrix in (
        ("R_rect0", event_rectification),
        ("R_rect1", image_rectification),
    ):
        if matrix.shape != (3, 3):
            raise ValueError(f"{name} must have shape (3, 3): {calibration_path}")

    event_rectified_transform = _homogeneous_rotation(event_rectification)
    image_rectified_transform = _homogeneous_rotation(image_rectification)
    event_to_image_transform = (
        image_rectified_transform
        @ transform_10
        @ np.linalg.inv(event_rectified_transform)
    )
    event_to_image_homography = (
        image_camera
        @ event_to_image_transform[:3, :3]
        @ np.linalg.inv(event_camera)
    )
    image_to_event_homography = np.linalg.inv(event_to_image_homography)
    if not np.all(np.isfinite(image_to_event_homography)):
        raise ValueError(f"Non-finite RGB-to-event homography: {calibration_path}")

    width, height = (int(value) for value in resolution)
    return DSECCameraAlignment(
        width=width,
        height=height,
        image_to_event_homography=image_to_event_homography,
        event_camera_matrix=event_camera,
        image_camera_matrix=image_camera,
    )


def _camera_matrix(value: Any) -> np.ndarray:
    parameters = np.asarray(value, dtype=np.float64)
    if parameters.shape == (3, 3):
        matrix = parameters
    elif parameters.shape == (4,):
        fx, fy, cx, cy = parameters
        matrix = np.asarray(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Camera intrinsics must contain [fx, fy, cx, cy], got {parameters}")
    if not np.all(np.isfinite(matrix)) or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError("Camera matrix is singular or contains non-finite values")
    return matrix


def _homogeneous_rotation(rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform
