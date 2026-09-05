from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from event_state.data.calibration import load_dsec_camera_alignment


def test_identity_calibration_produces_identity_homography(tmp_path: Path) -> None:
    calibration_path = tmp_path / "cam_to_cam.yaml"
    payload = {
        "intrinsics": {
            "camRect0": {"camera_matrix": [100.0, 100.0, 2.0, 2.0], "resolution": [5, 4]},
            "camRect1": {"camera_matrix": [100.0, 100.0, 2.0, 2.0]},
        },
        "extrinsics": {
            "T_10": np.eye(4).tolist(),
            "R_rect0": np.eye(3).tolist(),
            "R_rect1": np.eye(3).tolist(),
        },
    }
    calibration_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    alignment = load_dsec_camera_alignment(calibration_path)

    assert (alignment.width, alignment.height) == (5, 4)
    assert np.allclose(alignment.image_to_event_homography, np.eye(3))

