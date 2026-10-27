"""Official DSEC disparity and prepared M3ED LiDAR depth targets."""
from pathlib import Path

import numpy as np
from PIL import Image


def disparity_to_depth(disparity, q):
    """Homogeneous reprojection using the event stereo pair's calibrated Q.

    Do not assume identical principal points or substitute one global f*b.
    """
    d = np.asarray(disparity, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4, 4) or not np.isfinite(q).all():
        raise ValueError("Expected a finite 4x4 event disparity_to_depth matrix")
    y, x = np.indices(d.shape)
    numerator = q[2, 0] * x + q[2, 1] * y + q[2, 2] * d + q[2, 3]
    denominator = q[3, 0] * x + q[3, 1] * y + q[3, 2] * d + q[3, 3]
    result = np.full(d.shape, np.nan, dtype=np.float32)
    np.divide(numerator, denominator, out=result, where=(d > 0) & (denominator != 0))
    return result


def read_target(record, task):
    path = Path(record["target_path"])
    if record["encoding"] == "dsec_disparity_uint16_x256":
        with Image.open(path) as image:
            if image.mode not in {"I", "I;16", "I;16L", "I;16B"}:
                raise ValueError(f"Disparity must use 16-bit PNG encoding: {path}")
            raw = np.asarray(image)
        if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f"Not a single-channel integer disparity PNG: {path}")
        value = raw.astype(np.float32) / 256.0
        valid = raw > 0
        if task == "depth":
            value = disparity_to_depth(value, record["disparity_to_depth_q"])
    elif record["encoding"] == "m3ed_prepared_depth_m":
        import h5py
        with h5py.File(path, "r") as stream:
            row = record["target_row"]
            if int(stream["timestamps"][row]) != record["timestamp"]:
                raise ValueError("M3ED target timestamp changed")
            value = np.asarray(stream["depth_m"][row], dtype=np.float32)
            valid = np.asarray(stream["depth_valid"][row], dtype=bool)
    else:
        raise ValueError("Unknown official target encoding")
    valid &= np.isfinite(value) & (value > 0)
    return value, valid
