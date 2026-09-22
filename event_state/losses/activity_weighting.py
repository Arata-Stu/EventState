"""Optional event-only spatial weights for supervised downstream losses."""

import math

import torch
from torch import Tensor

ACTIVITY_FORMAT = "scale_event_uint8_two_resize_v1"


def validate_activity_weights(active: float, inactive: float) -> None:
    if not all(math.isfinite(v) and v >= 0 for v in (active, inactive)):
        raise ValueError("Activity loss weights must be finite and non-negative")
    if active + inactive <= 0:
        raise ValueError("At least one activity loss weight must be positive")


def spatial_activity_weights(activity: Tensor, active: float, inactive: float) -> Tensor:
    validate_activity_weights(active, inactive)
    if activity.dtype != torch.bool:
        raise ValueError("Expected a bool activity mask")
    return torch.where(activity, float(active), float(inactive)).float()


def require_activity(payload: dict, grid_size: tuple[int, int]) -> Tensor:
    value = payload.get("event_activity")
    if (payload.get("activity_format") != ACTIVITY_FORMAT or not isinstance(value, Tensor)
            or value.dtype != torch.bool or tuple(value.shape) != tuple(grid_size)):
        raise ValueError("Activity cache missing/incompatible; re-export with --include-activity")
    return value
