"""Main event-state representation model."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from .event_encoder import EventEncoder
from .projectors import TeacherProjection
from .temporal import TemporalBackbone, TemporalState


class EventStateModel(nn.Module):
    """Event encoder, temporal state model, and disposable teacher projectors.

    The frozen RGB teacher intentionally is not a child of this module.
    """

    def __init__(
        self,
        event_encoder: EventEncoder,
        temporal_model: TemporalBackbone,
        h_projector: nn.Module | None = None,
        z_projector: nn.Module | None = None,
        *,
        teacher_dim: int = 384,
        projector_hidden_dim: int = 768,
    ) -> None:
        super().__init__()
        if not isinstance(event_encoder, nn.Module):
            raise TypeError("event_encoder must be an nn.Module")
        if not isinstance(temporal_model, nn.Module):
            raise TypeError("temporal_model must be an nn.Module")
        event_dim = getattr(event_encoder, "embed_dim", None)
        temporal_input_dim = getattr(temporal_model, "input_dim", None)
        temporal_output_dim = getattr(temporal_model, "output_dim", None)
        if not isinstance(event_dim, int) or event_dim <= 0:
            raise ValueError("event_encoder must expose a positive integer embed_dim")
        if temporal_input_dim != event_dim:
            raise ValueError(
                f"temporal input_dim {temporal_input_dim!r} must match event embed_dim {event_dim}"
            )
        if not isinstance(temporal_output_dim, int) or temporal_output_dim <= 0:
            raise ValueError("temporal_model must expose a positive integer output_dim")
        if teacher_dim <= 0 or projector_hidden_dim <= 0:
            raise ValueError("teacher_dim and projector_hidden_dim must be positive")

        self.event_encoder = event_encoder
        self.temporal_model = temporal_model
        self.h_projector = (
            h_projector
            if h_projector is not None
            else TeacherProjection(
                input_dim=temporal_output_dim,
                hidden_dim=projector_hidden_dim,
                output_dim=teacher_dim,
            )
        )
        self.z_projector = (
            z_projector
            if z_projector is not None
            else TeacherProjection(
                input_dim=event_dim,
                hidden_dim=projector_hidden_dim,
                output_dim=teacher_dim,
            )
        )
        if not isinstance(self.h_projector, nn.Module) or not isinstance(
            self.z_projector, nn.Module
        ):
            raise TypeError("h_projector and z_projector must be nn.Module instances")
        for name, projector, expected_input_dim in (
            ("h_projector", self.h_projector, temporal_output_dim),
            ("z_projector", self.z_projector, event_dim),
        ):
            projector_input_dim = getattr(projector, "input_dim", expected_input_dim)
            if projector_input_dim != expected_input_dim:
                raise ValueError(
                    f"{name}.input_dim {projector_input_dim!r} must be {expected_input_dim}"
                )

        self.event_dim = event_dim
        self.temporal_dim = temporal_output_dim
        self.teacher_dim = int(teacher_dim)

    def encode_event(self, event: Tensor) -> Tensor:
        """Encode one frame from ``[B, C, H, W]`` to ``[B, N, D]``."""

        z = self.event_encoder(event)
        if not isinstance(z, Tensor) or z.ndim != 3:
            raise ValueError("event_encoder must return z with shape [B, N, D]")
        if z.shape[-1] != self.event_dim:
            raise ValueError(
                f"event_encoder returned feature dim {z.shape[-1]}, expected {self.event_dim}"
            )
        return z

    def update_state(
        self,
        z: Tensor,
        state: TemporalState = None,
    ) -> tuple[Tensor, TemporalState]:
        """Update recurrent state for one ``[B, N, D]`` frame."""

        if not isinstance(z, Tensor) or z.ndim != 3:
            shape = tuple(z.shape) if isinstance(z, Tensor) else type(z).__name__
            raise ValueError(f"z must have shape [B, N, D], got {shape}")
        if callable(getattr(self.temporal_model, "forward_step", None)):
            h, new_state = self.temporal_model.forward_step(z, state)
        else:  # A small compatibility path for user-supplied temporal modules.
            h_sequence, new_state = self.temporal_model(z.unsqueeze(1), state)
            h = h_sequence[:, 0]
        if not isinstance(h, Tensor) or h.ndim != 3 or h.shape[:2] != z.shape[:2]:
            raise ValueError("temporal_model must preserve [B, N] in step mode")
        if h.shape[-1] != self.temporal_dim:
            raise ValueError(
                f"temporal_model returned feature dim {h.shape[-1]}, "
                f"expected {self.temporal_dim}"
            )
        return h, new_state

    def forward_sequence(
        self,
        events: Tensor,
        state: TemporalState = None,
    ) -> dict[str, Any]:
        """Encode and update a complete ``[B, T, C, H, W]`` clip."""

        if not isinstance(events, Tensor) or events.ndim != 5:
            shape = tuple(events.shape) if isinstance(events, Tensor) else type(events).__name__
            raise ValueError(f"events must have shape [B, T, C, H, W], got {shape}")
        batch_size, sequence_length, channels, height, width = events.shape
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("events batch and sequence dimensions must be non-empty")
        flat_events = events.reshape(
            batch_size * sequence_length, channels, height, width
        )
        flat_z = self.encode_event(flat_events)
        z = flat_z.reshape(
            batch_size, sequence_length, flat_z.shape[1], flat_z.shape[2]
        )
        h, new_state = self.temporal_model(z, state)
        if not isinstance(h, Tensor) or h.ndim != 4:
            raise ValueError("temporal_model must return h with shape [B, T, N, D]")
        if h.shape[:3] != z.shape[:3]:
            raise ValueError(
                f"temporal_model changed [B, T, N] from {tuple(z.shape[:3])} "
                f"to {tuple(h.shape[:3])}"
            )
        if h.shape[-1] != self.temporal_dim:
            raise ValueError(
                f"temporal_model returned feature dim {h.shape[-1]}, "
                f"expected {self.temporal_dim}"
            )
        return {"z": z, "h": h, "state": new_state}

    def project_h(self, h: Tensor) -> Tensor:
        projected = self.h_projector(h)
        if not isinstance(projected, Tensor) or projected.shape[:-1] != h.shape[:-1]:
            raise ValueError("h_projector must preserve all leading token dimensions")
        return projected

    def project_z(self, z: Tensor) -> Tensor:
        projected = self.z_projector(z)
        if not isinstance(projected, Tensor) or projected.shape[:-1] != z.shape[:-1]:
            raise ValueError("z_projector must preserve all leading token dimensions")
        return projected

    def forward(
        self,
        events: Tensor,
        state: TemporalState = None,
    ) -> dict[str, Any]:
        return self.forward_sequence(events, state)


__all__ = ["EventStateModel"]
