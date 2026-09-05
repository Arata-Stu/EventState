"""GEP-style direct z-to-DINO objective."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from .distillation import DistillationLoss
from .z_none import ZObjective


class DirectDINOObjective(ZObjective):
    """Apply dense DINO distillation to an already projected z tensor."""

    def __init__(
        self,
        *,
        cosine_weight: float = 1.0,
        mse_weight: float = 1.0,
        weight: float = 1.0,
        eps: float = 1e-6,
        detach_target: bool = True,
    ) -> None:
        super().__init__()
        if weight < 0:
            raise ValueError("weight must be non-negative")
        self.weight = float(weight)
        self.distillation = DistillationLoss(
            cosine_weight=cosine_weight,
            mse_weight=mse_weight,
            eps=eps,
            detach_target=detach_target,
        )

    def components(
        self,
        z: Tensor,
        teacher_features: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        components = dict(self.distillation.components(z, teacher_features, mask))
        components["unweighted_loss"] = components["loss"]
        components["loss"] = self.weight * components["loss"]
        return components

    def forward(
        self,
        z: Tensor,
        teacher_features: Tensor | None = None,
        *,
        events: Tensor | None = None,
        metadata: Any = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        del events, metadata
        if teacher_features is None:
            raise ValueError("DirectDINOObjective requires teacher_features")
        return self.components(z, teacher_features, mask)["loss"]


DirectDinoObjective = DirectDINOObjective
DirectDinoZObjective = DirectDINOObjective


def build_z_objective(kind: str, **kwargs: object) -> ZObjective:
    normalized_kind = str(kind).lower()
    if normalized_kind in {"none", "disabled"}:
        from .z_none import NoZObjective

        return NoZObjective()
    if normalized_kind in {"direct_dino", "dino", "distill"}:
        return DirectDINOObjective(**kwargs)
    if normalized_kind in {"future", "cmax"}:
        raise NotImplementedError(f"z objective {normalized_kind!r} is reserved for a later phase")
    raise ValueError("z objective kind must be 'none' or 'direct_dino'")


__all__ = [
    "DirectDINOObjective",
    "DirectDinoObjective",
    "DirectDinoZObjective",
    "build_z_objective",
]
