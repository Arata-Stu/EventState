"""Activity-only ablation of the existing cosine + normalized squared L2 loss."""

import torch
import torch.nn.functional as F

from .distillation import DistillationLoss, _validate_pair


class ActivityDistillationLoss(DistillationLoss):
    """Keep baseline distances/reduction; allow an empty selected branch."""

    def components(self, prediction, target, mask=None):
        if mask is None:
            return super().components(prediction, target)
        _validate_pair(prediction, target)
        target = target.detach() if self.detach_target else target
        weights = torch.broadcast_to(mask, prediction.shape[:-1]).to(prediction.dtype)
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("Activity weights must be finite and non-negative")
        p = F.normalize(prediction, dim=-1, eps=self.eps)
        q = F.normalize(target, dim=-1, eps=self.eps)
        total = weights.sum()
        denominator = torch.where(total > 0, total, torch.ones_like(total))
        cosine = ((1 - (p * q).sum(-1)) * weights).sum() / denominator
        mse = ((p - q).square().sum(-1) * weights).sum() / denominator
        return {"loss": self.cosine_weight * cosine + self.mse_weight * mse,
                "cosine": cosine, "mse": mse}
