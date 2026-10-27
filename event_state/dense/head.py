"""Small, non-recurrent frozen probe; the decoder cannot supply its own memory."""
import torch.nn.functional as F
from torch import nn


class DenseRegressionHead(nn.Module):
    def __init__(self, in_channels, width=128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels, width, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(width, 1, 1))

    def forward(self, features, size):
        logits = self.net(features).float()
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        return F.softplus(logits[:, 0]) + 1e-6
