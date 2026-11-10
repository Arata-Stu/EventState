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


class FourLevelDepthHead(nn.Module):
    """DPT-style reassembly and coarse-to-fine residual fusion on four ViT maps."""
    def __init__(self, in_channels, width):
        super().__init__()
        if in_channels % 4:
            raise ValueError("Expected four equally sized patch feature maps")
        self.projections = nn.ModuleList(nn.Conv2d(in_channels // 4, width, 1) for _ in range(4))
        self.resizers = nn.ModuleList([
            nn.ConvTranspose2d(width, width, 4, stride=4),
            nn.ConvTranspose2d(width, width, 2, stride=2),
            nn.Identity(), nn.Conv2d(width, width, 3, stride=2, padding=1),
        ])
        self.refine = nn.ModuleList(nn.Sequential(
            nn.GELU(), nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(), nn.Conv2d(width, width, 3, padding=1),
        ) for _ in range(4))
        self.predict = nn.Conv2d(width, 1, 1)

    def forward(self, features, size):
        maps = features.chunk(4, dim=1)
        fused = None
        for i in reversed(range(4)):
            x = self.resizers[i](self.projections[i](maps[i]))
            if fused is not None:
                x = x + F.interpolate(fused, x.shape[-2:], mode="bilinear", align_corners=False)
            fused = x + self.refine[i](x)
        logits = F.interpolate(self.predict(fused).float(), size, mode="bilinear", align_corners=False)
        return F.softplus(logits[:, 0]) + 1e-6


def build_dense_head(feature, in_channels, width, decoder=None):
    if decoder == "dav2_vkitti_small":
        from .dav2 import MetricDPTSmall
        if feature != "stages" or in_channels != 1536:
            raise ValueError("DAv2 requires four ViT-S feature maps")
        return MetricDPTSmall()
    cls = FourLevelDepthHead if feature == "stages" else DenseRegressionHead
    return cls(in_channels, width)
