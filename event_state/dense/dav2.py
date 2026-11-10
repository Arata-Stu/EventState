"""VKITTI ViT-S metric DPT decoder for four EventState stride-16 feature maps."""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(F.relu(x))))


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.resConfUnit1 = ResidualUnit()
        self.resConfUnit2 = ResidualUnit()
        self.out_conv = nn.Conv2d(64, 64, 1)

    def forward(self, x, size, skip=None):
        if skip is not None:
            x = x + self.resConfUnit1(skip)
        return self.out_conv(F.interpolate(self.resConfUnit2(x), size=size,
                                            mode="bilinear", align_corners=True))


class MetricDPTSmall(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [48, 96, 192, 384]
        self.projects = nn.ModuleList(nn.Conv2d(384, c, 1) for c in channels)
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(48, 48, 4, stride=4), nn.ConvTranspose2d(96, 96, 2, stride=2),
            nn.Identity(), nn.Conv2d(384, 384, 3, stride=2, padding=1),
        ])
        self.scratch = nn.Module()
        for i, c in enumerate(channels, 1):
            setattr(self.scratch, f"layer{i}_rn", nn.Conv2d(c, 64, 3, padding=1, bias=False))
            setattr(self.scratch, f"refinenet{i}", Fusion())
        self.scratch.output_conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.scratch.output_conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1),
                                                  nn.ReLU(), nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def initialize_vkitti(self, path):
        weights = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict({k.removeprefix("depth_head."): v for k, v in weights.items()
                              if k.startswith("depth_head.")}, strict=True)

    def forward(self, features, size):
        if features.shape[1] != 4 * 384:
            raise ValueError("Expected four ViT-S maps, ordered by increasing block depth")
        maps = []
        for i, x in enumerate(features.chunk(4, dim=1)):
            x = self.resize_layers[i](self.projects[i](x))
            maps.append(getattr(self.scratch, f"layer{i+1}_rn")(x))
        x = maps[-1]
        for i in (3, 2, 1, 0):
            target = maps[i-1].shape[-2:] if i else tuple(2*n for n in maps[0].shape[-2:])
            x = getattr(self.scratch, f"refinenet{i+1}")(x, target, maps[i] if i < 3 else None)
        x = F.interpolate(self.scratch.output_conv1(x), size=size, mode="bilinear", align_corners=True)
        return 80.0 * self.scratch.output_conv2(x)[:, 0]
