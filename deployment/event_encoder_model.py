"""Copyable inference module. Dependencies: torch and the pinned official DINOv3 source.

Input to forward: normalized [B,20,240,320]. prepare_counts accepts already
rectified, temporally binned VGA counts and applies the training resize/normalization.
This module does not turn raw event packets into voxels.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

DINO_REPOSITORY = 'facebookresearch/dinov3:adc254450203739c8149213a7a69d8d905b4fcfa'


class PortableEventEncoder(nn.Module):
    def __init__(self, backbone, metadata):
        super().__init__()
        self.backbone = backbone
        self.metadata = metadata
        self.input_shape = tuple(metadata['input_shape'][1:])
        dataset = metadata['dataset']
        self.content_shape = (dataset.get('content_height', self.input_shape[1]), dataset.get('content_width', self.input_shape[2]))
        self.sensor_shape = (dataset['sensor_height'], dataset['sensor_width'])
        stats = dataset['representation']['fixed_normalization']
        mean, std = torch.tensor(stats['mean']), torch.tensor(stats['std'])
        if len(mean) != self.input_shape[0] or len(std) != len(mean):
            raise ValueError('Normalization channel mismatch')
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError('Invalid normalization')
        self.register_buffer('mean', mean.reshape(1, -1, 1, 1))
        self.register_buffer('std', std.reshape(1, -1, 1, 1))

    def prepare_counts(self, counts):
        """VGA linear voxels -> resized, normalized input; call exactly once."""
        if counts.ndim != 4 or tuple(counts.shape[1:]) != (self.input_shape[0], *self.sensor_shape):
            raise ValueError('Expected unnormalized sensor-resolution voxel counts')
        if self.sensor_shape[1] * self.content_shape[0] != self.sensor_shape[0] * self.content_shape[1]:
            raise ValueError('This portable recipe requires equal source/target aspect ratios')
        if not torch.isfinite(counts).all() or (counts < 0).any():
            raise ValueError('Counts must be finite and nonnegative')
        if self.metadata['dataset'].get('event_resize_mode', 'bilinear') == 'sum_pool':
            h, w = self.sensor_shape
            ch, cw = self.content_shape
            if h % ch or w % cw:
                raise ValueError('Histogram requires integer spatial scale')
            factor = (h // ch, w // cw)
            resized = F.avg_pool2d(counts.float(), factor, stride=factor) * (factor[0] * factor[1])
        else:
            resized = F.interpolate(counts.float(), size=self.content_shape, mode='bilinear', align_corners=False)
        return self.prepare_content(resized)

    def prepare_content(self, counts):
        """Unnormalized model-resolution counts -> padding + normalization."""
        if counts.ndim != 4 or tuple(counts.shape[1:]) != (self.input_shape[0], *self.content_shape):
            raise ValueError('Content tensor shape mismatch')
        if not torch.isfinite(counts).all() or (counts < 0).any():
            raise ValueError('Counts must be finite and nonnegative')
        resized = counts.float()
        dh, dw = self.input_shape[1] - self.content_shape[0], self.input_shape[2] - self.content_shape[1]
        if dh < 0 or dw < 0:
            raise ValueError('Content larger than padded input')
        resized = F.pad(resized, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))
        return (resized - self.mean) / self.std

    def forward(self, normalized_events):
        if normalized_events.ndim != 4 or tuple(normalized_events.shape[1:]) != self.input_shape:
            raise ValueError(f'Expected [B,{self.input_shape}] normalized events')
        tokens = self.backbone.forward_features(normalized_events)['x_norm_patchtokens']
        expected = (normalized_events.shape[0], (self.input_shape[1] // 16) * (self.input_shape[2] // 16), 384)
        if tuple(tokens.shape) != expected:
            raise ValueError(f'Unexpected feature shape: {tuple(tokens.shape)}')
        return tokens

    def feature_map(self, normalized_events):
        tokens = self(normalized_events)
        return tokens.transpose(1, 2).reshape(-1, 384, self.input_shape[1] // 16, self.input_shape[2] // 16)


def load_encoder(weights, *, device='cpu', dinov3_repo=None, backbone_factory=None):
    """No teacher weights needed. local dinov3_repo avoids network access.

    Use the pinned revision above for a local checkout. backbone_factory is a
    zero-argument alternative for applications managing the architecture themselves.
    """
    bundle = torch.load(Path(weights), map_location='cpu', weights_only=True)
    meta = bundle['metadata']
    if meta['format'] != 'eventstate_encoder_bundle_v1' or meta['architecture']['architecture'] != 'dinov3_vits16':
        raise ValueError('Unsupported encoder bundle')
    if backbone_factory is not None:
        backbone = backbone_factory()
    else:
        backbone = torch.hub.load(str(dinov3_repo) if dinov3_repo else DINO_REPOSITORY,
                                  'dinov3_vits16', source='local' if dinov3_repo else 'github',
                                  pretrained=False)
    old = backbone.patch_embed.proj
    channels = meta['input_shape'][1]
    if not isinstance(old, nn.Conv2d) or old.groups != 1:
        raise ValueError('Unsupported patch projection')
    backbone.patch_embed.proj = nn.Conv2d(channels, old.out_channels, old.kernel_size,
        old.stride, old.padding, old.dilation, bias=old.bias is not None,
        padding_mode=old.padding_mode)
    if hasattr(backbone.patch_embed, 'in_chans'):
        backbone.patch_embed.in_chans = channels
    backbone.load_state_dict(bundle['state_dict'], strict=True)
    return PortableEventEncoder(backbone, meta).to(device).eval()
