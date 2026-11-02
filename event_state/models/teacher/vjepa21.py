"""Frozen official V-JEPA 2.1 ViT-B encoder with endpoint-only video targets.

The encoder is bidirectional *inside* each trailing window. Never use its
earlier tubelets as targets: only the last tubelet is causal at the endpoint.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import subprocess
import sys

import torch
from torch import Tensor, nn
from event_state.causal_rgb import validate_context


def verify_source(repository: str, revision: str, checkpoint: str, sha256: str) -> Path:
    root = Path(repository).expanduser().resolve()
    if len(revision) != 40 or len(sha256) != 64:
        raise ValueError("Explicit full source commit and checkpoint SHA256 are required")
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"], text=True
    )
    if actual != revision or dirty:
        raise ValueError("V-JEPA source must be a clean checkout of the specified revision")
    digest = hashlib.sha256()
    with Path(checkpoint).expanduser().open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        raise ValueError("V-JEPA checkpoint SHA256 mismatch")
    return root


def load_official_encoder(repository, revision, checkpoint, checkpoint_sha256):
    root = verify_source(repository, revision, checkpoint, checkpoint_sha256)
    # Official model imports app.* and src.*. Reject collisions rather than
    # silently running an unrelated package with the same generic name.
    for name, module in tuple(sys.modules.items()):
        if name in {"app", "src"} or name.startswith(("app.", "src.")):
            locations = list(getattr(module, "__path__", []))
            if getattr(module, "__file__", None):
                locations.append(module.__file__)
            if not locations or any(not Path(p).resolve().is_relative_to(root) for p in locations):
                raise RuntimeError(f"V-JEPA import namespace collision: {name}")
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("app.vjepa_2_1.models.vision_transformer")
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError("V-JEPA encoder resolved outside the pinned repository")
        # Match official src/hub/backbones.py; omit the training predictor.
        encoder = module.vit_base(
            img_size=(384, 384), patch_size=16, num_frames=64, tubelet_size=2,
            use_sdpa=True, use_rope=True, uniform_power=False,
            img_temporal_dim_size=1, interpolate_rope=True,
        )
    finally:
        sys.path.pop(0)
    payload = torch.load(Path(checkpoint).expanduser(), map_location="cpu", weights_only=True)
    weights = payload["ema_encoder"]
    cleaned = {}
    for name, value in weights.items():
        key = name.replace("module.", "").replace("backbone.", "")
        if key in cleaned:
            raise ValueError(f"Duplicate encoder state key: {key}")
        cleaned[key] = value
    encoder.load_state_dict(cleaned, strict=True)
    return encoder


class FrozenVJEPA21Teacher(nn.Module):
    sequence_teacher = True
    embed_dim = 768
    patch_size = 16

    def __init__(self, *, repository=None, revision=None, checkpoint=None,
                 checkpoint_sha256=None, context_frames=8, micro_batch_size=1,
                 image_mean=(0.485, 0.456, 0.406), image_std=(0.229, 0.224, 0.225),
                 backbone=None, input_mode="video", sample_fps=None):
        super().__init__()
        validate_context(input_mode, context_frames, sample_fps)
        if micro_batch_size < 1:
            raise ValueError("Positive micro_batch_size required")
        self.input_mode = input_mode
        self.sample_fps = sample_fps
        self.context_frames = int(context_frames)
        self.micro_batch_size = int(micro_batch_size)
        self.backbone = backbone if backbone is not None else load_official_encoder(
            repository, revision, checkpoint, checkpoint_sha256
        )
        if getattr(self.backbone, "embed_dim", None) != self.embed_dim:
            raise ValueError("Expected ViT-B encoder features with 768 channels")
        self.register_buffer("mean", torch.tensor(image_mean).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor(image_std).view(1, 3, 1, 1, 1))
        self.requires_grad_(False)
        self.eval()
        self._reported_contract = False

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward_history(self, images: Tensor, history_timestamps: Tensor,
                        target_timestamps: Tensor, *, indices=None, image_counts=None,
                        requested_timestamps=None, prefix_padding=None,
                        sequence_start_timestamps=None) -> Tensor:
        """CPU unique frames [B,L,3,H,W] + [B,T,K] indices -> [B,T,N,768].

        Overlapping windows are materialized one micro-batch at a time so the
        entire B*T*K video expansion is never resident in GPU memory.
        """
        k = self.context_frames
        if images.ndim != 5 or images.shape[2] != 3 or target_timestamps.ndim != 2:
            raise ValueError("Invalid teacher history/image shape")
        b, t = target_timestamps.shape
        times = history_timestamps.detach().cpu()
        targets = target_timestamps.detach().cpu()
        if indices is None:
            # Compatibility with the original consecutive-frame smoke adapter.
            if self.sample_fps is not None:
                raise ValueError("Timestamp-sampled teacher requires explicit indices")
            if images.shape[:2] != (b, t + k - 1) or times.shape != images.shape[:2]:
                raise ValueError("Consecutive history length must be T+K-1")
            indices = (torch.arange(t)[:, None] + torch.arange(k)[None, :]).expand(b, -1, -1)
            times = torch.stack([times[:, j:j + k] for j in range(t)], dim=1)
        else:
            indices = indices.detach().cpu()
        if images.shape[0] != b or tuple(indices.shape) != (b, t, k) or times.shape != indices.shape:
            raise ValueError("Teacher indices/timestamps must have shape [B,T,K]")
        if indices.dtype != torch.int64 or (indices < 0).any() or (indices >= images.shape[1]).any():
            raise ValueError("Invalid teacher image index")
        if image_counts is not None and (indices >= image_counts.detach().cpu()[:, None, None]).any():
            raise ValueError("Teacher index refers to batch padding")
        if not torch.equal(times[:, :, -1], targets) or (times[:, :, 1:] < times[:, :, :-1]).any():
            raise ValueError("Noncausal or misaligned RGB history")
        if self.sample_fps is not None and any(x is None for x in
                (requested_timestamps, prefix_padding, sequence_start_timestamps)):
            raise ValueError("Sampled video requires timestamp and prefix provenance")
        if requested_timestamps is not None:
            if prefix_padding is None or sequence_start_timestamps is None:
                raise ValueError("Requested timestamps need prefix provenance")
            requested = requested_timestamps.detach().cpu()
            padded = prefix_padding.detach().cpu()
            first = sequence_start_timestamps.detach().cpu()[:, None, None]
            if requested.shape != times.shape or padded.shape != times.shape or padded.dtype != torch.bool:
                raise ValueError("Invalid requested timestamp/prefix shape")
            if (not torch.equal(requested[:, :, -1], targets)
                    or (requested[:, :, 1:] < requested[:, :, :-1]).any()
                    or (padded & ((requested >= first) | (times != first))).any()
                    or ((~padded) & (times > requested)).any()):
                raise ValueError("RGB exceeds requested sampling time outside sequence prefix padding")
        spans = times[:, :, -1] - times[:, :, 0]
        self.history_metrics = {"teacher_span_ms": spans.float().mean().item() / 1000}
        if k > 1:
            self.history_metrics["teacher_repeat_fraction"] = (times[:, :, 1:] == times[:, :, :-1]).float().mean().item()
        if requested_timestamps is not None:
            self.history_metrics["teacher_prefix_frame_fraction"] = padded.any(dim=-1).float().mean().item()
            lag = (requested - times)[~padded]
            self.history_metrics["teacher_sampling_lag_ms"] = lag.float().mean().item() / 1000 if lag.numel() else 0.0
        height, width = images.shape[-2:]
        if height % 16 or width % 16:
            raise ValueError("Teacher and student must share a patch-aligned spatial grid")
        n = (height // 16) * (width // 16)
        if not self._reported_contract:
            print(f"[vjepa21] mode={self.input_mode}; RGB K={k}; fps={self.sample_fps}; "
                  f"grid={height // 16}x{width // 16}; "
                  f"history span={spans.min().item() / 1000:.1f}..{spans.max().item() / 1000:.1f} ms; "
                  "teacher FP32; no feature cache", flush=True)
            self._reported_contract = True
        results = []
        device = self.mean.device
        with torch.autocast(device_type=device.type, enabled=False):
            for offset in range(0, b * t, self.micro_batch_size):
                windows = [images[i // t, indices[i // t, i % t]]
                           for i in range(offset, min(b * t, offset + self.micro_batch_size))]
                video = torch.stack(windows).to(device=device, dtype=torch.float32)
                video = video.permute(0, 2, 1, 3, 4).contiguous()
                tokens = self.backbone((video - self.mean) / self.std)
                expected = (len(windows), (1 if self.input_mode == "image" else k // 2) * n, self.embed_dim)
                if not isinstance(tokens, Tensor) or tuple(tokens.shape) != expected:
                    raise ValueError(f"Expected time-major dense video tokens {expected}")
                # Clone so retained targets don't hold the whole context storage.
                endpoint = tokens[:, -n:].float().clone()
                if not torch.isfinite(endpoint).all():
                    raise ValueError("Non-finite V-JEPA teacher features")
                results.append(endpoint)
        return torch.cat(results).reshape(b, t, n, self.embed_dim).detach()
