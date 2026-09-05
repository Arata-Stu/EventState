"""DINOv3 backbone loading shared by the teacher and event encoder.

The loader deliberately has no relationship with the temporary reference
checkout.  It can either import an installed DINOv3 package or ask
:mod:`torch.hub` to load a pinned repository revision.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

import torch
from torch import Tensor, nn

from event_state.data.cache_metadata import normalize_dinov3_checkpoint


DINO_V3_VITS16 = "dinov3_vits16"
DINO_V3_VITS16_EMBED_DIM = 384
DINO_V3_VITS16_PATCH_SIZE = 16
DINO_V3_DEFAULT_REPOSITORY = (
    "facebookresearch/dinov3:adc254450203739c8149213a7a69d8d905b4fcfa"
)

DinoV3Backend = Literal["package", "torch_hub"]
TorchHubSource = Literal["github", "local"]


def _pair(value: object, *, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        result = (value, value)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        result = (int(value[0]), int(value[1]))
    else:
        raise TypeError(f"{name} must be an int or a pair of ints, got {value!r}")
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _reject_disposable_reference_path(value: str | Path, *, name: str) -> None:
    """Prevent runtime artifacts from depending on the disposable checkout."""

    text = str(value)
    parsed = urlparse(text)
    if parsed.scheme in {"http", "https"}:
        return
    candidate = Path(unquote(parsed.path)) if parsed.scheme == "file" else Path(text)
    resolved = candidate.expanduser().resolve()
    reference_root = (Path(__file__).resolve().parents[2] / "reference_repo").resolve()
    try:
        resolved.relative_to(reference_root)
    except ValueError:
        return
    raise ValueError(
        f"{name} points inside the disposable reference checkout; "
        f"copy or clone it elsewhere first: {resolved}"
    )


def validate_dinov3_vits16(backbone: nn.Module) -> None:
    """Fail early when a loader returned an incompatible backbone."""

    if not callable(getattr(backbone, "forward_features", None)):
        raise TypeError("DINOv3 backbone must implement forward_features()")
    embed_dim = getattr(backbone, "embed_dim", None)
    if embed_dim != DINO_V3_VITS16_EMBED_DIM:
        raise ValueError(
            f"Expected DINOv3 ViT-S/16 embed_dim={DINO_V3_VITS16_EMBED_DIM}, "
            f"got {embed_dim!r}"
        )
    patch_size = _pair(getattr(backbone, "patch_size", None), name="backbone.patch_size")
    expected = (DINO_V3_VITS16_PATCH_SIZE, DINO_V3_VITS16_PATCH_SIZE)
    if patch_size != expected:
        raise ValueError(f"Expected DINOv3 ViT-S/16 patch size {expected}, got {patch_size}")
    patch_embed = getattr(backbone, "patch_embed", None)
    projection = getattr(patch_embed, "proj", None)
    if not isinstance(projection, nn.Conv2d):
        raise TypeError("DINOv3 backbone.patch_embed.proj must be nn.Conv2d")


def load_dinov3_backbone(
    *,
    backbone: str = DINO_V3_VITS16,
    checkpoint: str | Path | None = None,
    pretrained: bool = True,
    source: DinoV3Backend | str = "package",
    repository: str | Path = DINO_V3_DEFAULT_REPOSITORY,
    hub_source: TorchHubSource | str = "github",
) -> nn.Module:
    """Load the official DINOv3 ViT-S/16 architecture.

    Args:
        backbone: Official hub entry point.  Phase 1 supports only
            ``dinov3_vits16`` so accidental architecture drift is rejected.
        checkpoint: Optional official backbone checkpoint path or URL.  It is
            forwarded as the DINOv3 builder's ``weights`` argument.
        pretrained: Whether the official builder should load pretrained
            weights.  When true and ``checkpoint`` is omitted, the builder's
            official default weights are used.
        source: ``package`` imports ``dinov3.hub.backbones`` from an installed,
            pinned dependency; ``torch_hub`` loads ``repository``.
        repository: A torch-hub ``owner/repo[:ref]`` identifier or a separately
            managed local checkout.  The value is passed through unchanged.
        hub_source: ``github`` or ``local`` as understood by ``torch.hub.load``.
    """

    if backbone != DINO_V3_VITS16:
        raise ValueError(
            f"Unsupported DINOv3 backbone {backbone!r}; Phase 1 requires {DINO_V3_VITS16!r}"
        )
    if checkpoint is not None and not pretrained:
        raise ValueError("checkpoint cannot be supplied when pretrained=False")
    if checkpoint is not None:
        _reject_disposable_reference_path(checkpoint, name="checkpoint")

    backend = str(source).lower()
    weights = normalize_dinov3_checkpoint(checkpoint)
    builder_kwargs: dict[str, object] = {"pretrained": bool(pretrained)}
    if weights is not None:
        builder_kwargs["weights"] = weights

    if backend == "package":
        try:
            module = importlib.import_module("dinov3.hub.backbones")
        except ImportError as error:
            raise ImportError(
                "The package DINOv3 backend was requested, but `dinov3` is not installed. "
                "Install the pinned DINOv3 dependency or select source='torch_hub'."
            ) from error
        module_path = getattr(module, "__file__", None)
        if module_path is None:
            raise ImportError("Installed dinov3 package does not expose its source path")
        _reject_disposable_reference_path(module_path, name="installed DINOv3 package")
        builder = getattr(module, backbone, None)
        if not callable(builder):
            raise ImportError(f"Installed dinov3 package has no callable {backbone!r}")
        model = builder(**builder_kwargs)
    elif backend == "torch_hub":
        resolved_hub_source = str(hub_source).lower()
        if resolved_hub_source not in {"github", "local"}:
            raise ValueError("hub_source must be 'github' or 'local'")
        if resolved_hub_source == "local":
            _reject_disposable_reference_path(repository, name="DINOv3 repository")
        hub_kwargs = dict(builder_kwargs)
        if resolved_hub_source == "github":
            # Torch Hub validates only current branch/tag tips. This project
            # intentionally pins a known historical commit for reproducibility.
            hub_kwargs["skip_validation"] = True
        model = torch.hub.load(
            repo_or_dir=str(repository),
            model=backbone,
            source=resolved_hub_source,
            trust_repo=True,
            **hub_kwargs,
        )
    else:
        raise ValueError("source must be 'package' or 'torch_hub'")

    if not isinstance(model, nn.Module):
        raise TypeError(f"DINOv3 loader returned {type(model).__name__}, not nn.Module")
    validate_dinov3_vits16(model)
    return model


def extract_normalized_patch_tokens(backbone: nn.Module, inputs: Tensor) -> Tensor:
    """Return final normalized patch tokens, excluding CLS/storage tokens."""

    features = backbone.forward_features(inputs)
    if not isinstance(features, dict):
        raise TypeError("DINOv3 forward_features() must return a dictionary")
    tokens = features.get("x_norm_patchtokens")
    if not isinstance(tokens, Tensor):
        raise KeyError("DINOv3 features do not contain tensor 'x_norm_patchtokens'")
    if tokens.ndim != 3:
        raise ValueError(
            "DINOv3 x_norm_patchtokens must have shape [B, N, D], "
            f"got {tuple(tokens.shape)}"
        )
    return tokens


__all__ = [
    "DINO_V3_DEFAULT_REPOSITORY",
    "DINO_V3_VITS16",
    "DINO_V3_VITS16_EMBED_DIM",
    "DINO_V3_VITS16_PATCH_SIZE",
    "DinoV3Backend",
    "TorchHubSource",
    "extract_normalized_patch_tokens",
    "load_dinov3_backbone",
    "validate_dinov3_vits16",
]
