from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from event_state.models import EventEncoder, FrozenDinoV3Teacher, load_dinov3_backbone
from event_state.models.temporal import PatchwiseLSTM


class _FakePatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_chans = 3
        self.proj = nn.Conv2d(3, 384, kernel_size=16, stride=16)


class _FakeDinoV3(nn.Module):
    embed_dim = 384
    patch_size = 16

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = _FakePatchEmbed()
        self.norm = nn.LayerNorm(384)

    def forward_features(self, inputs: Tensor) -> dict[str, Tensor]:
        patches = self.patch_embed.proj(inputs).flatten(2).transpose(1, 2)
        return {"x_norm_patchtokens": self.norm(patches)}


def test_dinov3_loader_passes_file_uri_as_plain_local_path(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    checkpoint = tmp_path / "weights with spaces.pt"
    checkpoint.write_bytes(b"checkpoint")
    received: dict[str, Any] = {}

    def fake_hub_load(**kwargs: Any) -> nn.Module:
        received.update(kwargs)
        return _FakeDinoV3()

    monkeypatch.setattr(torch.hub, "load", fake_hub_load)
    load_dinov3_backbone(
        checkpoint=checkpoint.as_uri(),
        source="torch_hub",
    )

    assert received["weights"] == str(checkpoint.resolve())


def test_dinov3_patch_shape_at_blueprint_resolution() -> None:
    encoder = EventEncoder(
        backbone=_FakeDinoV3(),
        in_channels=3,
        patch_init="pretrained",
    )
    tokens = encoder(torch.randn(1, 3, 448, 640))
    assert tokens.shape == (1, 1120, 384)


def test_event_encoder_adapts_only_patch_projection_for_twenty_channels() -> None:
    backbone = _FakeDinoV3()
    norm_weight = backbone.norm.weight.detach().clone()
    encoder = EventEncoder(
        backbone=backbone,
        in_channels=20,
        patch_init="random",
    )
    assert encoder.backbone.patch_embed.proj.in_channels == 20
    assert torch.equal(encoder.backbone.norm.weight, norm_weight)
    tokens = encoder(torch.randn(2, 20, 32, 48))
    assert tokens.shape == (2, 6, 384)


def test_event_encoder_rejects_non_patch_aligned_resolution() -> None:
    encoder = EventEncoder(
        backbone=_FakeDinoV3(),
        in_channels=3,
        patch_init="pretrained",
    )
    try:
        encoder(torch.randn(1, 3, 31, 32))
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("Non-divisible input resolution was accepted")


def test_teacher_is_frozen_stays_in_eval_and_returns_patch_tokens() -> None:
    teacher = FrozenDinoV3Teacher(backbone=_FakeDinoV3())
    teacher.train(True)
    tokens = teacher(torch.rand(2, 3, 32, 48))
    assert tokens.shape == (2, 6, 384)
    assert not teacher.training
    assert not tokens.requires_grad
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_patchwise_lstm_shape_and_streaming_state() -> None:
    torch.manual_seed(0)
    temporal = PatchwiseLSTM(
        input_dim=8,
        hidden_dim=6,
        output_dim=8,
        num_layers=2,
        dropout=0.0,
    ).eval()
    z = torch.randn(2, 4, 3, 8)

    full, _ = temporal(z)
    first, state = temporal(z[:, :2])
    second, _ = temporal(z[:, 2:], state)
    reset, _ = temporal(z[:, :2], None)

    assert full.shape == z.shape
    assert torch.allclose(torch.cat((first, second), dim=1), full, atol=1e-6)
    assert torch.allclose(reset, first, atol=1e-6)
