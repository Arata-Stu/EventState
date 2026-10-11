"""Server preflight for the portable 20ch encoder recipe (no downloads)."""
import json
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir


def test_recipe_and_geometry():
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / 'configs'), version_base=None):
        cfg = compose(config_name='config', overrides=[
            'dataset=dsec_det_train41_voxel20', 'model=no_memory',
            'experiment=voxel20_encoder_320'])
    assert cfg.teacher.frozen and not cfg.teacher.cache_features
    assert cfg.dataset.event_cache_dir is None
    assert cfg.model.event_encoder.in_channels == 20
    assert cfg.model.temporal.type == 'identity'
    assert cfg.loss.z_objective.type == 'direct_dino'
    assert not cfg.loss.h_distill.enabled
    assert cfg.dataset.representation.fixed_normalization_required
    assert cfg.training.sampling.mode == 'mixed'
    from event_state.data.transforms import PairedSequenceTransform
    transform = PairedSequenceTransform(height=240, width=320)
    event = torch.ones(1, 20, 480, 640)
    rgb = torch.ones(1, 3, 480, 640)
    e, r = transform(event, rgb)
    assert e.shape == (1, 20, 240, 320)
    assert r.shape == (1, 3, 240, 320)
    assert (e.shape[-1] // 16) * (e.shape[-2] // 16) == 300


def test_export_preserves_weights_and_contract(tmp_path, monkeypatch):
    from tools.export_event_encoder import main
    checkpoint, output = tmp_path / 'checkpoint.pt', tmp_path / 'encoder.pt'
    weight = torch.randn(4, 20, 16, 16)
    config = dict(model=dict(event_encoder=dict(in_channels=20, architecture='dinov3_vits16')),
                  teacher=dict(frozen=True), dataset=dict(input_height=240, input_width=320,
                  representation=dict(fixed_normalization=dict(mean=[0.] * 20, std=[1.] * 20))))
    torch.save(dict(model={'event_encoder.backbone.patch_embed.proj.weight': weight,
                           'z_projector.weight': torch.ones(4)}, config=config, global_step=100000), checkpoint)
    monkeypatch.setattr(sys, 'argv', ['export', '--checkpoint', str(checkpoint), '--output', str(output)])
    main()
    bundle = torch.load(output, weights_only=True)
    assert set(bundle['state_dict']) == {'patch_embed.proj.weight'}
    assert torch.equal(bundle['state_dict']['patch_embed.proj.weight'], weight)
    meta = json.loads(output.with_suffix('.json').read_text())
    assert meta['input_shape'] == [1, 20, 240, 320]
    assert meta['dataset']['representation'] == config['dataset']['representation']
    assert len(meta['export_sha256']) == 64
    with pytest.raises(FileExistsError):
        main()


@pytest.mark.parametrize('height,width,content_h,content_w', [(240,320,240,320), (128,160,120,160)])
def test_portable_loader_and_preprocessing(tmp_path, height, width, content_h, content_w):
    from copy import deepcopy
    from deployment.event_encoder_model import load_encoder
    from event_state.data.transforms import PairedSequenceTransform

    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = torch.nn.Module()
            self.patch_embed.proj = torch.nn.Conv2d(3, 384, 16, 16)

        def forward_features(self, value):
            return {'x_norm_patchtokens': self.patch_embed.proj(value).flatten(2).transpose(1, 2)}

    original = TinyBackbone()
    original.patch_embed.proj = torch.nn.Conv2d(20, 384, 16, 16)
    meta = dict(format='eventstate_encoder_bundle_v1', input_shape=[1, 20, height, width],
                architecture=dict(architecture='dinov3_vits16'), dataset=dict(
                    sensor_height=480, sensor_width=640, content_height=content_h, content_width=content_w, representation=dict(
                        fixed_normalization=dict(mean=[.1]*20, std=[2.]*20))))
    path = tmp_path / 'portable.pt'
    torch.save(dict(state_dict=original.state_dict(), metadata=meta), path)
    loaded = load_encoder(path, backbone_factory=TinyBackbone)
    counts = torch.rand(1, 20, 480, 640)
    expected, _ = PairedSequenceTransform(height=content_h, width=content_w,
        output_height=height, output_width=width, event_mean=(.1,)*20, event_std=(2.,)*20)(counts, None)
    prepared = loaded.prepare_counts(counts)
    torch.testing.assert_close(prepared, expected, rtol=0, atol=0)
    with torch.inference_mode():
        torch.testing.assert_close(loaded(prepared), original.forward_features(expected)['x_norm_patchtokens'], rtol=0, atol=0)
        assert loaded.feature_map(prepared).shape == (1, 384, height // 16, width // 16)
    with pytest.raises(ValueError):
        loaded.prepare_counts(prepared)
    broken = deepcopy(original.state_dict())
    broken.pop('patch_embed.proj.bias')
    torch.save(dict(state_dict=broken, metadata=meta), path)
    with pytest.raises(RuntimeError):
        load_encoder(path, backbone_factory=TinyBackbone)


def test_160_recipe_pads_both_modalities():
    from event_state.data.transforms import PairedSequenceTransform
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / 'configs'), version_base=None):
        cfg = compose(config_name='config', overrides=[
            'dataset=dsec_det_train41_voxel20', 'model=no_memory',
            'experiment=voxel20_encoder_160'])
    assert (cfg.dataset.content_height, cfg.dataset.content_width) == (120, 160)
    assert (cfg.dataset.input_height, cfg.dataset.input_width) == (128, 160)
    transform = PairedSequenceTransform(height=120, width=160, output_height=128,
                                       output_width=160, event_mean=(1.,)*20, event_std=(2.,)*20)
    events, rgb = transform(torch.ones(1,20,480,640), torch.ones(1,3,480,640))
    assert events.shape == (1,20,128,160)
    assert torch.all(events[:,:,:4] == -.5)
    assert torch.all(events[:,:,4:124] == 0)
    assert torch.all(events[:,:,124:] == -.5)
    assert torch.all(rgb[:,:,:4] == 0)
    assert torch.all(rgb[:,:,4:124] == 1)
    assert torch.all(rgb[:,:,124:] == 0)
