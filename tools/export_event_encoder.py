#!/usr/bin/env python3
"""Export an EventState encoder with its input contract, without optimizer/LSTM/head."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New .pt file')
    a = p.parse_args()
    metadata_path = a.output.with_suffix('.json')
    loader_path = a.output.with_name(a.output.stem + '_model.py')
    if a.output.exists() or metadata_path.exists() or loader_path.exists():
        raise FileExistsError('Use new export paths')
    import torch
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    prefix = 'event_encoder.backbone.'
    state = {k[len(prefix):]: v for k, v in ck['model'].items() if k.startswith(prefix)}
    if not state or 'patch_embed.proj.weight' not in state:
        raise ValueError('Checkpoint has no DINO event encoder')
    config = ck['config']
    dataset = config['dataset']
    channels = config['model']['event_encoder']['in_channels']
    if state['patch_embed.proj.weight'].shape[1] != channels:
        raise ValueError('Patch projection/config channel mismatch')
    if not all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        raise ValueError('Nonfinite encoder weights')
    meta = dict(format='eventstate_encoder_bundle_v1', global_step=ck['global_step'],
        source_checkpoint=str(a.checkpoint.resolve()),
        architecture=config['model']['event_encoder'], teacher=config['teacher'],
        runtime=dict(loader='event_encoder_model.py',
                     dinov3_repository='facebookresearch/dinov3:adc254450203739c8149213a7a69d8d905b4fcfa',
                     torch_version=str(torch.__version__)),
        input_shape=[1, channels, dataset['input_height'], dataset['input_width']],
        dataset=dataset,
        preprocessing='EventState event representation, paired crop/bilinear resize, optional symmetric zero padding in raw space, then fixed normalization; no automatic kart compatibility',
        output='DINOv3 forward_features: x_norm_patchtokens; no EventState projector',
        state_dict_target='DINOv3 backbone with matching input-channel patch projection')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix(a.output.suffix + '.tmp')
    try:
        torch.save({'state_dict': state, 'metadata': meta}, tmp)
        reread = torch.load(tmp, map_location='cpu', weights_only=True)
        if set(reread['state_dict']) != set(state) or any(
            not torch.equal(v, reread['state_dict'][k]) for k, v in state.items()):
            raise ValueError('Export round-trip mismatch')
        tmp.replace(a.output)
    finally:
        tmp.unlink(missing_ok=True)
    meta['export_sha256'] = hashlib.sha256(a.output.read_bytes()).hexdigest()
    with metadata_path.open('x') as f:
        json.dump(meta, f, indent=2)
        f.write('\n')
    shutil.copyfile(Path(__file__).resolve().parents[1] / 'deployment/event_encoder_model.py', loader_path)
    print(f'[export] {a.output}\n[contract] {metadata_path}')


if __name__ == '__main__':
    main()
