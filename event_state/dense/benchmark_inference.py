"""Load an already evaluated frozen head/backbone pair; no test-time fitting."""
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from event_state.training import build_model, load_checkpoint, load_checkpoint_config_metadata
from event_state.models.tokens import tokens_to_map
from .protocol import digest, load_protocol, sha256_file, audit_pretraining
from .head import DenseRegressionHead


class DepthPredictor:
    def __init__(self, run, name, feature, device, teacher_weights=None):
        ds = name.split('_')[0]
        protocol = load_protocol(ds + '_f3')
        root = Path(run)
        backbone = root/'pretrain'/name/'checkpoints/step_00100000.pt'
        head = root/'heads'/name/feature/'best.pt'
        report = json.loads((head.parent/'continuous.json').read_text())
        self.identity = {'head_sha256': sha256_file(head), 'backbone_sha256': sha256_file(backbone),
                         'feature': feature, 'task': 'depth', 'role': 'val', 'state_policy': 'continuous',
                         'protocol_sha256': digest(protocol)}
        if any(report.get(k) != v for k, v in self.identity.items()):
            raise ValueError('Weights or protocol differ from evaluated public validation artifacts')
        hp = torch.load(head, map_location='cpu', weights_only=False)
        c = hp['config']
        if any(c[k] != self.identity[k] for k in ('feature','task','protocol_sha256','backbone_sha256')):
            raise ValueError('Head configuration mismatch')
        raw = load_checkpoint_config_metadata(backbone).config
        self.pretraining_audit = audit_pretraining(raw, protocol, statistics_sequences=c['normalization_audit']['statistics_sequences'])
        cfg = OmegaConf.create(raw)
        if teacher_weights:
            cfg.teacher.checkpoint = str(teacher_weights)
            cfg.model.event_encoder.checkpoint = str(teacher_weights)
        if cfg.dataset.representation.type != 'gep_rgb':
            raise ValueError('Requires the frozen GEP RGB model')
        self.cfg = cfg
        self.model = build_model(cfg, device=device)
        load_checkpoint(backbone, model=self.model, device=device, restore_rng=False)
        self.model.eval()
        self.head = DenseRegressionHead(c['in_channels'], c['width']).to(device).eval()
        self.head.load_state_dict(hp['model'], strict=True)
        self.feature, self.device, self.state = feature, device, None
        self.mean, self.std = [torch.tensor(list(v), device=device).view(3,1,1) for v in
                              (cfg.dataset.representation.normalize_mean, cfg.dataset.representation.normalize_std)]

    @torch.inference_mode()
    def step(self, gep, predict=True):
        height, width = gep.shape[-2:]
        if height % 16 or width % 16:
            raise ValueError('Unaligned EventState patch grid')
        out = self.model(((gep.to(self.device)-self.mean)/self.std)[None,None], state=self.state)
        self.state = out['state']
        if predict:
            maps = [tokens_to_map(out[k][0,0], (height//16,width//16)) for k in
                    (('z','h') if self.feature == 'concat' else (self.feature,))]
            # Cache-based val heads were trained/evaluated on FP16 maps cast to FP32.
            pred = self.head(torch.cat(maps).half().float()[None], (height,width))[0]
            if not torch.isfinite(pred).all() or (pred <= 0).any():
                raise ValueError('Invalid depth prediction')
            return pred
