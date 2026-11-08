"""Frozen common Stage 1 checkpoint, read-only, continuous state at every input."""
import torch
from omegaconf import OmegaConf
from event_state.dense.common_protocol import audit_common
from event_state.training import build_model, load_checkpoint, load_checkpoint_config_metadata
from event_state.models.tokens import tokens_to_map


class CommonFeatures:
    def __init__(self, checkpoint, dataset, device):
        raw=load_checkpoint_config_metadata(checkpoint).config
        self.audit=audit_common(raw)
        self.cfg=OmegaConf.create(raw)
        ds=self.cfg.dataset
        if ds.name!=dataset or ds.representation.type!='gep_rgb' or ds.get('event_window_us') is not None or ds.get('event_window_fraction',1)!=1:
            raise ValueError('Expected audited common RGB-interval GEP checkpoint')
        self.device=device
        self.model=build_model(self.cfg,device=device)
        load_checkpoint(checkpoint,model=self.model,device=device,restore_rng=False)
        self.model.eval()
        self.mean,self.std=[torch.tensor(list(v),device=device)[:,None,None] for v in
                            (ds.representation.normalize_mean,ds.representation.normalize_std)]
        self.state=None

    @torch.inference_mode()
    def step(self,image,keys):
        outputs=self.model(((image.to(self.device)-self.mean)/self.std)[None,None],state=self.state)
        self.state=outputs['state']
        grid=tuple(v//16 for v in image.shape[-2:])
        return {k:tokens_to_map(outputs[k][0,0],grid).half().cpu().contiguous() for k in keys}
