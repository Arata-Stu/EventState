"""torchrun support: rank-local streams and portable, unwrapped checkpoints."""
from __future__ import annotations

import os
from datetime import timedelta

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def active():
    return torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1


def setup():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1
    local = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available():
        raise RuntimeError("Production torchrun training requires CUDA; use synthetic Gloo tests on CPU")
    torch.cuda.set_device(local)
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=30))
    return torch.distributed.get_rank(), world


def detach(value):
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(detach(v) for v in value)
    if isinstance(value, list):
        return [detach(v) for v in value]
    return value


class ProjectedStudent(nn.Module):
    """Put all trainable paths inside DDP.forward, including the projectors.

    Only objective tensors retain their graph. Diagnostic/state outputs must not
    make unused-parameter detection think that disabled objectives are active.
    """
    def __init__(self, model, h_enabled, z_enabled):
        super().__init__()
        self.student = model
        self.h_enabled, self.z_enabled = h_enabled, z_enabled

    def forward(self, events, state=None):
        out = self.student.forward_sequence(events, state=state)
        z, h = out["z"], out["h"]
        zp = self.student.project_z(z) if self.z_enabled else None
        hp = self.student.project_h(h) if self.h_enabled else None
        with torch.no_grad():
            if zp is None:
                zp = self.student.project_z(z.detach())
            if hp is None:
                hp = self.student.project_h(h.detach())
        return dict(z=z.detach(), h=h.detach(), state=detach(out.get("state")),
                    z_prediction=zp, h_prediction=hp)


def wrap(model, h_enabled, z_enabled, device):
    return DistributedDataParallel(
        ProjectedStudent(model, h_enabled, z_enabled),
        device_ids=[device.index] if device.type == "cuda" else None,
        broadcast_buffers=False, find_unused_parameters=True,
    )


class NullLogger:
    def text(self, *a, **kw): pass
    def scalars(self, *a, **kw): pass
    def console(self, *a, **kw): pass
    def flush(self): pass
    def close(self): pass
    def write_metrics_json(self, *a, **kw): pass
