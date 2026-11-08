"""Frozen endpoint features -> forward displacement in native image pixels."""
import torch
from torch import nn
from torch.nn import functional as F


class FlowProbe(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.project=nn.Sequential(nn.Conv2d(channels,64,1),nn.GELU())
        self.decode=nn.Sequential(nn.Conv2d(64*3+25,128,3,padding=1),nn.GELU(),
                                  nn.Conv2d(128,64,3,padding=1),nn.GELU(),nn.Conv2d(64,2,1))

    def forward(self, first, last):
        a=self.project(first);b=self.project(last)
        an=F.normalize(a,dim=1); bn=F.pad(F.normalize(b,dim=1),(2,2,2,2))
        h,w=a.shape[-2:]
        cost=[(an*bn[:,:,dy:dy+h,dx:dx+w]).sum(1,keepdim=True) for dy in range(5) for dx in range(5)]
        small=self.decode(torch.cat([a,b,b-a,*cost],dim=1))
        return 16*F.interpolate(small,size=(480,640),mode='bilinear',align_corners=False)
