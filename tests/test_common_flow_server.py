"""Run in the training server env (PyTorch/OpenCV); synthetic signed native pixel flow."""
import cv2
import numpy as np
import torch
from event_state.flow.protocol import encode_png, read_gt
from event_state.flow.probe import FlowProbe


def test_official_rgb_codec(tmp_path):
    x=np.zeros((480,640,2),np.float32); x[:]=[-3.125,8.5]
    path=tmp_path/'000020.png';path.write_bytes(encode_png(x))
    buf=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    assert buf[0,0].tolist()==[1,33856,32368] # OpenCV B,G,R
    y,valid=read_gt(path)
    assert valid.all() and np.array_equal(y,x)


def test_endpoint_gradients():
    net=FlowProbe(4)
    a,b=[torch.randn(1,4,30,40,requires_grad=True) for _ in range(2)]
    flow=net(a,b)
    assert flow.shape==(1,2,480,640)
    flow.square().mean().backward()
    assert a.grad.abs().sum()>0 and b.grad.abs().sum()>0
