# Third-party notices

## ScaleEvent algorithm reference

The optional ScaleEvent loss and event-activation rule are native implementations
based on the released algorithm at https://github.com/zhiwen-xdu/ScaleEvent,
commit `92f005b0f2cbb19dfc8391e3019ca042b1a9f587`.
Paper: Chen et al., *Scaling Dense Event-Stream Pretraining from Visual Foundation
Models*, https://arxiv.org/abs/2603.03969.
The reference package is not imported by the training runtime or redistributed
as part of EventState. Tests use self-contained specification checks and do not
load a reference checkout.
The complementary h-branch objective is an EventState extension.

## DINOv3

**Built with DINOv3.**

EventState uses the DINOv3 ViT-S/16 architecture and pretrained backbone from
Meta AI as a frozen RGB teacher and as initialization for the event encoder.
DINOv3 code and model weights are provided by Meta under the DINOv3 License:

- Project: https://github.com/facebookresearch/dinov3
- License: https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md
- Paper: https://arxiv.org/abs/2508.10104

The DINOv3 source code and pretrained weights are not redistributed by this
repository. Users must obtain the weights from an authorized official source
and accept the applicable DINOv3 License terms themselves.

Before distributing a checkpoint initialized from DINOv3, review the license
applicable when the weights were obtained and include the required agreement
and attribution with the distributed material. Research publications using
EventState with DINOv3 must acknowledge and cite DINOv3.

## YOLOX design reference

The native EventState detection probe follows the published YOLOX design
(decoupled head and dynamic-k assignment). It does not vendor or import YOLOX,
DAGR, or RVT source code.

- YOLOX project: https://github.com/Megvii-BaseDetection/YOLOX
- YOLOX license: Apache License 2.0
- DAGR reference: https://github.com/uzh-rpg/dagr
- RVT reference: https://github.com/uzh-rpg/RVT

DAGR and RVT are consulted for the public DSEC-Detection data and evaluation
contracts only. Their GPL source code is not included in this repository.
