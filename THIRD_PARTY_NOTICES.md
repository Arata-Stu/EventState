# Third-party notices

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
