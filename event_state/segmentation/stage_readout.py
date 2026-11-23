"""Channel-wise readout of four aligned ViT block outputs for frozen Semantic probes."""

import torch

from event_state.models.tokens import tokens_to_map


STAGE_BLOCKS = (3, 6, 9, 12)


def stage_feature_map(levels, grid):
    """Pack one frame's four [1,1,N,D] outputs into a CPU fp16 [4D,H,W] map."""
    if len(levels) != len(STAGE_BLOCKS):
        raise ValueError("Semantic stages require exactly four block outputs")
    shape = levels[0].shape
    if len(shape) != 4 or shape[:2] != (1, 1) or any(level.shape != shape for level in levels):
        raise ValueError("Semantic stages require four aligned [1,1,N,D] tensors")
    tokens = torch.cat([level[0, 0] for level in levels], dim=-1)
    return tokens_to_map(tokens, grid).to(device="cpu", dtype=torch.float16).contiguous()
