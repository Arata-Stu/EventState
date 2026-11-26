"""Patchwise LSTM placements in the 12-block DINOv3 encoder, shared by both stages."""

PLACEMENTS = {
    'E1_b3': (3,),
    'E1_b6': (6,),
    'E1_b9': (9,),
    'E1_b9_12': (9, 12),
    'E1_b6_9_12': (6, 9, 12),
    'E1_b3_6_9_12': (3, 6, 9, 12),
}
SKIP_CONDITION = 'E1_b3_skip'
PLACEMENT_BLOCKS = {**PLACEMENTS, SKIP_CONDITION: (3,)}


def placement_label(condition: str) -> str:
    layers = '・'.join(str(block) for block in PLACEMENT_BLOCKS[condition])
    skip = '、学習可能な残差接続' if condition == SKIP_CONDITION else ''
    return f'E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM（{layers}層後{skip}）'
