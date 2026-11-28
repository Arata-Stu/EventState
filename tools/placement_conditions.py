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
SKIP_CONTROLS = {
    SKIP_CONDITION: 'E1_b3',
    'E1_b6_skip': 'E1_b6',
    'E1_b9_skip': 'E1_b9',
    'E1_b3_6_9_skip': 'E1_b3_6_9',
    'E1_b12_skip': 'E1',
    'E1_b3_6_9_12_skip': 'E1_b3_6_9_12',
}
SKIP_PLACEMENT_SUITE = ('E1_b3_6_9', 'E1_b6_skip', 'E1_b9_skip', 'E1_b3_6_9_skip')
FINAL_SKIP_SUITE = ('E1_b12_skip', 'E1_b3_6_9_12_skip')
PLACEMENT_BLOCKS = {**PLACEMENTS, 'E1_b3_6_9': (3, 6, 9)}
PLACEMENT_BLOCKS.update({skip: ((12,) if control == 'E1' else PLACEMENT_BLOCKS[control]) for skip, control in SKIP_CONTROLS.items()})


def placement_label(condition: str) -> str:
    layers = '・'.join(str(block) for block in PLACEMENT_BLOCKS[condition])
    connection = '、学習可能な残差接続' if condition in SKIP_CONTROLS else ''
    if condition == 'E1_b3_6_9':
        connection = '、置換対照'
    return f'E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM（{layers}層後{connection}）'
