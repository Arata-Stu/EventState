"""DSEC forward flow: public validation held out of common Stage 1."""
import hashlib
import json
from pathlib import Path
import zipfile

TIMES_SHA = '4871a77639fc6e5eb75e5ec81931193d4a7570560b0709215e57649d1570899a'
VAL = ('zurich_city_02_e','zurich_city_07_a','zurich_city_08_a','zurich_city_11_c')


def splits():
    path = Path(__file__).resolve().parents[2]/'docs/dsec_published_split_audit.json'
    official = json.loads(path.read_text())['protocols']['DSEC_Flow_official']
    return dict(train=[s for s in official['train'] if s not in VAL], val=list(VAL), test=official['test'])


def parse_rows(content, test):
    rows = []
    for line in content.splitlines():
        line = line.partition('#')[0].strip()
        if not line: continue
        values = tuple(int(v.strip()) for v in line.split(','))
        if len(values) != (3 if test else 2) or min(values)<0 or values[1]<=values[0]:
            raise ValueError(f'Invalid forward pair: {line}')
        if rows and (values[0]<=rows[-1][0] or (test and values[2]<=rows[-1][2])):
            raise ValueError('Unordered/duplicate flow reference times or file indices')
        rows.append(values)
    if not rows: raise ValueError('Empty forward-flow timestamps')
    return rows


def test_pairs(path):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=TIMES_SHA:
        raise ValueError('Expected unchanged official test_forward_optical_flow_timestamps.zip')
    with zipfile.ZipFile(path) as z:
        return {s:parse_rows(z.read(s+'.csv').decode(),True) for s in splits()['test']}


def labeled_pairs(root, seq):
    folder=root/seq
    times=list(folder.rglob('forward_timestamps.txt'))
    if len(times)!=1: raise ValueError(f'{seq}: require exactly one forward_timestamps.txt under {folder}')
    rows=parse_rows(times[0].read_text(),False)
    files=sorted((times[0].parent/'forward').glob('*.png'),key=lambda p:int(p.stem))
    if len(rows)!=len(files): raise ValueError(f'{seq}: GT count mismatch {len(rows)} vs {len(files)}')
    return [(start,end,str(p.resolve())) for (start,end),p in zip(rows,files)]


def encode_png(flow):
    import cv2
    import numpy as np
    if flow.shape!=(480,640,2) or not np.isfinite(flow).all(): raise ValueError('Invalid predicted flow')
    rgb=np.ones((480,640,3),np.uint16)
    rgb[:,:,:2]=np.clip(np.rint(flow*128+32768),0,65535).astype(np.uint16)
    ok,buf=cv2.imencode('.png',rgb[:,:,::-1])
    if not ok: raise OSError('PNG encode failed')
    return buf.tobytes()


def read_gt(path):
    import cv2
    import numpy as np
    bgr=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.dtype!=np.uint16 or bgr.shape!=(480,640,3) or (bgr[:,:,0]>1).any():
        raise ValueError(f'Invalid 16-bit flow GT: {path}')
    return (bgr[:,:,[2,1]].astype(np.float32)-32768)/128, bgr[:,:,0].astype(bool)
