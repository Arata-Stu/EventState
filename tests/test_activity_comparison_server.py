"""Small CPU-only rendering checks; no checkpoints or dataset downloads."""
import numpy as np
from PIL import ImageFont

from tools.render_activity_comparison import depth_color, draw_frame, frame_metric, semantic_color


def test_depth_display_limits_do_not_change_raw_metrics():
    values = np.array([[0, 40, 80, 120]], dtype=np.float32)
    pixels = depth_color(values, maximum=80)
    assert tuple(pixels[0, 0]) == (255, 0, 0)
    assert tuple(pixels[0, 2]) == (0, 0, 255)
    assert np.array_equal(pixels[0, 2], pixels[0, 3])
    assert values[0, 3] == 120
    target = np.array([[10, 20, 30, 40]], dtype=np.float32)
    valid = np.array([[False, True, True, False]])
    assert frame_metric('depth', values, target, valid) == 35
    assert frame_metric('depth', values, None, valid) is None
    assert frame_metric('depth', values, target, np.zeros_like(valid)) is None


def test_shared_semantic_palette_and_ignored_gt():
    labels = np.array([[0, 8, 255]], dtype=np.uint8)
    palette = semantic_color(labels)
    assert tuple(palette[0, 2]) == (35, 35, 35)
    assert np.array_equal(semantic_color(labels), palette)
    prediction = np.array([[0, 8, 3]], dtype=np.uint8)
    assert frame_metric('semantic', prediction, labels, labels != 255) == 1


def test_missing_gt_still_draws_predictions_and_native_source_time(tmp_path):
    # Pillow's shipped font path is sufficient for ASCII fixture labels. The
    # real CLI validates a CJK font before performing expensive inference.
    font = ImageFont.truetype('DejaVuSans.ttf', 16)
    source = np.zeros((32, 640, 3), dtype=np.uint8)
    prediction = np.ones((32, 640), dtype=np.float32)*20
    scene = dict(sequence='fixture', kind='low', start_us=1000000, end_us=1200000,
                 low_fraction=0.8, high_fraction=0.0)
    rows = [dict(timestamp=t, event_rate_hz=10.0, group='p00_20') for t in (1000000, 1050000, 1100000, 1150000)]
    canvas, metrics = draw_frame('depth', source, source, None,
        np.zeros((32, 640), dtype=bool), [prediction], ['E1 Encoder LSTM h'], scene,
        1050000, rows[1], rows, [20, 40, 60, 80], font.path, 80, 100)
    assert canvas.width == 1920 and canvas.height % 2 == 0
    assert metrics == [None]
    # GT slot is dark; the model panel still contains a dense prediction.
    pixels = np.asarray(canvas)
    assert np.array_equal(pixels[50, 1280+5], (35, 35, 35))
    assert not np.array_equal(pixels[32+42+50, 5], (0, 0, 0))


def test_renderer_replays_prefix_and_keeps_all_scene_inputs_with_missing_gt(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import torch
    from tools import render_activity_comparison as renderer
    from event_state.evaluation import comparison_semantic
    from event_state.dense import source
    from event_state import activity_quantiles, activity_reporting

    sequence = 'car_urban_day_ucity_small_loop'
    prepared = tmp_path/'prepared'
    (prepared/sequence).mkdir(parents=True)
    (prepared/sequence/'metadata.json').write_text('{}')
    stamps = [50000, 100000, 150000, 200000, 250000]
    records = {t:dict(sequence=sequence, timestamp=t, previous_timestamp=t-50000, window_us=50000,
                     event_rate_hz=20.0, group='p00_20') for t in stamps}
    identity = {'manifest_sha256': 'fixture', 'source_contract': {'dataset':'m3ed'}, 'cutpoints': [
        {'event_rate_hz': v} for v in (30, 40, 50, 60)]}
    index = SimpleNamespace(identity=identity, lookup=lambda s,t:records[t], timestamps=lambda s:stamps)
    monkeypatch.setattr(activity_quantiles.ActivityIndex, 'load', lambda p:index)
    monkeypatch.setattr(activity_reporting, 'validate_activity_source', lambda *a:None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda:False)
    font = ImageFont.truetype('DejaVuSans.ttf', 16)
    monkeypatch.setattr(renderer, 'cjk_font', lambda p:font.path)
    monkeypatch.setattr(renderer, 'ffmpeg_executable', lambda:'ffmpeg-fixture')
    monkeypatch.setattr(renderer, 'rgb_frame', lambda *a:np.zeros((352,640,3),dtype=np.uint8))
    calls = {}
    class Backend:
        def __init__(self, dataset, condition, *args):
            self.condition=condition;self.label=condition+' Encoder LSTM h'
            self.identity={'condition_label':self.label,'label_source':'fixture'};self.cfg=object();self.config={}
            self.prepared_root=prepared;self.frame_size=(352,640);calls[condition]=[]
        def reset(self): pass
        def close(self): pass
        def ground_truth(self, s,t): return None
        def step(self, frame, predict=True):
            calls[self.condition].append((frame['frame_index'],predict))
            return np.full(self.frame_size,8,dtype=np.uint8) if predict else None
    monkeypatch.setattr(comparison_semantic, 'SemanticBackend', Backend)
    def frames(*args):
        for i,t in enumerate(stamps,1):
            yield dict(sequence_name=sequence,frame_index=i,timestamp=t,previous_timestamp=t-50000,
                       event_count=1,events=torch.zeros((3,360,640)))
    monkeypatch.setattr(source,'prepared_frames',frames)
    def encode(command,check):
        from pathlib import Path
        Path(command[-1]).write_bytes(b'encoder fixture; native PTS independently tested')
    monkeypatch.setattr(renderer.subprocess,'run',encode)
    scene={'id':'fixture_low','kind':'low','sequence':sequence,'start_us':100000,'end_us':250000,
           'duration_us':150000,'timestamps_us':[100000,150000,200000],'frames':3,
           'low_fraction':1.0,'high_fraction':0.0}
    plan={'dataset':'m3ed','task':'semantic','activity_manifest':'fixture',
          'activity_manifest_sha256':'fixture','status':'complete','shortfalls':[],
          'selection':{'max_gap_seconds':0.25},'scenes':[scene]}
    selection=tmp_path/'scenes.json';selection.write_text(json.dumps(plan))
    output=tmp_path/'comparison'
    monkeypatch.setattr(renderer.sys,'argv',['renderer','--scenes',str(selection),
        '--task','semantic','--output',str(output),'--device','cpu'])
    renderer.main()
    assert (output/'_SUCCESS').exists()
    assert all(value==[(1,False),(2,True),(3,True),(4,True)] for value in calls.values())
    result=json.loads((output/'manifest.json').read_text())
    assert result['status']=='complete' and result['videos']['fixture_low']['original_frames']==3
    assert len(list((output/'fixture_low').glob('*.png')))==3
    assert len((output/'fixture_low/frames.csv').read_text().splitlines())==4
    assert len(list((output/'predictions/E1'/sequence).glob('*.npy')))==3
