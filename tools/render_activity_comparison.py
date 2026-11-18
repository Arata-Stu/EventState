#!/usr/bin/env python3
"""Render complete, real-time activity scenes with evaluated Semantic/Depth heads.

Scene selection never sees predictions. Inference replays every prepared input
from sequence start; only the selected display intervals are written to disk.
"""
import argparse
import csv
from contextlib import ExitStack
from functools import lru_cache
import gc
from itertools import chain, zip_longest
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_state.evaluation.comparison_timing import ffmpeg_executable

CONDITIONS = ('E0', 'E1', 'E2', 'E1_old_ddp', 'E1_new_ddp', 'E1_convlstm')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def cjk_font(path=None):
    from PIL import ImageFont
    candidates = ([Path(path)] if path else []) + [
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'),
        Path('/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'),
    ]
    if not path and shutil.which('fc-match'):
        result = subprocess.run(['fc-match', '-f', '%{file}', 'sans:lang=ja'],
                                capture_output=True, text=True, check=True)
        if result.stdout.strip():
            candidates.append(Path(result.stdout.strip()))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        font = ImageFont.truetype(str(candidate), 19)
        # Reject a missing-glyph box rather than silently drawing unreadable labels.
        if (font.getmask('蒸').size, bytes(font.getmask('蒸'))) != (
                font.getmask('\uffff').size, bytes(font.getmask('\uffff'))):
            return candidate
    raise FileNotFoundError('Japanese font required: pass --font /path/to/NotoSansCJK-Regular.ttc')


def rgb_frame(dataset, task, cfg, prepared_root, sequence, timestamp):
    from PIL import Image
    import numpy as np
    if dataset == 'm3ed':
        path = prepared_root / sequence / 'aligned_rgb' / f'{timestamp}.png'
        with Image.open(path) as source:
            image = source.convert('RGB')
            if image.size == (640, 360):
                image = image.crop((0, 4, 640, 356))
            if image.size != (640, 352):
                raise ValueError(f'Unexpected aligned RGB geometry: {path}')
    else:
        from tools.visualize_dsec_semantic_sequences import _find_aligned_rgb
        path = _find_aligned_rgb(Path(cfg.dataset.root), sequence, timestamp)
        with Image.open(path) as source:
            image = source.convert('RGB')
            if image.size != (640, 480):
                raise ValueError(f'Unexpected DSEC native RGB geometry: {path}')
            if task == 'semantic':
                image = image.crop((0, 0, 640, 440))
    return np.asarray(image).copy()


def event_frame(events, dataset, task):
    import numpy as np
    array = events.detach().cpu().float().numpy()
    if dataset == 'm3ed':
        array = array[:, 4:356]
    elif task == 'semantic':
        array = array[:, :440]
    return np.round(np.clip(array.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)


def depth_color(values, valid=None, maximum=80):
    import numpy as np
    normalized = np.clip(np.where(np.isfinite(values), values, 0) / maximum, 0, 1)
    rgb = np.round(np.stack((1-normalized, 1-np.abs(2*normalized-1), normalized), -1)*255).astype(np.uint8)
    if valid is not None:
        rgb[~valid] = 0
    return rgb


def semantic_color(values):
    import numpy as np
    from tools.visualize_dsec_semantic_sequences import PALETTE
    rgb = np.full((*values.shape, 3), 35, dtype=np.uint8)
    valid = (values >= 0) & (values < len(PALETTE))
    rgb[valid] = PALETTE[values[valid]]
    return rgb


def frame_metric(task, prediction, target, valid):
    import numpy as np
    if target is None or not valid.any():
        return None
    if task == 'depth':
        return float(np.abs(prediction[valid] - target[valid]).mean())
    matrix = np.bincount((target[valid].astype(np.int64)*11 + prediction[valid]).ravel(),
                         minlength=121).reshape(11, 11)
    tp = matrix.diagonal()
    union = matrix.sum(0) + matrix.sum(1) - tp
    return float((tp[union > 0] / union[union > 0]).mean())


@lru_cache(maxsize=4)
def _panel_font(font_path):
    from PIL import ImageFont
    return ImageFont.truetype(str(font_path), 16)


def activity_chart(width, scene, records, cutpoints, ymax):
    from PIL import Image, ImageDraw
    chart = Image.new('RGB', (width, 48), '#141414')
    draw = ImageDraw.Draw(chart)
    for boundary in (cutpoints[0], cutpoints[-1]):
        y = 47 * (1 - boundary / ymax)
        draw.line((12, y, width-12, y), fill='#555555')
    points = [(12 + (width-24) * (r['timestamp']-scene['start_us']) / (scene['end_us']-scene['start_us']),
               47 * (1 - r['event_rate_hz'] / ymax))
              for r in records if scene['start_us'] <= r['timestamp'] < scene['end_us']]
    if len(points) > 1:
        draw.line(points, fill='#65cfff', width=2)
    return chart


def draw_frame(task, rgb, events, target, valid, predictions, labels, scene, timestamp,
               activity_record, activity_records, cutpoints, font_path, depth_max, activity_ymax,
               *, chart=None):
    import numpy as np
    from PIL import Image, ImageDraw
    height, width = rgb.shape[:2]
    header, footer = 42, 138
    small = _panel_font(font_path)
    panels = [('RGB (reference only)', rgb), ('GEP events', events)]
    if target is None:
        missing = Image.new('RGB', (width, height), '#232323')
        ImageDraw.Draw(missing).text((20, 30), 'GT unavailable at this timestamp', font=small, fill='white')
        panels.append(('GT unavailable', np.asarray(missing)))
    else:
        panels.append(('GT (invalid pixels = gray)' if task == 'semantic' else 'GT (invalid pixels = black)',
                       semantic_color(target) if task == 'semantic' else depth_color(target, valid, depth_max)))
    metrics = []
    for prediction, label in zip(predictions, labels):
        value = frame_metric(task, prediction, target, valid)
        metrics.append(value)
        description = 'frame mIoU=N/A' if task == 'semantic' else 'frame MAE=N/A'
        if value is not None:
            description = f'frame mIoU={100*value:.2f}%' if task == 'semantic' else f'frame MAE={value:.3f}m'
        panels.append((label + ' | ' + description,
                       semantic_color(prediction) if task == 'semantic' else depth_color(prediction, maximum=depth_max)))
    columns, rows = 3, math.ceil(len(panels)/3)
    total_width = columns * width
    canvas = Image.new('RGB', (total_width, rows*(height+header)+footer), '#141414')
    draw = ImageDraw.Draw(canvas)
    for number, (label, pixels) in enumerate(panels):
        x, y = (number % columns)*width, (number // columns)*(height+header)
        # Two lines keep the full condition ID/name readable at native panel width.
        pieces = label.split(' | ', 1)
        draw.text((x+6, y+2), pieces[0], font=small, fill='white')
        if len(pieces) == 2:
            draw.text((x+6, y+21), pieces[1], font=small, fill='white')
        canvas.paste(Image.fromarray(pixels), (x, y+header))
    bottom = rows*(height+header)
    elapsed = (timestamp-scene['start_us'])/1e6
    rate = activity_record['event_rate_hz']/1e6
    draw.text((8, bottom+3), f"{task} | {scene['sequence']} | {scene['kind']} | t={elapsed:.3f}s | source={timestamp}us", font=small, fill='white')
    draw.text((8, bottom+25), f"Events={activity_record['event_count']:,} / {activity_record['window_us']/1000:.3f} ms | Rate={rate:.3f} M events/s | {activity_record['group']} | Q1 time={scene['low_fraction']:.1%} Q5 time={scene['high_fraction']:.1%}", font=small, fill='white')
    if task == 'depth':
        draw.text((8, bottom+47), f'Fixed depth scale: red=0m, blue>={depth_max:g}m; metrics use unclipped prediction', font=small, fill='white')
        ramp = np.repeat(np.linspace(0, depth_max, 240, dtype=np.float32)[None], 12, axis=0)
        canvas.paste(Image.fromarray(depth_color(ramp, maximum=depth_max)), (1100, bottom+49))
        draw.text((1064, bottom+45), '0m', font=small, fill='white')
        draw.text((1348, bottom+45), f'{depth_max:g}m', font=small, fill='white')
    else:
        from tools.visualize_dsec_semantic_sequences import PALETTE, DSEC_SEMANTIC_11_CLASSES
        x = 8
        for color, name in zip(PALETTE, DSEC_SEMANTIC_11_CLASSES):
            draw.rectangle((x, bottom+50, x+12, bottom+62), fill=tuple(int(c) for c in color))
            draw.text((x+17, bottom+46), name, font=small, fill='white')
            x += 18 + int(draw.textlength(name, font=small)) + 15
    # One event-rate axis shared by every selected clip and all compared models.
    if chart is None:
        chart = activity_chart(total_width, scene, activity_records, cutpoints, activity_ymax)
    canvas.paste(chart, (0, bottom+79))
    x = 12 + (total_width-24)*(timestamp-scene['start_us'])/(scene['end_us']-scene['start_us'])
    draw.line((x, bottom+79, x, bottom+126), fill='#ffcc55', width=2)
    draw.text((max(12, total_width-310), bottom+69), f'Activity axis: 0..{activity_ymax/1e6:.2f} M events/s', font=small, fill='white')
    # H.264 yuv420 requires even dimensions; pad rather than rescale predictions.
    if canvas.width % 2 or canvas.height % 2:
        padded = Image.new('RGB', (canvas.width+canvas.width%2, canvas.height+canvas.height%2), '#141414')
        padded.paste(canvas, (0, 0)); canvas = padded
    return canvas, metrics


def load_backend(args, dataset, condition):
    from event_state.evaluation.comparison_depth import DepthBackend
    from event_state.evaluation.comparison_semantic import SemanticBackend
    options = {}
    if condition == 'E1_convlstm':
        options.update(temporal_run=args.temporal_depth_run, temporal_stage1=args.convlstm_stage1_run)
    if args.task == 'semantic':
        if condition == 'E1_convlstm':
            options['m3ed_temporal_semantic'] = args.m3ed_convlstm_semantic_run
        return SemanticBackend(dataset, condition, ROOT, args.teacher_checkpoint, args.device, **options)
    return DepthBackend(dataset, condition, ROOT, args.depth_run, args.teacher_checkpoint, args.device, **options)


def render_full_sequences(args, scenes, records_by_sequence, activity, manifest, font_path, ffmpeg):
    """One chronological pass per recording, with independent state per model."""
    import numpy as np
    from event_state.activity_reporting import validate_activity_source
    from event_state.dense.source import prepared_frames
    from event_state.evaluation.comparison_depth import DepthTargets
    from event_state.evaluation.full_video import encode_native_video

    dataset, conditions = manifest['dataset'], manifest['conditions']
    cutpoints = [c['event_rate_hz'] for c in activity.identity['cutpoints']]
    with ExitStack() as stack:
        models = []
        labels = []
        for condition in conditions:
            model = load_backend(args, dataset, condition)
            if hasattr(model, 'close'):
                stack.callback(model.close)
            label = model.label
            if condition == 'E1':
                label = 'E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM'
            print(f'[comparison] {dataset}/{args.task} {label}', flush=True)
            models.append(model)
            labels.append(label)
            manifest['models'][condition] = dict(model.identity, condition_label=label)
        first = models[0]
        for model in models:
            if model.prepared_root != first.prepared_root or model.frame_size != first.frame_size:
                raise ValueError('Models use different prepared inputs or geometry')
            if model.cfg.dataset.representation.percentile != first.cfg.dataset.representation.percentile:
                raise ValueError('Models require different GEP inputs')

        for scene in scenes:
            seq = scene['sequence']
            records = records_by_sequence[seq]
            metadata = read(first.prepared_root/seq/'metadata.json')
            validate_activity_source(activity, seq, metadata)
            for model in models:
                if seq in getattr(model, 'config', {}).get('training_sequences', []):
                    raise ValueError(f'Training sequence in evaluation video: {seq}')
                model.reset()
            targets = DepthTargets(args.depth_run, dataset, seq) if args.task == 'depth' else None
            manifest['targets'][seq] = (targets.identity if targets else dict(
                label_source=first.identity['label_source'], timestamp_policy='exact match; unavailable stays unavailable',
                roots={key: value for key, value in first.config.items()
                       if key in ('labels_root', 'target_cache_dir', 'target_root', 'extra_target_root')}))
            height, width = first.frame_size
            ymax = max(cutpoints[-1], max(r['event_rate_hz'] for r in records), 1) * 1.05
            chart = activity_chart(3*width, scene, records, cutpoints, ymax)
            destination = args.output/scene['id']
            destination.mkdir()
            video = destination/'comparison.mp4'

            def frames(table):
                inputs = prepared_frames(first.prepared_root, seq, first.cfg, dataset)
                for count, (record, frame) in enumerate(zip_longest(records, inputs), 1):
                    if (frame is None or record is None or record['timestamp'] != frame['timestamp']
                            or record['previous_timestamp'] != frame['previous_timestamp']
                            or record['event_count'] != frame['event_count']):
                        raise ValueError(f'Prepared input/activity timeline differs at {seq}, frame {count}')
                    stamp = record['timestamp']
                    predictions = [model.step(frame) for model in models]
                    if any(p is None or p.shape != first.frame_size or not np.isfinite(p).all() for p in predictions):
                        raise ValueError(f'Missing/nonfinite prediction or changed geometry: {seq}/{stamp}')
                    target, valid = None, None
                    if targets:
                        pair = targets.get(stamp)
                        if pair is not None:
                            target, valid = pair
                    else:
                        target = first.ground_truth(seq, stamp)
                        for model in models[1:]:
                            other = model.ground_truth(seq, stamp)
                            if (other is None) != (target is None) or (target is not None and not np.array_equal(target, other)):
                                raise ValueError(f'Models reference different Semantic GT: {seq}/{stamp}')
                        for model in models:
                            scored = getattr(model, 'scored_frame_indices', {}).get(seq)
                            if scored is not None and (frame['frame_index'] in scored) != (target is not None):
                                raise ValueError(f'Semantic GT coverage differs from evaluated frames: {seq}/{stamp}')
                        if target is not None:
                            valid = target != 255
                    if valid is None:
                        valid = np.zeros(first.frame_size, dtype=bool)
                    rgb = rgb_frame(dataset, args.task, first.cfg, first.prepared_root, seq, stamp)
                    events = event_frame(frame['events'], dataset, args.task)
                    canvas, metrics = draw_frame(args.task, rgb, events, target, valid, predictions, labels,
                        scene, stamp, record, records, cutpoints, font_path, args.depth_display_max, ymax, chart=chart)
                    table.writerow([stamp, (stamp-scene['start_us'])/1e6, record['event_count'], record['window_us'],
                                    record['event_rate_hz'], record['group'], target is not None, int(valid.sum()), *metrics])
                    if count % 100 == 0:
                        print(f'[comparison] {seq} {count}/{len(records)} | {" / ".join(labels)}', flush=True)
                    yield canvas

            with (destination/'frames.csv').open('x', newline='', encoding='utf-8') as stream:
                table = csv.writer(stream)
                table.writerow(['source_timestamp_us', 'video_time_s', 'event_count', 'window_us', 'event_rate_hz',
                                'activity_group', 'gt_available', 'valid_pixels',
                                *[c+('_frame_mIoU' if args.task == 'semantic' else '_frame_MAE_m') for c in conditions]])
                canvases = frames(table)
                first_frame = next(canvases)
                size = first_frame.size
                rgb_frames = (image.tobytes() for image in chain([first_frame], canvases))
                encode_native_video(ffmpeg, video, scene['timestamps_us'], scene['end_us'], size, rgb_frames)
            manifest['videos'][scene['id']] = dict(path=str(video.resolve()), original_frames=len(records),
                duration_seconds=scene['duration_us']/1e6, terminal_duplicate=True, activity_axis_max_hz=ymax)
            save(args.output/'manifest.json', manifest)
            print(f'[comparison] saved: {video}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', type=Path, required=True)
    parser.add_argument('--task', choices=('semantic', 'depth'), required=True)
    parser.add_argument('--conditions', help='Comma-separated condition IDs; full recordings default to E0,E1,E1_convlstm')
    parser.add_argument('--sequence', help='Render one complete recording from a full-sequence plan')
    parser.add_argument('--depth-run', type=Path, default=ROOT/'outputs/stage2_common_depth_vkitti_20261110')
    parser.add_argument('--temporal-depth-run', type=Path, help='Completed ConvLSTM Depth and DSEC Semantic run')
    parser.add_argument('--convlstm-stage1-run', type=Path, help='ConvLSTM full pretraining output root')
    parser.add_argument('--m3ed-convlstm-semantic-run', type=Path, help='Completed M3ED ConvLSTM Semantic run')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--teacher-checkpoint', type=Path)
    parser.add_argument('--font', type=Path)
    parser.add_argument('--depth-display-max', type=float, default=80)
    args = parser.parse_args()
    plan = read(args.scenes)
    full = plan['selection'].get('mode') == 'full_sequences'
    conditions = args.conditions.split(',') if args.conditions is not None else (
        ['E0', 'E1', 'E1_convlstm'] if full else ['E0', 'E1', 'E2'])
    if not conditions or len(set(conditions)) != len(conditions) or any(c not in CONDITIONS for c in conditions):
        parser.error('Choose unique conditions: ' + ','.join(CONDITIONS))
    if not math.isfinite(args.depth_display_max) or args.depth_display_max <= 0:
        parser.error('depth-display-max must be positive and finite')
    if args.output.exists():
        raise FileExistsError('Existing comparison outputs are preserved; choose a fresh --output')
    from event_state.activity_quantiles import ActivityIndex, canonical_sha256
    from event_state.evaluation.activity_scenes import eligible_sequences, validate_full_sequence
    from event_state.evaluation.comparison_timing import validate_scene, write_concat, build_ffmpeg_command
    dataset = plan['dataset']
    if dataset not in ('dsec', 'm3ed') or (dataset == 'dsec' and plan['task'] != args.task):
        raise ValueError('DSEC Semantic test and Depth val scenes must remain task-specific')
    scenes = plan['scenes']
    if not scenes:
        raise ValueError(f"No eligible scenes; inspect selection shortfalls: {plan.get('shortfalls')}")
    if plan.get('status') != 'complete' or plan.get('shortfalls'):
        raise ValueError(f"Scene selection is incomplete; inspect shortfalls before rendering: {plan.get('shortfalls')}")
    if len({s['id'] for s in scenes}) != len(scenes) or any(Path(s['id']).name != s['id'] for s in scenes):
        raise ValueError('Scene IDs must be unique safe directory names')
    max_gap = round(float(plan['selection']['max_gap_seconds'])*1e6)
    allowed, split_source = eligible_sequences(dataset, args.task)
    if full and (plan['task'] != args.task or sorted(s['sequence'] for s in scenes) != sorted(allowed)):
        raise ValueError('Full plan must include each evaluation recording exactly once')
    if args.sequence is not None:
        if not full or args.sequence not in allowed:
            parser.error('--sequence must name one evaluation recording in a full-sequence plan')
        scenes = [s for s in scenes if s['sequence'] == args.sequence]
    for scene in scenes:
        validate_scene(scene, max_gap_us=max_gap)
        if scene['sequence'] not in allowed:
            raise ValueError('Selected scene is outside the current task evaluation split')
    activity = ActivityIndex.load(plan['activity_manifest'])
    if (activity.identity['manifest_sha256'] != plan['activity_manifest_sha256'] or
            activity.identity['source_contract']['dataset'] != dataset):
        raise ValueError('Scene/activity manifest identity mismatch')
    font_path, ffmpeg = cjk_font(args.font), ffmpeg_executable()
    import numpy as np
    import torch
    from event_state.activity_reporting import validate_activity_source
    from event_state.dense.source import prepared_frames
    from event_state.evaluation.comparison_depth import DepthTargets
    grouped = {}
    records_by_sequence = {}
    for scene in scenes:
        grouped.setdefault(scene['sequence'], []).append(scene)
        if scene['sequence'] not in records_by_sequence:
            records_by_sequence[scene['sequence']] = [activity.lookup(scene['sequence'], t)
                                                     for t in activity.timestamps(scene['sequence'])]
        if full:
            validate_full_sequence(scene, records_by_sequence[scene['sequence']])
            continue
        # Demand the complete original input timeline, never an intersection of model frames.
        expected = [r['timestamp'] for r in records_by_sequence[scene['sequence']]
                    if scene['start_us'] <= r['timestamp'] < scene['end_us']]
        if expected != scene['timestamps_us']:
            raise ValueError('Scene omits/reorders original input frames')
        intervals = [r for r in records_by_sequence[scene['sequence']]
                     if r['timestamp'] > scene['start_us'] and r['previous_timestamp'] < scene['end_us']]
        if (not intervals or intervals[0]['previous_timestamp'] != scene['start_us'] or
                intervals[-1]['timestamp'] < scene['end_us'] or
                any(r['window_us'] > max_gap for r in intervals) or
                any(b['previous_timestamp'] != a['timestamp'] for a, b in zip(intervals, intervals[1:]))):
            raise ValueError('Scene activity does not continuously cover its entire display interval')
    args.output.mkdir(parents=True)
    if not full:
        for sequence in grouped:
            (args.output/'assets'/sequence).mkdir(parents=True)
    manifest = dict(format_version=1, dataset=dataset, task=args.task, status='incomplete',
                    selection_file=str(args.scenes.resolve()), selection_sha256=canonical_sha256(plan),
                    source_activity=activity.identity, scenes=scenes, conditions=conditions,
                    state_context='Every prepared input from sequence start; no reset at clip start',
                    timing='Native timestamps, VFR; one final duplicate at endpoint holds the final frame',
                    depth_display_max_m=args.depth_display_max, models={}, targets={}, videos={})
    save(args.output/'manifest.json', manifest)
    if full:
        try:
            render_full_sequences(args, scenes, records_by_sequence, activity, manifest, font_path, ffmpeg)
        except BaseException:
            manifest['status'] = 'failed; partial outputs preserved'
            save(args.output/'manifest.json', manifest)
            raise
        manifest['status'] = 'complete'
        save(args.output/'manifest.json', manifest)
        (args.output/'_SUCCESS').write_text('All requested full sequences rendered at native timestamps.\n')
        return
    sources = {}
    try:
        for condition_number, condition in enumerate(conditions):
            backend = load_backend(args, dataset, condition)
            if args.task == 'semantic' and set(grouped).intersection(getattr(backend, 'config', {}).get('training_sequences', [])):
                raise ValueError('A selected scene was used to train this Semantic head')
            print(f'[comparison] ready: {dataset}/{args.task} {backend.label}; one model on {args.device}', flush=True)
            manifest['models'][condition] = backend.identity
            for sequence, selected in grouped.items():
                metadata = read(backend.prepared_root/sequence/'metadata.json')
                validate_activity_source(activity, sequence, metadata)
                if sequence in sources and sources[sequence] != str(backend.prepared_root):
                    raise ValueError('Models use different prepared input sources')
                sources[sequence] = str(backend.prepared_root)
                requested = sorted({t for scene in selected for t in scene['timestamps_us']})
                wanted, seen = set(requested), []
                backend.reset()
                targets = DepthTargets(args.depth_run, dataset, sequence) if args.task == 'depth' and condition_number == 0 else None
                if condition_number == 0:
                    if targets is not None:
                        manifest['targets'][sequence] = targets.identity
                    else:
                        config = backend.config
                        manifest['targets'][sequence] = dict(
                            label_source=backend.identity['label_source'],
                            roots={k:config[k] for k in ('labels_root','target_cache_dir','target_root','extra_target_root') if k in config},
                            prepared_metadata_sha256=canonical_sha256(metadata),
                            policy='Exact source timestamp only; actual labels retained in assets; later conditions checked equal')
                prediction_root = args.output/'predictions'/condition/sequence
                prediction_root.mkdir(parents=True)
                for frame in prepared_frames(backend.prepared_root, sequence, backend.cfg, dataset):
                    stamp = int(frame['timestamp'])
                    if stamp > requested[-1]:
                        break
                    prediction = backend.step(frame, predict=stamp in wanted)
                    if stamp not in wanted:
                        if int(frame['frame_index']) % 500 == 0:
                            print(f'[comparison] context: {backend.label} {sequence} frame={frame["frame_index"]} source={stamp}us (state updated; not exported)', flush=True)
                        continue
                    expected_record = activity.lookup(sequence, stamp)
                    if not math.isclose(expected_record['event_rate_hz'], frame['event_count']*1e6/(stamp-frame['previous_timestamp']), rel_tol=1e-9):
                        raise ValueError('Activity/input count differs')
                    if prediction is None or prediction.shape != backend.frame_size or not np.isfinite(prediction).all():
                        raise ValueError('Missing/nonfinite prediction or changed output geometry')
                    np.save(prediction_root/f'{stamp}.npy', prediction, allow_pickle=False)
                    asset = args.output/'assets'/sequence/f'{stamp}.npz'
                    if condition_number == 0:
                        target = backend.ground_truth(sequence, stamp) if args.task == 'semantic' else targets.get(stamp)
                        if target is None:
                            values = dict(gt_available=np.array(False))
                        elif args.task == 'semantic':
                            values = dict(gt_available=np.array(True), target=target, valid=target != 255)
                        else:
                            values = dict(gt_available=np.array(True), target=target[0], valid=target[1])
                        values.update(rgb=rgb_frame(dataset, args.task, backend.cfg, backend.prepared_root, sequence, stamp),
                                      events=event_frame(frame['events'], dataset, args.task))
                        np.savez_compressed(asset, **values)
                    elif args.task == 'semantic':
                        target = backend.ground_truth(sequence, stamp)
                        with np.load(asset, allow_pickle=False) as reference:
                            if bool(reference['gt_available']) != (target is not None) or (
                                    target is not None and not np.array_equal(target, reference['target'])):
                                raise ValueError('Models reference different Semantic GT')
                    seen.append(stamp)
                    if len(seen) % 100 == 0:
                        print(f'[comparison] {dataset}/{args.task} {backend.label} {sequence}: {len(seen)}/{len(requested)} displayed inputs', flush=True)
                if seen != requested:
                    raise ValueError('Prepared input coverage differs from selected scene timestamps')
            if hasattr(backend, 'close'):
                backend.close()
            del backend
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            save(args.output/'manifest.json', manifest)
        cutpoints = [c['event_rate_hz'] for c in activity.identity['cutpoints']]
        selected_activity = [r['event_rate_hz'] for records in records_by_sequence.values() for r in records
                             if any(r['sequence'] == s['sequence'] and s['start_us'] <= r['timestamp'] < s['end_us'] for s in scenes)]
        activity_ymax = max(cutpoints[-1]*1.05, max(selected_activity)*1.05, 1)
        manifest['activity_axis_max_hz'] = activity_ymax
        labels = [manifest['models'][c]['condition_label'] for c in conditions]
        for scene in scenes:
            destination = args.output/scene['id']
            destination.mkdir()
            images = []
            records = records_by_sequence[scene['sequence']]
            with (destination/'frames.csv').open('w', newline='') as stream:
                table = csv.writer(stream)
                table.writerow(['source_timestamp_us', 'video_time_s', 'event_count', 'window_us', 'event_rate_hz', 'activity_group', 'gt_available', 'valid_pixels',
                                *[c+('_frame_mIoU' if args.task == 'semantic' else '_frame_MAE_m') for c in conditions]])
                for number, stamp in enumerate(scene['timestamps_us']):
                    with np.load(args.output/'assets'/scene['sequence']/f'{stamp}.npz', allow_pickle=False) as asset:
                        available = bool(asset['gt_available'])
                        target = asset['target'] if available else None
                        valid = asset['valid'] if available else np.zeros(asset['rgb'].shape[:2], dtype=bool)
                        predictions = [np.load(args.output/'predictions'/c/scene['sequence']/f'{stamp}.npy', allow_pickle=False) for c in conditions]
                        record = activity.lookup(scene['sequence'], stamp)
                        canvas, metrics = draw_frame(args.task, asset['rgb'], asset['events'], target, valid, predictions,
                            labels, scene, stamp, record, records, cutpoints, font_path, args.depth_display_max, activity_ymax)
                        frame_file = destination/f'{number:06d}.png'
                        canvas.save(frame_file)
                        images.append(frame_file)
                        table.writerow([stamp, (stamp-scene['start_us'])/1e6, record['event_count'], record['window_us'], record['event_rate_hz'], record['group'],
                                        available, int(valid.sum()), *metrics])
            concat = destination/'frames.ffconcat'
            write_concat(concat, images, scene['timestamps_us'], scene['end_us'], max_gap_us=max_gap)
            partial = destination/'comparison.partial.mp4'
            subprocess.run(build_ffmpeg_command(ffmpeg, concat, partial), check=True)
            video = destination/'comparison.mp4'
            partial.rename(video)
            manifest['videos'][scene['id']] = dict(path=str(video.resolve()), original_frames=len(images),
                duration_seconds=(scene['end_us']-scene['start_us'])/1e6, terminal_duplicate=True)
            save(args.output/'manifest.json', manifest)
            print(f'[comparison] saved: {video}', flush=True)
    except BaseException:
        manifest['status'] = 'failed; partial outputs preserved'
        save(args.output/'manifest.json', manifest)
        raise
    manifest['status'] = 'complete'
    save(args.output/'manifest.json', manifest)
    (args.output/'_SUCCESS').write_text('All selected scenes rendered with complete native input coverage.\n')
    print(f'Completed: {args.output}', flush=True)


if __name__ == '__main__':
    main()
