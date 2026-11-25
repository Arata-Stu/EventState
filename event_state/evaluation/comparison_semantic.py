"""Continuous Semantic inference and exact-time GT for scene comparison videos.

The prediction interval is a display choice: every prepared input from a
sequence start must still update recurrent state. Runtime dependencies are
imported by the classes, not by this module's provenance helpers.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _resolve(value, project_root):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _safe_load(path, torch):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


class SemanticTargets:
    """Return only an exact timestamp's label, or None when no GT exists."""

    def __init__(self, dataset, config, cfg, project_root, prepared_root,
                 frame_size, known_sequences=()):
        self.dataset = dataset
        self.config = config
        self.cfg = cfg
        self.project_root = Path(project_root)
        self.prepared_root = Path(prepared_root)
        self.frame_size = tuple(frame_size)
        self.known_sequences = set(known_sequences)
        self._sequences = {}

    def _dsec(self, sequence):
        from event_state.segmentation import find_semantic_label_dir

        root = _resolve(self.config['labels_root'], self.project_root)
        directories = []
        for role in ('train', 'test'):
            base = root / role / sequence
            candidates = (base / '11classes', base / 'semantic/left/11classes/data',
                          base / 'semantic/left/11classes')
            found = [p for p in candidates if p.is_dir()]
            if len(found) > 1:
                raise ValueError(f'Ambiguous Semantic label directories: {base}')
            if found:
                directories.append(find_semantic_label_dir(root, role=role, sequence=sequence))
        if not directories:
            return None
        if len(directories) != 1:
            raise ValueError(f'Semantic sequence occurs in both train/test: {sequence}')
        dataset_root = _resolve(self.cfg.dataset.root, self.project_root)
        timelines = [dataset_root / part / sequence / 'images/timestamps.txt'
                     for part in ('train_images', 'test_images')]
        timelines = [p for p in timelines if p.is_file()]
        if len(timelines) != 1:
            raise FileNotFoundError(f'Expected one native RGB timeline for {sequence}')
        stamps = [int(line.strip()) for line in timelines[0].read_text().splitlines() if line.strip()]
        if not stamps or any(b <= a for a, b in zip(stamps, stamps[1:])):
            raise ValueError(f'Invalid Semantic RGB timestamps: {timelines[0]}')
        return {'directory': directories[0], 'indices': {t: i for i, t in enumerate(stamps)}}

    def _m3ed(self, sequence):
        import h5py
        import numpy as np
        try:
            import hdf5plugin  # noqa: F401; register optional HDF5 codecs
        except ImportError:
            pass

        roots = []
        for key in ('extra_target_root', 'extra_target_cache_dir', 'target_cache_dir', 'target_root'):
            if self.config.get(key):
                root = _resolve(self.config[key], self.project_root)
                if root not in roots:
                    roots.append(root)
        candidates = [root / sequence / 'targets.h5' for root in roots]
        for path in candidates:
            if not path.is_file():
                continue
            handle = h5py.File(path, 'r')
            if not {'semantics_11', 'semantics_frame_valid', 'timestamps'} <= set(handle):
                handle.close()
                continue
            try:
                stamps = np.asarray(handle['timestamps'], dtype=np.int64)
                valid = np.asarray(handle['semantics_frame_valid'], dtype=bool)
                labels = handle['semantics_11']
                if (stamps.ndim != 1 or valid.shape != stamps.shape or
                        labels.shape != (len(stamps), *self.frame_size) or
                        (len(stamps) > 1 and (np.diff(stamps) <= 0).any())):
                    raise ValueError(f'Invalid Semantic target geometry/timeline: {path}')
                prepared = _read(self.prepared_root / sequence / 'metadata.json')
                if not np.array_equal(stamps, np.asarray(prepared['timestamps'], dtype=np.int64)):
                    raise ValueError(f'Semantic GT/prepared timeline mismatch: {path}')
            except Exception:
                handle.close()
                raise
            return {'handle': handle, 'indices': {int(t): i for i, t in enumerate(stamps)},
                    'valid': valid}
        if sequence in self.known_sequences:
            raise FileNotFoundError(f'Missing Semantic HDF5 targets for {sequence}: {candidates}')
        return None

    def _source(self, sequence):
        if sequence not in self._sequences:
            self._sequences[sequence] = self._dsec(sequence) if self.dataset == 'dsec' else self._m3ed(sequence)
        return self._sequences[sequence]

    def labeled_timestamps(self, sequence):
        """Scored label timestamps; the first image has no preceding event interval."""
        source = self._source(sequence)
        if source is None:
            return frozenset()
        if self.dataset == 'dsec':
            labeled = {int(path.stem) for path in source['directory'].glob('*.png') if path.stem.isdecimal()}
            return frozenset(t for t, i in source['indices'].items() if i > 0 and i in labeled)
        return frozenset(t for t, i in source['indices'].items() if i > 0 and source['valid'][i])

    def get(self, sequence, timestamp):
        import numpy as np

        timestamp = int(timestamp)
        source = self._source(sequence)
        if source is None or timestamp not in source['indices']:
            return None
        index = source['indices'][timestamp]
        if self.dataset == 'dsec':
            path = source['directory'] / f'{index:06d}.png'
            if not path.is_file():
                return None
            from event_state.segmentation import load_semantic_label
            label = load_semantic_label(path).numpy().astype(np.uint8, copy=False)
        else:
            if not source['valid'][index]:
                return None
            label = np.asarray(source['handle']['semantics_11'][index], dtype=np.uint8)
        if label.shape != self.frame_size or ((label != 255) & (label >= 11)).any():
            raise ValueError(f'Invalid Semantic GT at {sequence}/{timestamp}')
        return label.copy()

    def close(self):
        for source in self._sequences.values():
            if source is not None and 'handle' in source:
                source['handle'].close()
        self._sequences.clear()


class SemanticBackend:
    """One frozen common checkpoint/head pair, with continuous state."""

    def __init__(self, dataset, condition, project_root, teacher_checkpoint=None, device='cuda',
                 *, temporal_run=None, temporal_stage1=None, m3ed_temporal_semantic=None):
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from event_state.dense.common_protocol import audit_common
        from event_state.segmentation import EventStateSegmentationHead
        from event_state.training import build_model, load_checkpoint, load_checkpoint_config_metadata
        from tools import run_common_depth, run_semantic_depth_activity
        from tools.temporal_downstream_semantic import LABEL as convlstm_label

        self.dataset = str(dataset).lower()
        self.condition = condition
        self.project_root = Path(project_root).expanduser().resolve()
        condition_info = {c: (label, feature) for c, label, feature in run_semantic_depth_activity.CONDITIONS}
        condition_info['E1_convlstm'] = (convlstm_label, 'h')
        if self.dataset not in ('dsec', 'm3ed') or condition not in condition_info:
            raise ValueError('Expected a DSEC/M3ED common E0/E1/E2/DDP condition')
        self.label, self.feature = condition_info[condition]
        if condition == 'E1_convlstm':
            if self.dataset == 'm3ed':
                run = _resolve(m3ed_temporal_semantic or 'outputs/stage2_m3ed_semantic_convlstm_20261117', self.project_root)
                activity_report = run / 'h/activity_groups.json'
            else:
                run = _resolve(temporal_run or 'outputs/stage2_temporal_remaining_20261117', self.project_root) / 'dsec_semantic'
                activity_report = run / 'activity_test.json'
        else:
            relative = run_semantic_depth_activity.semantic_run(self.dataset, condition).relative_to(
                run_semantic_depth_activity.ROOT)
            run = self.project_root / relative
            activity_report = self.project_root / 'outputs/semantic_depth_activity_20261110/semantic' / f'{self.dataset}_{condition}.json'
        head_dir = run / (self.feature if self.dataset == 'm3ed' else 'final')
        self.head_path = head_dir / ('best.pt' if self.dataset == 'm3ed' else 'last.pt')
        if not self.head_path.is_file() or not self.head_path.stat().st_size:
            raise FileNotFoundError(f'Missing or empty Semantic head: {self.head_path}')
        self.config = _read(head_dir / 'config.json')
        if self.config.get('feature') != self.feature:
            raise ValueError(f'Semantic readout differs from common condition: {head_dir}')
        self.frame_size = (440, 640) if self.dataset == 'dsec' else (352, 640)
        if tuple(self.config.get('label_size', self.frame_size)) != self.frame_size:
            raise ValueError('Semantic head label geometry differs from the established protocol')
        if int(self.config.get('num_classes', 11)) != 11:
            raise ValueError('Semantic comparison requires DSEC-11 labels')
        cache = _resolve(self.config['feature_cache_dir'], self.project_root)
        metadata = {p.parent.name: _read(p) for p in sorted(cache.glob('*/metadata.json'))}
        if not metadata:
            raise FileNotFoundError(f'No Semantic feature metadata to audit: {cache}')
        expected_hashes = set()
        paths = set()
        self.scored_frame_indices = {}
        for sequence, value in metadata.items():
            if (value.get('task') != self.dataset + '_semantic' or
                    value.get('state_policy') != 'continuous' or
                    tuple(value.get('label_size', ())) != self.frame_size or
                    int(value.get('patch_size', 0)) != 16 or
                    self.feature not in value.get('features', [])):
                raise ValueError(f'Incompatible Semantic cache metadata: {cache / sequence}')
            expected_hashes.add(value['checkpoint_sha256'])
            if 'cached_frame_indices' in value:
                self.scored_frame_indices[sequence] = frozenset(value['cached_frame_indices'])
            if value.get('checkpoint'):
                paths.add(_resolve(value['checkpoint'], self.project_root))
        if len(expected_hashes) != 1 or len(paths) > 1:
            raise ValueError('Semantic cache mixes backbone checkpoints')
        if condition == 'E1_convlstm':
            stage1 = _resolve(temporal_stage1 or 'outputs/stage1_temporal_full_20261115', self.project_root)
            self.backbone_path = stage1 / f'{self.dataset}_E1_convlstm_w1/checkpoints/step_00099999.pt'
        elif paths:
            self.backbone_path = next(iter(paths))
        else:
            spec = next(v for v in run_common_depth.CONDITIONS if v[0] == condition)
            relative = run_common_depth.checkpoint(self.dataset, spec).relative_to(run_common_depth.ROOT)
            self.backbone_path = self.project_root / relative
        backbone_sha = _sha256(self.backbone_path)
        if backbone_sha != next(iter(expected_hashes)):
            raise ValueError('Semantic cache/backbone SHA256 mismatch')
        evaluated = _read(activity_report)['activity_evaluation']
        head_sha = _sha256(self.head_path)
        if evaluated.get('condition_label') != self.label or evaluated.get('head_sha256') != head_sha:
            raise ValueError('Semantic head differs from the completed activity evaluation')
        raw = copy.deepcopy(load_checkpoint_config_metadata(self.backbone_path).config)
        audit_common(raw)
        if condition == 'E1_convlstm':
            from tools.run_temporal_downstream import validate_backbone_config
            validate_backbone_config(raw, self.dataset)
        self.cfg = OmegaConf.create(raw)
        if (self.cfg.dataset.name != self.dataset or
                self.cfg.dataset.representation.type != 'gep_rgb' or
                self.cfg.dataset.get('event_window_us') is not None or
                float(self.cfg.dataset.get('event_window_fraction', 1)) != 1):
            raise ValueError('Expected a common RGB-interval GEP checkpoint')
        expected_input = (448, 640) if self.dataset == 'dsec' else (352, 640)
        if tuple(int(self.cfg.dataset[k]) for k in ('input_height', 'input_width')) != expected_input:
            raise ValueError('Semantic backbone input geometry differs from the common protocol')
        if teacher_checkpoint is not None:
            teacher = _resolve(teacher_checkpoint, self.project_root)
            if not teacher.is_file():
                raise FileNotFoundError(teacher)
            OmegaConf.update(self.cfg, 'teacher.checkpoint', str(teacher), force_add=True)
            OmegaConf.update(self.cfg, 'model.event_encoder.checkpoint', str(teacher), force_add=True)
        prepared = self.cfg.dataset.get('prepared_root') if self.dataset == 'm3ed' else None
        self.prepared_root = _resolve(prepared or self.cfg.dataset.event_cache_dir, self.project_root)
        self._torch, self._np = torch, np
        self.device = torch.device(device)
        self.model = build_model(self.cfg, device=self.device)
        load_checkpoint(self.backbone_path, model=self.model, device=self.device, restore_rng=False)
        self.model.eval().requires_grad_(False)
        channels = self.model.event_dim if self.feature == 'z' else self.model.temporal_dim
        if int(self.config.get('in_channels', channels)) != channels:
            raise ValueError('Semantic head/backbone channel mismatch')
        self.head = EventStateSegmentationHead(
            in_channels=channels, num_classes=11, width=int(self.config.get('head_width', 192)),
            output_size=self.frame_size, head_type=self.config.get('head_type', 'linear')).to(self.device)
        payload = _safe_load(self.head_path, torch)
        for key in ('feature', 'in_channels', 'num_classes', 'head_type', 'head_width', 'label_size',
                    'feature_cache_dir', 'split_manifest', 'labels_root', 'target_cache_dir'):
            if key in payload.get('config', {}) and key in self.config and payload['config'][key] != self.config[key]:
                raise ValueError(f'Semantic saved head/config mismatch: {key}')
        self.head.load_state_dict(payload['model'], strict=True)
        self.head.eval().requires_grad_(False)
        self.use_amp = self.config.get('precision', 'fp16') == 'fp16' and self.device.type == 'cuda'
        self.targets = SemanticTargets(self.dataset, self.config, self.cfg, self.project_root,
                                       self.prepared_root, self.frame_size, metadata)
        self.identity = dict(task='semantic', dataset=self.dataset, condition=condition,
            condition_label=self.label, feature=self.feature, state_policy='continuous',
            backbone=str(self.backbone_path), backbone_sha256=backbone_sha,
            head=str(self.head_path), head_sha256=head_sha,
            activity_report=str(activity_report), activity_report_sha256=_sha256(activity_report),
            feature_cache=str(cache), input_size=list(expected_input), label_size=list(self.frame_size),
            feature_rounding='float16 then float32, matching the frozen feature cache',
            backbone_precision='fp32', head_precision=self.config.get('precision', 'fp16'),
            head_batch_size=1, label_source='official DSEC-11' if self.dataset == 'dsec' else 'InternImage pseudo-labels')
        self.reset()

    def reset(self):
        self.state = None
        self._sequence = None
        self._timestamp = None
        self._frame_index = 0

    def fork_without_history(self):
        """Start a fresh recurrent state at the next input, sharing weights and target reader.

        The original backend owns the shared resources and must outlive the branch.
        """
        branch = copy.copy(self)
        branch.state = None
        return branch

    def step(self, frame_dict, predict=True):
        torch, np = self._torch, self._np
        sequence = frame_dict['sequence_name']
        timestamp, previous = int(frame_dict['timestamp']), int(frame_dict['previous_timestamp'])
        index = int(frame_dict['frame_index'])
        if index != self._frame_index + 1 or timestamp <= previous:
            raise ValueError('Semantic inference must process contiguous inputs from sequence start')
        if self._sequence is not None and (sequence != self._sequence or previous != self._timestamp):
            raise ValueError('Semantic recurrent history changed; reset only between sequences')
        events = frame_dict['events'].float()
        expected = (3, 480 if self.dataset == 'dsec' else 360, 640)
        if tuple(events.shape) != expected or not torch.isfinite(events).all() or ((events < 0) | (events > 1)).any():
            raise ValueError('Semantic input must be a finite unnormalized native GEP frame')
        if self.dataset == 'dsec':
            from tools.cache_dsec_semantic_features import _semantic_transform
            value = _semantic_transform(events, self.cfg).to(self.device)
        else:
            value = events[:, 4:356].to(self.device)
            mean = value.new_tensor(list(self.cfg.dataset.representation.normalize_mean))[:, None, None]
            std = value.new_tensor(list(self.cfg.dataset.representation.normalize_std))[:, None, None]
            value = ((value - mean) / std)[None]
        with torch.inference_mode():
            outputs = self.model(value[:, None], state=self.state)
            self.state = outputs['state']
            self._sequence, self._timestamp, self._frame_index = sequence, timestamp, index
            if not predict:
                return None
            from event_state.models.tokens import tokens_to_map
            grid = tuple(v // int(self.model.event_encoder.patch_size) for v in value.shape[-2:])
            features = tokens_to_map(outputs[self.feature][0, 0], grid).half().float()[None]
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits = self.head(features)
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite Semantic logits; failed predictions must not be omitted')
            return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)

    def ground_truth(self, sequence, timestamp):
        return self.targets.get(sequence, timestamp)

    gt = ground_truth

    def close(self):
        self.targets.close()
        self.state = None
        self.model = None
        self.head = None
