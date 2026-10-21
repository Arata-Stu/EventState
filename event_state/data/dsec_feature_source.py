"""Online fixed-window GEP source for downstream feature exporters.

Uses the pretraining dataset's exact reader/rectification/window implementation.
No persistent event image cache is created.
"""
from pathlib import Path
from omegaconf import OmegaConf
from .dsec import DSECSequenceDataset, event_representation_metadata
from .transforms import PairedSequenceTransform


def validate_source(config, event_cache, raw_root):
    window = OmegaConf.select(config, 'dataset.event_window_us')
    if raw_root is not None:
        if event_cache is not None:
            raise ValueError('Choose raw-event-root OR event-cache-dir, never both')
        if window is None:
            raise ValueError('Raw source requires a fixed-window checkpoint')
        if OmegaConf.select(config, 'dataset.name') != 'dsec':
            raise ValueError('Raw fixed-window source requires DSEC')
        if (OmegaConf.select(config, 'dataset.representation.type') != 'gep_rgb'
                or not OmegaConf.select(config, 'dataset.rectify_events', default=True)):
            raise ValueError('Raw fixed-window source requires rectified GEP RGB')
    elif event_cache is None or window is not None:
        raise ValueError('Fixed-window checkpoint needs --raw-event-root; legacy checkpoint needs --event-cache-dir')


class RawGEPFeatureSource:
    def __init__(self, root, sequence, config):
        from event_state.training.data import build_event_representation
        self.root = Path(root).resolve()
        matches = [(s, None) for s in ('train', 'test') if
                   (self.root/f'{s}_events'/sequence).is_dir() and
                   (self.root/f'{s}_images'/sequence).is_dir()]
        matches += [(s, self.root/'dsec_det_extra'/s/sequence) for s in ('train', 'test')
                    if (self.root/'dsec_det_extra'/s/sequence/'images').is_dir()
                    and (self.root/'dsec_det_extra'/s/sequence/'events').is_dir()]
        if len(matches) != 1:
            raise ValueError(f'Expected one physical DSEC split for {sequence}: {matches}')
        split, extra_root = matches[0]
        representation = build_event_representation(config.dataset)
        self.dataset = DSECSequenceDataset(root=self.root, split=split, sequences=[sequence],
            **({"sequence_root": extra_root} if extra_root is not None else {}),
            sequence_length=1, event_representation=representation,
            transform=PairedSequenceTransform(height=480, width=640, training=False),
            image_directory=OmegaConf.select(config, 'dataset.image_directory', default='aligned_event'),
            rectify_events=True, load_images=False, event_cache_dir=None, feature_cache_dir=None,
            event_window_fraction=float(OmegaConf.select(config, 'dataset.event_window_fraction', default=1)),
            event_window_us=OmegaConf.select(config, 'dataset.event_window_us'))
        frames = self.dataset._frames_by_sequence[sequence][1:]
        self.records = {str(r.timestamp): r for r in frames}
        if len(self.records) != len(frames):
            raise ValueError('Duplicate frame timestamps')
        self.source_dir = extra_root if extra_root is not None else self.root/f'{split}_events'/sequence
        self.paths = [self.source_dir/f'{r.timestamp}.pt' for r in frames]  # identity only, not files
        self.metadata = dict(source='raw_fixed_window_gep_v1', physical_split=split,
            sequence_name=sequence, representation=event_representation_metadata(representation),
            event_window_us=int(config.dataset.event_window_us), event_window_boundary='(t-window_us,t]',
            timestamps=[r.timestamp for r in frames], frame_indices=[r.frame_index for r in frames],
            root=str(self.root), inputs={})
        # Input identity invalidates completed features if underlying raw sources change.
        for path in sorted(self.source_dir.rglob('*.h5')):
            st = path.stat()
            self.metadata['inputs'][str(path)] = [st.st_size, st.st_mtime_ns]
        image_base = extra_root/'images' if extra_root is not None else self.root/f'{split}_images'/sequence
        for path in sorted(image_base.rglob('*')):
            if path.is_file() and path.suffix in {'.txt', '.json'}:
                st = path.stat()
                self.metadata['inputs'][str(path)] = [st.st_size, st.st_mtime_ns]

    def load(self, path):
        record = self.records[path.stem]
        events, count = self.dataset._load_event_frame(record)
        return dict(events=events, event_count=count, frame_index=record.frame_index,
                    timestamp=record.timestamp, previous_timestamp=record.previous_timestamp,
                    event_window_us=self.dataset.event_window_us)

    def close(self):
        self.dataset.close()
