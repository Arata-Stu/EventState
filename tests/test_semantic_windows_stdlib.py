"""Portable guards; actual inference/video encoding must be tested on the server."""
import copy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'tools/visualize_dsec_semantic_windows.py'
spec = importlib.util.spec_from_file_location('semantic_windows', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WindowGuards(unittest.TestCase):
    def test_complete_ordered_timeline_required(self):
        paths = [Path('100.pt'), Path('200.pt')]
        self.assertEqual(module.paired_paths(paths, paths), list(zip(paths, paths)))
        for other in ([], paths[1:], paths[::-1], paths + [paths[-1]]):
            with self.assertRaises(ValueError):
                module.paired_paths(paths, other)
        with self.assertRaises(ValueError):
            module.paired_paths(paths[::-1], paths[::-1])

    def test_window_configuration(self):
        legacy = dict(dataset=dict(name='dsec', representation=dict(type='gep_rgb')))
        short = copy.deepcopy(legacy)
        short['dataset']['event_window_us'] = 20000
        module.validate_windows(legacy, short)
        for key, value in [('event_window_us', 50000), ('event_window_fraction', .4),
                           ('rectify_events', False)]:
            bad = copy.deepcopy(short)
            bad['dataset'][key] = value
            with self.assertRaises(ValueError):
                module.validate_windows(legacy, bad)
        with self.assertRaises(ValueError):
            module.validate_windows(short, short)
