"""Check orchestration without executing training or importing ML runtimes."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class NaturalPipeline(unittest.TestCase):
    def test_natural_has_no_reset_and_summary_needs_no_cache(self):
        script = Path(__file__).resolve().parents[1] / 'tools/run_dense_f3_pipeline.sh'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / 'bin'
            binary.mkdir()
            stub = binary / 'python'
            stub.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALLS"\n')
            stub.chmod(0o755)
            checkpoint = root / 'pretrain/dsec_baseline/checkpoints/step_00100000.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text('fixture, never loaded')
            calls = root / 'calls'
            env = dict(os.environ, PATH=f'{binary}:{os.environ["PATH"]}', CALLS=str(calls),
                       DATASETS='dsec', CONDITIONS='baseline')
            subprocess.run(['bash', str(script), 'natural', str(root)], env=env,
                           check=True, capture_output=True, text=True)
            rows = calls.read_text().splitlines()
            self.assertEqual(sum('cache_dense_features.py' in r for r in rows), 2)
            self.assertEqual(sum('train_dense_probe.py' in r for r in rows), 3)
            self.assertEqual(sum('evaluate_dense_probe.py' in r for r in rows), 3)
            self.assertEqual(sum('--mode readout' in r for r in rows), 2)
            self.assertFalse(any('--state-policy frame' in r or '/frame.json' in r for r in rows))
            calls.unlink()
            checkpoint.unlink()
            subprocess.run(['bash', str(script), 'natural-summary', str(root)], env=env,
                           check=True, capture_output=True, text=True)
            rows = calls.read_text().splitlines()
            self.assertEqual(len(rows), 2)
            self.assertTrue(all('--mode readout' in r for r in rows))
