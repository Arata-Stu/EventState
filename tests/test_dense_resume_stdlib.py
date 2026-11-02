import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('resume_dense', Path(__file__).resolve().parents[1] / 'tools/resume_dense_step.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ResumeTests(unittest.TestCase):
    def test_existing_head_uses_explicit_resume(self):
        with tempfile.TemporaryDirectory() as root:
            cmd = ['python', 'tools/train_dense_probe.py', '--output-dir', root]
            self.assertEqual(m.prepare(cmd), cmd + ['--resume'])

    def test_incomplete_cache_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(RuntimeError):
                m.prepare(['python', 'tools/cache_dense_features.py', '--output-dir', root])
            self.assertTrue(Path(root).exists())

    def test_evaluation_requires_same_head_and_cache(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            checkpoint = root / 'best.pt'; checkpoint.write_bytes(b'weights')
            cache = root / 'cache'; cache.mkdir()
            (cache / 'metadata.json').write_text('{}')
            output = root / 'eval.json'
            output.write_text(json.dumps(dict(head_sha256=m.sha(checkpoint), cache_metadata_sha256=m.sha(cache / 'metadata.json'))))
            cmd = ['python', 'tools/evaluate_dense_probe.py', '--checkpoint', str(checkpoint), '--cache', str(cache), '--output', str(output)]
            self.assertIsNone(m.prepare(cmd))
            checkpoint.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                m.prepare(cmd)


if __name__ == '__main__':
    unittest.main()
