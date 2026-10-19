import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile
import json

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('state_runner', TOOLS/'run_semantic_state_probe.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class StateProbePlanTest(unittest.TestCase):
    def test_validation_only_and_fixed_heads(self):
        for model in runner.default_models():
            for policy in ('continuous', 'frame'):
                stages = runner.commands(model, policy, Path('/scratch/cache'), Path('/out'), Path('/data'), Path('/teacher.pt'), .1)
                self.assertEqual(len(stages), 4)
                for label, command in stages:
                    self.assertNotIn('--epochs', command)
                    self.assertIn(policy, command)
                    if label == 'cache':
                        self.assertIn('--include-activity', command)
                        role = command[command.index('--role')+1]
                        self.assertIn(role, ('val', 'validation'))
                    else:
                        self.assertTrue(command[command.index('--checkpoint')+1].endswith(f'{label}/seed_0/best.pt'))


    def test_summary_pairs_and_rejects_changed_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = [dict(name='model')]
            for policy in ('continuous', 'frame'):
                directory = root/'model'/policy
                directory.mkdir(parents=True)
                for feature in ('z', 'h', 'concat'):
                    result = dict(samples_sha256='samples', activity_sha256='mask',
                        backbone_sha256='backbone', head_sha256=feature, low_activity_threshold=.1,
                        z_features_sha256='same', metrics={'all':dict(mIoU=.6 if policy == 'continuous' else .5,
                            class_support=[3, 4], evaluated_pixels=7)})
                    (directory/f'{feature}.json').write_text(json.dumps(result))
            runner.summarize(root, models)
            self.assertAlmostEqual(json.loads((root/'summary.json').read_text())[0]['delta_points'], 10)
            path = root/'model/frame/z.json'
            result = json.loads(path.read_text()); result['activity_sha256'] = 'changed'
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'activity_sha256'):
                runner.summarize(root, models)


if __name__ == '__main__':
    unittest.main()
