"""Check the four-stage Depth condition and its matched target indexes without Torch."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import run_common_depth as runner


class FourStageDepthTests(unittest.TestCase):
    def test_selects_only_the_full_four_stage_checkpoint_for_each_dataset(self):
        args = SimpleNamespace(four_stage_placement=True, baselines_only=False)
        condition, = runner._conditions(args)
        key = 'E1_b3_6_9_12'
        self.assertEqual(condition[0], key)
        self.assertIn(key, condition[1])
        self.assertIn('E1 Encoder＋LSTM（h蒸留）', condition[1])
        for dataset in ('dsec', 'm3ed'):
            expected = (runner.ROOT / 'outputs/stage1_placement_full_20261118'
                        / f'{dataset}_{key}_w1/checkpoints/step_00099999.pt')
            self.assertEqual(runner.checkpoint(dataset, condition), expected)
        for baselines_only in (False, True):
            legacy = runner._conditions(SimpleNamespace(baselines_only=baselines_only))
            self.assertEqual(len(legacy), 3 if baselines_only else 5)
            self.assertNotIn(key, [item[0] for item in legacy])

    def test_fresh_jobs_train_both_heads_score_activity_and_use_public_val(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                four_stage_placement=True, baselines_only=False, output=root/'run', data=root,
                dsec_activity_manifest=root/'dsec.json', m3ed_activity_manifest=root/'m3ed.json',
            )
            indexes = {}
            for ds in ('dsec', 'm3ed'):
                for role in ('train', 'val'):
                    path = root / f'{ds}_{role}.json'
                    path.write_text(json.dumps({'records': [{'sequence': f'{ds}_{role}'}]}))
                    indexes[ds, role] = path
            jobs = runner.build_jobs(args, indexes, {'dsec': object(), 'm3ed': object()})
            self.assertEqual({job['name'] for job in jobs},
                             {'dsec_E1_b3_6_9_12', 'm3ed_E1_b3_6_9_12'})
            self.assertEqual(len(jobs), 2)
            for job in jobs:
                ds = job['dataset']
                self.assertEqual(job['kind'], 'depth')
                self.assertEqual(job['roles'], ['train', 'val'])
                self.assertTrue(job['need_main'] and job['need_activity'])
                self.assertEqual(job['need_disparity'], ds == 'dsec')
                self.assertEqual(job['indexes'], {role: str(indexes[ds, role]) for role in ('train', 'val')})
                self.assertEqual(Path(job['head']), args.output/'heads'/job['name']/'stages')
            self.assertFalse(args.output.exists())

    def test_target_index_preview_copy_reuse_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'dsec_val.json'
            destination = root/'new_run/targets/dsec_val.json'
            original = b'{"dataset":"dsec","role":"val","records":[]}\n'
            source.write_bytes(original)
            self.assertEqual(runner._placement_index(source, destination, True), source)
            self.assertFalse(destination.parent.exists())
            runner._placement_index(source, destination, False)
            self.assertEqual(destination.read_bytes(), original)
            saved = destination.stat()
            runner._placement_index(source, destination, False)
            self.assertEqual(destination.stat().st_ino, saved.st_ino)
            self.assertEqual(destination.stat().st_mtime_ns, saved.st_mtime_ns)
            changed = b'{"dataset":"dsec","role":"val","records":[{}]}\n'
            destination.write_bytes(changed)
            for preflight in (True, False):
                with self.subTest(preflight=preflight), self.assertRaises(ValueError):
                    runner._placement_index(source, destination, preflight)
                self.assertEqual(destination.read_bytes(), changed)
                self.assertEqual(source.read_bytes(), original)

    def test_preflight_matches_controls_without_creating_a_new_run(self):
        from event_state.dense.protocol import sha256_file
        from tools.placement_depth_comparison import baseline_record
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = root / runner.COMMON_OUTPUT
            targets = control / 'targets'
            targets.mkdir(parents=True)
            weights = root / 'weights'
            weights.write_bytes(b'DPT weights')
            (control/'run.json').write_text(json.dumps({'weights': sha256_file(weights)}))
            args = SimpleNamespace(four_stage_placement=True, preflight=True, local_only=True, data=root,
                                   m3ed_targets=root, m3ed_val_targets=root, weights=weights, output=root/'new',
                                   dsec_activity_manifest=root/'dsec.json', m3ed_activity_manifest=root/'m3ed.json')
            activity = {}
            with patch.object(runner, 'ROOT', root), patch.object(runner, '_activity_indexes', return_value=activity):
                for ds in ('dsec', 'm3ed'):
                    ckpt = runner.checkpoint(ds, runner._conditions(args)[0])
                    ckpt.parent.mkdir(parents=True)
                    ckpt.write_bytes(b'Stage 1')
                    activity[ds] = SimpleNamespace(identity={'manifest_sha256': baseline_record(ds)['manifest_sha256']})
                    for role in ('train', 'val'):
                        (targets/f'{ds}_{role}.json').write_text('{"records":[{}]}')
                jobs, export = runner._prepare(args)
                self.assertIsNone(export)
                self.assertEqual(len(jobs), 2)
                self.assertFalse(args.output.exists())
                activity['dsec'].identity['manifest_sha256'] = 'other activity fit'
                with self.assertRaisesRegex(ValueError, 'activity manifest'):
                    runner._prepare(args)


if __name__ == '__main__':
    unittest.main()
