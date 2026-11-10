"""Exercise the full orchestration without GPU, checkpoint downloads or dataset writes."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
import unittest

from tools import run_common_depth as runner
from event_state.dense.depth_frame_metrics import F3_FRAME_PROTOCOL


class DepthRunTest(unittest.TestCase):
    @staticmethod
    def activity_payload(identity, label, *, frame_metrics=True):
        payload = dict(metrics=dict(all=dict(mae=9)), protocol=dict(dataset='dsec'),
            task='depth', role='val', head_sha256='fixture', backbone_sha256='fixture',
            activity_evaluation=dict(activity_manifest=identity, condition_label=label,
                groups=dict(all=dict(metrics=dict(mae=9)))))
        if frame_metrics:
            payload['depth_frame_metric_protocol'] = F3_FRAME_PROTOCOL
            payload['metrics_frame_mean_f3'] = dict(all=dict(mae=5))
            payload['activity_evaluation']['groups']['all']['metrics_frame_mean_f3'] = dict(mae=5)
        return payload

    def test_all_models_and_public_val_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(output=root, scratch=root, data=root, m3ed_targets=root, m3ed_val_targets=root/'val_gt',
                                   weights=root/'weights', baselines_only=False, local_only=False)
            trained, submitted = [], []
            def fake(tool, *cmd):
                def option(key):
                    return cmd[cmd.index(key)+1]
                if tool == 'prepare_dense_targets.py':
                    if option('--protocol') == 'm3ed_f3':
                        self.assertEqual(option('--target-root'), root/'val_gt' if option('--role') == 'val' else root)
                    f = option('--output')
                    f.parent.mkdir(exist_ok=True)
                    f.write_text('{"records": []}')
                elif tool == 'train_dense_probe.py':
                    trained.append(option('--output-dir').parent.name)
                    self.assertEqual(option('--dav2-weights'), args.weights)
                    option('--output-dir').mkdir(parents=True)
                    (option('--output-dir')/'best.pt').touch()
                    (option('--output-dir')/'_SUCCESS').touch()
                elif tool == 'evaluate_dense_probe.py':
                    option('--output').write_text('{"metrics":{"all":{"mae":1}}}')
                elif tool == 'evaluate_dsec_disparity_window.py':
                    value = 0.5 if option('--name') == 'dsec_E2' else 1
                    option('--output').write_text(json.dumps({'mean_sequence_mae_px':value}))
                elif tool == 'export_dsec_depth_submission.py' and '--preflight' not in cmd:
                    submitted.append(option('--name'))
                    option('--output').mkdir()
                    (option('--output')/'submission.zip').touch()
            with patch.object(runner,'verify_decoder'), \
                 patch('event_state.dense.protocol.sha256_file', return_value='fixture'), \
                 patch.object(runner,'timestamps',return_value=root), patch.object(runner,'run',fake):
                runner.pipeline(args)
                runner.pipeline(args)
            self.assertEqual(trained,[ds+'_'+c[0] for ds in ('dsec','m3ed') for c in runner.CONDITIONS])
            self.assertEqual(submitted,['dsec_E2'])

    def test_activity_only_reuses_heads_and_extracts_only_selected_validation(self):
        self.exercise_activity_only()

    def test_legacy_activity_report_is_preserved_and_frame_mean_sibling_is_reused(self):
        self.exercise_activity_only(legacy=True)

    def exercise_activity_only(self, legacy=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(output=root, scratch=root, data=root, m3ed_targets=root,
                m3ed_val_targets=root, weights=root/'missing-initial-weights', baselines_only=True,
                local_only=False, activity_only=True, dsec_activity_manifest=root/'activity.json',
                m3ed_activity_manifest=None)
            target_dir = root/'targets'
            target_dir.mkdir()
            (target_dir/'dsec_val.json').write_text('{"records": [{}]}')
            (root/'run.json').write_text('existing run identity')
            (root/'summary.json').write_text('existing summary')
            original_reports = {}
            fixture = SimpleNamespace(identity=dict(manifest_sha256='fixed-activity'))
            for condition in runner.CONDITIONS[:3]:
                head = root/'heads'/('dsec_'+condition[0])/'stages'
                head.mkdir(parents=True)
                (head/'best.pt').touch()
                (head/'_SUCCESS').touch()
                report = head/'continuous.json'
                report.write_text('{"metrics":{"all":{"mae":9}}}')
                original_reports[report] = report.read_bytes()
                if legacy:
                    old_activity = head/'activity_continuous.json'
                    old_activity.write_text(json.dumps(self.activity_payload(fixture.identity,
                        condition[1], frame_metrics=False)))
                    old_csv = old_activity.with_suffix('.csv')
                    old_csv.write_text('old pooled-only CSV')
                    for original in (old_activity, old_csv):
                        original_reports[original] = original.read_bytes()
            extracted, scored = [], []
            def fake(tool, *cmd):
                def option(key):
                    return cmd[cmd.index(key)+1]
                if tool == 'cache_dense_features.py':
                    extracted.append(option('--target-index').name)
                    self.assertEqual(option('--target-index'), target_dir/'dsec_val.json')
                    option('--output-dir').mkdir()
                elif tool == 'evaluate_dense_probe.py':
                    self.assertIn('--activity-manifest', cmd)
                    self.assertEqual(option('--activity-manifest'), args.dsec_activity_manifest)
                    scored.append(option('--output').name)
                    option('--output').write_text(json.dumps(self.activity_payload(fixture.identity,
                        option('--condition-label'))))
                else:
                    self.fail(f'Activity-only evaluation must not call {tool}')
            with patch.object(runner, 'run', fake), patch.object(runner, 'verify_decoder') as verify, \
                 patch.object(runner, 'timestamps') as timestamps, \
                 patch('event_state.dense.protocol.sha256_file', return_value='fixture'), \
                 patch('event_state.activity_reporting.load_activity_options', return_value=fixture):
                runner.pipeline(args)
                runner.pipeline(args)
            self.assertEqual(extracted, ['dsec_val.json'] * 3)
            expected_name = 'activity_continuous_frame_mean.json' if legacy else 'activity_continuous.json'
            self.assertEqual(scored, [expected_name] * 3)
            verify.assert_not_called()
            timestamps.assert_not_called()
            self.assertEqual((root/'run.json').read_text(), 'existing run identity')
            self.assertEqual((root/'summary.json').read_text(), 'existing summary')
            for path, content in original_reports.items():
                self.assertEqual(path.read_bytes(), content)
            summary = json.loads((root/'activity_summary.json').read_text())
            self.assertEqual(set(summary), {'dsec_E0', 'dsec_E1', 'dsec_E2'})
            self.assertIn('activity_groups', summary['dsec_E1'])
            self.assertEqual(summary['dsec_E1']['overall_frame_mean_f3'], dict(mae=5))
            self.assertEqual(Path(summary['dsec_E1']['activity_report']).name, expected_name)

    def test_activity_only_requires_manifest_and_completed_existing_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(output=root, scratch=root, data=root, baselines_only=True,
                activity_only=True, dsec_activity_manifest=None, m3ed_activity_manifest=None)
            with self.assertRaisesRegex(ValueError, 'at least one'):
                runner.pipeline(args)
            args.dsec_activity_manifest = root/'activity.json'
            args.local_only = True
            args.m3ed_targets = root
            args.m3ed_val_targets = root
            (root/'targets').mkdir()
            (root/'targets/dsec_val.json').write_text('{"records": []}')
            with patch('event_state.activity_reporting.load_activity_options',
                       return_value=SimpleNamespace(identity={})), \
                 patch.object(runner, 'run') as launch:
                with self.assertRaisesRegex(ValueError, 'completed existing head'):
                    runner.pipeline(args)
                launch.assert_not_called()

    def test_cached_activity_report_rejects_stale_model_or_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(output=root, scratch=root, data=root, baselines_only=True,
                activity_only=True, dsec_activity_manifest=root/'activity.json', m3ed_activity_manifest=None,
                local_only=True, m3ed_targets=root, m3ed_val_targets=root)
            (root/'targets').mkdir()
            (root/'targets/dsec_val.json').write_text('{"records": []}')
            head = root/'heads/dsec_E0/stages'
            head.mkdir(parents=True)
            (head/'best.pt').touch()
            (head/'_SUCCESS').touch()
            identity = dict(manifest_sha256='fixed')
            payload = dict(protocol=dict(dataset='dsec'), task='depth', role='val',
                head_sha256='fixture', backbone_sha256='fixture',
                activity_evaluation=dict(activity_manifest=identity, condition_label=runner.CONDITIONS[0][1]))
            for field, value in [('head_sha256', 'old'), ('backbone_sha256', 'old'),
                                 ('task', 'disparity'), ('role', 'test'),
                                 ('protocol', dict(dataset='m3ed'))]:
                with self.subTest(field=field):
                    (head/'activity_continuous.json').write_text(json.dumps({**payload, field: value}))
                    with patch('event_state.activity_reporting.load_activity_options',
                               return_value=SimpleNamespace(identity=identity)), \
                         patch('event_state.dense.protocol.sha256_file', return_value='fixture'), \
                         patch.object(runner, 'run') as launch:
                        with self.assertRaisesRegex(ValueError, 'stale head/backbone'):
                            runner.pipeline(args)
                        launch.assert_not_called()

    def test_activity_report_rejects_changed_frame_metric_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            head = Path(tmp)
            identity = dict(manifest_sha256='fixed')
            payload = self.activity_payload(identity, runner.CONDITIONS[0][1])
            payload['depth_frame_metric_protocol'] = dict(version='different')
            report = head/'activity_continuous.json'
            report.write_text(json.dumps(payload))
            with patch('event_state.dense.protocol.sha256_file', return_value='fixture'):
                with self.assertRaisesRegex(ValueError, 'different depth frame metric protocol'):
                    runner.activity_report_path(head, dataset='dsec', label=runner.CONDITIONS[0][1],
                        activity_index=SimpleNamespace(identity=identity), backbone=head/'backbone.pt')
