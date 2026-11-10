"""Exercise the full orchestration without GPU, checkpoint downloads or dataset writes."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
import unittest

from tools import run_common_depth as runner


class DepthRunTest(unittest.TestCase):
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
