import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from run_hybrid_target_downstream import model_plan


class GEP20PlanTest(unittest.TestCase):
    def test_raw_source_twelve_heads_activity_and_validation(self):
        total = 0
        for name in ('baseline','alpha1'):
            stages, results = model_plan('dsec_gep20_'+name, Path('/pre/checkpoint.pt'),Path('/data'),
                                        Path('/teacher.pt'),Path('/scratch'),Path('/output'))
            self.assertEqual(len(results),6)
            total += len(results)
            self.assertEqual(sum('semantic_activity_' in label for label,_,_ in stages),3)
            for label, cmd, _ in stages:
                if '_cache_' in label:
                    self.assertIn('--raw-event-root',cmd)
                    self.assertNotIn('--event-cache-dir',cmd)
                    self.assertIn(cmd[cmd.index('--role')+1],('train','val'))
                    if label.startswith('semantic'): self.assertIn('--include-activity',cmd)
                if '_train_' in label:
                    self.assertEqual(cmd[cmd.index('--epochs')+1],'50')
                    self.assertEqual(cmd[cmd.index('--seed')+1],'0')
            self.assertTrue(all(r[-1]=='val' for r in results))
        self.assertEqual(total,12)

    def test_legacy_plan_unchanged_source(self):
        stages,_ = model_plan('dsec_z_h',Path('/pre'),Path('/data'),Path('/teacher'),Path('/scratch'),Path('/output'))
        for label,cmd,_ in stages:
            if '_cache_' in label:
                self.assertIn('--event-cache-dir',cmd)
                self.assertNotIn('--raw-event-root',cmd)

if __name__ == '__main__': unittest.main()
