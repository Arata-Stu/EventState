import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from compare_semantic_activity_probes import compare
from run_semantic_state_probe import summarize


class ActivityComparisonTest(unittest.TestCase):
    def test_separate_weights_same_population_and_empty_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, score in [('base', .35), ('alpha', .37)]:
                directory = root/name/'continuous'; directory.mkdir(parents=True)
                for feature in ('z','h','concat'):
                    metric = dict(mIoU=score, class_support=[4,5], class_IoU=[score,score], evaluated_pixels=9, frames=1)
                    empty = dict(mIoU=None, class_support=[0,0], class_IoU=[None,None], evaluated_pixels=0, frames=0)
                    result = dict(dataset='m3ed',role='validation',sequences=['seq'],state_policy='continuous',feature=feature,
                        samples_sha256='same',activity_sha256='same',low_activity_threshold=.1,
                        backbone_sha256=name,head_sha256=name+feature,metrics=dict(all=metric,low_activity_frames=empty))
                    (directory/f'{feature}.json').write_text(json.dumps(result))
            rows = compare(root/'base/continuous',root/'alpha/continuous')
            self.assertAlmostEqual(rows[0]['delta_points'],2)
            self.assertIsNone(rows[1]['delta_points'])
            summarize(root,[dict(name='alpha')],['continuous'])
            self.assertEqual(len(json.loads((root/'summary.json').read_text())),6)
            path = root/'alpha/continuous/z.json'
            result = json.loads(path.read_text());result['activity_sha256']='different'
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'activity_sha256'):
                compare(root/'base/continuous',root/'alpha/continuous')

if __name__ == '__main__': unittest.main()
