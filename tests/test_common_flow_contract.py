import sys
import unittest
from pathlib import Path
from event_state.flow.protocol import parse_rows,splits
from event_state.dense.common_protocol import load_common

class FlowContract(unittest.TestCase):
    def test_development_unseen(self):
        s=splits()
        self.assertEqual([len(s[k]) for k in ('train','val','test')],[14,4,7])
        self.assertTrue(set(s['train']).isdisjoint(s['val']))
        for k in ('val','test'):
            self.assertTrue(set(s[k]).isdisjoint(load_common('dsec')['splits']['train']))
    def test_reference_target_index(self):
        self.assertEqual(parse_rows('# times\n100, 201, 80\n600, 698, 90',True),[(100,201,80),(600,698,90)])
        for line in ('100, 100, 80','100, 95, 80','100, 200','100, 200, 80\n600, 700, 80'):
            with self.assertRaises(ValueError): parse_rows(line,True)

    def test_gt_pair_map_uses_numeric_image_order(self):
        import tempfile
        from event_state.flow.protocol import labeled_pairs
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'sample'/'flow'
            (folder/'forward').mkdir(parents=True)
            (folder/'forward_timestamps.txt').write_text('# from, to\n10, 110\n110, 210\n')
            for i in ('10','2'): (folder/'forward'/f'{i}.png').touch()
            rows=labeled_pairs(root,'sample')
            self.assertEqual([(u,v,Path(f).stem) for u,v,f in rows],[(10,110,'2'),(110,210,'10')])
            (folder/'forward'/'2.png').unlink()
            with self.assertRaises(ValueError): labeled_pairs(root,'sample')
