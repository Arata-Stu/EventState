import argparse
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from export_dsec_depth_submission import parse_window_ms, window_bounds
from event_state.dense.protocol import audit_pretraining,load_protocol


class SubmissionRules(unittest.TestCase):
    def test_irregular_updates_and_longer_window(self):
        self.assertEqual(window_bounds(101000,149000,parse_window_ms('20')), (129000,149000))
        self.assertEqual(window_bounds(101000,149000,parse_window_ms('70')), (79000,149000))
        self.assertEqual(window_bounds(101000,149000,parse_window_ms('rgb-interval')), (101000,149000))
        for bad in ('0','-3','20,50','nan'):
            with self.assertRaises(argparse.ArgumentTypeError): parse_window_ms(bad)

    def test_hidden_test_in_unlabeled_earlier_training_rejected(self):
        protocol=load_protocol('dsec_f3')
        train=protocol['splits']['train']
        config={'dataset':{'name':'dsec','train_sequences':train},
                'training':{'validation_enabled':False},
                'pretraining_history':[{'sequences':[protocol['splits']['test'][0]]}]}
        with self.assertRaisesRegex(ValueError,'overlaps downstream holdout'):
            audit_pretraining(config,protocol,statistics_sequences=train)
