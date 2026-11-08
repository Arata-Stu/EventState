"""Small CPU-only benchmark contract tests, no training runtime or user data."""
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from event_state.dense.targets import depth_to_disparity, disparity_to_depth
from tools.eval_m3ed_f3_native import make_left_index


class BenchmarkTests(unittest.TestCase):
    def test_inverse_q_nonidentical_principal_points(self):
        q=np.array([[1,0,0,-320],[0,1,0,-240],[0,0,0,400],[0,0,5,-.75]])
        d=np.array([[0.75,2.0],[18.1,45.0]])
        np.testing.assert_allclose(depth_to_disparity(disparity_to_depth(d,q),q), d,rtol=1e-6)

    def test_official_left_index_boundaries(self):
        ts=np.array([0,19,20,20,20,39,61,80,81,99,121])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'idx.npy'
            make_left_index(ts,p,block=3)
            idx=np.load(p,allow_pickle=True).item()['left']
            np.testing.assert_array_equal(idx,np.searchsorted(ts,np.arange(int(ts[-1])//20)*20))

    def test_dsec_png_contract(self):
        disp=np.broadcast_to(np.arange(640,dtype=float)/10,(480,640))
        arr=(disp*256).astype(np.uint16)
        buffer=io.BytesIO()
        Image.fromarray(arr).save(buffer,format='PNG')
        self.assertEqual(buffer.getvalue()[16:26],bytes.fromhex('00000280000001e01000'))
        buffer.seek(0)
        np.testing.assert_array_equal(np.array(Image.open(buffer)),arr)
