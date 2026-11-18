"""Exercise the real chunked H.264 encoder without NumPy, Pillow or Torch."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from event_state.evaluation.comparison_timing import verify_video_timestamps
from event_state.evaluation.full_video import encode_native_video


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools unavailable")
class FullVideoTests(unittest.TestCase):
    def test_chunk_boundaries_preserve_every_pts_and_keyframes(self):
        stamps = [1_000_000 + i*49_997 + i % 7 for i in range(131)]
        end = stamps[-1] + 52_019
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "full.mp4"
            frames = (bytes([i])*16*16*3 for i in range(len(stamps)))
            encode_native_video("ffmpeg", path, stamps, end, (16, 16), frames)
            verify_video_timestamps(path, stamps, end)
            packets = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "packet=pts,flags", "-of", "csv=p=0", str(path)], text=True)
            key_times = [int(row.split(",")[0]) for row in packets.splitlines() if "K" in row]
            self.assertEqual(key_times[0], 0)
            self.assertTrue(all(b-a <= 2_050_000 for a, b in zip(key_times, key_times[1:])))
            self.assertEqual(list(Path(temp).iterdir()), [path])
            for marker in (b"moov", b"mdat"):
                self.assertIn(marker, path.read_bytes())
            self.assertLess(path.read_bytes().index(b"moov"), path.read_bytes().index(b"mdat"))

    def test_missing_source_frame_preserves_no_partial_video(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "full.mp4"
            with self.assertRaisesRegex(ValueError, "Missing RGB frame"):
                encode_native_video("ffmpeg", path, [100, 200], 300, (16, 16), [bytes(768)])
            self.assertEqual(list(Path(temp).iterdir()), [])
