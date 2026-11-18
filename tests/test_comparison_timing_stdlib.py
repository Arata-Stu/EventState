"""Native video timing checks; no Torch, NumPy or Pillow dependency."""
from decimal import Decimal
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zlib

from event_state.evaluation.comparison_timing import (
    build_ffmpeg_command, validate_scene, verify_video_timestamps, write_concat,
)


def png(path, color):
    def chunk(name, value):
        return struct.pack("!I", len(value)) + name + value + struct.pack("!I", zlib.crc32(name + value) & 0xffffffff)
    header = struct.pack("!IIBBBBB", 16, 16, 8, 2, 0, 0, 0)
    pixels = b"".join(b"\0" + bytes(color) * 16 for _ in range(16))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                     + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


class ComparisonTimingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.stamps = [0, 50001, 100120, 149900, 201000]
        self.scene = dict(start_us=0, end_us=250000, timestamps_us=self.stamps,
                          frames=5, duration_us=250000)
        self.files = [self.root / f"frame_{i:06d}.png" for i in range(5)]
        for i, file in enumerate(self.files):
            png(file, (i * 40, 20, 200))

    def test_scene_preserves_all_native_timestamps(self):
        self.assertEqual(validate_scene(self.scene), tuple(self.stamps))
        self.assertEqual(self.scene["timestamps_us"], self.stamps)

    def test_single_frame_scene_retains_its_duration(self):
        self.assertEqual(validate_scene(dict(start_us=10, end_us=200010, timestamps_us=[10])), (10,))

    def test_scene_rejects_bounds_count_duration_and_future_frames(self):
        alterations = [dict(start_us=-1), dict(end_us=0), dict(start_us=1),
                       dict(timestamps_us=[]), dict(timestamps_us=[0, 250000]),
                       dict(timestamps_us=[0, 250001]), dict(frames=4), dict(duration_us=249999),
                       dict(timestamps_us=[0, 100000, 50000]), dict(timestamps_us=[0, 0])]
        for change in alterations:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_scene(dict(self.scene, **change))

    def test_scene_rejects_noninteger_and_boolean_values(self):
        for field, value in (("start_us", 0.0), ("end_us", True), ("frames", True),
                             ("duration_us", "250000"), ("timestamps_us", [0, 1.5])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_scene(dict(self.scene, **{field: value}))
        for gap in (0, -1, True, 250000.0):
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                validate_scene(self.scene, max_gap_us=gap)

    def test_scene_rejects_internal_and_terminal_gaps(self):
        with self.assertRaisesRegex(ValueError, "gap"):
            validate_scene(dict(start_us=0, end_us=400000, timestamps_us=[0, 300000]))
        with self.assertRaisesRegex(ValueError, "gap"):
            validate_scene(dict(start_us=0, end_us=400000, timestamps_us=[0, 100000]))
        self.assertEqual(validate_scene(dict(start_us=0, end_us=400000, timestamps_us=[0, 100000]),
                                        max_gap_us=300000), (0, 100000))

    def test_concat_encodes_exact_microsecond_durations_and_terminal_duplicate(self):
        path = write_concat(self.root / "frames.ffconcat", self.files, self.stamps, 250000)
        lines = path.read_text().splitlines()
        self.assertEqual(lines[0], "ffconcat version 1.0")
        self.assertEqual([line for line in lines if line.startswith("duration")],
                         ["duration 0.050001", "duration 0.050119", "duration 0.049780",
                          "duration 0.051100", "duration 0.049000"])
        self.assertEqual(sum(line == "option framerate 1000000" for line in lines), 6)
        self.assertEqual(sum(line.startswith("file ") for line in lines), 6)
        self.assertEqual(lines[-2:], ["file 'frame_000004.png'", "option framerate 1000000"])
        self.assertNotIn(str(self.root), path.read_text())

    def test_concat_accepts_safe_relative_basenames(self):
        path = write_concat(self.root / "frames.ffconcat", [f.name for f in self.files], self.stamps, 250000)
        self.assertTrue(path.is_file())

    def test_concat_validates_before_creating_output(self):
        path = self.root / "frames.ffconcat"
        for files, stamps, end in ((self.files[:-1], self.stamps, 250000),
                                  (self.files, self.stamps, 201000),
                                  (self.files, self.stamps, 600000)):
            with self.assertRaises(ValueError):
                write_concat(path, files, stamps, end)
            self.assertFalse(path.exists())

    def test_concat_rejects_unsafe_missing_duplicate_and_outside_pngs(self):
        for filename in ("frame one.png", "bad'quote.png", "frame\nline.png"):
            file = self.root / filename
            file.write_bytes(b"unused")
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                write_concat(self.root / "frames.ffconcat", [file], [0], 100000)
        outside = self.root / "other"
        outside.mkdir()
        file = outside / "image.png"
        file.write_bytes(b"unused")
        with self.assertRaises(ValueError):
            write_concat(self.root / "frames.ffconcat", [file], [0], 100000)
        with self.assertRaises(FileNotFoundError):
            write_concat(self.root / "frames.ffconcat", [self.root / "missing.png"], [0], 100000)
        with self.assertRaises(ValueError):
            write_concat(self.root / "frames.ffconcat", [self.files[0]] * 2, [0, 50000], 100000)

    def test_concat_does_not_overwrite_existing_file(self):
        path = self.root / "frames.ffconcat"
        path.write_text("preserve")
        with self.assertRaises(FileExistsError):
            write_concat(path, self.files, self.stamps, 250000)
        self.assertEqual(path.read_text(), "preserve")

    def test_ffmpeg_command_keeps_vfr_and_refuses_overwrite(self):
        command = build_ffmpeg_command("ffmpeg", self.root / "frames.ffconcat", self.root / "movie.mp4")
        for forbidden in ("-y", "-r", "-t", "-vsync"):
            self.assertNotIn(forbidden, command)
        self.assertIn("-n", command)
        for option, value in (("-fps_mode", "vfr"), ("-enc_time_base", "1:1000000"),
                              ("-bf", "0"), ("-video_track_timescale", "1000000")):
            self.assertEqual(command[command.index(option) + 1], value)
        with self.assertRaises(ValueError):
            build_ffmpeg_command("ffmpeg", self.root / "same", self.root / "same")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg unavailable")
    def test_packet_validation_checks_all_native_pts_for_absolute_source_times(self):
        ffmpeg = shutil.which("ffmpeg")
        concat = write_concat(self.root / "frames.ffconcat", self.files, self.stamps, 250000)
        output = self.root / "native.mp4"
        subprocess.run(build_ffmpeg_command(ffmpeg, concat, output), check=True, capture_output=True, timeout=30)
        start = 1_650_000_000_000_000
        stamps = [start + delta for delta in self.stamps]
        end = start + 250_000
        for ffprobe in dict.fromkeys([shutil.which("ffprobe"), None]):
            with self.subTest(ffprobe=ffprobe), patch("shutil.which", side_effect={"ffmpeg": ffmpeg, "ffprobe": ffprobe}.get):
                report = verify_video_timestamps(output, stamps, end)
                self.assertEqual(report["native_frames"], len(stamps))
                self.assertEqual(report["encoded_packets"], len(stamps)+1)
                self.assertEqual(report["first_pts_us"], 0)
                self.assertEqual(report["terminal_pts_us"], 250_000)
                shifted = stamps.copy()
                shifted[2] += 1
                with self.assertRaisesRegex(ValueError, "packet 2 PTS"):
                    verify_video_timestamps(output, shifted, end)
                with self.assertRaisesRegex(ValueError, "packet 5 PTS"):
                    verify_video_timestamps(output, stamps, end+1)
                with self.assertRaisesRegex(ValueError, "video packets"):
                    verify_video_timestamps(output, stamps[:-1], end)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe unavailable")
    def test_encoded_video_preserves_every_native_pts_and_end_marker(self):
        concat = write_concat(self.root / "frames.ffconcat", self.files, self.stamps, 250000)
        output = self.root / "native.mp4"
        subprocess.run(build_ffmpeg_command(shutil.which("ffmpeg"), concat, output),
                       check=True, capture_output=True, timeout=30)
        result = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0",
                                 "-show_frames", "-show_streams", "-show_format", "-show_entries",
                                 "frame=pts_time:stream=time_base,duration,nb_frames,level:format=duration",
                                 "-of", "json", str(output)],
                                check=True, capture_output=True, text=True, timeout=30)
        probe = json.loads(result.stdout)
        timestamps = [int(Decimal(frame["pts_time"]) * 1_000_000) for frame in probe["frames"]]
        self.assertEqual(timestamps, self.stamps + [250000])
        self.assertEqual(probe["streams"][0]["time_base"], "1/1000000")
        self.assertEqual(int(Decimal(probe["format"]["duration"]) * 1_000_000), 250001)
        self.assertEqual(int(probe["streams"][0]["nb_frames"]), 6)
        self.assertLessEqual(int(probe["streams"][0]["level"]), 42)


if __name__ == "__main__":
    unittest.main()
