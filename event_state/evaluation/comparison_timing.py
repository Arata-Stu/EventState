"""Preserve native scene timestamps in a variable-frame-rate comparison video.

The final duplicate is an explicit end marker, not a resampled source frame.
All durations are formatted from integer microseconds without float rounding.
"""
from __future__ import annotations

from fractions import Fraction
import json
from numbers import Integral
import os
from pathlib import Path
import re
import shutil
import subprocess


_SAFE_PNG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.png\Z")


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be integer microseconds")
    return int(value)


def validate_scene(scene, max_gap_us=250000):
    """Validate a native ``[start_us,end_us)`` scene and return its timestamps.

    No frame is selected, synthesized or discarded by this function. Optional
    ``frames`` and ``duration_us`` fields must agree with the original timeline.
    """
    if not isinstance(scene, dict):
        raise ValueError("Scene must be a mapping")
    start = _integer(scene.get("start_us"), "start_us")
    end = _integer(scene.get("end_us"), "end_us")
    gap = _integer(max_gap_us, "max_gap_us")
    if start < 0 or end <= start or gap <= 0:
        raise ValueError("Expected nonnegative start, later end and positive maximum gap")
    source = scene.get("timestamps_us")
    if not isinstance(source, (list, tuple)) or not source:
        raise ValueError("Scene must contain a nonempty native timestamp list")
    stamps = tuple(_integer(value, "source timestamp") for value in source)
    if stamps[0] != start:
        raise ValueError("The first source timestamp must equal scene start_us")
    if any(value < start or value >= end for value in stamps):
        raise ValueError("Source timestamps must remain within [start_us,end_us); no future frame")
    if any(current <= previous for previous, current in zip(stamps, stamps[1:])):
        raise ValueError("Native source timestamps must be strictly increasing")
    if any(current - previous > gap for previous, current in zip(stamps, (*stamps[1:], end))):
        raise ValueError("Native scene has an excessive timestamp gap, including its final hold")
    if "frames" in scene and _integer(scene["frames"], "frames") != len(stamps):
        raise ValueError("Scene frame count does not match its native timestamps")
    if "duration_us" in scene and _integer(scene["duration_us"], "duration_us") != end - start:
        raise ValueError("Scene duration_us does not match its bounds")
    return stamps


def _seconds(microseconds):
    return f"{microseconds // 1_000_000}.{microseconds % 1_000_000:06d}"


def write_concat(path, frame_files, timestamps_us, end_us, *, max_gap_us=250000):
    """Write a fresh ffconcat file beside all supplied safe-named PNGs.

    Every native input frame occurs once, followed by one terminal duplicate at
    scene end. The encoder's one-microsecond terminal packet makes the container
    duration end-start+1us while preserving every native PTS and the end marker.
    """
    destination = Path(path).expanduser().resolve()
    files = tuple(frame_files)
    stamps = tuple(timestamps_us)
    if len(files) != len(stamps) or not files:
        raise ValueError("PNG count must equal the nonempty native timestamp count")
    stamps = validate_scene(dict(start_us=stamps[0], end_us=end_us,
                                 timestamps_us=stamps), max_gap_us=max_gap_us)
    end = _integer(end_us, "end_us")
    names = []
    for value in files:
        file = Path(value).expanduser()
        if not file.is_absolute() and len(file.parts) == 1:
            file = destination.parent / file
        file = file.resolve()
        if file.parent != destination.parent or not _SAFE_PNG.fullmatch(file.name):
            raise ValueError("All PNGs must have safe basenames in the ffconcat directory")
        if not file.is_file():
            raise FileNotFoundError(file)
        names.append(file.name)
    if len(set(names)) != len(names):
        raise ValueError("Each native source frame needs its own PNG file")
    lines = ["ffconcat version 1.0"]
    for name, previous, following in zip(names, stamps, (*stamps[1:], end)):
        lines.extend((f"file '{name}'", "option framerate 1000000",
                      f"duration {_seconds(following - previous)}"))
    lines.extend((f"file '{names[-1]}'", "option framerate 1000000"))
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
    return destination


def build_ffmpeg_command(exe, concat_path, output_path):
    """Return an overwrite-protected VFR H.264/MP4 command with microsecond PTS."""
    executable = os.fspath(exe)
    if not isinstance(executable, str) or not executable.strip():
        raise ValueError("An ffmpeg executable is required")
    source, output = (Path(value).expanduser().resolve() for value in (concat_path, output_path))
    if source == output:
        raise ValueError("The video output cannot replace its ffconcat input")
    # A sane nominal rate is only for x264 level negotiation. Without it the
    # image demuxer's 1us time base advertises 1,000,000 fps and selects level
    # 6.2 even for a small clip. force-cfr=0 and VFR retain every native PTS.
    return [executable, "-hide_banner", "-loglevel", "error", "-n",
            "-f", "concat", "-safe", "0", "-i", str(source),
            "-fps_mode", "vfr", "-enc_time_base", "1:1000000",
            "-c:v", "libx264", "-x264-params", "fps=20/1:force-cfr=0",
            "-bf", "0", "-pix_fmt", "yuv420p",
            "-video_track_timescale", "1000000", "-movflags", "+faststart", str(output)]


def ffmpeg_executable():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _encoded_packet_timestamps(video):
    probe = shutil.which("ffprobe")
    if probe:
        command = [probe, "-v", "error", "-select_streams", "v:0", "-show_packets",
                   "-show_entries", "packet=pts:stream=time_base", "-of", "json", str(video)]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        report = json.loads(result.stdout)
        if len(report.get("streams", [])) != 1:
            raise ValueError(f"Expected a video stream in {video}")
        return report["streams"][0]["time_base"], [int(p["pts"]) for p in report.get("packets", [])]

    # Stream-copy into a packet log so the imageio-distributed ffmpeg binary can
    # verify real PTS without an ffprobe installation or decoding the recording.
    command = [ffmpeg_executable(), "-nostdin", "-v", "error", "-copyts", "-i", str(video),
               "-map", "0:v:0", "-c:v", "copy", "-copytb", "1", "-avoid_negative_ts", "disabled",
               "-f", "framecrc", "-"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    lines = result.stdout.splitlines()
    for line in lines:
        if line.startswith("#tb 0:"):
            time_base = line.split(":", 1)[1].strip()
            break
    else:
        raise ValueError(f"Missing video stream time base in packet log for {video}")
    # framecrc packet columns are stream index, DTS, PTS, duration, size, CRC.
    return time_base, [int(line.split(",", 3)[2]) for line in lines if line and not line.startswith("#")]


def verify_video_timestamps(video: Path, timestamps_us: list[int], end_us: int):
    """Inspect encoded packet PTS without decoding this renderer's B-frame-free video."""
    if not timestamps_us:
        raise ValueError("Video validation needs at least one native source timestamp")
    end = _integer(end_us, "end_us")
    start = _integer(timestamps_us[0], "first source timestamp")
    stamps = validate_scene(dict(start_us=start, end_us=end, timestamps_us=timestamps_us),
                            max_gap_us=end-start)
    time_base, pts = _encoded_packet_timestamps(Path(video).resolve())
    if len(pts) != len(stamps) + 1:
        raise ValueError(f"{video}: expected {len(stamps) + 1} video packets including the endpoint, got {len(pts)}")
    units = Fraction(time_base) * 1_000_000
    if units <= 0:
        raise ValueError(f"Invalid video time base {time_base}")
    for number, (tick, source) in enumerate(zip(pts, (*stamps, end))):
        expected = source - start
        if tick * units.numerator != expected * units.denominator:
            raise ValueError(f"{video}: packet {number} PTS {tick} at {time_base} differs from expected {expected} us")
    return dict(native_frames=len(stamps), encoded_packets=len(pts), first_pts_us=0,
                terminal_pts_us=end-start, time_base=time_base)
