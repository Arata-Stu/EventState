"""Stream native-timestamp RGB frames into seekable H.264 with bounded scratch."""
from __future__ import annotations

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


def encode_native_video(ffmpeg, output, stamps, end_us, size, rgb_frames, *, chunk_size=64):
    """Keep only compressed chunks on disk. The terminal duplicate closes the last hold.

    A raw RGB pipe cannot carry timestamps. Each small encoder invocation gets its
    own exact PTS lookup; the final stream copy uses the native chunk start offsets.
    A fresh encoder also guarantees an IDR at every chunk boundary.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not stamps or end_us <= stamps[-1] or any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("Expected increasing input timestamps and a positive terminal hold")
    if not 1 <= chunk_size <= 64:
        raise ValueError("Expected 1 to 64 frames per encoding chunk")
    width, height = size
    if min(width, height) <= 0 or width % 2 or height % 2:
        raise ValueError("H.264 panels need even, positive geometry")
    frames = iter(rgb_frames)
    packet_times = [*stamps, end_us]
    previous = None
    with TemporaryDirectory(prefix=".encode-", dir=output.parent) as temporary:
        folder = Path(temporary)
        entries = ["ffconcat version 1.0"]
        for start in range(0, len(packet_times), chunk_size):
            times = packet_times[start:start + chunk_size]
            chunk = folder / f"part-{start:08d}.mp4"
            pts = "+".join(f"eq(N,{i})*{t-times[0]}" for i, t in enumerate(times))
            command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-n",
                       "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
                       "-framerate", "1000000", "-i", "pipe:0", "-vf", f"setpts='{pts}'",
                       "-fps_mode", "vfr", "-enc_time_base", "1:1000000", "-c:v", "libx264",
                       "-preset", "veryfast", "-crf", "18",
                       "-x264-params", "fps=20/1:force-cfr=0", "-force_key_frames", "expr:gte(t,n_forced*2)",
                       "-bf", "0", "-pix_fmt", "yuv420p", "-video_track_timescale", "1000000", str(chunk)]
            with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
                try:
                    for index in range(start, start + len(times)):
                        if index < len(stamps):
                            previous = next(frames, None)
                            if previous is None or len(previous) != width * height * 3:
                                raise ValueError("Missing RGB frame or changed panel geometry")
                        encoder.stdin.write(previous)
                    encoder.stdin.close()
                    if encoder.wait():
                        raise subprocess.CalledProcessError(encoder.returncode, command)
                except BaseException:
                    encoder.kill()
                    raise
            entries.append(f"file '{chunk.name}'")
            following = start + len(times)
            if following < len(packet_times):
                delta = packet_times[following] - times[0]
                entries.append(f"duration {delta // 1_000_000}.{delta % 1_000_000:06d}")
        if next(frames, None) is not None:
            raise ValueError("More RGB frames than native input timestamps")
        concat = folder / "parts.ffconcat"
        concat.write_text("\n".join(entries) + "\n", encoding="utf-8")
        partial = folder / "comparison.mp4"
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-f", "concat", "-i", str(concat),
                        "-c", "copy", "-video_track_timescale", "1000000", "-movflags", "+faststart",
                        str(partial)], check=True)
        partial.rename(output)
