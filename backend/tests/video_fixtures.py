"""Synthetic video builders only: no competition recordings in tests."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


def ffmpeg_tools() -> tuple[str, str]:
    ffmpeg = os.environ.get("FORTE_FFMPEG_BIN") or shutil.which("ffmpeg")
    ffprobe = os.environ.get("FORTE_FFPROBE_BIN") or shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("Install FFmpeg/ffprobe or set FORTE_FFMPEG_BIN and FORTE_FFPROBE_BIN")
    return ffmpeg, ffprobe


def make_video(path: Path, ffmpeg: str, *, duration: float = 1.0, rate: int = 10) -> Path:
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size=96x64:rate={rate}:duration={duration}",
        "-an", "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(path),
    ], check=True, capture_output=True, timeout=60)
    return path
