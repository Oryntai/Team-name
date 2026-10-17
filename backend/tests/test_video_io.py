import io
from pathlib import Path
import subprocess
import time

from PIL import Image
import pytest

from backend.video_io import VideoDeadlineExceeded, VideoError, VideoIO
from backend.tests.video_fixtures import ffmpeg_tools, make_video


def video_io(path: Path) -> VideoIO:
    ffmpeg, ffprobe = ffmpeg_tools()
    return VideoIO(path, ffmpeg_bin=ffmpeg, ffprobe_bin=ffprobe, deadline_at=time.time() + 60)


def test_frames_are_actual_decodable_jpegs(tmp_path):
    ffmpeg, _ = ffmpeg_tools()
    video = video_io(make_video(tmp_path / "synthetic.mov", ffmpeg))
    metadata = video.validate()
    assert metadata.width == 96 and metadata.height == 64
    assert metadata.frame_pts == pytest.approx([n / 10 for n in range(10)])
    frames = video.sample_frames(0.3)
    assert [frame.timestamp_sec for frame in frames] == pytest.approx([0, 0.3, 0.6, 0.9])
    for frame in frames:
        with Image.open(io.BytesIO(frame.jpeg_bytes)) as image:
            image.load()
            assert image.format == "JPEG"
            assert image.size == (96, 64)
    assert video.frame_at(0.39).timestamp_sec == pytest.approx(0.4)


def test_vfr_pts_nonzero_start_and_matching_frame_content(tmp_path):
    ffmpeg, _ = ffmpeg_tools()
    path = tmp_path / "vfr.mov"
    filters = (
        "drawbox=x=0:y=0:w=iw:h=ih:color=red:t=fill:enable='eq(n,0)',"
        "drawbox=x=0:y=0:w=iw:h=ih:color=green:t=fill:enable='eq(n,1)',"
        "drawbox=x=0:y=0:w=iw:h=ih:color=blue:t=fill:enable='eq(n,4)',"
        "drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='eq(n,9)',"
        "select='eq(n,0)+eq(n,1)+eq(n,4)+eq(n,9)'"
    )
    subprocess.run([
        ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
        "color=c=black:size=96x64:rate=10:duration=1", "-vf", filters,
        "-fps_mode", "vfr", "-c:v", "libx264", "-threads", "1", "-bf", "0",
        # Offset at the muxer, after encoding: on FFmpeg 7, shifting PTS
        # inside the filter creates an edit list ending before the last frame.
        # That malformed fixture declares four frames but exposes only three.
        "-g", "1", "-video_track_timescale", "1000", "-output_ts_offset", "2", str(path),
    ], check=True, capture_output=True, timeout=60)
    video = video_io(path)
    metadata = video.validate()
    assert metadata.start_time_sec == pytest.approx(2.0)
    assert metadata.duration_sec == pytest.approx(1.0)
    assert metadata.frame_pts == pytest.approx([0, 0.1, 0.4, 0.9])
    expected = [(250, 0, 0), (0, 125, 0), (0, 0, 250), (250, 250, 250)]
    for pts, color in zip(metadata.frame_pts, expected):
        frame = video.frame_at(pts)
        assert frame.timestamp_sec == pytest.approx(pts)
        with Image.open(io.BytesIO(frame.jpeg_bytes)) as image:
            pixel = image.convert("RGB").getpixel((48, 32))
        assert all(abs(actual - target) < 15 for actual, target in zip(pixel, color))
    assert video.frame_at(0.35).timestamp_sec == pytest.approx(0.4)


def test_display_rotation_is_applied_to_evidence(tmp_path):
    ffmpeg, _ = ffmpeg_tools()
    original = make_video(tmp_path / "original.mov", ffmpeg)
    rotated = tmp_path / "rotated.mov"
    subprocess.run([
        ffmpeg, "-v", "error", "-nostdin", "-y", "-display_rotation:v:0", "90", "-i", str(original),
        "-c", "copy", str(rotated),
    ], check=True, capture_output=True, timeout=60)
    video = video_io(rotated)
    metadata = video.validate()
    assert (metadata.width, metadata.height) == (64, 96)
    assert abs(metadata.rotation_degrees) == 90
    frame = video.frame_at(0)
    assert (frame.width, frame.height) == (64, 96)


def test_rejects_corrupt_and_truncated_video(tmp_path):
    ffmpeg, _ = ffmpeg_tools()
    corrupt = tmp_path / "corrupt.mov"
    corrupt.write_bytes(b"this is not a MOV video")
    with pytest.raises(VideoError):
        video_io(corrupt).validate()
    original = make_video(tmp_path / "original.mov", ffmpeg)
    data = original.read_bytes()
    truncated = tmp_path / "truncated.mov"
    truncated.write_bytes(data[:len(data) // 2])
    with pytest.raises(VideoError):
        video_io(truncated).validate()


def test_expired_deadline_and_invalid_timestamps(tmp_path):
    ffmpeg, ffprobe = ffmpeg_tools()
    path = make_video(tmp_path / "synthetic.mov", ffmpeg)
    expired = VideoIO(path, ffmpeg_bin=ffmpeg, ffprobe_bin=ffprobe, deadline_at=time.time() - 1)
    with pytest.raises(VideoDeadlineExceeded):
        expired.probe()
    video = video_io(path)
    for value in (-1, float("nan"), float("inf"), 100):
        with pytest.raises(ValueError):
            video.frame_at(value)
    with pytest.raises(ValueError):
        video.sample_frames(0)


def test_rejects_playlist_disguised_as_mov(tmp_path):
    ffmpeg, _ = ffmpeg_tools()
    original = make_video(tmp_path / "original.mov", ffmpeg)
    playlist = tmp_path / "playlist.mov"
    playlist.write_text(f"ffconcat version 1.0\nfile '{original.as_posix()}'\n", encoding="utf-8")
    with pytest.raises(VideoError):
        video_io(playlist).validate()


def test_longest_case_duration_can_be_fully_decoded(tmp_path):
    ffmpeg, _ = ffmpeg_tools()
    # A local synthetic recording with the longest dataset clip's duration.
    video = video_io(make_video(tmp_path / "long.mov", ffmpeg, duration=142.534, rate=30))
    metadata = video.validate()
    assert metadata.duration_sec >= 142.5
    assert len(metadata.frame_pts) >= 4275
    frames = video.sample_frames(30)
    assert len(frames) == 5
    assert frames[-1].timestamp_sec == pytest.approx(120)


def test_long_irregular_frame_request_is_split_for_ffmpeg_expression_limit(tmp_path):
    """FFmpeg rejects select expressions over 100 terms; 250 isolated frames must still decode."""
    ffmpeg, _ = ffmpeg_tools()
    video = video_io(make_video(tmp_path / "long.mov", ffmpeg, duration=60.0, rate=10))
    pts = video.probe().frame_pts
    wanted = [pts[i] for i in range(0, 500, 2)] + [pts[i] for i in range(520, 560)]  # 250 одиночных + один подряд
    frames = video.frames_at(wanted[:256])
    assert [f.timestamp_sec for f in frames] == pytest.approx(wanted[:256])
    frames = video.frames_at(wanted[256:])
    assert [f.timestamp_sec for f in frames] == pytest.approx(wanted[256:])
