"""Адаптер backend на синтетике: ObservationBatch строгой схемы backend, времена — только из frame_pts."""

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
contracts = pytest.importorskip("backend.contracts")

from conftest import EVENT  # noqa: E402
from forte_vision import backend_adapter  # noqa: E402
from backend.video_io import VideoIO, VideoDeadlineExceeded  # noqa: E402
from backend.tests.video_fixtures import ffmpeg_tools  # noqa: E402
from forte_vision.vlm import StubBackend  # noqa: E402


def actual_video(path):
    ffmpeg, ffprobe = ffmpeg_tools()
    return VideoIO(
        path, ffmpeg_bin=ffmpeg, ffprobe_bin=ffprobe, deadline_at=time.time() + 120
    )


def _respond(times, question):
    if "Categories:" in question:
        hit = any(EVENT[0] <= t < EVENT[1] for t in times)
        return (
            json.dumps(
                [{"category": "cash_counting", "start": EVENT[0], "end": EVENT[1]}]
            )
            if hit
            else "[]"
        )
    return json.dumps(
        {"cash_counting_observed": "yes", "counting_interval_fully_visible": "no"}
    )


def test_adapter_returns_backend_batch(synthetic_video):
    a = backend_adapter.VisionAnalyzer.__new__(backend_adapter.VisionAnalyzer)
    a.backend, a.detector, a.reserve = StubBackend(_respond), None, 0.0
    video = actual_video(synthetic_video)
    ctx = contracts.AnalysisContext(
        job_id="j1", video_path=synthetic_video, deadline_at=time.time() + 120
    )
    batch = a.analyze(ctx, video)
    assert isinstance(batch, contracts.ObservationBatch)
    assert batch.complete and batch.events
    canon = set(video.probe().frame_pts)
    for ev in batch.events:
        assert all(
            p in canon for p in ev.frame_pts
        ), "время доказательства — только из frame_pts VideoIO"
        assert all(ev.t_start <= p <= ev.t_end for p in ev.frame_pts)
        assert set(ev.sources) <= {"vlm", "motion", "detector"}
        assert ev.visibility == {}
    hit = [ev for ev in batch.events if ev.t_start <= 9.0 <= ev.t_end]
    assert hit and hit[0].observations["cash_counting_observed"] == "observed"
    assert hit[0].observations["counting_interval_fully_visible"] == "not_observed"
    assert all(
        isinstance(v, (str, int, float, bool)) or v is None
        for v in batch.diagnostics.values()
    )


def test_adapter_refuses_without_time(synthetic_video):
    a = backend_adapter.VisionAnalyzer.__new__(backend_adapter.VisionAnalyzer)
    a.backend, a.detector, a.reserve = StubBackend(_respond), None, 30.0
    ctx = contracts.AnalysisContext(
        job_id="j2", video_path=synthetic_video, deadline_at=time.time() + 5
    )
    with pytest.raises(contracts.IncompleteAnalysis):
        a.analyze(ctx, actual_video(synthetic_video))


def test_source_pixels_equal_actual_jpeg_at_exact_pts(synthetic_video):
    import io
    from PIL import Image, ImageChops

    video = actual_video(synthetic_video)
    source = backend_adapter.VideoIOSource(video, synthetic_video)
    frames = list(source.sample(fps=2, size=None, end=1))
    assert frames and all(pts in video.probe().frame_pts for pts, _ in frames)
    assert all(0 <= pts < 1 for pts, _ in frames)
    for pts, image in frames:
        frame = video.frame_at(pts)
        assert frame.timestamp_sec == pts
        with Image.open(io.BytesIO(frame.jpeg_bytes)) as raw:
            expected = raw.convert("RGB")
        assert image.size == expected.size
        assert ImageChops.difference(image, expected).getbbox() is None
    original_bytes = video.frame_at(frames[0][0]).jpeg_bytes
    resized = source.grab_at([frames[0][0]], size=(32, 24))[0][1]
    assert resized.size == (32, 24)
    assert video.frame_at(frames[0][0]).jpeg_bytes == original_bytes


def test_source_grab_ceil_dedup_tail_and_empty(synthetic_video):
    video = actual_video(synthetic_video)
    source = backend_adapter.VideoIOSource(video, synthetic_video)
    pts = video.probe().frame_pts
    between = (pts[1] + pts[2]) / 2
    frames = source.grab_at([between, between, pts[-1] + 10, pts[0]], size=(20, 10))
    assert [t for t, _ in frames] == [pts[0], pts[2], pts[-1]]
    assert all(image.size == (20, 10) for _, image in frames)
    assert source.grab_at([]) == []
    assert list(source.sample(fps=2, start=1, end=1)) == []
    assert list(source.sample(fps=2, start=source.info.duration_sec + 1)) == []
    bounded = list(source.sample(fps=2, start=0.2, end=0.8))
    assert bounded and all(0.2 <= t < 0.8 for t, _ in bounded)


def test_source_grab_batches_actual_video_over_256(synthetic_video, monkeypatch):
    video = actual_video(synthetic_video)
    source = backend_adapter.VideoIOSource(video, synthetic_video)
    requested = list(video.probe().frame_pts[:257])
    calls = []
    original = video.frames_at

    def measured(times):
        calls.append(len(times))
        return original(times)

    monkeypatch.setattr(video, "frames_at", measured)
    frames = source.grab_at(requested, size=(8, 8))
    assert calls == [256, 1]
    assert [t for t, _ in frames] == requested


@pytest.mark.parametrize("fps", [0, -1, True, float("nan"), float("inf")])
def test_source_invalid_fps(synthetic_video, fps):
    source = backend_adapter.VideoIOSource(
        actual_video(synthetic_video), synthetic_video
    )
    with pytest.raises(ValueError):
        list(source.sample(fps))


def test_source_deadline_enforced_after_cached_frames(synthetic_video):
    video = actual_video(synthetic_video)
    source = backend_adapter.VideoIOSource(video, synthetic_video)
    iterator = source.sample(fps=2, end=1)
    next(iterator)
    video.deadline_at = time.time() - 1
    with pytest.raises(VideoDeadlineExceeded):
        next(iterator)
    with pytest.raises(VideoDeadlineExceeded):
        source.grab_at([0])


def test_source_sample_continues_past_256(synthetic_video):
    video = actual_video(synthetic_video)
    source = backend_adapter.VideoIOSource(video, synthetic_video)
    frames = list(source.sample(fps=30, size=(8, 8)))
    canonical = video.probe().frame_pts
    assert len(frames) > 256
    assert frames[-1][0] == canonical[-1]
    assert [t for t, _ in frames] == sorted({t for t, _ in frames})
    assert all(t in canonical for t, _ in frames)


@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf")])
def test_source_grab_rejects_invalid_times(synthetic_video, value):
    source = backend_adapter.VideoIOSource(
        actual_video(synthetic_video), synthetic_video
    )
    with pytest.raises(ValueError):
        source.grab_at([value])
