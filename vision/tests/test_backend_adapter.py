"""Адаптер backend на синтетике: ObservationBatch строгой схемы backend, времена — только из frame_pts."""

import json
import os
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
contracts = pytest.importorskip("backend.contracts")

from conftest import EVENT  # noqa: E402
from forte_vision import backend_adapter  # noqa: E402
from forte_vision.video import probe  # noqa: E402
from forte_vision.vlm import StubBackend  # noqa: E402


class FakeVideoIO:
    """Минимум VideoIO: probe() с canonical frame_pts (здесь — из PyAV)."""

    def __init__(self, path):
        import av

        with av.open(path) as c:
            s = c.streams.video[0]
            self.frame_pts = tuple(float(f.pts * s.time_base) for f in c.decode(s))
        info = probe(path)
        self.meta = SimpleNamespace(duration_sec=info.duration_sec, width=info.width, height=info.height,
                                    frame_pts=self.frame_pts)

    def probe(self):
        return self.meta


def _respond(times, question):
    if "Categories:" in question:
        hit = any(EVENT[0] <= t < EVENT[1] for t in times)
        return json.dumps([{"category": "cash_counting", "start": EVENT[0], "end": EVENT[1]}]) if hit else "[]"
    return json.dumps({"cash_counting_observed": "yes", "counting_interval_fully_visible": "no"})


def test_adapter_returns_backend_batch(synthetic_video):
    a = backend_adapter.VisionAnalyzer.__new__(backend_adapter.VisionAnalyzer)
    a.backend, a.detector, a.reserve = StubBackend(_respond), None, 0.0
    video = FakeVideoIO(synthetic_video)
    ctx = contracts.AnalysisContext(job_id="j1", video_path=synthetic_video, deadline_at=time.time() + 120)
    batch = a.analyze(ctx, video)
    assert isinstance(batch, contracts.ObservationBatch)
    assert batch.complete and batch.events
    canon = set(video.frame_pts)
    for ev in batch.events:
        assert all(p in canon for p in ev.frame_pts), "время доказательства — только из frame_pts VideoIO"
        assert all(ev.t_start <= p <= ev.t_end for p in ev.frame_pts)
        assert set(ev.sources) <= {"vlm", "motion", "detector"}
        assert ev.visibility == {}
    hit = [ev for ev in batch.events if ev.t_start <= 9.0 <= ev.t_end]
    assert hit and hit[0].observations["cash_counting_observed"] == "observed"
    assert hit[0].observations["counting_interval_fully_visible"] == "not_observed"
    assert all(isinstance(v, (str, int, float, bool)) or v is None for v in batch.diagnostics.values())


def test_adapter_refuses_without_time(synthetic_video):
    a = backend_adapter.VisionAnalyzer.__new__(backend_adapter.VisionAnalyzer)
    a.backend, a.detector, a.reserve = StubBackend(_respond), None, 30.0
    ctx = contracts.AnalysisContext(job_id="j2", video_path=synthetic_video, deadline_at=time.time() + 5)
    with pytest.raises(contracts.IncompleteAnalysis):
        a.analyze(ctx, FakeVideoIO(synthetic_video))
