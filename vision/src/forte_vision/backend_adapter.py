"""Фабрика для backend: FORTE_ANALYZER_FACTORY=forte_vision.backend_adapter:create_analyzer.

analyze(context, video) -> backend.contracts.ObservationBatch. Длительность и canonical PTS берутся из
переданного VideoIO (video.probe()); каждое время в событиях — значение из его frame_pts.
Плотная выборка для движения и детектора пока читается PyAV из того же провалидированного файла
(публичный VideoIO.sample_frames ограничен 256 кадрами от начала), PTS привязываются к frame_pts.
Когда в VideoIO появится потоковая выборка, меняется только VideoIOSource.

Переменные окружения: FORTE_VLM_BACKEND (mlx | transformers), FORTE_VLM_MODEL, FORTE_DETECTOR
(none | nano | small | medium), FORTE_VISION_PROFILE (fast | full; иначе context.profile, если он
совпадает с именем профиля), FORTE_VISION_RESERVE_SEC — запас до дедлайна на правила и JPEG.
"""

from __future__ import annotations

import bisect
import os
import time

from .observer import PROFILES, analyze_video
from .video import VideoInfo
from .video import grab_at as _grab_at
from .video import sample as _sample

VALID_SOURCES = {"vlm", "motion", "detector"}


class VideoIOSource:
    """probe — из общего VideoIO; кадры — PyAV, с привязкой времени к canonical frame_pts."""

    def __init__(self, video, path: str):
        self.video, self.path = video, str(path)
        meta = video.probe()
        self.pts = tuple(float(p) for p in meta.frame_pts)
        fps = (len(self.pts) - 1) / (self.pts[-1] - self.pts[0]) if len(self.pts) > 1 and self.pts[-1] > self.pts[0] else 30.0
        self.info = VideoInfo(duration_sec=float(meta.duration_sec), fps=fps, width=int(meta.width),
                              height=int(meta.height), frames=len(self.pts))

    def snap(self, t: float) -> float:
        i = bisect.bisect_left(self.pts, t)
        if i == len(self.pts):
            return self.pts[-1]
        if i and t - self.pts[i - 1] <= self.pts[i] - t:
            return self.pts[i - 1]
        return self.pts[i]

    def probe(self) -> VideoInfo:
        return self.info

    def sample(self, fps, size=None, start=0.0, end=None):
        for t, im in _sample(self.path, fps, size, start, end):
            yield self.snap(t), im

    def grab_at(self, times, size=None):
        return [(self.snap(t), im) for t, im in _grab_at(self.path, times, size)]


class VisionAnalyzer:
    def __init__(self):
        from .vlm import make_backend

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.backend = make_backend(os.environ.get("FORTE_VLM_BACKEND", "mlx"), os.environ.get("FORTE_VLM_MODEL"))
        det = os.environ.get("FORTE_DETECTOR", "nano")
        self.detector = None
        if det != "none":
            from .scene import Detector

            self.detector = Detector(det)
        self.reserve = float(os.environ.get("FORTE_VISION_RESERVE_SEC", "20"))

    def analyze(self, context, video):
        from backend.contracts import IncompleteAnalysis, ObservationBatch, ObservedEvent

        prof = os.environ.get("FORTE_VISION_PROFILE") or (context.profile if context.profile in PROFILES else "fast")
        budget = context.deadline_at - time.time() - self.reserve
        if budget <= 0:
            raise IncompleteAnalysis("no time left for analysis before the deadline")
        src = VideoIOSource(video, context.video_path)
        obs, diag = analyze_video(str(context.video_path), self.backend, video_id=context.job_id, profile=prof,
                                  deadline_sec=budget, detector=self.detector, source=src)
        events = []
        for c in obs["candidates"]:
            sources = [s for s in c["source"].split("+") if s in VALID_SOURCES]
            # Observation округляет время до мс; в batch — точные значения из frame_pts VideoIO,
            # интервал расширяется до них, чтобы кадр на границе не выпал из события
            pts = sorted({src.snap(e["pts_sec"]) for e in c["evidence"]})
            t_start, t_end = min(c["start_sec"], pts[0]), max(c["end_sec"], pts[-1])
            events.append(ObservedEvent(
                t_start=t_start, t_end=t_end, t_main=min(max(c["main_sec"], t_start), t_end),
                frame_pts=pts,
                # именованные tri-state видимости — под своими именами рядом с предикатами;
                # поле visibility (clear/partial/unusable) — другое утверждение, его не заполняем
                observations={**c["predicates"], **c["visibility"]},
                sources=sources or ["motion"],
            ))
        timings = diag.get("stage_timings", {})
        flat = {f"t_{k}": float(v) for k, v in timings.items()}
        flat.update({
            "vision_model": str(diag.get("model_id")),
            "vision_revision": str(diag.get("revision")),
            "vision_detector": str(diag.get("detector")),
            "vision_profile": prof,
            "analysis_status": obs["analysis_status"],
            "coverage_complete": bool(obs["coverage_complete"]),
            "candidates": len(events),
            "vlm_calls": int(diag.get("vlm", {}).get("calls", 0)),
        })
        complete = obs["analysis_status"] == "complete" and obs["coverage_complete"]
        return ObservationBatch(complete=complete, events=events, diagnostics=flat)


def create_analyzer() -> VisionAnalyzer:
    return VisionAnalyzer()
