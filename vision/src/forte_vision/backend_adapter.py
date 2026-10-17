"""Фабрика для backend: FORTE_ANALYZER_FACTORY=forte_vision.backend_adapter:create_analyzer.

analyze(context, video) -> backend.contracts.ObservationBatch. Длительность и canonical PTS берутся из
переданного VideoIO (video.probe()); каждое время в событиях — значение из его frame_pts.
Все изображения берутся из общего VideoIO: iter_frames для выборки и frames_at для перепроверки.
PIL декодирует оригинальный JPEG и уменьшает копию для inference; время изображения не переименовывается.

Переменные окружения: FORTE_VLM_BACKEND (mlx | transformers | openai; для openai — FORTE_VLM_URL
локального сервера и FORTE_VLM_KEY_FILE), FORTE_VLM_MODEL, FORTE_DETECTOR
(none | nano | small | medium), FORTE_VISION_PROFILE (fast | full; иначе context.profile, если он
совпадает с именем профиля), FORTE_VISION_RESERVE_SEC — запас до дедлайна на правила и JPEG.
"""

from __future__ import annotations

import bisect
import io
import math

from PIL import Image
import os
import time

from .observer import PROFILES, analyze_video
from .video import VideoInfo

VALID_SOURCES = {"vlm", "motion", "detector"}


def _bounded_observer_time(value: float, duration: float) -> float:
    """Repair only the observer's known 3-decimal rounding at the video end."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("invalid observer timestamp")
    if value <= duration:
        return value
    # Observer times are rounded to milliseconds. A legitimate end overshoot
    # must equal rounded actual duration, not merely be an arbitrary small excess.
    if value != round(duration, 3) or value - duration > 0.0005 + 1e-9:
        raise ValueError("observer timestamp exceeds actual video duration")
    return duration


class VideoIOSource:
    """Canonical VideoIO JPEG pixels and PTS; resizing affects inference copies only."""

    def __init__(self, video, path: str):
        self.video, self.path = video, str(path)
        meta = video.probe()
        self.pts = tuple(float(p) for p in meta.frame_pts)
        if not self.pts:
            raise ValueError("VideoIO has no canonical frames")
        fps = (
            (len(self.pts) - 1) / (self.pts[-1] - self.pts[0])
            if len(self.pts) > 1 and self.pts[-1] > self.pts[0]
            else 30.0
        )
        self.info = VideoInfo(
            duration_sec=float(meta.duration_sec),
            fps=fps,
            width=int(meta.width),
            height=int(meta.height),
            frames=len(self.pts),
        )

    def snap(self, t: float) -> float:
        i = bisect.bisect_left(self.pts, t)
        if i == len(self.pts):
            return self.pts[-1]
        if i and t - self.pts[i - 1] <= self.pts[i] - t:
            return self.pts[i - 1]
        return self.pts[i]

    def probe(self) -> VideoInfo:
        return self.info

    @staticmethod
    def _time(value):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("time must be finite and nonnegative")
        return float(value)

    @staticmethod
    def _size(size):
        if size is not None and (
            not isinstance(size, (tuple, list))
            or len(size) != 2
            or any(type(value) is not int or value <= 0 for value in size)
        ):
            raise ValueError("size must be positive integer width/height")
        return tuple(size) if size is not None else None

    @staticmethod
    def _image(frame, size):
        with Image.open(io.BytesIO(frame.jpeg_bytes)) as decoded:
            if decoded.format != "JPEG":
                raise ValueError("VideoIO evidence must be JPEG")
            image = decoded.convert("RGB")
        return image.resize(size, Image.Resampling.BICUBIC) if size else image

    def sample(self, fps, size=None, start=0.0, end=None):
        """Shared zero-anchored interval samples restricted to [start, end)."""
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isfinite(fps)
            or fps <= 0
        ):
            raise ValueError("fps must be finite and positive")
        interval = 1.0 / fps
        if not math.isfinite(interval) or interval < 1e-6:
            raise ValueError("unsupported sampling interval")
        size = self._size(size)
        start = self._time(start)
        end = self.info.duration_sec if end is None else self._time(end)
        if end <= start or start >= self.info.duration_sec:
            return
        for frame in self.video.iter_frames(interval_sec=interval):
            if frame.timestamp_sec >= end:
                break
            if frame.timestamp_sec < start:
                continue
            image = self._image(frame, size)
            # Public probe enforces the same backend deadline after PIL work.
            self.video.probe()
            yield frame.timestamp_sec, image

    def grab_at(self, times, size=None):
        """First canonical frame at/after each target; dedup and clamp beyond tail."""
        size = self._size(size)
        if isinstance(times, (str, bytes)):
            raise ValueError("times must be a sequence")
        targets = sorted(set(self._time(value) for value in times))
        if not targets:
            return []
        canonical = sorted(
            {
                self.pts[min(bisect.bisect_left(self.pts, target), len(self.pts) - 1)]
                for target in targets
            }
        )
        output = []
        for offset in range(0, len(canonical), 256):
            frames = self.video.frames_at(canonical[offset : offset + 256])
            for frame in frames:
                image = self._image(frame, size)
                self.video.probe()
                output.append((frame.timestamp_sec, image))
        return output


class VisionAnalyzer:
    def __init__(self):
        from .vlm import make_backend

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.backend = make_backend(
            os.environ.get("FORTE_VLM_BACKEND", "mlx"),
            os.environ.get("FORTE_VLM_MODEL"),
        )
        det = os.environ.get("FORTE_DETECTOR", "nano")
        self.detector = None
        if det != "none":
            from .scene import Detector

            self.detector = Detector(det)
        self.reserve = float(os.environ.get("FORTE_VISION_RESERVE_SEC", "20"))

    def analyze(self, context, video):
        from backend.contracts import (
            IncompleteAnalysis,
            ObservationBatch,
            ObservedEvent,
        )

        prof = os.environ.get("FORTE_VISION_PROFILE") or (
            context.profile if context.profile in PROFILES else "fast"
        )
        budget = context.deadline_at - time.time() - self.reserve
        if budget <= 0:
            raise IncompleteAnalysis("no time left for analysis before the deadline")
        src = VideoIOSource(video, context.video_path)
        obs, diag = analyze_video(
            str(context.video_path),
            self.backend,
            video_id=context.job_id,
            profile=prof,
            deadline_sec=budget,
            detector=self.detector,
            source=src,
        )
        events = []
        for c in obs["candidates"]:
            sources = [s for s in c["source"].split("+") if s in VALID_SOURCES]
            # Observation округляет время до мс; в batch — точные значения из frame_pts VideoIO,
            # интервал расширяется до них, чтобы кадр на границе не выпал из события
            pts = sorted({src.snap(e["pts_sec"]) for e in c["evidence"]})
            duration = src.info.duration_sec
            start = _bounded_observer_time(c["start_sec"], duration)
            end = _bounded_observer_time(c["end_sec"], duration)
            main = _bounded_observer_time(c["main_sec"], duration)
            t_start, t_end = min(start, pts[0]), max(end, pts[-1])
            events.append(
                ObservedEvent(
                    t_start=t_start,
                    t_end=t_end,
                    t_main=min(max(main, t_start), t_end),
                    frame_pts=pts,
                    # именованные tri-state видимости — под своими именами рядом с предикатами;
                    # поле visibility (clear/partial/unusable) — другое утверждение, его не заполняем
                    observations={**c["predicates"], **c["visibility"]},
                    sources=sources or ["motion"],
                )
            )
        timings = diag.get("stage_timings", {})
        flat = {f"t_{k}": float(v) for k, v in timings.items()}
        flat.update(
            {
                "vision_model": str(diag.get("model_id")),
                "vision_revision": str(diag.get("revision")),
                "vision_detector": str(diag.get("detector")),
                "vision_profile": prof,
                "analysis_status": obs["analysis_status"],
                "coverage_complete": bool(obs["coverage_complete"]),
                "candidates": len(events),
                "vlm_calls": int(diag.get("vlm", {}).get("calls", 0)),
                # счётчики переключения GPU-сервер -> локальная модель (накопительные за жизнь worker)
                **{f"vlm_{k}": int(v) for k, v in (diag.get("vlm_routing") or {}).items()},
            }
        )
        complete = obs["analysis_status"] == "complete" and obs["coverage_complete"]
        return ObservationBatch(complete=complete, events=events, diagnostics=flat)


def create_analyzer() -> VisionAnalyzer:
    return VisionAnalyzer()
