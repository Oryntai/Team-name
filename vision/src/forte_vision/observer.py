"""analyze_video: файл .mov -> Observation (контракт decision/) + diagnostics (внутреннее).

Конвейер (гибрид, согласован с decision/):
  1. движение: top-K пиков энергии — дешёвый источник кандидатов;
  2. скрининг VLM: всё видео редкой выборкой, перекрывающимися окнами — открытый набор категорий;
  3. объединение кандидатов из всех источников, ни один источник не фильтрует другой;
  4. перепроверка кандидата: кадры полного разрешения, вопросы по предикатам и видимости.
Покрытие честное: если скрининг не прошёл всё видео до дедлайна, coverage_complete=False,
статус partial; если кандидат не перепроверен, его предикаты не заполняются (= unclear).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import motion, predicates
from .contract import validate_observation
from .video import EVIDENCE_SOURCE, frame_id, grab_at, probe, sample
from .vlm import VLMBackend


@dataclass(frozen=True)
class Profile:
    name: str
    motion_fps: float
    motion_sec_per_peak: float  # один пик движения на столько секунд ролика
    screen_fps: float
    screen_window_sec: float
    screen_overlap_sec: float
    recheck_frames: int
    max_rechecks: int
    max_candidate_sec: float = 6.0  # длиннее — 4–6 кадров перепроверки уже ничего не доказывают
    screen_size: tuple[int, int] = (224, 416)
    recheck_size: tuple[int, int] | None = None  # None — исходное разрешение


# Стартовые параметры, не оптимум: подбираются замерами и качеством на dev.
PROFILES: dict[str, Profile] = {
    "fast": Profile("fast", motion_fps=5.0, motion_sec_per_peak=10.0, screen_fps=0.5, screen_window_sec=20.0,
                    screen_overlap_sec=2.0, recheck_frames=4, max_rechecks=8),
    "full": Profile("full", motion_fps=5.0, motion_sec_per_peak=5.0, screen_fps=1.0, screen_window_sec=12.0,
                    screen_overlap_sec=2.0, recheck_frames=6, max_rechecks=20),
}

# Чем выше, тем раньше кандидат перепроверяется при нехватке времени
_PRIORITY = {"vlm": 2, "motion": 1}


@dataclass
class _Cand:
    start: float
    end: float
    main: float
    sources: set[str]
    priority: float
    categories: set[str] = field(default_factory=set)
    motion_score: float = 0.0


def _windows(duration: float, win: float, overlap: float) -> list[tuple[float, float]]:
    out, s = [], 0.0
    step = max(win - overlap, 1.0)
    while s < duration:
        out.append((s, min(duration, s + win)))
        if s + win >= duration:
            break
        s += step
    return out


def _split(c: _Cand, max_len: float) -> list[_Cand]:
    """Длинный интервал VLM -> куски не длиннее max_len, каждый со своей главной секундой."""
    n = max(1, int(np.ceil((c.end - c.start) / max_len)))
    if n == 1:
        return [c]
    edges = np.linspace(c.start, c.end, n + 1)
    return [
        _Cand(float(a), float(b), float((a + b) / 2), set(c.sources), c.priority, set(c.categories), c.motion_score)
        for a, b in zip(edges[:-1], edges[1:])
    ]


def _merge(cands: list[_Cand], max_len: float, gap: float = 0.5) -> list[_Cand]:
    """Сливаем пересекающиеся кандидаты, пока итог не длиннее max_len; иначе оставляем рядом (пересечение допустимо)."""
    cands = sorted(cands, key=lambda c: c.start)
    out: list[_Cand] = []
    for c in cands:
        if out and c.start <= out[-1].end + gap and max(out[-1].end, c.end) - out[-1].start <= max_len:
            m = out[-1]
            # главная секунда — от пика движения, если он есть: у VLM время грубее
            if "motion" in c.sources and ("motion" not in m.sources or c.motion_score > m.motion_score):
                m.main = c.main
            m.start, m.end = min(m.start, c.start), max(m.end, c.end)
            m.sources |= c.sources
            m.categories |= c.categories
            m.priority = max(m.priority, c.priority)
            m.motion_score = max(m.motion_score, c.motion_score)
        else:
            out.append(c)
    return out


def _nearest(pts_pool: np.ndarray, t: float, lo: float, hi: float) -> float | None:
    inside = pts_pool[(pts_pool >= lo - 1e-6) & (pts_pool <= hi + 1e-6)]
    if len(inside) == 0:
        return None
    return float(inside[np.argmin(np.abs(inside - t))])


def analyze_video(
    path: str,
    backend: VLMBackend,
    video_id: str | None = None,
    profile: str | Profile = "fast",
    deadline_sec: float = 240.0,
) -> tuple[dict, dict]:
    """Возвращает (observation, diagnostics). Observation всегда проходит validate_observation."""
    prof = PROFILES[profile] if isinstance(profile, str) else profile
    t_start = time.monotonic()
    deadline = t_start + deadline_sec
    timings: dict[str, float] = {}
    vid = video_id or path.rsplit("/", 1)[-1].rsplit(".", 1)[0]

    def mark(stage: str, t0: float) -> None:
        timings[stage] = round(time.monotonic() - t0, 2)

    t0 = time.monotonic()
    info = probe(path)
    dur = info.duration_sec
    mark("probe", t0)

    # 1. движение
    t0 = time.monotonic()
    m_times, m_energy = motion.motion_energy(path, fps=prof.motion_fps)
    top_k = max(2, int(np.ceil(dur / prof.motion_sec_per_peak)))
    peaks = motion.find_peaks(m_times, m_energy, dur, top_k=top_k)
    cands = [
        _Cand(p.start_sec, p.end_sec, p.main_sec, {"motion"}, _PRIORITY["motion"] + p.score, motion_score=p.score)
        for p in peaks
    ]
    pts_pool = m_times.copy()
    mark("motion", t0)

    # 2. скрининг всего видео
    t0 = time.monotonic()
    screen_frames = list(sample(path, fps=prof.screen_fps, size=prof.screen_size))
    pts_pool = np.unique(np.concatenate([pts_pool, [t for t, _ in screen_frames]]))
    windows = _windows(dur, prof.screen_window_sec, prof.screen_overlap_sec)
    covered_until, screen_items = 0.0, []
    q_screen = predicates.screen_question()
    for ws, we in windows:
        if time.monotonic() > deadline:
            break
        frames = [(t, im) for t, im in screen_frames if ws - 1e-6 <= t < we]
        if frames:
            ans = backend.ask(frames, q_screen, max_tokens=200)
            for cat, s, e in predicates.parse_screen(ans.text, ws, we):
                screen_items.append((cat, s, e))
                cands.extend(
                    _split(_Cand(s, max(e, s), (s + e) / 2, {"vlm"}, _PRIORITY["vlm"], categories={cat}),
                           prof.max_candidate_sec)
                )
        covered_until = we
    coverage_complete = covered_until >= dur - 1e-6
    mark("screen", t0)

    # 3. объединение
    merged = _merge(cands, prof.max_candidate_sec)

    # 4. перепроверка, самые приоритетные первыми
    t0 = time.monotonic()
    order = sorted(range(len(merged)), key=lambda i: merged[i].priority, reverse=True)
    rechecked: dict[int, tuple[dict, dict, list[float]]] = {}
    names_p, names_v = list(predicates.PREDICATES), list(predicates.VISIBILITY)
    table = {**predicates.PREDICATES, **predicates.VISIBILITY}
    q_pred = predicates.predicate_question(names_p + names_v, table)
    for i in order[: prof.max_rechecks]:
        if time.monotonic() > deadline:
            break
        c = merged[i]
        times = sorted(set(np.linspace(c.start, c.end, prof.recheck_frames).round(3).tolist()) | {round(c.main, 3)})
        frames = grab_at(path, times, size=prof.recheck_size)
        ans = backend.ask(frames, q_pred, max_tokens=220)
        tri = predicates.parse_tri_state(ans.text, names_p + names_v)
        rechecked[i] = (
            {n: tri[n] for n in names_p},
            {n: tri[n] for n in names_v},
            [t for t, _ in frames],
        )
    mark("recheck", t0)

    # 5. сборка Observation
    out_cands, diag_cands = [], {}
    for k, i in enumerate(sorted(range(len(merged)), key=lambda j: merged[j].start)):
        c = merged[i]
        pred, vis, seen = rechecked.get(i, ({}, {}, []))
        pool = np.unique(np.concatenate([pts_pool, seen])) if seen else pts_pool
        ev_pts: list[float] = []
        for target in (c.main, c.start, c.end):
            p = _nearest(pool, target, c.start, c.end)
            if p is not None and p not in ev_pts:
                ev_pts.append(p)
        if not ev_pts:  # интервал уже шага выборки — берём ближайший кадр и расширяем интервал до него
            p = float(pool[np.argmin(np.abs(pool - c.main))])
            c.start, c.end = min(c.start, p), max(c.end, p)
            ev_pts = [p]
        c.main = min(max(c.main, c.start), c.end)
        cid = f"c{k:03d}"
        out_cands.append(
            {
                "candidate_id": cid,
                "start_sec": round(c.start, 3),
                "end_sec": round(c.end, 3),
                "main_sec": round(c.main, 3),
                "source": "+".join(sorted(c.sources)),
                "evidence": [
                    {"frame_id": frame_id(p), "pts_sec": round(p, 3), "source": EVIDENCE_SOURCE}
                    for p in sorted(ev_pts)[:3]
                ],
                "predicates": pred,
                "visibility": vis,
            }
        )
        diag_cands[cid] = {
            "screen_categories": sorted(c.categories),
            "motion_score": round(c.motion_score, 5),
            "rechecked": i in rechecked,
        }

    all_rechecked = len(rechecked) == len(merged)
    obs = {
        "video_id": vid,
        "duration_sec": round(dur, 3),
        "analysis_status": "complete" if coverage_complete and all_rechecked else "partial",
        "coverage_complete": coverage_complete,
        "candidates": out_cands,
    }
    validate_observation(obs)
    timings["total"] = round(time.monotonic() - t_start, 2)
    diag = {
        "model_id": backend.model_id,
        "revision": backend.revision,
        "profile": asdict(prof),
        "stage_timings": timings,
        "vlm": asdict(backend.stats),
        "screen_windows": len(windows),
        "screen_covered_until": round(covered_until, 3),
        "candidates": diag_cands,
        "video": asdict(info),
    }
    return obs, diag


def failed_observation(video_id: str, duration_sec: float | None) -> dict:
    """Для ошибок вне конвейера (битый файл и т.п.): честный failed без кандидатов."""
    return {
        "video_id": video_id,
        "duration_sec": duration_sec if duration_sec and duration_sec > 0 else 1e-3,
        "analysis_status": "failed",
        "coverage_complete": False,
        "candidates": [],
    }
