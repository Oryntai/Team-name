"""analyze_video: файл .mov -> Observation (контракт decision/) + diagnostics (внутреннее).

Конвейер (гибрид, согласован с decision/):
  1. сцена (детектор): стол, место кассира, люди и предметы -> события-кандидаты и зона стола;
  2. движение: пики по зоне стола/рук и немного по всему кадру;
  3. скрининг VLM: всё видео редкой выборкой, перекрывающимися окнами — открытый набор категорий;
  4. объединение кандидатов, ни один источник не фильтрует другой;
  5. отбор под бюджет: лучший кандидат в каждом отрезке времени, затем по приоритету с разнесением;
  6. перепроверка: кадры полного разрешения (для VLM — с рамками ролей), предикаты и видимость.
Покрытие честное: если скрининг не прошёл всё видео до дедлайна, coverage_complete=False, partial;
если отобранный кандидат не успели перепроверить — partial, его предикаты не заполняются (= unclear).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import motion, predicates, scene
from .contract import validate_observation
from .video import EVIDENCE_SOURCE, PyAVSource, frame_id
from .vlm import VLMBackend
from .audit import save_audit


@dataclass(frozen=True)
class Profile:
    name: str
    motion_fps: float
    zone_sec_per_peak: float  # пики движения в зоне стола: один на столько секунд ролика
    global_sec_per_peak: float  # пики по всему кадру (ходьба, вход/выход)
    det_fps: float
    screen_fps: float
    screen_window_sec: float
    screen_overlap_sec: float
    recheck_frames: int
    max_rechecks: int
    spread_seg_sec: float  # гарантированно по кандидату на каждый такой отрезок (если он там есть)
    min_gap_sec: float = 2.5
    max_candidate_sec: float = 6.0  # длиннее — 4–6 кадров перепроверки уже ничего не доказывают
    screen: bool = True
    role_marks: bool = True
    screen_size: tuple[int, int] = (224, 416)
    recheck_size: tuple[int, int] | None = None  # None — исходное разрешение


# Стартовые параметры, не оптимум: подбираются замерами и качеством на dev.
PROFILES: dict[str, Profile] = {
    "fast": Profile("fast", motion_fps=5.0, zone_sec_per_peak=8.0, global_sec_per_peak=20.0, det_fps=2.0,
                    screen_fps=0.5, screen_window_sec=20.0, screen_overlap_sec=2.0, recheck_frames=4,
                    max_rechecks=8, spread_seg_sec=15.0),
    "full": Profile("full", motion_fps=5.0, zone_sec_per_peak=4.0, global_sec_per_peak=10.0, det_fps=3.0,
                    screen_fps=1.0, screen_window_sec=12.0, screen_overlap_sec=2.0, recheck_frames=6,
                    max_rechecks=20, spread_seg_sec=8.0),
}


@dataclass
class _Cand:
    start: float
    end: float
    main: float
    sources: set[str]
    priority: float
    kinds: set[str] = field(default_factory=set)  # подробности источника для diagnostics
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
    """Длинный интервал -> куски не длиннее max_len, каждый со своей главной секундой."""
    n = max(1, int(np.ceil((c.end - c.start) / max_len)))
    if n == 1:
        return [c]
    edges = np.linspace(c.start, c.end, n + 1)
    return [
        _Cand(float(a), float(b), float((a + b) / 2), set(c.sources), c.priority, set(c.kinds), c.motion_score)
        for a, b in zip(edges[:-1], edges[1:])
    ]


def _merge(cands: list[_Cand], max_len: float, gap: float = 0.5) -> list[_Cand]:
    """Сливаем пересекающиеся кандидаты, пока итог не длиннее max_len; согласие источников повышает приоритет."""
    cands = sorted(cands, key=lambda c: c.start)
    out: list[_Cand] = []
    for c in cands:
        if out and c.start <= out[-1].end + gap and max(out[-1].end, c.end) - out[-1].start <= max_len:
            m = out[-1]
            # главная секунда: детектор и движение точнее по времени, чем VLM
            if ("vlm" in m.sources and not (m.sources - {"vlm"})) or c.priority > m.priority and "vlm" not in c.sources:
                m.main = c.main
            m.start, m.end = min(m.start, c.start), max(m.end, c.end)
            new = c.sources - m.sources
            m.sources |= c.sources
            m.kinds |= c.kinds
            m.priority = max(m.priority, c.priority) + 0.5 * len(new)
            m.motion_score = max(m.motion_score, c.motion_score)
        else:
            out.append(c)
    return out


def _select(cands: list[_Cand], max_n: int, seg_len: float, min_gap: float, duration: float) -> list[_Cand]:
    """Под бюджет перепроверок: сначала лучший в каждом отрезке (разнесение), потом по приоритету."""
    chosen: list[_Cand] = []

    def free(c: _Cand) -> bool:
        return all(abs(c.main - o.main) >= min_gap for o in chosen)

    for k in range(int(np.ceil(duration / seg_len))):
        seg = [c for c in cands if k * seg_len <= c.main < (k + 1) * seg_len]
        if seg:
            best = max(seg, key=lambda c: c.priority)
            if free(best):
                chosen.append(best)
    if len(chosen) > max_n:
        chosen = sorted(chosen, key=lambda c: c.priority, reverse=True)[:max_n]
    for c in sorted(cands, key=lambda c: c.priority, reverse=True):
        if len(chosen) >= max_n:
            break
        if all(c is not o for o in chosen) and free(c):
            chosen.append(c)
    return chosen


def _nearest(pts_pool: np.ndarray, t: float, lo: float, hi: float) -> float | None:
    inside = pts_pool[(pts_pool >= lo - 1e-6) & (pts_pool <= hi + 1e-6)]
    if len(inside) == 0:
        return None
    return float(inside[np.argmin(np.abs(inside - t))])


def _peaks(times, energy, duration, sec_per_peak, base, kind) -> list[_Cand]:
    if energy is None or len(energy) == 0:
        return []
    top_k = max(1, int(np.ceil(duration / sec_per_peak)))
    peaks = motion.find_peaks(times, energy, duration, top_k=top_k)
    hi = max((p.score for p in peaks), default=1.0) or 1.0
    return [
        _Cand(p.start_sec, p.end_sec, p.main_sec, {"motion"}, base + p.score / hi, {kind}, p.score)
        for p in peaks
    ]


ROLE_HINT = (
    "Boxes drawn by a person detector: green CASHIER = person at the cashier workstation, "
    "orange PERSON = other people. Boxes can be wrong; rely on what you actually see.\n"
)


def analyze_video(
    path: str,
    backend: VLMBackend,
    video_id: str | None = None,
    profile: str | Profile = "fast",
    deadline_sec: float = 240.0,
    detector: scene.Detector | None = None,
    source=None,
) -> tuple[dict, dict]:
    """Возвращает (observation, diagnostics). Observation всегда проходит validate_observation.
    source — источник кадров с probe/sample/grab_at; по умолчанию PyAV по path."""
    prof = PROFILES[profile] if isinstance(profile, str) else profile
    t_start = time.monotonic()
    deadline = t_start + deadline_sec
    timings: dict[str, float] = {}
    vid = video_id or path.rsplit("/", 1)[-1].rsplit(".", 1)[0]

    def mark(stage: str, t0: float) -> None:
        timings[stage] = round(time.monotonic() - t0, 2)

    src = source or PyAVSource(path)
    t0 = time.monotonic()
    info = src.probe()
    dur = info.duration_sec
    mark("probe", t0)
    cands: list[_Cand] = []
    pts_pool: list[float] = []

    # 1. сцена
    sc = None
    if detector is not None:
        t0 = time.monotonic()
        dets = detector.detect(src.sample(fps=prof.det_fps))
        sc = scene.build_scene(dets, info.width, info.height)
        for e in scene.scene_events(sc, dur):
            cands.append(_Cand(e.start_sec, e.end_sec, e.main_sec, {"detector"}, e.priority, {e.kind}))
        pts_pool.extend(d.t for d in dets)
        mark("scene", t0)

    # 2. движение: зона стола важнее всего кадра
    t0 = time.monotonic()
    zone = sc.zone() if sc is not None else None
    m_times, m_energy, z_energy = motion.motion_energy(
        src, fps=prof.motion_fps, zone=zone, frame_size=(info.width, info.height)
    )
    cands += _peaks(m_times, z_energy, dur, prof.zone_sec_per_peak, 1.2, "desk_motion")
    cands += _peaks(m_times, m_energy, dur, prof.global_sec_per_peak if zone else prof.zone_sec_per_peak,
                    0.6 if zone else 1.2, "frame_motion")
    pts_pool.extend(m_times.tolist())
    mark("motion", t0)

    # 3. скрининг всего видео
    t0 = time.monotonic()
    windows = _windows(dur, prof.screen_window_sec, prof.screen_overlap_sec) if prof.screen else []
    covered_until = 0.0 if prof.screen else dur
    invalid_screen_windows = 0
    empty_screen_windows = 0
    valid_screen_windows = 0
    if prof.screen:
        screen_frames = list(src.sample(fps=prof.screen_fps, size=prof.screen_size))
        pts_pool.extend(t for t, _ in screen_frames)
        q_screen = predicates.screen_question()
        for ws, we in windows:
            if time.monotonic() > deadline:
                break
            frames = [(t, im) for t, im in screen_frames if ws - 1e-6 <= t < we]
            if not frames:
                empty_screen_windows += 1
                continue
            ans = backend.ask(frames, q_screen, max_tokens=200)
            parsed, valid = predicates.parse_screen_checked(ans.text, ws, we)
            for cat, s, e in parsed:
                cands.extend(_split(_Cand(s, e, (s + e) / 2, {"vlm"}, 2.0, {f"screen:{cat}"}),
                                    prof.max_candidate_sec))
            if not valid:
                invalid_screen_windows += 1
                continue
            valid_screen_windows += 1
            covered_until = we
    coverage_complete = (covered_until >= dur - 1e-6 and invalid_screen_windows == 0
                         and empty_screen_windows == 0)
    mark("screen", t0)

    # 4–5. объединение и отбор
    raw_intervals = [(c.start, c.end) for c in cands]
    pool = _merge(cands, prof.max_candidate_sec)
    pooled_intervals = [(c.start, c.end) for c in pool]
    selected = sorted(_select(pool, prof.max_rechecks, prof.spread_seg_sec, prof.min_gap_sec, dur),
                      key=lambda c: c.priority, reverse=True)

    # 6. перепроверка
    t0 = time.monotonic()
    rechecked: dict[int, tuple[dict, dict, list[float]]] = {}
    names_p, names_v = list(predicates.PREDICATES), list(predicates.VISIBILITY)
    table = {**predicates.PREDICATES, **predicates.VISIBILITY}
    q_pred = predicates.predicate_question(names_p + names_v, table)
    for c in selected:
        if time.monotonic() > deadline:
            break
        times = sorted(set(np.linspace(c.start, c.end, prof.recheck_frames).round(3).tolist()) | {round(c.main, 3)})
        frames = src.grab_at(times, size=prof.recheck_size)
        question = q_pred
        if detector is not None and prof.role_marks and sc is not None and sc.cashier_slot is not None:
            frames_for_vlm = scene.mark_roles(frames, detector.detect(frames), sc.cashier_slot)
            question = ROLE_HINT + q_pred
        else:
            frames_for_vlm = frames
        ans = backend.ask(frames_for_vlm, question, max_tokens=220)
        tri = predicates.parse_tri_state(ans.text, names_p + names_v)
        rechecked[id(c)] = ({n: tri[n] for n in names_p}, {n: tri[n] for n in names_v}, [t for t, _ in frames])
    mark("recheck", t0)

    # 7. сборка Observation: только отобранные кандидаты
    base_pool = np.unique(np.asarray(pts_pool, dtype=float)) if pts_pool else np.asarray([0.0])
    out_cands, diag_cands = [], {}
    for k, c in enumerate(sorted(selected, key=lambda c: c.start)):
        pred, vis, seen = rechecked.get(id(c), ({}, {}, []))
        ppool = np.unique(np.concatenate([base_pool, seen])) if seen else base_pool
        ev_pts: list[float] = []
        for target in (c.main, c.start, c.end):
            p = _nearest(ppool, target, c.start, c.end)
            if p is not None and p not in ev_pts:
                ev_pts.append(p)
        if not ev_pts:  # интервал уже шага выборки — берём ближайший кадр и расширяем интервал до него
            p = float(ppool[np.argmin(np.abs(ppool - c.main))])
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
        diag_cands[cid] = {"kinds": sorted(c.kinds), "priority": round(c.priority, 3),
                           "motion_score": round(c.motion_score, 5), "rechecked": id(c) in rechecked}

    all_rechecked = len(rechecked) == len(selected)
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
        "detector": None if detector is None else f"rfdetr-{detector.size}@{detector.device}",
        "profile": asdict(prof),
        "stage_timings": timings,
        "vlm": asdict(backend.stats),
        "screen_windows": len(windows),
        "valid_screen_windows": valid_screen_windows,
        "invalid_screen_windows": invalid_screen_windows,
        "empty_screen_windows": empty_screen_windows,
        "pool_selected_count": len(selected),
        "pool_dropped_count": len(pool) - len(selected),
        "screen_covered_until": round(covered_until, 3),
        "pool_size": len(pool),
        "raw_candidates": len(cands),
        "scene": None if sc is None else {
            "desk_found": sc.desk_box is not None,
            "cashier_slot_found": sc.cashier_slot is not None,
            "cashier_present_frac": round(float(sc.cashier_present.mean()), 3) if len(sc.times) else None,
        },
        "candidates": diag_cands,
        "video": asdict(info),
    }
    save_audit(vid, obs, diag, raw_intervals, pooled_intervals)
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
