"""Анализатор «по якорям» под контракт сервиса: FORTE_ANALYZER_FACTORY=forte_vision.anchored.analyzer:create_analyzer.

Порядок: плотный проход детектора (люди на каждом кадре) -> роли и события -> зоны сцены (один вопрос
модели) -> пакет вопросов по вырезкам в моменты событий -> нарушения. Модель не видит ролик целиком:
она отвечает «да/нет» про конкретную зону в конкретный момент, который нашёл детектор.

Неуверенность не превращается в failed: не подтвердилось на двух кадрах — нарушения нет, не успели
спросить — тоже нет (в диагностике coverage). Предикаты выдаются только из загруженного каталога правил.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from backend.contracts import AnalysisContext, ObservationBatch, ObservedEvent

from . import decide
from .types import CropAnswer

ROOT = Path(__file__).resolve().parents[4]  # корень репозитория: vision/src/forte_vision/anchored -> ../../../..


def _known_predicates() -> set[str]:
    raw = os.environ.get("FORTE_RULE_CATALOGS")
    paths = json.loads(raw) if raw else ["rules/cash_checks.json"]
    out: set[str] = set()
    for p in paths:
        p = Path(p)
        p = p if p.is_absolute() else ROOT / p
        data = json.loads(p.read_text(encoding="utf-8"))
        rules = data if isinstance(data, list) else data.get("rules", [])
        for r in rules:
            out.update(r.get("required", []))
    return out


class AnchoredAnalyzer:
    def __init__(self, detector, asker, *, reserve_sec: float = 20.0, scan_fps: float = 5.0):
        self.detector, self.asker = detector, asker
        self.reserve, self.scan_fps = reserve_sec, scan_fps
        self.known = _known_predicates()

    def analyze(self, context: AnalysisContext, video) -> ObservationBatch:
        from forte_vision.backend_adapter import VideoIOSource

        from .desk import desk_blobs
        from .detect import dense_scan
        from .people import analyze_people
        from .zones import find_zones

        t0 = time.monotonic()
        deadline = t0 + max(10.0, context.deadline_at - time.time() - self.reserve)
        src = VideoIOSource(video, context.video_path)
        dur = float(src.info.duration_sec)
        path = str(context.video_path)
        timings = {}

        # 1) плотный проход детектора не дольше трети остатка бюджета
        scan = dense_scan(path, fps=self.scan_fps, detector=self.detector,
                          deadline=t0 + 0.35 * (deadline - t0))
        timings["t_scan"] = round(time.monotonic() - t0, 2)

        # 2) зоны: 2–4 вопроса модели на ролик
        try:
            zones = find_zones(path, scan, self.asker)
        except Exception as e:  # модель недоступна — работаем с тем, что дал детектор
            from .types import Zones
            zones = Zones(boxes={}, source={}, diag={"error": repr(e)[:200]})
        timings["t_zones"] = round(time.monotonic() - t0, 2)

        # 3) люди, роли, события (стол от модели — подсказка стороны)
        people = analyze_people(scan, desk=zones.boxes.get("desk"))
        timings["t_people"] = round(time.monotonic() - t0, 2)

        # 4) новые предметы на столе по пикселям (вместо слепого обхода «есть ли сумка/еда»)
        blobs = None
        if scan.hi is not None:
            try:
                blobs = desk_blobs(scan, zones)
            except Exception as e:  # датчик не обязателен: без него — прежний обход
                timings["blobs_error"] = repr(e)[:200]
        timings["t_blobs"] = round(time.monotonic() - t0, 2)

        # 5) вопросы по вырезкам: сначала события, потом обход; не успели — не спрашиваем
        door = decide.door_state_fn(scan, zones, people)
        probes = decide.plan(people, zones, dur, known_predicates=self.known, door_open=door, blobs=blobs)
        event_items = decide.all_items([p for p in probes if p.reason != "обход по ролику"])
        sweep_items = decide.all_items([p for p in probes if p.reason == "обход по ролику"])
        answers: dict[str, CropAnswer] = {}
        asked = 0
        ctx = _context_text(people)
        for batch in _chunks(event_items, 8) + _chunks(sweep_items, 8):
            if time.monotonic() > deadline - 12.0:
                break
            try:
                for a in self.asker.ask(path, batch, ctx):
                    answers[a.key] = a
                asked += len(batch)
            except Exception as e:
                timings["ask_error"] = repr(e)[:200]
                break
        timings["t_ask"] = round(time.monotonic() - t0, 2)
        found = decide.judge(probes, answers)

        events = []
        for f in found:
            pred = decide.PREDICATE[f.check]
            if pred not in self.known:
                continue
            m = src.snap(f.t_main)
            pts = sorted({m, *(src.snap(t) for t in f.evidence)})[:3]
            if m not in pts:
                pts = sorted({m, *pts[:2]})
            lo, hi = min(pts + [src.snap(f.t_start)]), max(pts + [src.snap(min(f.t_end, dur))])
            events.append(ObservedEvent(t_start=lo, t_end=hi, t_main=m, frame_pts=pts,
                                        observations={pred: "observed"}, sources=["detector", "vlm"]))

        _write_trace(path, zones, people, probes, answers, found, blobs)
        n_items = len(event_items) + len(sweep_items)
        diag = {
            "analyzer": "anchored",
            "duration": round(dur, 3),
            "scan_fps": round(float(scan.fps), 2),
            "events": ";".join(f"{e.kind}@{e.t:.1f}" for e in people.events)[:600],
            "roles_uncertain": bool(people.roles_uncertain),
            "zones": ";".join(f"{k}:{zones.source.get(k, '-')}" for k, v in zones.boxes.items() if v)[:300],
            "probes": len(probes),
            "items": n_items,
            "items_asked": asked,
            "coverage_complete": asked >= n_items,
            "findings": ";".join(f"{f.check}@{f.t_main:.1f}" for f in found)[:300],
            "funnel": decide.funnel(probes, answers, found)[:400],  # CHECK:проб/без_зоны/спрошено/да/подтв.
            "unclear": sum(1 for a in answers.values() if a.answer == "unclear"),
            "desk_objects": _objects_text(blobs, probes, answers)[:400],
            **timings,
        }
        return ObservationBatch(complete=True, events=events, diagnostics=diag)


def _objects_text(blobs, probes, answers) -> str:
    """Новые предметы на столе: «26.0–58.5:cash(bundle of banknotes)» через «;» — для диагностики и показа."""
    if blobs is None:
        return "off"
    out = []
    for i, b in enumerate(blobs):
        a = [answers[k] for k in sorted(answers) if k.startswith(f"OBJ{i}@")]
        what = ",".join(dict.fromkeys(x.answer for x in a)) or "-"
        note = next((x.note for x in a if x.note), "")
        out.append(f"{b.t_on:.1f}-{b.t_off:.1f}:{what}" + (f"({note})" if note else ""))
    return ";".join(out)


def _write_trace(path, zones, people, probes, answers, found, blobs=None) -> None:
    """Подробный след ролика для разбора ошибок — только при офлайн-оценке (FORTE_ANCHORED_TRACE)."""
    out = os.environ.get("FORTE_ANCHORED_TRACE")
    if not out:
        return
    rec = {
        "video": Path(path).name,
        "zones": {k: v for k, v in zones.boxes.items() if v}, "zones_diag": {k: v for k, v in zones.diag.items()
                                                                            if isinstance(v, (list, tuple, str, float, int))},
        "events": [{"kind": e.kind, "t": e.t, "t_end": e.t_end, "meta": e.meta} for e in people.events],
        "present": people.cashier_present, "visits": people.visits, "desk_dir": people.desk_dir,
        "probes": [{"check": p.check, "t_main": p.t_main, "reason": p.reason, "fixed": p.fixed,
                    "items": [[it.key, it.t, it.zone, list(it.box), answers[it.key].answer if it.key in answers else None,
                               answers[it.key].raw[:200] if it.key in answers else None] for it in p.items + p.veto]}
                   for p in probes],
        "found": [[f.check, f.t_main] for f in found],
        "blobs": None if blobs is None else [{"box": b.box, "t_on": b.t_on, "t_off": b.t_off, "area": b.area}
                                             for b in blobs],
        "notes": {k: a.note for k, a in answers.items() if a.note},
    }
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / (Path(path).stem + ".json")).write_text(json.dumps(rec, ensure_ascii=False, default=str, indent=1))


def _chunks(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def _context_text(people) -> str:
    """Надёжные факты от детектора — модели как контекст. Её собственные догадки сюда не кладём."""
    parts = []
    for e in people.events:
        name = {"cashier_leaves": "the cashier left her workplace", "cashier_enters": "the cashier entered the room",
                "cashier_sits": "the cashier sat down at her workplace", "client_arrives": "a client came to the desk",
                "client_leaves": "the client left the desk", "client_change": "one client left and another came",
                "third_person": "a third person appeared in the room"}.get(e.kind)
        if name:
            parts.append(f"{e.t:.1f}s: {name}")
    return ("Known facts from a person detector (times in seconds): " + "; ".join(parts) + ".") if parts else ""


def create_analyzer():
    from .ask import CropAsker
    from .detect import make_detector

    url = os.environ.get("FORTE_VLM_URL", "http://127.0.0.1:18080")
    key = os.environ.get("FORTE_VLM_KEY_FILE")
    cache = os.environ.get("FORTE_ANCHORED_CACHE")  # только для офлайн-оценки; в проде пусто
    asker = CropAsker(url=url, key_file=key, model=os.environ.get("FORTE_VLM_MODEL", "local"), cache_dir=cache,
                      timeout=float(os.environ.get("FORTE_VLM_TIMEOUT", "45")))
    return AnchoredAnalyzer(make_detector(os.environ.get("FORTE_DETECTOR", "nano")), asker,
                            reserve_sec=float(os.environ.get("FORTE_VISION_RESERVE_SEC", "20")))
