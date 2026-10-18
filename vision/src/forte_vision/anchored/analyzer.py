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


# зоны (~11 с), стол (~5 с), одна пачка вопросов (~8 с) и запас: столько нужно после детектора
SCAN_RESERVE_SEC = 30.0


class AnchoredAnalyzer:
    def __init__(self, detector, asker, *, reserve_sec: float = 20.0, scan_fps: float = 5.0):
        self.detector, self.asker = detector, asker
        self.reserve, self.scan_fps = reserve_sec, scan_fps
        self.known = _known_predicates()
        # worker ставит сюда f(жёсткий срок, epoch) -> до какого момента (epoch) разбирать ролик, чтобы очередь
        # за ним тоже успела; без очереди срок не меняется
        self.time_share = None

    def analyze(self, context: AnalysisContext, video) -> ObservationBatch:
        from forte_vision.backend_adapter import VideoIOSource

        from .desk import desk_blobs
        from .detect import dense_scan
        from .people import analyze_people
        from .zones import find_zones

        t0 = time.monotonic()
        deadline = t0 + max(10.0, context.deadline_at - time.time() - self.reserve)

        share = {"end": deadline}

        def share_end() -> float:
            """Доля этого ролика (monotonic): ролики, загруженные вместе, делят одно окно в 5 минут.
            Конец доли только сокращается (подъехали новые ролики) и никогда не отодвигается: пересчёт «части
            остатка» от текущего момента иначе продлевал бы его на каждом этапе, и залп снова не укладывался."""
            if self.time_share is None:
                return deadline
            try:
                soft = float(self.time_share(context.deadline_at))
            except Exception:
                return share["end"]
            share["end"] = min(share["end"], time.monotonic() + soft - time.time())
            return share["end"]

        _forget(self.asker)  # ответы по прошлому ролику в памяти процесса не держим
        st0 = getattr(self.asker, "stats", None)
        fb0 = (getattr(st0, "fallback_calls", 0), getattr(st0, "primary_failures", 0))
        src = VideoIOSource(video, context.video_path)
        dur = float(src.info.duration_sec)
        path = str(context.video_path)
        timings = {"share_sec": round(share_end() - t0, 1)}

        # 1) плотный проход детектора не дольше трети доли ролика
        # детектору — треть доли, но не больше, чем остаётся после зон, стола и хотя бы одной пачки вопросов
        # (~30 с): при залпе иначе он съедал долю целиком и вопросов уже не оставалось
        budget = share_end() - t0
        scan = dense_scan(path, fps=self.scan_fps, detector=self.detector,
                          deadline=t0 + min(0.35 * budget, max(3.0, budget - SCAN_RESERVE_SEC)))
        timings["t_scan"] = round(time.monotonic() - t0, 2)

        # 2) зоны: 2–4 вопроса модели на ролик
        try:
            # замок сейфа нужен только проверке «ключ в замке»; нет её в каталоге — лишний вопрос модели не задаём
            zones = find_zones(path, scan, self.asker, find_lock=decide.PREDICATE["KEYS"] in self.known)
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
        batch_sec = 8.0  # сколько идёт пачка вопросов; уточняется по факту
        for i, batch in enumerate(_chunks(event_items, 8) + _chunks(sweep_items, 8)):
            # первая пачка ограничена только жёстким сроком: без неё ролик остался бы без вердикта;
            # дальше пачку начинаем, только если она успеет закончиться в доле ролика (иначе перерасход
            # копится, и последнему ролику залпа не остаётся времени даже на один вопрос)
            stop = deadline - 12.0 if i == 0 else min(deadline - 12.0, share_end() - batch_sec)
            if time.monotonic() > stop:
                break
            t_batch = time.monotonic()
            try:
                got = self.asker.ask(path, batch, ctx)
                # сервер модели моргнул (туннель переподключается, перегрузка) — переспросить, пока есть время
                for wait in (3.0, 8.0):
                    if not any(_failed(a) for a in got) or time.monotonic() + wait > stop:
                        break
                    time.sleep(wait)
                    got = self.asker.ask(path, batch, ctx)
                for a in got:
                    answers[a.key] = a
                asked += len(batch)
                batch_sec = max(4.0, time.monotonic() - t_batch)
            except Exception as e:
                timings["ask_error"] = repr(e)[:200]
                break
        timings["t_ask"] = round(time.monotonic() - t0, 2)
        found = decide.judge(probes, answers)
        complete = trustworthy(found, event_items, answers, zones)

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
                                        observations={pred: "observed"}, sources=["detector", "vlm"],
                                        note=(f.seen or f.reason)[:600]))

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
            "vlm_errors": sum(1 for a in answers.values() if _failed(a)),
            "vlm_ground_error": str(zones.diag.get("error") or "")[:200],
            **_fallback_diag(self.asker, fb0),
            **timings,
        }
        return ObservationBatch(complete=complete, events=events if complete else [], diagnostics=diag)


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


def _failed(a: CropAnswer) -> bool:
    return a.raw.startswith("error:")


def trustworthy(found, event_items, answers: dict[str, CropAnswer], zones) -> bool:
    """Можно ли выдать вердикт. Нашли нарушение — да: вердикт «нарушение» верен, даже если часть вопросов
    не дошла до модели. Ничего не нашли — «чисто» можно сказать, только если модель ответила хоть на что-то
    и сорвалось меньше половины вопросов по событиям: иначе «чисто» значило бы «модель лежала», а пропуск
    нарушения хуже, чем честное «не обработано» (complete=False -> failed)."""
    if found:
        return True
    ok = [a for a in answers.values() if not _failed(a)]
    if answers and not ok:
        return False  # модель не ответила ни на один вопрос
    if not answers and zones.diag.get("error"):
        return False  # даже зоны не нашли: модель недоступна с самого начала
    asked = [answers[it.key] for it in event_items if it.key in answers]
    if event_items and not asked:
        return False  # были моменты для проверки, но ни один вопрос не задан (время кончилось раньше)
    # одиночный сорвавшийся вопрос не делает почти полный разбор «необработанным»; половина и больше — делает
    return sum(_failed(a) for a in asked) < 0.5 * len(asked) if asked else True


def _fallback_diag(asker, before: tuple[int, int]) -> dict:
    """Сколько запросов этого ролика ответила запасная локальная модель и сколько раз падал основной сервер."""
    st = getattr(asker, "stats", None)
    if st is None or getattr(asker, "_fallback", None) is None:
        return {}
    return {"vlm_fallback_calls": int(getattr(st, "fallback_calls", 0)) - before[0],
            "vlm_primary_failures": int(getattr(st, "primary_failures", 0)) - before[1],
            "vlm_fallback_reason": str(getattr(st, "fallback_reason", ""))[:200]}


def _forget(asker) -> None:
    mem = getattr(asker, "_mem", None)
    if isinstance(mem, dict):
        mem.clear()


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
    from .ask import CropAsker, MLXChat
    from .detect import make_detector

    url = os.environ.get("FORTE_VLM_URL", "http://127.0.0.1:18080")
    key = os.environ.get("FORTE_VLM_KEY_FILE")
    cache = os.environ.get("FORTE_ANCHORED_CACHE")  # только для офлайн-оценки; в проде пусто
    # запасная локальная MLX-модель на случай недоступного сервера модели; веса грузятся при первом сбое
    fb_model = os.environ.get("FORTE_VLM_FALLBACK_MODEL", "").strip()
    asker = CropAsker(url=url, key_file=key, model=os.environ.get("FORTE_VLM_MODEL", "local"), cache_dir=cache,
                      timeout=float(os.environ.get("FORTE_VLM_TIMEOUT", "45")),
                      fallback=MLXChat(fb_model) if fb_model else None,
                      retry_sec=float(os.environ.get("FORTE_VLM_RETRY_SEC", "30")))
    return AnchoredAnalyzer(make_detector(os.environ.get("FORTE_DETECTOR", "nano")), asker,
                            reserve_sec=float(os.environ.get("FORTE_VISION_RESERVE_SEC", "20")))
