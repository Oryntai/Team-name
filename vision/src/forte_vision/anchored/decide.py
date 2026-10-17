"""События людей + зоны -> вопросы VLM по вырезкам -> нарушения с таймкодом.

Каждая проверка привязана к событию, которое видит детектор (не модель): кассир ушла, вошла, сменился
клиент, клиент ушёл. В этот момент модель получает увеличенную вырезку нужной зоны и вопрос да/нет.
Нарушение — только если ответ «да» повторился на двух разных кадрах: одиночный ответ не доказательство.
Таймкоды выверены по парам «нарушение / чистый близнец» открытой выборки (см. отчёт cue_table):
эталон ставит момент, когда кассир встаёт / клиент садится, а не момент, когда она скрылась за дверью.

План (plan) и решение (judge) разделены, чтобы вопросы одного ролика ушли пачкой, а решение
пересчитывалось по кешу ответов без повторных вызовов модели.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import Box, CropAnswer, CropItem, Event, People, Zones

# Проверка -> предикат каталога rules/cash_checks.json (тот же словарь, что timeline.CHECKS).
PREDICATE = {
    "LOCK": "left_with_computer_unlocked",
    "LEAVE": "left_valuables_on_desk",
    "SAFE": "safe_left_open",
    "DOOR": "cash_room_door_left_open",
    "MIX": "client_cash_mixed_or_left_for_next_client",
    "NOCLIENT": "counted_after_client_left",
    "ITEMS": "personal_items_on_desk",
    "FOOD": "food_or_drink_at_desk",
    "KEYS": "keys_left_in_safe_lock",  # выдаётся, только если правило есть в загруженном каталоге
}

Q = {
    "screen_on": "Is this laptop or monitor screen switched on and showing a document, a table or a desktop "
                 "(NOT a lock screen, NOT black/off, lid NOT closed)?",
    "desk_valuables": "Are banknotes, a cash bundle, coin rolls, documents, a folder or a cash bag lying on this desk?",
    "safe_open": "Is the door of this safe or metal cabinet standing open?",
    "safe_key": "Is a key sticking out of the lock of this safe?",
    "door_open": "Is this door standing open (door leaf swung away, an open gap / doorway visible)?",
    "desk_cash": "Is a stack of banknotes lying on this desk?",
    "counting": "Is someone putting banknotes into this counting machine or counting banknotes by hand here?",
    "desk_items": "Is a handbag, a bag, a coat or a jacket lying on this desk?",
    "desk_food": "Is there a cup, mug, glass, bottle or food on this desk?",
    # один вопрос с вариантами вместо отдельных «да/нет»: выбирая, модель не путает пачку денег с едой (X-38)
    "desk_what": "Which of these are lying on this desk (not held in someone's hands)? "
                 "cash = paper banknotes or a bundle of banknotes, also in a paper band or a plastic pack; "
                 "documents = papers, forms, a passport, a folder; bag = a handbag, bag, backpack, coat or jacket; "
                 "food = any food or snack; drink = a cup, mug, glass or bottle; phone = a mobile phone. "
                 "A laptop, keyboard, mouse, desk telephone and counting machine do not count.",
}
DESK_CHOICES = ("cash", "documents", "bag", "food", "drink", "phone", "none")


@dataclass
class Probe:
    """Одна проверка: какие вырезки спросить и как из ответов получить нарушение."""

    check: str
    t_main: float  # таймкод нарушения, если оно подтвердится
    items: list[CropItem]
    need_yes: int = 2  # сколько «да» нужно (из items)
    veto: list[CropItem] = field(default_factory=list)  # «да» на любом из них отменяет нарушение
    t_start: float | None = None
    t_end: float | None = None
    reason: str = ""
    fixed: bool | None = None  # решено без модели (пиксели): True — нарушение, False — нет
    want: tuple[str, ...] = ()  # вопрос с вариантами: «да» = в ответе есть что-то из want


@dataclass
class Finding:
    check: str
    t_main: float
    t_start: float
    t_end: float
    evidence: list[float]
    reason: str


def _clip(t: float, dur: float) -> float:
    return round(min(max(t, 0.0), max(dur - 0.05, 0.0)), 3)


def _items(tag: str, zone: str, box: Box | None, times, q: str, dur: float, zoom: float = 1.0) -> list[CropItem]:
    if box is None:
        return []
    ch = DESK_CHOICES if q == "desk_what" else ()
    return [CropItem(key=f"{tag}@{_clip(t, dur):g}", t=_clip(t, dur), zone=zone, box=box, question=Q[q], zoom=zoom,
                     choices=ch) for t in times]


def _yes(answers: dict[str, CropAnswer], it: CropItem, want=()) -> bool:
    a = answers.get(it.key)
    return a is not None and a.has(want)


def _first(events: list[Event], kind: str, after: float = -1.0, before: float = 1e9) -> Event | None:
    for e in sorted(events, key=lambda e: e.t):
        if e.kind == kind and after <= e.t <= before:
            return e
    return None


def _leave_time(e: Event) -> float:
    """Таймкод нарушения при уходе. Эталон ставит его, когда кассир встала и пошла (X-26: встала 9, вышла 13–15),
    но не на раннее вставание задолго до ухода (X-11: встала 12–13, ушла 18) — тогда чуть раньше ухода с места."""
    st = e.meta.get("t_stand")
    if st is not None and 0.0 <= e.t - st <= 4.0:
        return float(st)
    return max(0.0, e.t - 0.5)


def plan(people: People, zones: Zones, dur: float, *, known_predicates: set[str] | None = None,
         periodic_step: float = 4.0, door_open=None) -> list[Probe]:
    """Список проверок для ролика. Ничего не спрашивает у модели.
    door_open(t) -> bool | None — состояние двери по пикселям (VLM открытую дверь на вырезке не различает)."""
    z = zones.boxes
    ev = people.events
    probes: list[Probe] = []
    allow = (lambda c: known_predicates is None or PREDICATE[c] in known_predicates)

    # 1) Кассир ушла: экран, стол, сейф, ключ — когда её уже нет в кадре рабочего места.
    for e in [e for e in ev if e.kind == "cashier_leaves"]:
        gone = float(e.meta.get("t_exit", e.t + 2.0))
        back = _first(ev, "cashier_enters", after=gone) or _first(ev, "cashier_sits", after=gone)
        hi = min(dur, back.t - 0.5) if back else dur
        if hi - gone < 1.0:
            gone = e.t + 1.0
        ts = [gone + 0.5, min(gone + 3.0, hi), min(gone + 6.0, hi)]
        t_main = _leave_time(e)
        for check, zone, q in (("LOCK", "screen", "screen_on"), ("LEAVE", "desk", "desk_valuables"),
                               ("SAFE", "safe", "safe_open")):
            if allow(check):
                probes.append(Probe(check, t_main, _items(check, zone, z.get(zone), ts, q, dur),
                                    t_start=t_main, t_end=_clip(ts[-1], dur), reason=f"после ухода кассира в {e.t:.1f} с"))
        if allow("KEYS"):
            probes.append(Probe("KEYS", t_main, _items("KEYS", "safe", z.get("safe"), ts[:2], "safe_key", dur, zoom=6.0),
                                t_start=t_main, t_end=_clip(ts[1], dur), reason="ключ в замке после ухода"))

    # 2) Кассир вошла: дверь за собой, личные вещи на столе.
    for e in [e for e in ev if e.kind in ("cashier_enters", "cashier_sits")]:
        if e.kind == "cashier_sits" and _first(ev, "cashier_enters", after=e.t - 8.0, before=e.t):
            continue  # вход уже учтён
        t_in = e.t
        if allow("DOOR") and door_open is not None:
            # открыта и через 5, и через 8 с после входа: вошла и закрыла (X-31) — дверь открыта лишь пару секунд
            late = [t for t in (t_in + 5.0, t_in + 8.0) if t < dur - 0.2]
            states = [door_open(t) for t in late]
            if late and None not in states:
                probes.append(Probe("DOOR", t_in + 1.0, [], t_start=t_in, t_end=_clip(late[-1], dur),
                                    reason=f"дверь после входа кассира в {t_in:.1f} с", fixed=all(states)))

    # 3) Смена клиентов: деньги предыдущего на столе, когда садится следующий.
    for e in [e for e in ev if e.kind == "client_change"]:
        t2 = e.t_end if e.t_end is not None else e.t
        if allow("MIX"):
            probes.append(Probe("MIX", t2, _items("MIX", "desk", z.get("desk"), [t2, t2 + 2.0], "desk_what", dur),
                                t_start=e.t, t_end=_clip(t2 + 2.0, dur), reason="смена клиентов", want=("cash",)))

    # 4) Клиент ушёл, кассир осталась: пересчёт без клиента (только если до ухода не считали).
    for e in [e for e in ev if e.kind == "client_leaves"]:
        if e.t > dur - 2.0 or not _present(people, e.t + 2.0):
            continue
        nxt = _first(ev, "client_arrives", after=e.t + 0.5)
        if nxt and nxt.t - e.t < 4.0:
            continue  # это смена клиентов, а не уход
        if allow("NOCLIENT"):
            probes.append(Probe("NOCLIENT", e.t + 1.5,
                                _items("NOCL", "counter", z.get("counter"), [e.t + 1.5, e.t + 4.0], "counting", dur),
                                veto=_items("NOCLpre", "counter", z.get("counter"), [e.t - 2.0], "counting", dur),
                                t_start=e.t, t_end=_clip(e.t + 4.0, dur), reason="пересчёт после ухода клиента"))

    # 5) Состояния без события людей: сейф открыт при кассире не у сейфа, вещи и еда на столе.
    #    Редкий обход по всему ролику, подтверждение — два соседних «да».
    grid = [0.5] + [round(periodic_step * k, 3) for k in range(1, int(dur // periodic_step) + 1)]
    if dur - 1.0 - grid[-1] > 1.5:
        grid.append(round(dur - 1.0, 3))
    present = [t for t in grid if _present(people, t)]
    # ITEMS и FOOD читают один и тот же ответ «что лежит на столе» (общие ключи DESKp@t)
    for check, zone, q, tag, want in (("SAFE", "safe", "safe_open", "SAFEp", ()),
                                      ("ITEMS", "desk", "desk_what", "DESKp", ("bag",)),
                                      ("FOOD", "desk", "desk_what", "DESKp", ("food", "drink"))):
        if not allow(check) or z.get(zone) is None:
            continue
        ts = [t for t in present if not (check == "SAFE" and _near(people, t, z.get("safe")))]
        if ts:
            probes.append(Probe(check, ts[0], _items(tag, zone, z.get(zone), ts, q, dur),
                                need_yes=2, t_start=ts[0], t_end=ts[-1], reason="обход по ролику", want=want))
    return probes


def _present(people: People, t: float) -> bool:
    return any(a - 0.25 <= t <= b + 0.25 for a, b in people.cashier_present)


def _box_at(track, t: float):
    if track is None or not track.t:
        return None
    i = min(range(len(track.t)), key=lambda k: abs(track.t[k] - t))
    return track.boxes[i] if abs(track.t[i] - t) <= 1.0 else None


def _near(people: People, t: float, box: Box | None) -> bool:
    """Кассир у сейфа: её рамка пересекает расширенную рамку сейфа."""
    b = _box_at(people.cashier, t)
    if b is None or box is None:
        return False
    w, h = box[2] - box[0], box[3] - box[1]
    x1, y1, x2, y2 = box[0] - 0.5 * w, box[1] - 0.5 * h, box[2] + 0.5 * w, box[3] + 0.5 * h
    return not (b[2] < x1 or b[0] > x2 or b[3] < y1 or b[1] > y2)


def judge(probes: list[Probe], answers: dict[str, CropAnswer]) -> list[Finding]:
    """Ответы модели -> подтверждённые нарушения. «unclear» — не «да»."""
    out: list[Finding] = []
    for p in probes:
        if p.fixed is not None:
            if p.fixed:
                out.append(Finding(p.check, p.t_main, p.t_start or p.t_main, p.t_end or p.t_main,
                                   [p.t_main, p.t_end or p.t_main], p.reason))
            continue
        if not p.items:
            continue
        yes = [it for it in p.items if _yes(answers, it, p.want)]
        if any(_yes(answers, v) for v in p.veto):
            continue
        if p.reason == "обход по ролику":
            # два соседних «да» подряд; таймкод — первое из них
            ys = [it.t for it in yes]
            run = [a for a, b in zip(ys, ys[1:]) if b - a <= 2.01 * (p.items[1].t - p.items[0].t if len(p.items) > 1 else 4.0)]
            if not run:
                continue
            t0 = run[0]
            out.append(Finding(p.check, t0, t0, t0 + 4.0, [t0], f"{p.reason}: подтверждено на двух кадрах"))
            continue
        if len(yes) < min(p.need_yes, len(p.items)):
            continue
        ev_t = [p.t_main] + [it.t for it in yes[:2]]
        out.append(Finding(p.check, p.t_main, min([p.t_start or p.t_main] + ev_t), max([p.t_end or p.t_main] + ev_t),
                           ev_t, p.reason))
    return _dedupe(out)


def _dedupe(found: list[Finding]) -> list[Finding]:
    """Одна проверка — одно нарушение на событие: близкие (≤6 с) сливаем, оставляя раннее.
    SAFE из обхода и SAFE после ухода — одно и то же событие."""
    found = sorted(found, key=lambda f: (f.check, f.t_main))
    out: list[Finding] = []
    for f in found:
        if out and out[-1].check == f.check and f.t_main - out[-1].t_main <= 6.0:
            continue
        out.append(f)
    return sorted(out, key=lambda f: f.t_main)


def door_state_fn(scan, zones: Zones, people: People, thr: float = 0.25):
    """Открыта ли дверь в момент t — по пикселям створки против кадров до входа кассира, люди закрашены.
    Возвращает функцию t -> bool | None (None: двери не нашли / всё закрыто людьми / нет опорных кадров)."""
    import numpy as np

    box = zones.boxes.get("door") or zones.diag.get("door_wall")
    G = None if scan is None else scan.gray
    if box is None or G is None or len(G) == 0:
        return None
    G = np.asarray(G, np.float32)
    T = np.asarray(scan.gray_t if len(scan.gray_t) == len(G) else [f.t for f in scan.frames], float)
    _, gh, gw = G.shape
    x1, y1 = int(box[0] * gw), int(box[1] * gh)
    x2, y2 = max(x1 + 1, int(np.ceil(box[2] * gw))), max(y1 + 1, int(np.ceil(box[3] * gh)))
    seen = [float(e.meta.get("t_seen", e.t)) for e in people.events if e.kind == "cashier_enters"]
    ref_end = max(0.6, (min(seen) - 0.5) if seen else 0.6)
    ref_idx = np.where(T <= ref_end)[0]
    if len(ref_idx) == 0:
        return None
    ref = np.median(G[ref_idx], axis=0)

    def person_mask(k: int) -> np.ndarray:
        m = np.zeros((gh, gw), bool)
        for p in scan.frames[k].persons:
            if p.conf < 0.3:
                continue
            b, w, h = p.box, p.box[2] - p.box[0], p.box[3] - p.box[1]
            m[max(0, int((b[1] - 0.1 * h) * gh)):int(np.ceil((b[3] + 0.1 * h) * gh)),
              max(0, int((b[0] - 0.15 * w) * gw)):int(np.ceil((b[2] + 0.15 * w) * gw))] = True
        return m

    def state(t: float):
        idx = np.where(np.abs(T - t) <= 0.6)[0]
        vals = []
        for k in idx:
            d = np.abs(G[k] - ref)
            d -= np.median(d)  # смена освещения всего кадра
            keep = ~person_mask(k)[y1:y2, x1:x2]
            if keep.mean() < 0.3:
                continue
            vals.append(float((d[y1:y2, x1:x2][keep] > 25).mean()))
        return None if not vals else float(np.median(vals)) >= thr

    state.ratio_debug = True
    return state


def funnel(probes: list[Probe], answers: dict[str, CropAnswer], found: list[Finding]) -> str:
    """Где теряется проверка: CHECK:проб/без_зоны/спрошено/да/подтверждено, через «;» (плоско для диагностики)."""
    stats: dict[str, list[int]] = {}
    for p in probes:
        s = stats.setdefault(p.check, [0, 0, 0, 0, 0])
        s[0] += 1
        s[1] += not p.items
        s[2] += sum(1 for it in p.items if it.key in answers)
        s[3] += sum(1 for it in p.items if _yes(answers, it, p.want))
    for f in found:
        stats.setdefault(f.check, [0, 0, 0, 0, 0])[4] += 1
    return ";".join(f"{c}:{'/'.join(map(str, s))}" for c, s in sorted(stats.items()))


def all_items(probes: list[Probe]) -> list[CropItem]:
    seen, out = set(), []
    for p in probes:
        for it in p.items + p.veto:
            if it.key not in seen:
                seen.add(it.key)
                out.append(it)
    return out
