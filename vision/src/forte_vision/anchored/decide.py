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

from .types import Box, CropAnswer, CropItem, DeskBlob, Event, People, Zones

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
    "QUE": "served_out_of_queue",
    "BAG": "collection_bag_left_uncounted",
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
                 "documents = papers, forms, a passport, a folder; "
                 "cash_bag = a cloth cash-collection sack (often grey or white, with a seal, tag or zip); "
                 "bag = a personal handbag or backpack, or a coat or jacket nobody is wearing; "
                 "food = any food or snack; drink = a cup, mug, glass or bottle; phone = a mobile phone. "
                 "A laptop (open or closed), keyboard, mouse, desk telephone, counting machine and clothes worn by "
                 "people do not count.",
    # экран после ухода: выбор вместо «включён ли». Экран входа Windows (синий) модель считала «включённым»,
    # и проверка срабатывала на чистых близнецах X-23/X-29; документ на экране — белое окно (X-11, X-28)
    "screen_what": "What does this laptop or monitor screen show? document = a white page, a table, a form or a "
                   "program window; desktop = a desktop with icons and a taskbar; lock = a lock or sign-in screen "
                   "(a full-screen picture or plain colour, maybe a big clock or a user name, no windows); "
                   "off = black or switched off, or the lid is closed.",
}
DESK_CHOICES = ("cash", "documents", "cash_bag", "bag", "food", "drink", "phone", "none")
# новый предмет на столе (пятно от desk.desk_blobs): сначала «что это» словами, потом группа
Q["desk_obj"] = ("Look at the object in the middle of this crop, lying on the desk (ignore people and hands). "
                 "What is it? cash = paper banknotes or a bundle of banknotes, also in a paper band or a plastic pack; "
                 "documents = papers, forms, a passport, a folder; "
                 "cash_bag = a cloth cash-collection sack (often grey or white, with a seal, tag or zip); "
                 "bag = a personal handbag or backpack, or a coat or jacket nobody is wearing; "
                 "food = any food or snack; drink = a cup, mug, glass or bottle; phone = a mobile phone; "
                 "other = any other object; none = no object, just the desk surface, a shadow or equipment "
                 "(laptop, telephone, counting machine).")
OBJ_CHOICES = ("cash", "documents", "cash_bag", "bag", "food", "drink", "phone", "other", "none")
SCREEN_CHOICES = ("document", "desktop", "lock", "off")


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
    ch = {"desk_what": DESK_CHOICES, "desk_obj": OBJ_CHOICES, "screen_what": SCREEN_CHOICES}.get(q, ())
    return [CropItem(key=f"{tag}@{_clip(t, dur):g}", t=_clip(t, dur), zone=zone, box=box, question=Q[q], zoom=zoom,
                     choices=ch, open=q == "desk_obj") for t in times]


def _grow(box: Box, min_w: float = 0.12, min_h: float = 0.07) -> Box:
    """Рамка вокруг пятна с контекстом: предмет в середине, но видно, что это стол (пачка денег — 15 px)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = max(min_w, 2.0 * (box[2] - box[0])), max(min_h, 2.0 * (box[3] - box[1]))
    x1, y1 = min(max(0.0, cx - w / 2), 1.0 - w), min(max(0.0, cy - h / 2), 1.0 - h)
    return (round(max(0.0, x1), 4), round(max(0.0, y1), 4), round(min(1.0, x1 + w), 4), round(min(1.0, y1 + h), 4))


def blob_items(blobs: list[DeskBlob], dur: float) -> list[tuple[DeskBlob, list[CropItem]]]:
    """Две вырезки на каждое пятно: вскоре после появления и попозже (пока лежит)."""
    out = []
    for i, b in enumerate(blobs):
        t1 = b.t_on + 1.0
        t2 = max(t1 + 1.0, min(b.t_on + 3.5, b.t_off - 0.5))
        out.append((b, _items(f"OBJ{i}", "desk", _grow(b.box), [t1, t2], "desk_obj", dur)))
    return out


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
         periodic_step: float = 4.0, door_open=None, blobs: list[DeskBlob] | None = None) -> list[Probe]:
    """Список проверок для ролика. Ничего не спрашивает у модели.
    door_open(t) -> bool | None — состояние двери по пикселям (VLM открытую дверь на вырезке не различает).
    blobs — новые предметы на столе по пикселям (desk.desk_blobs). Если заданы (даже пустой список), вещи и еда
    проверяются только по ним, а слепой обход стола выключен: он давал ложные тревоги на чистых роликах."""
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
        # LOCK: только открытый документ/окно (белое) — синий экран входа и рабочий стол на 464 px не различить
        for check, zone, q, want in (("LOCK", "screen", "screen_what", ("document",)),
                                     ("LEAVE", "desk", "desk_valuables", ()), ("SAFE", "safe", "safe_open", ())):
            if allow(check):
                probes.append(Probe(check, t_main, _items(check, zone, z.get(zone), ts, q, dur),
                                    t_start=t_main, t_end=_clip(ts[-1], dur), reason=f"после ухода кассира в {e.t:.1f} с",
                                    want=want))
        if allow("KEYS"):
            probes.append(Probe("KEYS", t_main, _items("KEYS", "safe", z.get("safe"), ts[:2], "safe_key", dur, zoom=6.0),
                                t_start=t_main, t_end=_clip(ts[1], dur), reason="ключ в замке после ухода"))
        # 1б) Ценность так и лежит на столе после ухода: пятно появилось до ухода и не исчезло. Что это — по плотной
        #     вырезке: когда положили (OBJi@) и уже без кассира (OBJix@). X-15 — зелёный инкассаторский мешок, X-13 —
        #     пачка купюр; широкий вопрос «есть ли на столе ценности» по всему столу их не видел (0 «да» из 3).
        for i, (b, its) in enumerate(blob_items(blobs or [], dur)):
            if not its or b.t_on > gone - 0.5 or b.t_off < min(gone + 1.0, dur - 0.5):
                continue
            after = _items(f"OBJ{i}x", "desk", its[0].box, [min(gone + 1.0, max(gone, hi - 0.2))], "desk_obj", dur)
            for check, want in (("LEAVE", ("cash", "documents")), ("BAG", ("cash_bag",))):
                if allow(check):
                    probes.append(Probe(check, t_main, its + after, need_yes=2, t_start=t_main,
                                        t_end=_clip(gone + 1.0, dur), want=want,
                                        reason=f"на столе осталось после ухода кассира в {e.t:.1f} с"))

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

    # 3) Смена клиентов: деньги предыдущего клиента ещё на столе, когда подходит следующий.
    #    Смотрим в конце промежутка между клиентами, до того как новый что-то отдал: после его прихода на столе
    #    уже его собственные деньги (X-5 — ложная тревога). Таймкод — приход следующего (X-38: эталон 33, пришёл 34.8).
    for e in [e for e in ev if e.kind == "client_change"]:
        hi = min(float(e.meta.get("t_seen", e.t)), e.t) - 0.3
        lo = float(e.meta.get("prev_exit", hi - 3.0)) + 0.5
        ts = [max(lo, hi - 2.0), hi] if hi - lo >= 1.0 else [max(0.0, e.t - 1.0), e.t]
        if allow("MIX"):
            probes.append(Probe("MIX", e.t, _items("MIX", "desk", z.get("desk"), ts, "desk_what", dur),
                                t_start=ts[0], t_end=_clip(e.t_end if e.t_end is not None else e.t, dur),
                                reason="смена клиентов", want=("cash",)))

    # 3б) Второй посетитель «через стол», пока первого ещё обслуживают (X-18; в чистом X-19 второй ждёт за
    #     стеклом, пока первый не выйдет). Решает детектор людей, без модели: на 40 роликах событие есть только
    #     в роликах с нарушением. Таймкод — второй дошёл до стола и остановился (X-18: появился 15.4, эталон 20).
    for e in [e for e in ev if e.kind == "third_person" and e.meta.get("side") == "across"]:
        if allow("QUE"):
            t_main = _settle_time(people, e)
            probes.append(Probe("QUE", t_main, [], t_start=e.t, t_end=_clip(e.t_end if e.t_end else t_main, dur),
                                reason=f"второй посетитель у стола во время визита с {e.t:.1f} с", fixed=True))

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
    sweep = [("SAFE", "safe", "safe_open", "SAFEp", ())]
    if blobs is None:
        sweep += [("ITEMS", "desk", "desk_what", "DESKp", ("bag",)),
                  ("FOOD", "desk", "desk_what", "DESKp", ("food", "drink"))]
    else:
        # 6) Новый предмет на столе: что это — по плотной вырезке, на двух кадрах (OBJi@t общие для проверок)
        for b, its in blob_items(blobs, dur):
            for check, want in (("ITEMS", ("bag",)), ("FOOD", ("food", "drink"))):
                if allow(check) and its:
                    probes.append(Probe(check, b.t_on, its, need_yes=2, t_start=b.t_on, t_end=min(b.t_off, dur),
                                        reason=f"новый предмет на столе {b.t_on:.1f}–{b.t_off:.1f} с", want=want))
    for check, zone, q, tag, want in sweep:
        if not allow(check) or z.get(zone) is None:
            continue
        ts = [t for t in present if not (check == "SAFE" and _near(people, t, z.get("safe")))]
        if ts:
            probes.append(Probe(check, ts[0], _items(tag, zone, z.get(zone), ts, q, dur),
                                need_yes=2, t_start=ts[0], t_end=ts[-1], reason="обход по ролику", want=want))
    return probes


def _settle_time(people: People, e: Event, still: float = 0.02, hold: float = 1.0) -> float:
    """Когда трек посетителя из события перестал идти: центр рамки за hold секунд сдвинулся меньше still."""
    tr = next((v for v in people.visitors if v.tid == e.meta.get("tid")), None)
    if tr is None or len(tr.t) < 3:
        return e.t
    c = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in tr.boxes]
    for i, t in enumerate(tr.t):
        if t < tr.start + 1.0:
            continue
        j = next((k for k in range(i, len(tr.t)) if tr.t[k] >= t + hold), None)
        if j is None:
            break
        if max(abs(c[k][0] - c[i][0]) + abs(c[k][1] - c[i][1]) for k in range(i, j + 1)) < still:
            return float(t)
    return e.t


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
