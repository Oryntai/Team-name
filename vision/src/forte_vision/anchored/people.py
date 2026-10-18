"""Люди по детектору: треки -> кто в комнате -> кассир -> посетители и их сторона стола -> события.

Без лиц и без внешности: только рамки, уверенность, время и смена пикселей рядом с человеком. Всё в долях
кадра и в единицах h — высоты рамки человека, поэтому правила не зависят от разрешения и масштаба ракурса.
Шаги (номера — из исследования ролей):
  R0 дубли рамок в кадре (DETR: целиком + верх тела) -> бо́льшая рамка;
  R1 жадный трекер: очки IoU - 0.5·d (d — расстояние центров в высотах), дыры до GAP_S;
  R2 «в комнате» = медиана уверенности трека >= TAU_ROOM (за стеклом детектор в основном неуверен);
  R3 места = кластеры рамок по IoU, frac — доля кадров;
  R4 кассир = трек с наибольшим числом кадров на местах frac >= 0.30 + склейка обрывков;
  R5 сторона стола: подсказка desk снаружи -> предметы на столе у кассира -> самый долгий посетитель;
  R6 сторона посетителя: sigma = (y2 - y2_места)·dir / h_места;
  R7 визиты клиента = занятость стороны «через стол», разрез по пустоте и по смене сидящего;
  R8 третий: посетитель рядом/за спиной кассира или второй «через стол» во время визита;
  R9 присутствие кассира = кадры цепочки с дырами <= PRESENT_GAP.
Человек за прозрачной дверью виден так же уверенно, как в комнате, поэтому момент прохода через дверь
уточняется «дверной активностью»: смена пикселей ВНЕ рамок людей рядом с человеком (створка двери).
"""

from __future__ import annotations

import numpy as np

from .types import Box, DenseScan, Event, People, Track

DEDUP_IOU, DEDUP_CONT = 0.5, 0.7
BOX_MIN = 0.3  # рамки слабее в трекер не идут
GAP_S = 1.5
TAU_ROOM = 0.80  # nano, порог 0.25, 5 к/с: 0.78–0.82 дают верное число людей 40/40, 0.75 и 0.85 — нет
SLOT_IOU, SLOT_FRAC = 0.45, 0.30
LINK_S, LINK_H = 2.5, 1.2
UNCERTAIN = 0.85
SIDE = 0.45
AXIS_RATIO = 2.0  # горизонтальная ось «кассир — клиент», если по x смещение больше чем вдвое
MIN_VISITOR_S = 1.5
VISIT_GAP = 3.0
PRESENT_GAP = 1.5
ABSENT_MIN = 3.0  # отсутствие короче — не уход (провал детектора, наклон под стол)
EDGE_TOL = 1.0  # кассир «был с начала» / «до конца», если цепочка ближе к краю ролика
WORK_IOU = 0.5  # на рабочем месте: IoU с местом (стоя у стола в ракурсе «лицом» IoU ~0.35)
DWELL_S = 2.0  # пребывание на месте не короче
AWAY_S = 1.5  # встала: вне места не меньше
SEAT_IOU = 0.3  # место клиента
DOOR_ACT = 0.2  # смена пикселей вне людей рядом с человеком, в долях площади его места
DOOR_MIN_S = 0.4
DOOR_JOIN_S = 0.81  # дыры между кадрами активности внутри одного прохода
LINGER_S = 4.0  # после двери кассир виден за стеклом не дольше — иначе это не выход (сейф, шкаф)
DESK_OBJ = ("tv", "keyboard", "mouse", "suitcase", "cell phone", "book", "remote")


# --- геометрия ---

def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _h(b: Box) -> float:
    return max(1e-6, b[3] - b[1])


def _dist(a: Box, b: Box) -> float:
    """Расстояние центров в высотах бо́льшей рамки."""
    d = np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)
    return float(d / max(_h(a), _h(b)))


def _cdist(a: Box, b: Box, h: float) -> float:
    """Расстояние центров в единицах h."""
    return float(np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2) / max(h, 1e-6))


def _iou_mat(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    ab = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def _mode_box(boxes) -> Box:
    """Самое частое положение: медиана рамок вокруг рамки с наибольшим числом соседей по IoU."""
    B = np.array(boxes, float)
    M = _iou_mat(B, B) >= SLOT_IOU
    return tuple(float(v) for v in np.median(B[M[int(np.argmax(M.sum(1)))]], axis=0))


def _runs(ts, gap: float) -> list[tuple[float, float]]:
    """Отрезки времени по возрастающим отсчётам ts с дырами <= gap."""
    out: list[list[float]] = []
    for t in ts:
        if out and t - out[-1][1] <= gap + 1e-6:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [(a, b) for a, b in out]


def dedupe(dets: list[tuple[Box, float]]) -> list[tuple[Box, float]]:
    """R0: дубли одной фигуры -> бо́льшая рамка с наибольшей уверенностью из группы."""
    keep: list[list] = []
    for b, c in sorted(dets, key=lambda x: -_area(x[0])):
        for k in keep:
            kb = k[0]
            ix = max(0.0, min(b[2], kb[2]) - max(b[0], kb[0]))
            iy = max(0.0, min(b[3], kb[3]) - max(b[1], kb[1]))
            if iou(b, kb) >= DEDUP_IOU or ix * iy / max(min(_area(b), _area(kb)), 1e-9) >= DEDUP_CONT:
                k[1] = max(k[1], c)
                break
        else:
            keep.append([b, c])
    return [(k[0], k[1]) for k in keep]


# --- R1: трекер ---

def _dmax(dt: float) -> float:
    """Допустимый сдвиг центра в высотах за dt секунд (5 к/с -> 0.4, 2 к/с -> 0.55, дыра 1.5 с -> 1.0)."""
    return min(1.0, 0.3 + 0.5 * dt)


def track(frames: list[tuple[float, list[tuple[Box, float]]]], gap_s: float = GAP_S) -> list[Track]:
    tracks: list[Track] = []
    for t, dets in frames:
        act = [tr for tr in tracks if t - tr.t[-1] <= gap_s]
        cand = []
        for i, tr in enumerate(act):
            lb, dt = tr.boxes[-1], t - tr.t[-1]
            for j, (b, _) in enumerate(dets):
                io, d, sr = iou(lb, b), _dist(lb, b), _h(b) / _h(lb)
                if (io >= 0.2 or d <= _dmax(dt)) and 0.4 <= sr <= 2.5:
                    cand.append((io - 0.5 * d, i, j))
        used_i, used_j = set(), set()
        for _, i, j in sorted(cand, reverse=True):
            if i in used_i or j in used_j:
                continue
            act[i].t.append(t)
            act[i].boxes.append(dets[j][0])
            act[i].conf.append(dets[j][1])
            used_i.add(i)
            used_j.add(j)
        for j, (b, c) in enumerate(dets):
            if j not in used_j:
                tracks.append(Track(len(tracks), [t], [b], [c]))
    return tracks


# --- R3: места ---

def slots(boxes: list[tuple[int, Box]], n_frames: int, iou_thr: float = SLOT_IOU, max_slots: int = 8,
          min_frames: int = 3) -> list[dict]:
    """Жадные кластеры рамок (k — номер кадра): затравка с наибольшей поддержкой, пересборка вокруг медианы."""
    if not boxes:
        return []
    B = np.array([b for _, b in boxes], float)
    K = np.array([k for k, _ in boxes])
    M = _iou_mat(B, B) >= iou_thr
    free = np.ones(len(B), bool)
    out = []
    while free.any() and len(out) < max_slots:
        sup = np.array([len(set(K[M[i] & free])) if free[i] else 0 for i in range(len(B))])
        i = int(np.argmax(sup))
        if sup[i] < min_frames:
            break
        med = np.median(B[M[i] & free], axis=0)
        mem = (_iou_mat(med[None], B)[0] >= iou_thr) & free
        if not mem.any():
            mem = M[i] & free
        box = tuple(float(v) for v in np.median(B[mem], axis=0))
        out.append({"box": box, "frac": len(set(K[mem])) / max(n_frames, 1), "n": int(mem.sum())})
        free &= ~mem
    return out


# --- R4: цепочка кассира ---

def _settled(tr: Track, slot_boxes: np.ndarray) -> int:
    if not len(slot_boxes):
        return 0
    return int((_iou_mat(np.array(tr.boxes, float), slot_boxes) >= SLOT_IOU).any(1).sum())


def _chain(tracks: list[Track], best: int) -> list[int]:
    """Склейка обрывков к лучшему треку: продолжение <= LINK_S после конца / до начала, на расстоянии <= LINK_H."""
    chain, used = [best], {best}
    while True:
        fa = tracks[min(chain, key=lambda i: tracks[i].start)]
        la = tracks[max(chain, key=lambda i: tracks[i].end)]
        cand = None
        for i, tr in enumerate(tracks):
            if i in used:
                continue
            ok = (0 < tr.start - la.end <= LINK_S and _dist(tr.boxes[0], la.boxes[-1]) <= LINK_H) or \
                 (0 < fa.start - tr.end <= LINK_S and _dist(tr.boxes[-1], fa.boxes[0]) <= LINK_H)
            if ok and (cand is None or len(tr.t) > len(tracks[cand].t)):
                cand = i
        if cand is None:
            return chain
        chain.append(cand)
        used.add(cand)


def _merge(tracks: list[Track], ids: list[int]) -> Track:
    rows = sorted((t, b, c) for i in ids for t, b, c in zip(tracks[i].t, tracks[i].boxes, tracks[i].conf))
    return Track(tracks[ids[0]].tid, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])


# --- дверная активность ---

def door_activity(scan: DenseScan, tr: Track, ref_area: float) -> list[tuple[float, float]]:
    """Отрезки, когда рядом с человеком (±1 h по горизонтали, ±0.5 h по вертикали) меняются пиксели ВНЕ рамок
    людей: открывается/закрывается дверь. Смена всего кадра (свет, экспозиция) не считается."""
    G = scan.gray
    if G is None or len(G) < 2 or len(scan.gray_t) != len(scan.frames):
        return []
    G = np.asarray(G, np.float32)
    _, gh, gw = G.shape
    box_at = dict(zip(tr.t, tr.boxes))
    need = DOOR_ACT * ref_area * gh * gw
    hits = []
    for k in range(1, len(G)):
        t = scan.frames[k].t
        b = box_at.get(t)
        if b is None:
            continue
        d = np.abs(G[k] - G[k - 1])
        r = d - np.median(d) > 25
        for f in (scan.frames[k], scan.frames[k - 1]):
            for p in f.persons:
                pb, w, h = p.box, p.box[2] - p.box[0], p.box[3] - p.box[1]
                r[max(0, int((pb[1] - 0.15 * h) * gh)):int(np.ceil((pb[3] + 0.15 * h) * gh)),
                  max(0, int((pb[0] - 0.25 * w) * gw)):int(np.ceil((pb[2] + 0.25 * w) * gw))] = False
        if r.mean() > 0.15:
            continue
        h = _h(b)
        ww = h * gh / gw  # h в долях ширины
        win = r[max(0, int((b[1] - 0.5 * h) * gh)):int(np.ceil(min(1.0, b[3] + 0.5 * h) * gh)),
                max(0, int((b[0] - ww) * gw)):int(np.ceil(min(1.0, b[2] + ww) * gw))]
        if win.sum() >= need:
            hits.append(t)
    # створка движется рывками: отдельные кадры активности внутри одного прохода разделены 1–3 кадрами покоя
    dt = float(np.median(np.diff(scan.gray_t)))
    return [r for r in _runs(hits, max(DOOR_JOIN_S, 1.5 * dt)) if r[1] - r[0] >= max(DOOR_MIN_S, dt) - 0.011]


# --- главный вход ---

def analyze_people(scan: DenseScan, *, desk: Box | None = None, tau_room: float = TAU_ROOM) -> People:
    """Роли и события людей. desk — рамка рабочего стола кассира (например, от VLM): задаёт сторону стола.

    cashier_present — интервалы, когда кассир в комнате (от входа до выхода, дыры <= PRESENT_GAP).
    События (t / t_end / meta):
      cashier_enters  t — прошла дверь (конец дверной активности) или первое появление; t_end — дошла до
                      рабочего места; meta: t_seen (первое появление, может быть за стеклом), by = door | track
      cashier_sits    t — заняла своё место (после входа или после того, как вставала)
      cashier_stands  t — встала со своего места, t_end — вернулась или пошла к выходу; на уход не влияет
      cashier_leaves  t — последний раз у своего места перед выходом, t_end = meta.t_exit — дверь закрылась
                      за ней (или цепочка кончилась); meta: t_stand (встала), by = door | track
      client_arrives  t — вошёл (дверь) или появился, t_end — сел на место клиента; meta: t_seen, t_seated, by
      client_leaves   t — встал с места клиента, t_end = meta.t_exit — вышел; by = door | track
      client_change   t — пришёл следующий клиент, t_end — сел; meta: gap (от выхода прежнего), prev_exit
      third_person    t..t_end — посторонний; meta.side: level (рядом с кассиром), behind (за её спиной),
                      across (второй посетитель «через стол» во время визита)
    Визит, идущий с первого кадра или до последнего, не даёт client_arrives / client_leaves.
    """
    times = [f.t for f in scan.frames]
    n = len(times)
    dt = float(np.median(np.diff(times))) if n > 1 else 0.2
    frames = [(f.t, dedupe([(d.box, d.conf) for d in f.persons if d.conf >= BOX_MIN])) for f in scan.frames]
    tracks = track(frames, max(GAP_S, 2.5 * dt))  # при прореженных кадрах (дедлайн) дыры длиннее
    room = [tr for tr in tracks if np.median(tr.conf) >= tau_room]
    diag: dict = {"tracks": len(tracks), "in_room": len(room), "tau_room": tau_room, "dt": round(dt, 3)}
    if not room:
        return People(None, [], [], [], 0, [], roles_uncertain=True, diag=diag)

    # R3 места по рамкам «в комнате», R4 кассир
    kidx = {t: k for k, t in enumerate(times)}
    S = slots([(kidx[t], b) for tr in room for t, b in zip(tr.t, tr.boxes)], n)
    big = np.array([s["box"] for s in S if s["frac"] >= SLOT_FRAC]).reshape(-1, 4)
    score = [_settled(tr, big) for tr in room]
    order = sorted(range(len(room)), key=lambda i: (-score[i], -len(room[i].t)))
    chain = _chain(room, order[0])
    cashier = _merge(room, chain)
    second = next((score[i] for i in order[1:] if i not in chain), 0)
    margin = second / max(score[order[0]], 1)
    seat = _mode_box(cashier.boxes)
    h = _h(seat)
    diag.update(margin=round(margin, 2), chain=len(chain), seat=tuple(round(v, 3) for v in seat),
                slots=[(tuple(round(v, 3) for v in s["box"]), round(s["frac"], 2)) for s in S])

    # R9 присутствие и события кассира
    doors = door_activity(scan, cashier, _area(seat))
    diag["cashier_doors"] = [(round(a, 1), round(b, 1)) for a, b in doors]
    events, present = _cashier_events(cashier, seat, doors, times[0], times[-1], dt)

    # R5, R6 посетители и их сторона
    others = [tr for i, tr in enumerate(room) if i not in chain and tr.end - tr.start >= MIN_VISITOR_S]
    axis = _desk_axis(seat, desk, others)
    ddir, dsrc = _desk_dir(scan, seat, desk, others, axis)
    diag["desk_dir_src"] = dsrc
    diag["desk_axis"] = "x" if axis == 0 else "y"
    side, sigma = {}, {}
    for tr in others:
        if ddir == 0:
            side[tr.tid] = "unknown"
            continue
        if axis == 1:  # камера сверху: глубина — по нижнему краю рамки (ноги, стул)
            sigma[tr.tid] = float(np.median([(b[3] - seat[3]) * ddir / h for b in tr.boxes]))
        else:  # камера сбоку: клиент по другую сторону стола — по горизонтали от кассира
            sigma[tr.tid] = float(np.median([((b[0] + b[2]) - (seat[0] + seat[2])) / 2 * ddir / h for b in tr.boxes]))
        s = sigma[tr.tid]
        side[tr.tid] = "across" if s > SIDE else "behind" if s < -SIDE else "level"
    diag["visitors"] = [(tr.tid, round(tr.start, 1), round(tr.end, 1), side[tr.tid], round(sigma.get(tr.tid, 0), 2))
                        for tr in others]

    # R7 визиты, R8 третий человек
    across = [tr for tr in others if side[tr.tid] in ("across", "unknown")]
    visits, vev = _visits(scan, across, seat, times[0], times[-1], dt)
    events += vev
    cset = set(cashier.t)
    events += _third(others, side, across, {tr.tid: sum(t in cset for t in tr.t) * dt for tr in others}, dt)
    events.sort(key=lambda e: e.t)
    return People(cashier, present, visits, others, ddir, events, margin >= UNCERTAIN, diag)


def _workplace(cashier: Track, seat: Box, dt: float) -> np.ndarray:
    """Рабочие места кассира: её положения рядом со своим местом (центр ближе 1 h места), где она провела
    >= 10% времени и >= DWELL_S: сидя у стола, стоя у счётчика. Дверь и коридор за стеклом сюда не попадают."""
    S = slots(list(enumerate(cashier.boxes)), len(cashier.t))
    keep = [s["box"] for s in S if s["frac"] >= 0.10 and s["n"] * dt >= DWELL_S and _cdist(s["box"], seat, _h(seat)) <= 1.0]
    return np.array(keep or [seat], float).reshape(-1, 4)


def _cashier_events(cashier: Track, seat: Box, doors, t_first: float, t_last: float,
                    dt: float) -> tuple[list[Event], list[tuple[float, float]]]:
    """События кассира и интервалы её присутствия в комнате (обрезаны по входу и выходу).

    Вход и выход привязаны к своему месту (seat — самое частое положение): дверная активность до первого
    пребывания на месте — вход, после последнего — выход. Рабочие места шире (стоя у счётчика и т.п.)
    и дают только моменты «дошла до рабочего места» и «пошла к выходу»: в кадре коридор за стеклом
    перекрывается с ними, поэтому для решения «вошла/ушла» они не годятся.
    """
    ev: list[Event] = []
    B = np.array(cashier.boxes, float)
    at_work = (_iou_mat(B, _workplace(cashier, seat, dt)) >= SLOT_IOU).any(1)
    work_t = [t for t, w in zip(cashier.t, at_work) if w]
    on_seat = [t for t, b in zip(cashier.t, cashier.boxes) if iou(b, seat) >= WORK_IOU]
    wg, pg = max(1.0, 1.5 * dt), max(PRESENT_GAP, 2.5 * dt)
    seat_dw = [r for r in _runs(on_seat, wg) if r[1] - r[0] >= DWELL_S]

    def dwell(lo, hi):
        return [r for r in _runs([t for t in work_t if lo <= t <= hi], wg) if r[1] - r[0] >= DWELL_S]

    blocks: list[list[float]] = []
    for a, b in _runs(cashier.t, pg):
        if blocks and a - blocks[-1][1] < ABSENT_MIN:
            blocks[-1][1] = b
        else:
            blocks.append([a, b])
    present = []
    for a, b in blocks:
        sd = [r for r in seat_dw if a <= r[0] <= b]
        s0 = sd[0][0] if sd else b
        # вход: появилась позже начала ролика, либо пришла из-за двери (активность до первого «на месте»)
        din = next((d for d in doors if a - 0.5 <= d[0] and d[1] <= s0 + 0.5 and d[1] - a >= 1.0), None)
        entered = a > t_first + EDGE_TOL or din is not None
        t_in = a
        if entered:
            t_in = din[1] if din else a
            arrive = next(iter(dwell(t_in, b)), (s0, s0))[0]
            ev.append(Event("cashier_enters", t_in, arrive, {"t_seen": a, "by": "door" if din else "track"}))
            ev.append(Event("cashier_sits", arrive, None, {"after": "enter"}))
        elif sd and sd[0][0] > a + EDGE_TOL:
            ev.append(Event("cashier_sits", sd[0][0]))
        # встала / села на своё место без ухода
        for (_, e0), (s1, _) in zip(sd, sd[1:]):
            if s1 - e0 >= AWAY_S:
                ev.append(Event("cashier_stands", e0, s1))
                ev.append(Event("cashier_sits", s1))
        # уход: дверная активность после последнего «на месте», после которой кассир видна (за стеклом)
        # не дольше LINGER_S; либо цепочка кончилась раньше конца ролика
        e_last = sd[-1][1] if sd else t_in
        dout = [d for d in doors if d[0] >= e_last - 0.5 and d is not din and b - d[1] <= LINGER_S]
        t_out = b
        if dout or b < t_last - EDGE_TOL:
            t_out = dout[-1][1] if dout else b  # дверь закрылась за ней (или цепочка кончилась)
            t_door = dout[0][0] if dout else b  # дверь начала открываться
            # ушла с рабочего места: последний раз у своего места (стоя у стола IoU ~0.35) до двери
            near = [t for t, bx in zip(cashier.t, cashier.boxes)
                    if e_last <= t <= t_door and iou(bx, seat) >= SEAT_IOU]
            t_leave = near[-1] if near else e_last
            if sd and t_leave - e_last >= AWAY_S:
                ev.append(Event("cashier_stands", e_last, t_leave, {"before": "leave"}))
            ev.append(Event("cashier_leaves", t_leave, t_out,
                            {"t_exit": t_out, "t_stand": e_last if sd else None, "by": "door" if dout else "track"}))
        # R9: отрезки с дырами <= PRESENT_GAP внутри [вход, выход]
        for x, y in _runs([t for t in cashier.t if a <= t <= b], pg):
            if min(y, t_out) > max(x, t_in):
                present.append((max(x, t_in), min(y, t_out)))
    return ev, present


def _desk_axis(seat: Box, desk: Box | None, visitors: list[Track]) -> int:
    """Ось «кассир — клиент»: 1 — вертикаль кадра (камера сверху, как на всех открытых роликах), 0 — горизонталь
    (камера сбоку: стол и клиент слева или справа от кассира). Горизонталь — только если стол от модели смещён от
    места кассира по x больше чем вдвое сильнее, чем по y, и самый долгий посетитель с этим не спорит; иначе —
    прежняя вертикальная логика."""
    if desk is None:
        return 1
    cx, cy = (seat[0] + seat[2]) / 2, (seat[1] + seat[3]) / 2
    dx, dy = (desk[0] + desk[2]) / 2 - cx, (desk[1] + desk[3]) / 2 - cy
    if abs(dx) <= AXIS_RATIO * abs(dy):
        return 1
    if visitors:
        tr = max(visitors, key=lambda x: x.end - x.start)
        vx = float(np.median([(b[0] + b[2]) / 2 for b in tr.boxes])) - cx
        vy = float(np.median([(b[1] + b[3]) / 2 for b in tr.boxes])) - cy
        if abs(vx) < abs(vy):
            return 1
    return 0


def _desk_dir(scan: DenseScan, seat: Box, desk: Box | None, visitors: list[Track], axis: int = 1) -> tuple[int, str]:
    """Знак стороны стола вдоль оси axis (1: +1 — стол ниже кассира по кадру; 0: +1 — правее).
    Подсказка desk -> ноутбук у кассира -> прочая техника -> посетитель."""
    if axis == 0:
        return _desk_dir_x(scan, seat, desk, visitors)
    if desk is not None:
        return (1 if (desk[1] + desk[3]) / 2 >= seat[3] else -1), "desk"
    h = _h(seat)
    cx, cy = (seat[0] + seat[2]) / 2, (seat[1] + seat[3]) / 2
    n = len(scan.frames)
    for names, cmin in ((("laptop",), 0.3), (DESK_OBJ, 0.25)):
        ys = [(d.box[1] + d.box[3]) / 2 for f in scan.frames for d in f.objects
              if d.name in names and d.conf >= cmin
              and np.hypot((d.box[0] + d.box[2]) / 2 - cx, (d.box[1] + d.box[3]) / 2 - cy) <= 2.0 * h]
        if len(ys) >= 0.2 * n:
            y = float(np.median(ys))
            if abs(y - cy) >= 0.1 * h:
                return int(np.sign(y - cy)), names[0]
    if visitors:
        tr = max(visitors, key=lambda x: x.end - x.start)
        y2 = float(np.median([b[3] for b in tr.boxes]))
        if abs(y2 - seat[3]) > SIDE * h:
            return int(np.sign(y2 - seat[3])), "visitor"
    return 0, "none"


def _desk_dir_x(scan: DenseScan, seat: Box, desk: Box | None, visitors: list[Track]) -> tuple[int, str]:
    """Камера сбоку: стол (и клиент за ним) правее (+1) или левее (-1) кассира."""
    h = _h(seat)
    cx, cy = (seat[0] + seat[2]) / 2, (seat[1] + seat[3]) / 2
    if desk is not None and abs((desk[0] + desk[2]) / 2 - cx) >= 0.1 * h:
        return int(np.sign((desk[0] + desk[2]) / 2 - cx)), "desk_x"
    n = len(scan.frames)
    for names, cmin in ((("laptop",), 0.3), (DESK_OBJ, 0.25)):
        xs = [(d.box[0] + d.box[2]) / 2 for f in scan.frames for d in f.objects
              if d.name in names and d.conf >= cmin
              and np.hypot((d.box[0] + d.box[2]) / 2 - cx, (d.box[1] + d.box[3]) / 2 - cy) <= 2.0 * h]
        if len(xs) >= 0.2 * n:
            x = float(np.median(xs))
            if abs(x - cx) >= 0.1 * h:
                return int(np.sign(x - cx)), names[0] + "_x"
    if visitors:
        tr = max(visitors, key=lambda v: v.end - v.start)
        x2 = float(np.median([(b[0] + b[2]) / 2 for b in tr.boxes]))
        if abs(x2 - cx) > SIDE * h:
            return int(np.sign(x2 - cx)), "visitor_x"
    return 0, "none"


def _client_seat(across: list[Track], seat: Box, dt: float) -> Box:
    """Место клиента: из долгих положений посетителей «через стол» (>= 2 с и >= 30% самого долгого) —
    ближайшее к кассиру. Самое частое не годится: у двери посетитель стоит дважды (вошёл и встал уходить)."""
    pts = [b for tr in across for b in tr.boxes]
    S = [x for x in slots(list(enumerate(pts)), len(pts)) if x["n"] * dt >= 2.0]
    if not S:
        return _mode_box(pts)
    top = max(x["n"] for x in S)
    return min((x["box"] for x in S if x["n"] >= 0.3 * top), key=lambda b: _cdist(b, seat, 1.0))


def _visits(scan: DenseScan, across: list[Track], seat: Box, t_first: float, t_last: float,
            dt: float) -> tuple[list[tuple[float, float]], list[Event]]:
    """Занятость стороны «через стол»; разрез по пустоте >= VISIT_GAP или по уходу сидящего к двери."""
    if not across:
        return [], []
    ts = sorted({t for tr in across for t in tr.t})
    runs = [r for r in _runs(ts, VISIT_GAP) if r[1] - r[0] >= MIN_VISITOR_S]
    cseat = _client_seat(across, seat, dt)
    hc = _h(cseat)
    pts = sorted((t, b) for tr in across for t, b in zip(tr.t, tr.boxes))
    on_seat = sorted({t for t, b in pts if iou(b, cseat) >= SEAT_IOU})
    seat_runs = [r for r in _runs(on_seat, max(1.0, 1.5 * dt)) if r[1] - r[0] >= 2.0]
    visits: list[tuple[float, float]] = []
    for a, b in runs:
        cur = a
        sr = [r for r in seat_runs if a <= r[0] and r[1] <= b]
        for (_, e0), (s1, _) in zip(sr, sr[1:]):
            mid = [(t, bb) for t, bb in pts if e0 < t < s1]
            far = max((_cdist(bb, cseat, hc) for _, bb in mid), default=0.0)
            if s1 - e0 < VISIT_GAP or far < 1.0:
                continue  # наклонился / пересел — тот же клиент
            # смена: самая длинная дыра в наблюдениях посетителей (ушёл за дверь, следующий ещё не вошёл)
            ts_mid = [e0] + sorted({t for t, _ in mid}) + [s1]
            g0, g1 = max(zip(ts_mid, ts_mid[1:]), key=lambda x: x[1] - x[0])
            if g1 - g0 < 0.6:
                g0 = g1 = max(mid, key=lambda x: _cdist(x[1], cseat, hc))[0]  # дыры нет — самая дальняя точка
            visits.append((cur, g0))
            cur = g1
        visits.append((cur, b))
    ev: list[Event] = []
    out: list[tuple[float, float]] = []
    for i, (a, b) in enumerate(visits):
        sit = next((max(s, a) for s, e in seat_runs if e > a and s < b), None)
        up = next((min(e, b) for s, e in reversed(seat_runs) if s < b and e > a), None)
        # приход и уход точнее по двери: активность рядом с человеком до того, как сел, и после того, как встал
        arr, by_in = a, "track"
        tr0 = min((tr for tr in across if tr.start <= a + 1.0 and tr.end >= a), key=lambda x: x.start, default=None)
        if tr0 is not None and sit is not None:
            dd = [d for d in door_activity(scan, tr0, _area(cseat)) if a - 0.5 <= d[0] and d[1] <= sit]
            if dd:
                arr, by_in = dd[0][1], "door"
        gone, by_out = b, "track"
        tr1 = max((tr for tr in across if tr.start <= b and tr.end >= b - 1.0), key=lambda x: x.end, default=None)
        if tr1 is not None and up is not None:
            dd = [d for d in door_activity(scan, tr1, _area(cseat))
                  if up - 0.5 <= d[0] and d[1] <= b + 1.0 and b - d[1] <= LINGER_S]
            if dd:
                gone, by_out = dd[-1][1], "door"
        if i:
            ev.append(Event("client_change", arr, sit, {"gap": round(arr - out[-1][1], 2),
                                                        "prev_exit": out[-1][1], "t_seen": a}))
        if i or by_in == "door" or a > t_first + EDGE_TOL:  # клиент, сидящий с первого кадра, не «пришёл»
            ev.append(Event("client_arrives", arr, sit, {"t_seen": a, "t_seated": sit, "by": by_in}))
        if by_out == "door" or b < t_last - EDGE_TOL:  # визит, идущий до конца ролика, не «закончился»
            ev.append(Event("client_leaves", up if up is not None else b, gone, {"t_exit": gone, "by": by_out}))
        out.append((arr, gone))
    return out, ev


def _third(others, side, across, with_cashier, dt) -> list[Event]:
    ev = []
    for tr in others:
        s = side[tr.tid]
        if s in ("level", "behind") and with_cashier[tr.tid] >= MIN_VISITOR_S:
            ev.append(Event("third_person", tr.start, tr.end, {"side": s, "tid": tr.tid}))
    # второй «через стол», пока идёт визит: два посетителя одновременно >= MIN_VISITOR_S
    for i, a in enumerate(across):
        for b in across[i + 1:]:
            if len(set(a.t) & set(b.t)) * dt >= MIN_VISITOR_S:
                late = b if b.start >= a.start else a
                ev.append(Event("third_person", late.start, min(a.end, b.end), {"side": "across", "tid": late.tid}))
    return ev
