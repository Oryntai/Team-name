"""Зоны сцены (сейф, экран, стол кассира, дверь, счётная машинка) — grounding VLM на одном кадре + детектор.

Камера неподвижна, поэтому зоны ищутся один раз на ролик: освещённый кадр с наименьшим числом людей,
один запрос grounding Qwen3-VL (+ повтор только для сейфа и поиск замка внутри сейфа), затем проверка
здравым смыслом и сверка с медианами детектора из плотного прохода:
  - рамка > 50% кадра или вырожденная — отбрасывается;
  - на «столе» должна стоять техника (низ рамки экрана/машинки — на столе), иначе это пустой стол
    переднего плана (Qwen его тоже отдаёт как desk, вторым кандидатом);
  - одна рамка под двумя метками (сейф = машинка, экран = машинка): решают ноутбук детектора и стол;
  - COCO «laptop» — экран; COCO «tv» срабатывает и на счётную машинку, и на сейф: на столе — машинка,
    вне стола — сейф.
Всё в долях кадра; источник каждой зоны — в Zones.source ("vlm" | "detector" | "fallback").
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

import numpy as np

from .ask import CropAsker
from .types import ZONES, Box, DenseScan, Zones

# зона -> (метка в промпте, описание). Естественные слова (laptop, money_counter) Qwen находит точнее.
PROMPT = {
    "safe": ("safe", "grey metal safe for cash: a metal cabinet or box with a door and a keyhole, standing next to "
                     "the desk (not a counting machine, not a white drawer unit)"),
    "screen": ("laptop", "the laptop computer or monitor the cashier works on (its screen)"),
    "desk": ("cash_desk", "the cashier's work desk with the laptop and the devices on it (not an empty table)"),
    "door": ("door", "the glass door of this room"),
    "counter": ("money_counter", "banknote counting machine: a black box-shaped device with a banknote tray, "
                                 "on the desk"),
}
LABELS = dict(PROMPT.values())
SAFE_RETRY = {"safe": "grey metal safe: a tall grey metal cabinet with a door and a keyhole, standing on the floor "
                      "or on a stand (not the counting machine, not the laptop, not a white drawer unit)"}
LOCK_LABEL = {"lock": "the keyhole or lock on the safe door (a key may be inserted in it)"}
# на полном кадре Qwen отдаёт дверью всю стеклянную стену (~половина кадра); внутри неё — створку у ручки
DOOR_LEAF = {"door_leaf": "the door leaf: the glass panel of the door that opens (the one with the door handle), "
                          "not the fixed glass wall"}

MAX_AREA, MIN_AREA, MIN_SIDE = 0.5, 4e-4, 0.01
PERSON_CONF = 0.6  # люди за стеклом детектируются с уверенностью 0.3–0.7, в комнате 0.8–0.9
DET_MIN_FRAC = 0.15  # предмет детектора считается частью сцены, если виден в >= 15% кадров
SAFE_DET_FRAC = 0.4  # tv детектора как сейф — только устойчивый (телевизор в коридоре за стеклом мигает)
DARK = 0.8  # кадр темнее 80% от p90 яркости ролика (выключен свет: 0.65) для grounding не берём


# ---------- геометрия ----------

def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a: Box, b: Box) -> float:
    i = inter(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def center_in(b: Box, z: Box, margin: float = 0.0) -> bool:
    dx, dy = (z[2] - z[0]) * margin, (z[3] - z[1]) * margin
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return z[0] - dx <= cx <= z[2] + dx and z[1] - dy <= cy <= z[3] + dy


def stands_on(e: Box, d: Box, margin: float = 0.1) -> bool:
    """Предмет стоит на поверхности d: центр по ширине и НИЖНИЙ край — в пределах d (экран торчит над столом)."""
    dw, dh = d[2] - d[0], d[3] - d[1]
    cx = (e[0] + e[2]) / 2
    return d[0] - margin * dw <= cx <= d[2] + margin * dw and d[1] - margin * dh <= e[3] <= d[3] + margin * dh


def expand(b: Box, f: float) -> Box:
    dx, dy = (b[2] - b[0]) * f, (b[3] - b[1]) * f
    return max(0.0, b[0] - dx), max(0.0, b[1] - dy), min(1.0, b[2] + dx), min(1.0, b[3] + dy)


def union(boxes: list[Box]) -> Box:
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def distinct(b: Box, o: Box | None) -> bool:
    """b — другой предмет, чем o: не та же рамка и не почти целиком внутри o.
    Сейф за машинкой (X-30..33, X-43) законно перекрыт ею на треть, поэтому порог не строже IoU 0.5."""
    return o is None or (iou(b, o) <= 0.5 and inter(b, o) <= 0.7 * area(b))


def sane(b: Box) -> str | None:
    """Причина отказа или None."""
    if b[2] - b[0] < MIN_SIDE or b[3] - b[1] < MIN_SIDE or area(b) < MIN_AREA:
        return "degenerate"
    if area(b) > MAX_AREA:
        return "huge"
    return None


# ---------- детектор ----------

def object_clusters(scan: DenseScan | None, name: str, min_frac: float = DET_MIN_FRAC,
                    min_conf: float = 0.3) -> list[tuple[Box, float]]:
    """Устойчивые предметы класса name: жадные кластеры рамок по IoU >= 0.3 -> (медианная рамка, доля кадров)."""
    if scan is None or not scan.frames:
        return []
    clusters: list[dict] = []
    for k, fr in enumerate(scan.frames):
        for d in fr.objects:
            if d.name != name or d.conf < min_conf:
                continue
            best = max(clusters, key=lambda c: iou(c["ref"], d.box), default=None)
            if best is not None and iou(best["ref"], d.box) >= 0.3:
                best["boxes"].append(d.box)
                best["frames"].add(k)
            else:
                clusters.append({"ref": d.box, "boxes": [d.box], "frames": {k}})
    out = []
    for c in clusters:
        frac = len(c["frames"]) / len(scan.frames)
        if frac >= min_frac:
            out.append((tuple(float(v) for v in np.median(np.array(c["boxes"]), axis=0)), frac))
    return sorted(out, key=lambda x: -x[1])


# ---------- выбор кадра ----------

def _lit(br: np.ndarray) -> np.ndarray:
    """Маска освещённых кадров: выключенный свет даёт ~0.65 яркости, автоэкспозиция камеры — меньше 0.1."""
    return br >= DARK * np.percentile(br, 90)


def pick_time(path: str, scan: DenseScan | None = None,
              people_boxes_at: Callable[[float], list] | None = None) -> tuple[float, dict]:
    """Кадр для grounding: свет включён, меньше всего людей, середина самого длинного такого отрезка."""
    if scan is not None and scan.frames:
        frames = scan.frames
        if scan.gray is not None and len(scan.gray_t):
            g = np.asarray(scan.gray)
            br = g.reshape(len(g), -1).mean(axis=1)
            lit, ts = _lit(br), np.asarray(scan.gray_t)
            frames = [f for f in frames if lit[int(np.argmin(np.abs(ts - f.t)))]] or frames
        score = [(sum(d.conf >= PERSON_CONF for d in f.persons),
                  round(sum(area(d.box) for d in f.persons if d.conf >= PERSON_CONF), 2)) for f in frames]
        best = min(score)
        # самый длинный подряд идущий отрезок лучших кадров — сцена спокойна, без смаза
        runs, cur = [], []
        for f, s in zip(frames, score):
            if s == best:
                cur.append(f.t)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        run = max(runs, key=len)
        return run[len(run) // 2], {"how": "scan", "persons": best[0], "run": [run[0], run[-1]]}

    # без плотного прохода: 1 к/с серые кадры, кадр ближе всего к медианному фону = меньше всего чужого
    from forte_vision.video import sample

    ts, grays = [], []
    for t, im in sample(str(path), 1.0):
        w = 64
        grays.append(np.asarray(im.convert("L").resize((w, max(1, round(w * im.height / im.width)))), dtype=np.float32))
        ts.append(t)
    if not ts:
        raise ValueError(f"в файле нет видеокадров: {path}")
    g = np.stack(grays)
    lit = _lit(g.reshape(len(g), -1).mean(axis=1))
    diff = np.abs(g - np.median(g[lit] if lit.any() else g, axis=0)).reshape(len(g), -1).mean(axis=1)
    idx = [i for i in range(len(ts)) if lit[i]] or list(range(len(ts)))
    if people_boxes_at is not None:
        idx = idx[::max(1, len(idx) // 12)]
        cnt = {i: len(people_boxes_at(ts[i])) for i in idx}
        i = min(idx, key=lambda i: (cnt[i], diff[i]))
        return float(ts[i]), {"how": "people_boxes_at", "persons": cnt[i]}
    i = min(idx, key=lambda i: diff[i])
    return float(ts[i]), {"how": "median_bg", "diff": round(float(diff[i]), 2)}


# ---------- сведение VLM и детектора ----------

def resolve_zones(vlm: dict[str, list[Box]], det: dict[str, list[tuple[Box, float]]]) -> Zones:
    """Чистая функция: кандидаты grounding по зонам + кластеры детектора -> Zones с источником и отказами."""
    boxes: dict[str, Box | None] = {z: None for z in ZONES}
    source: dict[str, str] = {}
    rejected: dict[str, list[str]] = {}
    cand: dict[str, list[Box]] = {z: [] for z in ZONES}
    for z in ZONES:
        for b in vlm.get(z, []):
            why = sane(b)
            if why:
                rejected.setdefault(z, []).append(why)
            else:
                cand[z].append(tuple(b))
    lap = det.get("laptop", [])
    laptops = [b for b, _ in lap]
    tvs = det.get("tv", [])
    tables = [b for b, _ in det.get("dining table", [])]

    def put(z: str, b: Box | None, src: str) -> None:
        if b is not None:
            boxes[z], source[z] = tuple(round(float(v), 4) for v in b), src

    # 1. стол: на нём стоит техника; пустой стол переднего плана отбрасывается
    equip = cand["screen"] + cand["counter"] + laptops + [b for b, _ in tvs]
    for b in cand["desk"]:
        if not equip or any(stands_on(e, b) or center_in(e, b) for e in equip):
            put("desk", b, "vlm")
            break
        rejected.setdefault("desk", []).append("no_equipment_on_it")
    if boxes["desk"] is None:
        t = next((b for b in tables if area(b) <= MAX_AREA and any(stands_on(e, b) for e in equip)), None)
        core = (cand["screen"] + cand["counter"])[:2] or laptops[:1]
        if t is not None:
            put("desk", t, "detector")
        elif core and area(expand(union(core), 0.5)) <= 0.25:  # грубый стол: техника с запасом
            put("desk", expand(union(core), 0.5), "fallback")
    desk = boxes["desk"]

    def on_desk(b: Box) -> bool:
        return desk is not None and (stands_on(b, desk) or center_in(b, desk))

    def is_laptop(b: Box) -> bool:
        return any(inter(b, l) >= 0.3 * min(area(b), area(l)) for l in laptops)

    # 2. одна рамка под двумя метками: ноутбук детектора -> экран, на столе -> машинка, иначе сейф
    for a, b in combinations(("safe", "screen", "counter"), 2):
        if cand[a] and cand[b] and iou(cand[a][0], cand[b][0]) > 0.5:
            box = cand[a][0]
            kind = "screen" if is_laptop(box) else "counter" if on_desk(box) else "safe"
            for z in (a, b):
                if z != kind:
                    cand[z].pop(0)
                    rejected.setdefault(z, []).append(f"same_as_{kind}")
            if kind not in (a, b):
                cand[kind].insert(0, box)

    # 3. экран: VLM, сверенный с ноутбуком; ноутбук, видимый полролика, важнее несогласного VLM.
    # Экран не у стола (монитор в коридоре за стеклом, X-31) без подтверждения детектора — не наш.
    if desk is not None:
        near = [b for b in cand["screen"] if is_laptop(b) or stands_on(b, desk, 0.2) or center_in(b, desk, 0.2)]
        if len(near) < len(cand["screen"]):
            rejected.setdefault("screen", []).append("off_desk")
        cand["screen"] = near
    scr = next((b for b in cand["screen"] if is_laptop(b)), None)
    if scr is not None:
        put("screen", scr, "vlm")
    elif lap and (not cand["screen"] or lap[0][1] >= 0.5):
        if cand["screen"]:
            rejected.setdefault("screen", []).append("disagrees_with_laptop")
        put("screen", lap[0][0], "detector")
    elif cand["screen"]:
        put("screen", cand["screen"][0], "vlm")
    scr = boxes["screen"]

    # 4. машинка: VLM на столе, иначе любая VLM, иначе tv детектора на столе (не экран)
    if cand["counter"]:
        put("counter", next((b for b in cand["counter"] if on_desk(b)), cand["counter"][0]), "vlm")
    else:
        tv = next((b for b, _ in tvs if on_desk(b) and not (scr and iou(b, scr) > 0.5)), None)
        put("counter", tv, "detector")
    cnt = boxes["counter"]

    # 5. сейф: VLM (не машинка и не экран), иначе устойчивый крупный tv детектора вне стола
    safes = [b for b in cand["safe"] if distinct(b, cnt) and distinct(b, scr)]
    if safes:
        put("safe", safes[0], "vlm")
    else:
        rest = [b for b, f in tvs if f >= SAFE_DET_FRAC and not on_desk(b) and not (cnt and iou(b, cnt) > 0.5)]
        put("safe", max(rest, key=area, default=None), "detector")

    if cand["door"]:
        put("door", cand["door"][0], "vlm")
    return Zones(boxes=boxes, source=source, diag={"rejected": rejected})


def find_zones(path: str, scan: DenseScan | None, asker: CropAsker,
               people_boxes_at: Callable[[float], list] | None = None, find_lock: bool = True,
               refine_door: bool = True) -> Zones:
    """Зоны ролика. people_boxes_at(t) -> рамки людей в момент t (если нет плотного прохода).

    Запросы VLM: 1 grounding полного кадра; +1, если сейф не найден (повтор с уточнённым описанием);
    +1 на замок сейфа внутри увеличенной вырезки сейфа (find_lock) -> diag["safe_lock"], для вопроса
    «ключ в замке» (вырезка с zoom 6); +1 на створку двери внутри найденной двери (refine_door):
    boxes["door"] = створка у ручки, вся стеклянная стена — в diag["door_wall"].
    """
    t, pick = pick_time(path, scan, people_boxes_at)
    found = asker.ground_candidates(path, t, LABELS)
    raw, err = asker.last_ground.get("raw", ""), asker.last_ground.get("error")
    vlm = {z: found.get(lab, []) for z, (lab, _) in PROMPT.items()}
    det = {n: object_clusters(scan, n) for n in ("laptop", "tv", "dining table")}
    zones = resolve_zones(vlm, det)
    zones.diag.update({"t": t, "pick": pick, "vlm": vlm, "raw": raw, "error": err,
                       "det": {n: [(list(b), round(f, 2)) for b, f in c] for n, c in det.items()}})

    if zones.source.get("safe") != "vlm":
        desk = zones.boxes["desk"]
        retry = asker.ground_candidates(path, t, SAFE_RETRY)["safe"]
        ok = [b for b in retry if sane(b) is None and distinct(b, zones.boxes["screen"])
              and distinct(b, zones.boxes["counter"]) and not (desk and iou(b, desk) > 0.5)]
        zones.diag["safe_retry"] = retry
        if ok:
            zones.boxes["safe"], zones.source["safe"] = tuple(round(v, 4) for v in ok[0]), "vlm"

    wall = zones.boxes["door"]
    if refine_door and wall is not None:
        leaf = asker.ground(path, t, DOOR_LEAF, region=wall)["door_leaf"]
        zones.diag["door_wall"] = wall
        if leaf is not None and sane(leaf) is None and center_in(leaf, wall) and area(leaf) <= 0.8 * area(wall):
            zones.boxes["door"] = tuple(round(v, 4) for v in leaf)
        else:
            zones.diag["door_leaf_rejected"] = leaf

    if find_lock and zones.boxes["safe"] is not None:
        safe = zones.boxes["safe"]
        lock = asker.ground(path, t, LOCK_LABEL, region=expand(safe, 0.15))["lock"]
        if lock is not None and center_in(lock, safe) and area(lock) <= 0.25 * area(safe):
            zones.diag["safe_lock"] = tuple(round(v, 4) for v in lock)
        else:
            zones.diag["safe_lock_rejected"] = lock
    return zones
