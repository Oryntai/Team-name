"""Новые предметы на столе по пикселям: что положили и оставили.

Слепой обход «есть ли на столе сумка / еда?» каждые 4 с давал ложные тревоги на чистых роликах (dev: 4 из 7):
модель, которую раз за разом спрашивают одно и то же, рано или поздно отвечает «да». Здесь модель
спрашивают только там, где на столе что-то реально изменилось и осталось лежать: медиана пикселей за
несколько секунд после момента против нескольких секунд до, люди закрашены (рамки детектора с полями),
экран и счётная машинка исключены. Каждое такое пятно — «что-то положили или убрали»; что именно, модель
говорит по плотной вырезке вокруг пятна (см. decide.plan, вопрос desk_obj).

Всё в долях кадра; кадры берутся из DenseScan.hi (цвет, ~половина разрешения, ~2 к/с). Разница — по цвету,
а не по яркости: голубая пачка денег на светлом дереве в сером почти не видна (X-38).
"""

from __future__ import annotations

import warnings

import numpy as np

from .types import Box, DenseScan, DeskBlob, Zones


def _px(box: Box, gw: int, gh: int, pad: float = 0.0) -> tuple[int, int, int, int]:
    x1, y1 = max(0, int((box[0] - pad) * gw)), max(0, int((box[1] - pad) * gh))
    x2, y2 = min(gw, int(np.ceil((box[2] + pad) * gw))), min(gh, int(np.ceil((box[3] + pad) * gh)))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def _occlusion(scan: DenseScan, T: np.ndarray, reg: tuple[int, int, int, int], gw: int, gh: int,
               mx: float = 0.25, my: float = 0.12) -> np.ndarray:
    """(M, rh, rw): пиксель стола закрыт человеком (ближайший кадр детектора, рамка с полями: руки и тень)."""
    x1, y1, x2, y2 = reg
    ft = np.asarray([f.t for f in scan.frames], float)
    out = np.zeros((len(T), y2 - y1, x2 - x1), bool)
    for k, t in enumerate(T):
        j = int(np.clip(np.searchsorted(ft, t), 0, len(ft) - 1))
        if j > 0 and abs(ft[j - 1] - t) < abs(ft[j] - t):
            j -= 1
        for p in scan.frames[j].persons:
            if p.conf < 0.3:
                continue
            b, w, h = p.box, p.box[2] - p.box[0], p.box[3] - p.box[1]
            a = _px((b[0] - mx * w, b[1] - my * h, b[2] + mx * w, b[3] + my * h), gw, gh)
            out[k, max(0, a[1] - y1):max(0, a[3] - y1), max(0, a[0] - x1):max(0, a[2] - x1)] = True
    return out


def _nanmedian(a: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def _nanmax(a: np.ndarray, axis: int = -1) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(a, axis=axis)


def desk_blobs(scan: DenseScan, zones: Zones, *, win: float = 3.0, step: float = 1.0, thr: float = 28.0,
               min_area: float = 1.5e-4, min_stay: float = 3.0, max_blobs: int = 6,
               margin: tuple[float, float] = (0.08, 0.05)) -> list[DeskBlob]:
    """Пятна на столе, которые изменились и держались не меньше min_stay секунд.

    win — окно медианы до/после момента, thr — порог разницы яркости (0..255) после вычитания общей
    смены освещения, min_area — минимальная площадь пятна в долях кадра (пачка денег X-38 — ~4e-4),
    margin — поля рамки человека (доля её ширины, высоты): шире — теряем предметы рядом с кассиром (X-38).
    Возвращает не больше max_blobs самых крупных и долгих пятен, по времени появления.
    """
    from scipy import ndimage

    box = zones.boxes.get("desk")
    if box is None or scan.hi is None or len(scan.hi) < 8:
        return []
    H = np.asarray(scan.hi)
    if H.ndim == 3:  # старый кеш: серые кадры
        H = H[..., None]
    T = np.asarray(scan.hi_t, float)
    _, gh, gw, _ = H.shape
    reg = _px(box, gw, gh, pad=0.02)
    x1, y1, x2, y2 = reg
    V = H[:, y1:y2, x1:x2].astype(np.float32)
    V[_occlusion(scan, T, reg, gw, gh, *margin)] = np.nan
    excl = np.zeros(V.shape[1:3], bool)
    for z in ("screen", "counter"):
        b = zones.boxes.get(z)
        if b:
            a = _px(b, gw, gh, pad=0.01)
            excl[max(0, a[1] - y1):max(0, a[3] - y1), max(0, a[0] - x1):max(0, a[2] - x1)] = True
    min_px = max(12, int(min_area * gh * gw))

    def state(lo: float, hi: float):
        idx = (T >= lo) & (T <= hi)
        if idx.sum() < 2:
            return None, None
        return _nanmedian(V[idx]), (~np.isnan(V[idx, ..., 0])).sum(0)

    cands = []  # (t, (sy, sx), area)
    for tg in np.arange(T[0] + win, T[-1] - win + 1e-6, step):
        before, nb = state(tg - win, tg - 0.5)
        after, na = state(tg + 0.5, tg + win)
        if before is None or after is None:
            continue
        valid = (nb >= 2) & (na >= 2) & ~excl
        if valid.mean() < 0.15:
            continue
        dc = np.abs(after - before)  # (h, w, C)
        dc = dc - _nanmedian(dc[valid], axis=0)  # общая смена освещения по каждому каналу
        d = _nanmax(dc, axis=-1)
        ch = valid & np.nan_to_num(d > thr)
        ch = ndimage.binary_opening(ch, iterations=1)
        lab, n = ndimage.label(ch)
        for k, sl in enumerate(ndimage.find_objects(lab), 1):
            if sl is None:
                continue
            area = int((lab[sl] == k).sum())
            if area >= min_px:
                cands.append((float(tg), sl, area))

    # одно изменение видно на нескольких соседних шагах сетки: склеиваем пересекающиеся пятна
    groups: list[list] = []
    for c in sorted(cands, key=lambda c: c[0]):
        for g in groups:
            last = g[-1]
            if c[0] - last[0] <= 1.5 * step and _iou(c[1], last[1]) >= 0.2:
                g.append(c)
                break
        else:
            groups.append([c])

    out: list[DeskBlob] = []
    for g in groups:
        t_on = float(np.mean([c[0] for c in g]))  # шаг до/после по обе стороны момента — середина серии
        sl = max(g, key=lambda c: c[2])[1]
        sy, sx = sl
        ref, _ = state(t_on - win - 1.0, t_on - 1.0)
        if ref is None:
            continue
        # точный момент появления: середина серии смещена окнами до/после — ищем первый кадр, где пятно уже другое
        for k in np.where((T >= t_on - win) & (T <= t_on + win + 1.0))[0]:
            cur = V[k, sy, sx]
            ok = ~np.isnan(cur[..., 0]) & ~excl[sy, sx]
            if ok.mean() >= 0.5 and np.mean(_nanmax(np.abs(cur - ref[sy, sx]), axis=-1)[ok] > thr) >= 0.5:
                t_on = float(T[k])
                break
        t_off, t = None, t_on + 0.5
        while t <= T[-1]:
            cur, n = state(t, t + 1.0)
            if cur is not None:
                m = (n[sy, sx] >= 1) & ~excl[sy, sx]
                if m.mean() >= 0.3:
                    dd = _nanmax(np.abs(cur[sy, sx] - ref[sy, sx]), axis=-1)[m]
                    if np.mean(dd > thr) < 0.3:  # вернулось к прежнему: предмет убрали
                        t_off = t
                        break
            t += step
        t_off = float(T[-1]) if t_off is None else float(t_off)
        if t_off - t_on < min_stay:
            continue
        bx = ((x1 + sx.start) / gw, (y1 + sy.start) / gh, (x1 + sx.stop) / gw, (y1 + sy.stop) / gh)
        out.append(DeskBlob(box=tuple(round(v, 4) for v in bx), t_on=round(t_on, 2), t_off=round(t_off, 2),
                            area=round(max(c[2] for c in g) / (gh * gw), 5), meta={"steps": len(g)}))
    out = _merge(out)
    out.sort(key=lambda b: -(b.area * min(b.t_off - b.t_on, 20.0)))
    return sorted(out[:max_blobs], key=lambda b: b.t_on)


def _box_iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _merge(blobs: list[DeskBlob]) -> list[DeskBlob]:
    """Один предмет, распавшийся на несколько пятен (X-2: подлокотник 4 раза): пересекаются по месту
    (IoU ≥ 0.2 или центр одного внутри другого) и по времени — склеиваем: общая рамка, раннее появление."""
    def same(a: DeskBlob, b: DeskBlob) -> bool:
        if a.t_off < b.t_on - 1.0 or b.t_off < a.t_on - 1.0:
            return False
        if _box_iou(a.box, b.box) >= 0.2:
            return True
        for x, y in ((a, b), (b, a)):
            cx, cy = (x.box[0] + x.box[2]) / 2, (x.box[1] + x.box[3]) / 2
            if y.box[0] <= cx <= y.box[2] and y.box[1] <= cy <= y.box[3]:
                return True
        return False

    out: list[DeskBlob] = []
    for b in sorted(blobs, key=lambda b: b.t_on):
        m = next((o for o in out if same(o, b)), None)
        if m is None:
            out.append(DeskBlob(b.box, b.t_on, b.t_off, b.area, dict(b.meta)))
            continue
        m.box = (min(m.box[0], b.box[0]), min(m.box[1], b.box[1]), max(m.box[2], b.box[2]), max(m.box[3], b.box[3]))
        m.t_on, m.t_off, m.area = min(m.t_on, b.t_on), max(m.t_off, b.t_off), max(m.area, b.area)
        m.meta["merged"] = m.meta.get("merged", 1) + 1
    return out


def _iou(a, b) -> float:
    (ay, ax), (by, bx) = a, b
    iy = max(0, min(ay.stop, by.stop) - max(ay.start, by.start))
    ix = max(0, min(ax.stop, bx.stop) - max(ax.start, bx.start))
    inter = iy * ix
    ua = (ay.stop - ay.start) * (ax.stop - ax.start) + (by.stop - by.start) * (bx.stop - bx.start) - inter
    return inter / ua if ua > 0 else 0.0
