"""Детектор сцены: стол, место кассира, присутствие людей и предметов — источник кандидатов и подсказка ролей.

RF-DETR (Apache-2.0, только размеры N–L) с весами COCO. Без трекинга по ID: роль кассира задаётся
«местом кассира» — областью, где за ролик дольше всего находится человек рядом со столом. Это
пересчитывается для каждого видео, поэтому не зависит от угла камеры и не требует лиц.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

# id классов COCO в RF-DETR
COCO = {1: "person", 73: "laptop", 67: "dining table", 76: "keyboard", 62: "chair", 31: "handbag",
        27: "backpack", 33: "suitcase", 77: "cell phone", 47: "cup", 44: "bottle", 84: "book"}
DESK = {"laptop", "dining table", "keyboard"}
ITEMS = {"handbag", "backpack", "suitcase", "cell phone", "cup", "bottle"}

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 в пикселях исходного кадра


@dataclass
class FrameDets:
    t: float
    persons: list[Box]
    objects: list[tuple[str, Box]]


@dataclass
class Scene:
    width: int
    height: int
    desk_box: Box | None
    cashier_slot: Box | None
    times: np.ndarray
    cashier_present: np.ndarray
    n_persons: np.ndarray
    items_on_desk: dict[str, np.ndarray] = field(default_factory=dict)

    def zone(self) -> Box | None:
        """Зона рук и стола для движения: стол, иначе место кассира; с запасом."""
        base = self.desk_box or self.cashier_slot
        return None if base is None else _expand(base, 0.2, self.width, self.height)


@dataclass(frozen=True)
class SceneEvent:
    kind: str  # cashier_departure | persons_change | item_on_desk:<класс>
    start_sec: float
    main_sec: float
    end_sec: float
    priority: float


def _expand(b: Box, frac: float, w: int, h: int) -> Box:
    dx, dy = (b[2] - b[0]) * frac, (b[3] - b[1]) * frac
    return max(0.0, b[0] - dx), max(0.0, b[1] - dy), min(w, b[2] + dx), min(h, b[3] + dy)


def _iou(a: Box, b: Box) -> float:
    ix, iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _center_in(b: Box, z: Box) -> bool:
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return z[0] <= cx <= z[2] and z[1] <= cy <= z[3]


class Detector:
    def __init__(self, size: str = "nano", threshold: float = 0.4, device: str | None = None):
        import torch
        from rfdetr import RFDETRMedium, RFDETRNano, RFDETRSmall

        if size not in {"nano", "small", "medium"}:
            raise ValueError("только Apache-размеры RF-DETR: nano | small | medium")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium}[size](device=device)
        self.size, self.threshold, self.device = size, threshold, device

    def detect(self, frames, batch: int = 8) -> list[FrameDets]:
        """frames — любой итератор (t, PIL.Image): кадры полного разрешения не держим в памяти все сразу."""
        out: list[FrameDets] = []
        it = iter(frames)
        while True:
            chunk = [x for _, x in zip(range(batch), it)]
            if not chunk:
                break
            dets = self.model.predict([im for _, im in chunk], threshold=self.threshold)
            if not isinstance(dets, list):
                dets = [dets]
            for (t, _), d in zip(chunk, dets):
                persons, objects = [], []
                for box, cid in zip(d.xyxy, d.class_id):
                    name = COCO.get(int(cid))
                    if name == "person":
                        persons.append(tuple(map(float, box)))
                    elif name:
                        objects.append((name, tuple(map(float, box))))
                out.append(FrameDets(t, persons, objects))
        return out


def build_scene(dets: list[FrameDets], width: int, height: int) -> Scene:
    times = np.array([d.t for d in dets])
    n = len(dets)
    # стол: медианная рамка каждого «настольного» класса, встречающегося хотя бы в 20% кадров
    desk_parts = []
    for cls in DESK:
        boxes = [b for d in dets for name, b in d.objects if name == cls]
        frames_with = sum(any(name == cls for name, _ in d.objects) for d in dets)
        if boxes and frames_with >= 0.2 * n:
            desk_parts.append(np.median(np.array(boxes), axis=0))
    desk = None
    if desk_parts:
        a = np.array(desk_parts)
        desk = (float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 2].max()), float(a[:, 3].max()))

    # место кассира: мода центров людей (ближе к столу — весомее), затем медиана рамок вокруг моды
    slot = None
    boxes = [b for d in dets for b in d.persons]
    if boxes:
        arr = np.array(boxes)
        cx, cy = (arr[:, 0] + arr[:, 2]) / 2, (arr[:, 1] + arr[:, 3]) / 2
        gx, gy = np.clip((cx / width * 12).astype(int), 0, 11), np.clip((cy / height * 20).astype(int), 0, 19)
        weight = np.ones(len(arr))
        if desk is not None:
            dz = _expand(desk, 0.5, width, height)
            weight += np.array([2.0 if _center_in(tuple(b), dz) else 0.0 for b in arr])
        hist = np.zeros((20, 12))
        np.add.at(hist, (gy, gx), weight)
        my, mx = np.unravel_index(np.argmax(hist), hist.shape)
        mcx, mcy = (mx + 0.5) * width / 12, (my + 0.5) * height / 20
        near = np.hypot(cx - mcx, cy - mcy) <= 0.12 * height
        slot = tuple(float(v) for v in np.median(arr[near], axis=0))

    present = np.array([slot is not None and any(_iou(b, slot) >= 0.3 for b in d.persons) for d in dets])
    n_persons = np.array([len(d.persons) for d in dets])
    items: dict[str, np.ndarray] = {}
    if desk is not None:
        dz = _expand(desk, 0.15, width, height)
        for cls in ITEMS:
            items[cls] = np.array([any(name == cls and _center_in(b, dz) for name, b in d.objects) for d in dets])
    return Scene(width, height, desk, slot, times, present, n_persons, items)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Отрезки подряд идущих True: [(i_start, i_end_inclusive)]."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def scene_events(scene: Scene, duration: float, min_absence_sec: float = 2.0, min_item_sec: float = 1.0) -> list[SceneEvent]:
    t, ev = scene.times, []
    if len(t) < 2:
        return ev
    clip = lambda a, b: (max(0.0, a), min(duration, b))  # noqa: E731
    # уход с места кассира: отсутствие дольше min_absence_sec после присутствия
    if scene.cashier_slot is not None and scene.cashier_present.any():
        for i0, i1 in _runs(~scene.cashier_present):
            if i0 == 0 or t[i1] - t[i0] < min_absence_sec:
                continue
            m = float((t[i0 - 1] + t[i0]) / 2)
            s, e = clip(m - 2.0, m + 3.0)
            ev.append(SceneEvent("cashier_departure", s, m, e, 2.5))
    # смена числа людей в кадре (устойчивая: два отсчёта подряд)
    nps = scene.n_persons
    for i in range(2, len(nps)):
        if nps[i] == nps[i - 1] != nps[i - 2]:
            m = float(t[i - 1])
            s, e = clip(m - 2.0, m + 2.0)
            ev.append(SceneEvent("persons_change", s, m, e, 1.5))
    # предмет на столе: появление, держится дольше min_item_sec
    for cls, mask in scene.items_on_desk.items():
        for i0, i1 in _runs(mask):
            if t[i1] - t[i0] >= min_item_sec:
                m = float(t[i0])
                s, e = clip(m - 1.0, m + 3.0)
                ev.append(SceneEvent(f"item_on_desk:{cls}", s, m, e, 2.0))
    return ev


def mark_roles(frames: list[tuple[float, Image.Image]], dets: list[FrameDets], slot: Box | None) -> list[tuple[float, Image.Image]]:
    """Копии кадров для VLM с рамками ролей: зелёная — человек на месте кассира, оранжевая — остальные."""
    out = []
    for (t, im), d in zip(frames, dets):
        im = im.convert("RGB").copy()
        dr = ImageDraw.Draw(im)
        for b in d.persons:
            cashier = slot is not None and _iou(b, slot) >= 0.3
            dr.rectangle(b, outline=(0, 200, 0) if cashier else (255, 140, 0), width=3)
            dr.text((b[0] + 3, b[1] + 3), "CASHIER" if cashier else "PERSON", fill=(255, 255, 255))
        out.append((t, im))
    return out
