"""Плотный проход: RF-DETR по кадрам ~5 к/с с уверенностью + маленькие серые кадры для датчика изменений.

Все рамки — в долях кадра. Кадры читаются PyAV по реальным PTS с той же сеткой, что video.sample (первый
кадр с PTS >= очередной точки сетки), но шаг сетки может расти на ходу: если по замеру скорости проход не
успевает к дедлайну, частота снижается, но не ниже 1 к/с — ролик всегда покрыт целиком.
Детектор — любой вызываемый объект: list[PIL.Image] -> list[list[Det]] (рамки в долях); в тестах это заглушка.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import av
import numpy as np

from .types import Det, DenseScan, Frame


def upright_k(frame) -> int:
    """Сколько раз повернуть кадр на 90° против часовой, чтобы он стоял так, как его покажет проигрыватель:
    телефон пишет видео боком и ставит тег поворота. Кадры-доказательства (VideoIO, ffmpeg) поворачиваются так
    же, поэтому вырезки и таймкоды анализатора совпадают с ними."""
    return int(round(float(getattr(frame, "rotation", 0) or 0) / 90.0)) % 4


def upright_image(frame, k: int):
    from PIL import Image

    im = frame.to_image()
    return im if not k else im.transpose({1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180,
                                          3: Image.Transpose.ROTATE_270}[k])


def _upright_array(frame, width: int, height: int, fmt: str, k: int) -> np.ndarray:
    """Кадр в формате fmt размером width×height уже ПОСЛЕ поворота (для k нечётного ширина и высота хранения
    меняются местами)."""
    if k % 2:
        arr = frame.reformat(width=height, height=width, format=fmt).to_ndarray()
    else:
        arr = frame.reformat(width=width, height=height, format=fmt).to_ndarray()
    return np.ascontiguousarray(np.rot90(arr, k)) if k else arr.copy()

# классы COCO, которые держим: люди + предметы рабочего места (tv — это счётная машинка/принтер/сейф)
KEEP = {"person", "laptop", "tv", "chair", "dining table", "keyboard", "handbag", "backpack", "suitcase",
        "cup", "bottle", "cell phone", "book", "mouse", "remote"}
MIN_FPS = 1.0
# априорная цена до первого замера: декодирование на секунду ролика и детектор на кадр (M1 Pro, nano, MPS)
PRIOR_DECODE_PER_SEC, PRIOR_DET_PER_FRAME = 0.02, 0.03


class RFDetector:
    """RF-DETR (Apache-2.0, nano/small/medium) с весами COCO; отдаёт уверенность и только нужные классы."""

    def __init__(self, size: str = "nano", threshold: float = 0.25, device: str | None = None, batch: int = 16):
        import torch
        from rfdetr import RFDETRMedium, RFDETRNano, RFDETRSmall
        from rfdetr.assets.coco_classes import COCO_CLASSES

        if size not in {"nano", "small", "medium"}:
            raise ValueError("только Apache-размеры RF-DETR: nano | small | medium")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium}[size](device=device)
        self.size, self.threshold, self.device, self.batch = size, threshold, device, batch
        self._names = COCO_CLASSES

    def __call__(self, images) -> list[list[Det]]:
        if not images:
            return []
        res = self.model.predict(list(images), threshold=self.threshold, include_source_image=False)
        res = res if isinstance(res, list) else [res]
        out = []
        for im, d in zip(images, res):
            w, h = im.size
            names = d.data.get("class_name") if d.data else None
            dets = []
            for k, (b, c, cid) in enumerate(zip(d.xyxy, d.confidence, d.class_id)):
                name = str(names[k]) if names is not None else self._names.get(int(cid), "")
                if name in KEEP:
                    box = (max(0.0, b[0] / w), max(0.0, b[1] / h), min(1.0, b[2] / w), min(1.0, b[3] / h))
                    dets.append(Det(tuple(round(float(v), 5) for v in box), round(float(c), 4), name))
            out.append(dets)
        return out


def make_detector(size: str = "nano", threshold: float = 0.25, device: str | None = None,
                  batch: int = 16) -> RFDetector:
    """Создать детектор и прогреть его одним пакетом (первый вызов на MPS заметно медленнее)."""
    from PIL import Image

    det = RFDetector(size, threshold, device, batch)
    det([Image.new("RGB", (464, 848))] * min(batch, 4))
    return det


def _split(t: float, dets: list[Det]) -> Frame:
    return Frame(t, [d for d in dets if d.name == "person"], [d for d in dets if d.name != "person"])


def dense_scan(path: str, fps: float = 5.0, *, detector, deadline: float | None = None, gray_width: int = 96,
               min_fps: float = MIN_FPS, hi_width: int = 232, hi_fps: float = 2.0) -> DenseScan:
    """Один проход по файлу: детекции на сетке ~fps и серый кадр ширины gray_width на каждом взятом кадре,
    плюс ЦВЕТНОЙ кадр ширины hi_width с частотой hi_fps для предметов на столе: голубая пачка денег на светлом
    дереве в сером почти не видна (разница яркости 8–10 из 255, X-38), а по цвету — 50–70.

    deadline — момент time.monotonic(), к которому проход должен закончиться; частота снижается по замеру.
    """
    batch_n = int(getattr(detector, "batch", 16))
    frames: list[Frame] = []
    grays: list[np.ndarray] = []
    his: list[np.ndarray] = []
    hi_t: list[float] = []
    nxt_hi = 0.0
    pend: list[tuple[float, object]] = []
    cur_fps = fps
    t_start = time.monotonic()
    det_sec = 0.0
    last_pts, nxt = 0.0, 0.0

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        W, H = s.codec_context.width, s.codec_context.height
        gw = max(8, int(gray_width))
        gh = max(8, round(gw * H / W))
        rot = None  # поворот по тегу видео; размеры ниже — для стоящего кадра
        rate = float(s.average_rate) if s.average_rate else 30.0
        meta_dur = float(s.duration * s.time_base) if s.duration else (c.duration / 1e6 if c.duration else None)

        def flush():
            nonlocal det_sec, cur_fps
            if not pend:
                return
            t0 = time.monotonic()
            dets = detector([im for _, im in pend])
            det_sec += time.monotonic() - t0
            frames.extend(_split(t, d) for (t, _), d in zip(pend, dets))
            covered = pend[-1][0]
            pend.clear()
            if deadline is None or not meta_dur:
                return
            # прогноз: декодирование ∝ секундам ролика, детектор ∝ числу кадров
            el = time.monotonic() - t_start
            a = max(0.0, el - det_sec) / max(covered, 1e-3)
            b = det_sec / max(len(frames), 1)
            rem = max(0.0, meta_dur - covered)
            left = deadline - time.monotonic()
            if rem <= 0:
                return
            want = (left / rem - a) / b if b > 0 else fps
            cur_fps = float(min(fps, max(min_fps, want)))

        # до первого замера — априорная цена
        if deadline is not None and meta_dur:
            left = deadline - time.monotonic()
            want = (left / meta_dur - PRIOR_DECODE_PER_SEC) / PRIOR_DET_PER_FRAME
            cur_fps = float(min(fps, max(min_fps, want)))

        for f in c.decode(s):
            if f.pts is None:
                continue
            t = float(f.pts * s.time_base)
            last_pts = max(last_pts, t)
            if rot is None:
                rot = upright_k(f)
                if rot % 2:
                    W, H = H, W
                    gh = max(8, round(gw * H / W))
            if t + 1e-6 < nxt:
                continue
            while nxt <= t + 1e-6:
                nxt += 1.0 / cur_fps
            grays.append(_upright_array(f, gw, gh, "gray", rot))
            if hi_width and t + 1e-6 >= nxt_hi:
                hh = max(8, round(hi_width * H / W))
                his.append(_upright_array(f, hi_width, hh, "rgb24", rot))
                hi_t.append(t)
                nxt_hi = t + 1.0 / max(hi_fps, 0.1) - 1e-3
            pend.append((t, upright_image(f, rot)))
            if len(pend) >= batch_n:
                flush()
        flush()

    if not frames:
        raise ValueError(f"в файле нет видеокадров: {path}")
    duration = last_pts + 1.0 / rate
    eff = len(frames) / duration if duration > 0 else fps
    return DenseScan(duration=duration, width=W, height=H, fps=round(fps if eff >= 0.9 * fps else eff, 3),
                     frames=frames, gray_t=[fr.t for fr in frames], gray=np.stack(grays).astype(np.uint8),
                     hi_t=hi_t, hi=np.stack(his).astype(np.uint8) if his else None)


# --- кеш: <stem>.json (детекции) + <stem>.npy (серые кадры) ---

def _dets(ds: list[Det]) -> list:
    return [[d.name, *d.box, d.conf] for d in ds]


def save_scan(scan: DenseScan, stem: str | Path) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    doc = {"duration": scan.duration, "width": scan.width, "height": scan.height, "fps": scan.fps,
           "gray_t": list(scan.gray_t), "hi_t": list(scan.hi_t),
           "frames": [{"t": f.t, "d": _dets(f.persons) + _dets(f.objects)} for f in scan.frames]}
    stem.with_suffix(".json").write_text(json.dumps(doc, separators=(",", ":")))
    if scan.gray is not None:
        np.save(stem.with_suffix(".npy"), np.asarray(scan.gray, dtype=np.uint8))
    if scan.hi is not None:
        np.save(stem.with_suffix(".hi.npy"), np.asarray(scan.hi, dtype=np.uint8))


def load_scan(stem: str | Path) -> DenseScan:
    stem = Path(stem)
    doc = json.loads(stem.with_suffix(".json").read_text())
    frames = [_split(f["t"], [Det(tuple(x[1:5]), x[5], x[0]) for x in f["d"]]) for f in doc["frames"]]
    npy = stem.with_suffix(".npy")
    gray = np.load(npy) if npy.exists() else None
    hi_npy = stem.with_suffix(".hi.npy")
    hi = np.load(hi_npy) if hi_npy.exists() else None
    return DenseScan(doc["duration"], doc["width"], doc["height"], doc["fps"], frames, doc.get("gray_t", []), gray,
                     doc.get("hi_t", []) if hi is not None else [], hi)
