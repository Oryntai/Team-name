"""Чтение видео с реальными PTS (не frame_index/fps).

Временный адаптер vision: когда backend выложит общий video_io, функции здесь заменяются им
один в один — наружу торчат только probe / sample / grab_at / frame_id.
Полное декодирование ролика 142 с (4273 кадра 464x848) на M1 Pro занимает ~1.3 с, поэтому каждая
стадия просто проходит файл заново, без кеша кадров в памяти.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import av
from PIL import Image

EVIDENCE_SOURCE = "decoded"


@dataclass(frozen=True)
class VideoInfo:
    duration_sec: float
    fps: float
    width: int
    height: int
    frames: int


def frame_id(pts_sec: float) -> str:
    """Стабильный id кадра в пределах видео: один и тот же PTS даёт один и тот же id."""
    return f"pts_{round(pts_sec * 1000):09d}"


def _open(path: str):
    c = av.open(path)
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    return c, s


def probe(path: str) -> VideoInfo:
    """Длительность считаем по последнему кадру: метаданные контейнера бывают неточны."""
    c, s = _open(path)
    with c:
        last, n = 0.0, 0
        for f in c.decode(s):
            if f.pts is None:
                continue
            last = max(last, float(f.pts * s.time_base))
            n += 1
        rate = float(s.average_rate) if s.average_rate else (n / last if last > 0 else 30.0)
        w, h = s.codec_context.width, s.codec_context.height
    if n == 0:
        raise ValueError(f"в файле нет видеокадров: {path}")
    return VideoInfo(duration_sec=last + 1.0 / rate, fps=rate, width=w, height=h, frames=n)


def sample(
    path: str,
    fps: float,
    size: tuple[int, int] | None = None,
    start: float = 0.0,
    end: float | None = None,
) -> Iterator[tuple[float, Image.Image]]:
    """Кадры с шагом 1/fps по реальному времени в [start, end); size — (ширина, высота)."""
    step = 1.0 / fps
    nxt = start
    c, s = _open(path)
    with c:
        for f in c.decode(s):
            if f.pts is None:
                continue
            t = float(f.pts * s.time_base)
            if end is not None and t >= end:
                break
            if t + 1e-6 < nxt:
                continue
            while nxt <= t + 1e-6:
                nxt += step
            im = f.to_image()
            yield t, (im.resize(size, Image.BICUBIC) if size else im)


def grab_at(path: str, times: Sequence[float], size: tuple[int, int] | None = None) -> list[tuple[float, Image.Image]]:
    """Первые кадры с PTS >= каждого запрошенного времени; возвращает фактические PTS."""
    targets = sorted(set(times))
    out: list[tuple[float, Image.Image]] = []
    if not targets:
        return out
    c, s = _open(path)
    with c:
        last = None
        for f in c.decode(s):
            if f.pts is None:
                continue
            t = float(f.pts * s.time_base)
            last = (t, f)
            while targets and t + 1e-6 >= targets[0]:
                targets.pop(0)
                im = f.to_image()
                out.append((t, im.resize(size, Image.BICUBIC) if size else im))
            if not targets:
                break
        # Запрошенное время позже последнего кадра — отдаём последний кадр
        if targets and last is not None:
            im = last[1].to_image()
            out.append((last[0], im.resize(size, Image.BICUBIC) if size else im))
    # один и тот же кадр мог закрыть несколько запросов
    uniq: dict[float, Image.Image] = {}
    for t, im in out:
        uniq.setdefault(t, im)
    return sorted(uniq.items())
