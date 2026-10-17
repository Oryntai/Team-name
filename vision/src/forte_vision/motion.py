"""Кандидаты по всплескам движения: дешёвый источник, не зависящий ни от детектора, ни от VLM.

Энергия движения — доля пикселей, заметно изменившихся между соседними кадрами (макс. по каналам
RGB, низкое разрешение). Кандидат — окно вокруг локального пика энергии. В кадре кассы почти всегда кто-то
двигается, поэтому берём не «всё, что выше порога», а top-K пиков с минимальным расстоянием —
так число перепроверок VLM ограничено сверху.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .video import PyAVSource


@dataclass(frozen=True)
class MotionPeak:
    start_sec: float
    main_sec: float
    end_sec: float
    score: float


def motion_energy(
    src,
    fps: float = 5.0,
    size: tuple[int, int] = (116, 212),
    pixel_thr: int = 18,
    zone: tuple[float, float, float, float] | None = None,
    frame_size: tuple[int, int] | None = None,
):
    """src — путь к файлу или источник кадров (PyAVSource / адаптер backend).
    (times, energy, zone_energy): energy[i] — доля изменившихся пикселей между кадрами i-1 и i по всему
    кадру, zone_energy — то же внутри зоны стола/рук (None, если зона не задана). zone — в пикселях
    исходного кадра размера frame_size (ширина, высота)."""
    mask = None
    if zone is not None and frame_size is not None:
        sx, sy = size[0] / frame_size[0], size[1] / frame_size[1]
        x1, y1 = int(zone[0] * sx), int(zone[1] * sy)
        x2, y2 = max(x1 + 1, int(np.ceil(zone[2] * sx))), max(y1 + 1, int(np.ceil(zone[3] * sy)))
        mask = np.zeros((size[1], size[0]), dtype=bool)
        mask[y1:y2, x1:x2] = True
    times, energy, zone_energy, prev = [], [], [], None
    if isinstance(src, str):
        src = PyAVSource(src)
    for t, im in src.sample(fps=fps, size=size):
        # по каналам RGB, а не по яркости: цветной объект на фоне той же яркости в сером не виден
        g = np.asarray(im.convert("RGB"), dtype=np.int16)
        if prev is not None:
            changed = np.abs(g - prev).max(axis=2) > pixel_thr
            energy.append(float(changed.mean()))
            if mask is not None:
                zone_energy.append(float(changed[mask].mean()))
            times.append(t)
        prev = g
    return np.asarray(times), np.asarray(energy), (np.asarray(zone_energy) if mask is not None else None)


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) < k or k <= 1:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def find_peaks(
    times: np.ndarray,
    energy: np.ndarray,
    duration: float,
    top_k: int = 12,
    min_separation_sec: float = 3.0,
    half_window_sec: float = 2.0,
    k_mad: float = 2.0,
    floor: float = 0.002,
) -> list[MotionPeak]:
    if len(energy) == 0:
        return []
    e = _smooth(energy, 3)
    med = float(np.median(e))
    mad = float(np.median(np.abs(e - med))) or 1e-6
    thr = max(floor, med + k_mad * mad)
    # локальные максимумы выше порога
    idx = [i for i in range(len(e)) if e[i] >= thr and e[i] == e[max(0, i - 2) : i + 3].max()]
    idx.sort(key=lambda i: e[i], reverse=True)
    chosen: list[int] = []
    for i in idx:
        if all(abs(times[i] - times[j]) >= min_separation_sec for j in chosen):
            chosen.append(i)
        if len(chosen) >= top_k:
            break
    peaks = [
        MotionPeak(
            start_sec=max(0.0, float(times[i]) - half_window_sec),
            main_sec=float(times[i]),
            end_sec=min(duration, float(times[i]) + half_window_sec),
            score=float(e[i]),
        )
        for i in chosen
    ]
    return sorted(peaks, key=lambda p: p.main_sec)
