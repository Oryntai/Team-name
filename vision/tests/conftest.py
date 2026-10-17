"""Синтетическое видео вместо роликов кейса: тестам не нужны и не разрешены реальные данные в git."""

import av
import numpy as np
import pytest

W, H, FPS, DURATION = 464, 848, 30, 20.0
EVENT = (8.0, 10.0)  # секунды, когда в кадре двигается объект


def make_video(path: str, duration: float = DURATION, event=EVENT) -> str:
    with av.open(path, "w", format="mov") as c:
        s = c.add_stream("mpeg4", rate=FPS)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.bit_rate = 2_000_000
        base = np.full((H, W, 3), 90, dtype=np.uint8)
        base[600:700, 50:400] = 160  # «стол»
        for i in range(int(duration * FPS)):
            t = i / FPS
            img = base.copy()
            if event[0] <= t < event[1]:
                x = int(40 + (t - event[0]) / (event[1] - event[0]) * 300)
                img[300:420, x : x + 80] = (220, 40, 40)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    return path


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> str:
    return make_video(str(tmp_path_factory.mktemp("vid") / "synthetic.mov"))
