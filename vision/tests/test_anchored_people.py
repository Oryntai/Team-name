"""Плотный проход и роли людей на синтетике: без GPU, без весов, без VLM и без роликов кейса."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# запуск с PYTHONPATH=. из корня: forte_vision берём из этого же дерева, а не из установленного пакета
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import forte_vision  # noqa: E402

if not (Path(forte_vision.__path__[0]) / "anchored").is_dir():
    forte_vision.__path__.insert(0, str(_SRC / "forte_vision"))

from forte_vision.anchored import people as P  # noqa: E402
from forte_vision.anchored.detect import RFDetector, dense_scan, load_scan, save_scan  # noqa: E402
from forte_vision.anchored.types import Det, DenseScan, Frame  # noqa: E402

FPS = 5.0
GW, GH = 96, 175
SEAT = (0.30, 0.35, 0.50, 0.50)  # кассир сидит, h = 0.15
STAND = (0.36, 0.22, 0.54, 0.48)  # встала рядом с местом (IoU с местом ~0.3)
DOOR = (0.02, 0.25, 0.16, 0.50)  # стоит у двери
CORRIDOR = (0.03, 0.12, 0.12, 0.30)  # за стеклом
LAPTOP = (0.42, 0.47, 0.54, 0.53)  # стол ниже кассира по кадру -> dir = +1
CLIENT = (0.30, 0.56, 0.50, 0.71)  # напротив, через стол
CL_DOOR = (0.05, 0.50, 0.20, 0.75)


def lerp(a, b, u):
    u = min(1.0, max(0.0, u))
    return tuple(x + (y - x) * u for x, y in zip(a, b))


def path(*keys):
    """keys: (t, box) ... ; между ключами — линейно; вне [t0, tN] человека нет."""
    def f(t):
        if t < keys[0][0] or t > keys[-1][0]:
            return None
        for (t0, b0), (t1, b1) in zip(keys, keys[1:]):
            if t0 <= t <= t1:
                return lerp(b0, b1, (t - t0) / max(t1 - t0, 1e-9))
        return keys[-1][1]
    return f


def make_scan(duration, actors, objects=((LAPTOP, "laptop"),), gray=None, fps=FPS):
    """actors: [(функция t -> рамка | None, уверенность)]; gray(t, k) -> кадр (GH, GW) или None."""
    frames, grays, ts = [], [], []
    for k in range(int(duration * fps)):
        t = round(k / fps, 3)
        persons = [Det(b, c, "person") for f, c in actors if (b := f(t)) is not None]
        frames.append(Frame(t, persons, [Det(b, 0.9, n) for b, n in objects]))
        ts.append(t)
        grays.append(np.full((GH, GW), 100, np.uint8) if gray is None else gray(t, k))
    return DenseScan(duration, 464, 848, fps, frames, ts, np.stack(grays))


def kinds(pp):
    return [e.kind for e in pp.events]


def one(pp, kind):
    ev = [e for e in pp.events if e.kind == kind]
    assert len(ev) == 1, (kind, kinds(pp))
    return ev[0]


def door_gray(t0, t1, region=(0.20, 0.10, 0.30, 0.45)):
    """Створка двери «мигает» в [t0, t1]: соседние кадры отличаются в области двери (вне рамок людей)."""
    x0, y0, x1, y1 = region

    def g(t, k):
        im = np.full((GH, GW), 100, np.uint8)
        if t0 <= t <= t1:
            im[int(y0 * GH):int(y1 * GH), int(x0 * GW):int(x1 * GW)] = 200 if k % 2 else 40
        return im
    return g


# --- R0, геометрия ---

def test_dedupe_keeps_bigger_box_and_max_conf():
    full, upper = (0.3, 0.3, 0.5, 0.6), (0.31, 0.3, 0.49, 0.42)
    out = P.dedupe([(upper, 0.95), (full, 0.85), ((0.7, 0.3, 0.8, 0.5), 0.9)])
    assert len(out) == 2
    assert (full, 0.95) in out


# --- кассир ---

def test_cashier_seated_whole_video_glass_walker_ignored():
    walker = path((2, CORRIDOR), (8, (0.40, 0.12, 0.49, 0.30)))
    pp = P.analyze_people(make_scan(20, [(lambda t: SEAT, 0.9), (walker, 0.45)]))
    assert pp.cashier is not None and pp.cashier.start == 0.0
    assert pp.cashier_present == [(0.0, pytest.approx(19.8))]
    assert pp.visitors == [] and pp.visits == []
    assert not any(k.startswith(("cashier_leaves", "cashier_enters", "third")) for k in kinds(pp))
    assert pp.desk_dir == 1


def test_cashier_leaves_without_door_view():
    cashier = path((0, SEAT), (10, SEAT), (11, STAND), (13, DOOR))
    pp = P.analyze_people(make_scan(20, [(cashier, 0.9)]))
    ev = one(pp, "cashier_leaves")
    assert 9.5 <= ev.t <= 12.0
    assert ev.meta["t_exit"] == pytest.approx(13.0)
    assert pp.cashier_present[-1][1] == pytest.approx(13.0)


def test_cashier_leaves_through_door_but_stays_visible_behind_glass():
    # за стеклом её видно до конца ролика так же уверенно; выход ловится по створке двери
    cashier = path((0, SEAT), (10, SEAT), (12, DOOR), (13, CORRIDOR), (15.8, CORRIDOR))
    pp = P.analyze_people(make_scan(16, [(cashier, 0.9)], gray=door_gray(12.4, 13.2)))
    ev = one(pp, "cashier_leaves")
    assert ev.meta["by"] == "door"
    assert ev.meta["t_exit"] == pytest.approx(13.2, abs=0.25)
    assert 9.5 <= ev.t <= 12.5
    assert pp.cashier_present[-1][1] == pytest.approx(13.2, abs=0.25)


def test_cashier_enters_late_and_sits():
    cashier = path((3, DOOR), (5, SEAT), (20, SEAT))
    pp = P.analyze_people(make_scan(20, [(cashier, 0.9)]))
    ev = one(pp, "cashier_enters")
    assert ev.t == pytest.approx(3.0)
    sits = [e.t for e in pp.events if e.kind == "cashier_sits"]
    assert sits and 4.0 <= sits[0] <= 5.5
    assert pp.cashier_present[0][0] == pytest.approx(3.0)


def test_cashier_enters_from_behind_glass_door_timed_by_door():
    # видна за стеклянной дверью с 0.4 с (как в комнате), вход — по створке двери
    cashier = path((0.4, CORRIDOR), (2.4, DOOR), (3.6, DOOR), (5, SEAT), (16, SEAT))
    pp = P.analyze_people(make_scan(16, [(cashier, 0.9)], gray=door_gray(2.6, 3.4)))
    ev = one(pp, "cashier_enters")
    assert ev.meta["by"] == "door" and ev.t == pytest.approx(3.4, abs=0.25)
    assert ev.meta["t_seen"] == pytest.approx(0.4)


def test_standing_up_is_not_leaving():
    cashier = path((0, SEAT), (8, SEAT), (8.4, STAND), (12, STAND), (12.4, SEAT), (20, SEAT))
    pp = P.analyze_people(make_scan(20, [(cashier, 0.9)]))
    assert "cashier_leaves" not in kinds(pp) and "cashier_enters" not in kinds(pp)
    st = one(pp, "cashier_stands")
    assert 7.8 <= st.t <= 8.6
    assert pp.cashier_present == [(0.0, pytest.approx(19.8))]


# --- посетители ---

def test_client_visits_and_change():
    c1 = path((2, CL_DOOR), (4, CLIENT), (20, CLIENT), (22, CL_DOOR))
    c2 = path((28, CL_DOOR), (30, CLIENT), (40, CLIENT))
    pp = P.analyze_people(make_scan(40, [(lambda t: SEAT, 0.9), (c1, 0.9), (c2, 0.9)]))
    assert len(pp.visits) == 2
    (a1, b1), (a2, b2) = pp.visits
    assert a1 == pytest.approx(2, abs=0.5) and b1 == pytest.approx(22, abs=0.5)
    assert a2 == pytest.approx(28, abs=0.5)
    ch = one(pp, "client_change")
    assert ch.t == pytest.approx(28, abs=0.5) and ch.meta["gap"] == pytest.approx(6, abs=1.0)
    lv = one(pp, "client_leaves")  # второй сидит до конца — его уход не выдумываем
    assert 19.5 <= lv.t <= 21 and lv.t_end == pytest.approx(22, abs=0.5)
    assert "third_person" not in kinds(pp)


def test_client_change_when_tracker_glues_two_clients_at_door():
    # первый выходит в дверь, второй входит через секунду: трекер склеивает их в один трек
    one_track = path((2, CL_DOOR), (4, CLIENT), (20, CLIENT), (23, CL_DOOR), (24.5, CL_DOOR), (27, CLIENT), (40, CLIENT))
    pp = P.analyze_people(make_scan(40, [(lambda t: SEAT, 0.9), (one_track, 0.9)]))
    assert len(pp.visits) == 2
    assert 21 <= pp.visits[0][1] <= 25.5
    assert "client_change" in kinds(pp)


def test_third_person_next_to_cashier():
    other = path((5, DOOR), (7, (0.12, 0.30, 0.26, 0.52)), (14, (0.12, 0.30, 0.26, 0.52)))
    pp = P.analyze_people(make_scan(16, [(lambda t: SEAT, 0.9), (other, 0.9)]))
    ev = one(pp, "third_person")
    assert ev.meta["side"] == "level" and ev.t == pytest.approx(5.0)
    assert pp.visits == []


def test_second_visitor_during_visit_is_third_person():
    c1 = path((1, CL_DOOR), (3, CLIENT), (20, CLIENT))
    c2 = path((10, CL_DOOR), (12, (0.55, 0.56, 0.70, 0.74)), (20, (0.55, 0.56, 0.70, 0.74)))
    pp = P.analyze_people(make_scan(20, [(lambda t: SEAT, 0.9), (c1, 0.9), (c2, 0.9)]))
    ev = one(pp, "third_person")
    assert ev.meta["side"] == "across" and 10.0 <= ev.t <= 12.0  # пока проходит мимо клиента, рамки сливаются


def test_mirrored_scene_gives_same_events():
    c1 = path((2, CL_DOOR), (4, CLIENT), (20, CLIENT), (22, CL_DOOR))
    cashier = path((0, SEAT), (24, SEAT), (25, STAND), (27, DOOR))
    actors = [(cashier, 0.9), (c1, 0.9)]
    a = P.analyze_people(make_scan(30, actors))

    def mir(f):
        return lambda t: None if (b := f(t)) is None else (1 - b[2], b[1], 1 - b[0], b[3])
    lap = ((1 - LAPTOP[2], LAPTOP[1], 1 - LAPTOP[0], LAPTOP[3]), "laptop")
    b = P.analyze_people(make_scan(30, [(mir(f), c) for f, c in actors], objects=(lap,)))
    assert [(e.kind, round(e.t, 1)) for e in a.events] == [(e.kind, round(e.t, 1)) for e in b.events]


def test_lower_fps_keeps_main_events():
    cashier = path((0, SEAT), (10, SEAT), (11, STAND), (13, DOOR))
    c1 = path((1, CL_DOOR), (3, CLIENT), (9, CLIENT), (11, CL_DOOR))
    pp = P.analyze_people(make_scan(20, [(cashier, 0.9), (c1, 0.9)], fps=2.0))
    assert "cashier_leaves" in kinds(pp) and len(pp.visits) == 1


def test_empty_scene():
    pp = P.analyze_people(make_scan(5, [], objects=()))
    assert pp.cashier is None and pp.events == [] and pp.roles_uncertain


# --- дверная активность ---

def test_door_activity_ignores_global_light_change():
    tr = P.Track(0, [k / FPS for k in range(50)], [SEAT] * 50, [0.9] * 50)

    def light(t, k):  # весь кадр меняется: свет выключили / экспозиция
        im = np.full((GH, GW), 100 if t < 5 else 30, np.uint8)
        im[:, : GW // 2] = 100 if t < 5 or k % 2 else 160
        return im
    scan = make_scan(10, [(lambda t: SEAT, 0.9)], gray=light)
    assert P.door_activity(scan, tr, P._area(SEAT)) == []
    scan = make_scan(10, [(lambda t: SEAT, 0.9)], gray=door_gray(4, 5.4, region=(0.12, 0.30, 0.22, 0.50)))
    (a, b), = P.door_activity(scan, tr, P._area(SEAT))
    assert a == pytest.approx(4.2, abs=0.25) and b == pytest.approx(5.4, abs=0.25)


# --- плотный проход ---

class StubDetector:
    batch = 16

    def __init__(self, delay=0.0):
        self.delay, self.calls = delay, 0

    def __call__(self, images):
        self.calls += 1
        time.sleep(self.delay * len(images))
        return [[Det(SEAT, 0.9, "person"), Det(LAPTOP, 0.8, "laptop")] for _ in images]


def test_dense_scan_grid_gray_and_cache(synthetic_video, tmp_path):
    det = StubDetector()
    scan = dense_scan(synthetic_video, 5.0, detector=det)
    assert scan.width == 464 and scan.height == 848 and scan.duration == pytest.approx(20.0, abs=0.05)
    assert 99 <= len(scan.frames) <= 101 and det.calls == -(-len(scan.frames) // 16)
    assert scan.gray.shape == (len(scan.frames), 175, 96) and scan.gray.dtype == np.uint8
    assert scan.gray_t == [f.t for f in scan.frames]
    assert np.diff(scan.gray_t).max() <= 0.2 + 1 / 30 + 1e-6
    f = scan.frames[3]
    assert [d.name for d in f.persons] == ["person"] and [d.name for d in f.objects] == ["laptop"]
    # серый кадр — уменьшенная яркость исходного: «стол» синтетики (y 600–700, x 50–400 из 848x464) светлее фона
    g = scan.gray[0].astype(int)
    assert abs(g[130, 45] - 160) <= 12 and abs(g[40, 45] - 90) <= 12
    save_scan(scan, tmp_path / "s")
    back = load_scan(tmp_path / "s")
    assert back.frames == scan.frames and back.duration == scan.duration and back.fps == scan.fps
    assert np.array_equal(back.gray, scan.gray) and back.gray_t == scan.gray_t


def test_dense_scan_deadline_lowers_fps_but_covers_video(synthetic_video):
    scan = dense_scan(synthetic_video, 5.0, detector=StubDetector(delay=0.01), deadline=time.monotonic() + 0.3)
    ts = scan.gray_t
    assert len(scan.frames) < 60
    assert ts[0] < 0.05 and ts[-1] >= 19.0
    assert np.diff(ts).max() <= 1.0 + 1 / 30 + 1e-6
    assert scan.fps < 5.0


def test_rf_detector_output_to_fractions():
    class D:
        xyxy = np.array([[46.4, 84.8, 232.0, 424.0], [0, 0, 10, 10], [100, 100, 200, 200]])
        confidence = np.array([0.91, 0.5, 0.7])
        class_id = np.array([1, 3, 73])
        data = {"class_name": np.array(["person", "car", "laptop"])}

    class M:
        def predict(self, images, threshold, include_source_image):
            assert include_source_image is False
            return [D() for _ in images]

    from PIL import Image
    det = RFDetector.__new__(RFDetector)
    det.model, det.threshold, det._names, det.batch = M(), 0.25, {}, 16
    (dets,) = det([Image.new("RGB", (464, 848))])
    assert [d.name for d in dets] == ["person", "laptop"]
    assert dets[0].box == pytest.approx((0.1, 0.1, 0.5, 0.5)) and dets[0].conf == pytest.approx(0.91)
