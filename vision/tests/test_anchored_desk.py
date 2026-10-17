"""Новые предметы на столе по пикселям: синтетические цветные кадры, люди — рамки детектора."""

import numpy as np

from forte_vision.anchored import decide
from forte_vision.anchored.desk import desk_blobs
from forte_vision.anchored.types import CropAnswer, Det, DenseScan, Frame, People, Zones

GW, GH = 232, 424
DESK = (0.1, 0.4, 0.7, 0.6)


def _scan(n_sec=30.0, fps=2.0, paint=None, person=None):
    """Стол светлого дерева; paint(t, img) рисует предметы; person(t) -> рамка человека или None."""
    ts = [round(k / fps, 3) for k in range(int(n_sec * fps))]
    rng = np.random.default_rng(0)
    imgs = []
    for t in ts:
        im = np.zeros((GH, GW, 3), np.float32) + (205, 175, 142)
        im += rng.normal(0, 3, im.shape)
        if paint:
            paint(t, im)
        imgs.append(np.clip(im, 0, 255).astype(np.uint8))
    frames = [Frame(t, [Det(person(t), 0.9, "person")] if person and person(t) else []) for t in ts]
    return DenseScan(duration=n_sec, width=464, height=848, fps=fps, frames=frames, hi_t=ts, hi=np.stack(imgs))


def _box(im, b, color):
    im[int(b[1] * GH):int(b[3] * GH), int(b[0] * GW):int(b[2] * GW)] = color


def test_pale_cash_pack_left_on_desk_is_found_and_dates_its_arrival():
    pack = (0.30, 0.46, 0.33, 0.48)  # X-38: бледно-голубая пачка ~7x8 px на светлом дереве
    scan = _scan(paint=lambda t, im: _box(im, pack, (197, 194, 181)) if t >= 12.0 else None)
    bl = desk_blobs(scan, Zones(boxes={"desk": DESK}))
    assert len(bl) == 1
    assert abs(bl[0].t_on - 12.0) <= 1.0 and bl[0].t_off >= 28.0
    b = bl[0].box
    assert b[0] <= 0.31 <= b[2] and b[1] <= 0.47 <= b[3]


def test_person_moving_over_desk_and_brief_object_are_ignored():
    hand = lambda t: (0.2 + 0.01 * t, 0.35, 0.4 + 0.01 * t, 0.62)  # человек ходит над столом
    def paint(t, im):
        _box(im, hand(t), (60, 60, 70))
        if 10.0 <= t < 11.5:  # предмет положили и сразу забрали
            _box(im, (0.55, 0.45, 0.6, 0.5), (240, 240, 240))
    scan = _scan(paint=paint, person=hand)
    assert desk_blobs(scan, Zones(boxes={"desk": DESK})) == []


def test_screen_changes_do_not_count_and_no_desk_means_no_blobs():
    screen = (0.12, 0.42, 0.25, 0.5)
    scan = _scan(paint=lambda t, im: _box(im, screen, (30, 60, 200) if t >= 10 else (20, 20, 20)))
    assert desk_blobs(scan, Zones(boxes={"desk": DESK, "screen": screen})) == []
    assert desk_blobs(scan, Zones(boxes={})) == []


def test_blob_probe_needs_both_crops_and_replaces_blind_sweep():
    from forte_vision.anchored.types import DeskBlob, Track
    cashier = Track(tid=1, t=[0.0, 30.0], boxes=[(0.4, 0.3, 0.55, 0.55)] * 2, conf=[0.9, 0.9])
    people = People(cashier=cashier, cashier_present=[(0.0, 30.0)], visits=[], visitors=[], desk_dir=1, events=[])
    z = Zones(boxes={"desk": DESK, "safe": (0.75, 0.3, 0.9, 0.5)})
    blob = DeskBlob(box=(0.3, 0.46, 0.33, 0.48), t_on=12.0, t_off=29.0, area=4e-4)
    probes = decide.plan(people, z, 30.0, blobs=[blob])
    keys = [it.key for it in decide.all_items(probes)]
    assert not any(k.startswith("DESKp@") for k in keys)  # слепого обхода стола нет
    obj = [it for it in decide.all_items(probes) if it.key.startswith("OBJ0@")]
    assert len(obj) == 2 and all(it.open and "other" in it.choices for it in obj)
    assert obj[0].box[2] - obj[0].box[0] >= 0.12  # вырезка с контекстом, а не 15 px
    ans = {k: CropAnswer(k, "no") for k in keys}
    ans[obj[0].key] = CropAnswer(obj[0].key, "drink", note="coffee cup")
    assert decide.judge(probes, ans) == []
    ans[obj[1].key] = CropAnswer(obj[1].key, "drink")
    assert [(f.check, f.t_main) for f in decide.judge(probes, ans)] == [("FOOD", 12.0)]
    assert decide.plan(people, z, 30.0, blobs=[]) and not any(
        it.key.startswith(("OBJ", "DESKp")) for it in decide.all_items(decide.plan(people, z, 30.0, blobs=[])))


def test_fragments_of_one_object_are_merged():
    from forte_vision.anchored.desk import _merge
    from forte_vision.anchored.types import DeskBlob
    parts = [DeskBlob((0.30, 0.46, 0.33, 0.48), 13.8, 16.8, 2e-4), DeskBlob((0.31, 0.465, 0.335, 0.485), 13.8, 16.8, 3e-4),
             DeskBlob((0.32, 0.47, 0.34, 0.49), 14.0, 17.0, 1e-4), DeskBlob((0.60, 0.46, 0.63, 0.48), 14.0, 17.0, 2e-4),
             DeskBlob((0.30, 0.46, 0.33, 0.48), 40.0, 45.0, 2e-4)]
    m = _merge(parts)
    assert len(m) == 3  # один предмет из трёх кусков + другой предмет рядом по времени + тот же стол позже
    assert m[0].t_on == 13.8 and m[0].t_off == 17.0 and m[0].box == (0.30, 0.46, 0.34, 0.49)
