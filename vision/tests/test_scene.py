"""Сцена и роли на поддельном детекторе: без весов RF-DETR и без реальных видео."""

import json

import numpy as np

from conftest import DURATION
from forte_vision import analyze_video, validate_observation
from forte_vision.scene import FrameDets, build_scene, mark_roles, scene_events
from forte_vision.vlm import StubBackend

W, H = 464, 848
DESK = (60.0, 560.0, 420.0, 720.0)
CASHIER = (150.0, 380.0, 330.0, 640.0)
CLIENT = (180.0, 700.0, 320.0, 840.0)
LEAVE_AT = 12.0


def fake_frames(fps=2.0, duration=DURATION):
    out = []
    for i in range(int(duration * fps)):
        t = i / fps
        persons = [CASHIER] if t < LEAVE_AT else []
        if 4.0 <= t < 16.0:
            persons.append(CLIENT)
        objects = [("dining table", DESK), ("laptop", (300.0, 580.0, 400.0, 650.0))]
        if t >= 6.0:
            objects.append(("handbag", (100.0, 600.0, 160.0, 660.0)))
        out.append(FrameDets(t, persons, objects))
    return out


class FakeDetector:
    size, device = "fake", "cpu"

    def detect(self, frames, batch=8):
        frames = list(frames)
        by_t = {d.t: d for d in fake_frames(fps=30.0)}
        out = []
        for t, _ in frames:
            k = min(by_t, key=lambda x: abs(x - t))
            out.append(FrameDets(t, by_t[k].persons, by_t[k].objects))
        return out


def test_scene_finds_desk_slot_and_departure():
    dets = fake_frames()
    sc = build_scene(dets, W, H)
    assert sc.desk_box is not None and sc.cashier_slot is not None
    # место кассира — там, где кассир, а не клиент
    cy = (sc.cashier_slot[1] + sc.cashier_slot[3]) / 2
    assert abs(cy - (CASHIER[1] + CASHIER[3]) / 2) < 40
    ev = scene_events(sc, DURATION)
    dep = [e for e in ev if e.kind == "cashier_departure"]
    assert dep and abs(dep[0].main_sec - LEAVE_AT) <= 0.6
    assert any(e.kind == "item_on_desk:handbag" and abs(e.main_sec - 6.0) <= 0.6 for e in ev)
    assert any(e.kind == "persons_change" for e in ev)
    for e in ev:
        assert 0 <= e.start_sec <= e.main_sec <= e.end_sec <= DURATION


def test_mark_roles_returns_copies():
    from PIL import Image

    im = Image.new("RGB", (W, H), (90, 90, 90))
    frames = [(1.0, im)]
    marked = mark_roles(frames, [FrameDets(1.0, [CASHIER, CLIENT], [])], CASHIER)
    assert marked[0][1] is not im
    assert np.asarray(im).max() == 90  # оригинал не тронут: доказательства идут без рамок


def test_observer_uses_detector_and_role_hint(synthetic_video):
    questions = []

    def respond(times, question):
        questions.append(question)
        return "[]" if "Categories:" in question else json.dumps({"cashier_left_workstation": "yes"})

    obs, diag = analyze_video(synthetic_video, StubBackend(respond), video_id="syn", profile="fast",
                              detector=FakeDetector())
    validate_observation(obs)
    assert diag["scene"]["desk_found"] and diag["scene"]["cashier_slot_found"]
    near = [c for c in obs["candidates"] if c["start_sec"] <= LEAVE_AT <= c["end_sec"]]
    assert near and "detector" in near[0]["source"]
    assert any("CASHIER" in q for q in questions), "перепроверка должна получать подсказку ролей"
