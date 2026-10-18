"""ask.py и zones.py без сервера VLM, без GPU и без роликов кейса: заглушка транспорта и синтетическое видео."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# при запуске с PYTHONPATH=. forte_vision берётся из editable-установки другого каталога — берём свой src
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if "forte_vision" in sys.modules and str(_SRC / "forte_vision") not in list(sys.modules["forte_vision"].__path__):
    sys.modules["forte_vision"].__path__.insert(0, str(_SRC / "forte_vision"))

from forte_vision.anchored.ask import (  # noqa: E402
    MAX_ITEMS,
    CropAsker,
    crop_region,
    grab_frames,
    norm_answer,
    norm_choices,
    parse_answers,
    parse_boxes,
)
from forte_vision.anchored.types import ZONES, CropItem, Det, DenseScan, Frame  # noqa: E402
from forte_vision.anchored.zones import PROMPT, find_zones, object_clusters, pick_time, resolve_zones  # noqa: E402


# ---------- разбор ответов ----------

@pytest.mark.parametrize("text, want", [
    ('```json\n{"a": "yes", "b": "no"}\n```', {"a": "yes", "b": "no", "c": "unclear"}),
    ('Sure! Here is the answer: {"a": "Yes.", "b": false, "c": "not visible"} Hope it helps.',
     {"a": "yes", "b": "no", "c": "unclear"}),
    ("{'A': 'TRUE', 'image 2': 'no'}", {"a": "yes", "b": "no", "c": "unclear"}),
    ('{"a": {"answer": "no"}, "c": "Unclear"}', {"a": "no", "b": "unclear", "c": "unclear"}),
    ("a: yes\nb - no", {"a": "yes", "b": "no", "c": "unclear"}),
    ("I cannot see anything.", {"a": "unclear", "b": "unclear", "c": "unclear"}),
    ("", {"a": "unclear", "b": "unclear", "c": "unclear"}),
])
def test_parse_answers(text, want):
    assert parse_answers(text, ["a", "b", "c"]) == want


def test_parse_answers_long_keys_and_no_positional_shift():
    keys = ["safe_open@3.0", "safe_open@5.0", "safe_open@7.0"]
    # модель пропустила средний ключ — остальные не съезжают
    got = parse_answers('{"safe_open@3.0": "yes", "safe_open@7.0": "no"}', keys)
    assert got == {"safe_open@3.0": "yes", "safe_open@5.0": "unclear", "safe_open@7.0": "no"}
    assert parse_answers('{"SAFE_OPEN_3_0": "yes"}', keys)["safe_open@3.0"] == "yes"


@pytest.mark.parametrize("v, want", [(True, "yes"), (False, "no"), (None, "unclear"), ("No, it is closed", "no"),
                                     ("да", "yes"), ("maybe", "unclear"), (1, "yes"), ("not sure", "unclear")])
def test_norm_answer(v, want):
    assert norm_answer(v) == want


def test_parse_boxes_rel1000_and_labels():
    text = ('```json\n[{"bbox_2d": [100, 200, 300, 500], "label": "safe"},'
            '{"bbox_2d": [500, 500, 400, 600], "label": "Laptop Screen"},'
            '{"bbox_2d": [0, 0, 0, 10], "label": "door"}, {"bbox_2d": [1, 2], "label": "desk"}]\n```')
    got = parse_boxes(text, ["safe", "screen", "door", "desk"])
    assert got["safe"] == [(0.1, 0.2, 0.3, 0.5)]
    assert got["screen"] == [(0.4, 0.5, 0.5, 0.6)]  # координаты упорядочены
    assert got["door"] == [] and got["desk"] == []  # вырожденная и битая рамки отброшены


def test_parse_boxes_label_with_echoed_description():
    # Qwen повторяет «метка: описание»; описание стола упоминает laptop — рамка всё равно должна уйти в cash_desk
    text = ('[{"bbox_2d": [62, 467, 630, 657], "label": "cash_desk: the desk with the laptop on it"},'
            '{"bbox_2d": [506, 435, 638, 517], "label": "laptop: the laptop computer"}]')
    got = parse_boxes(text, ["safe", "laptop", "cash_desk"])
    assert got["cash_desk"] == [(0.062, 0.467, 0.63, 0.657)] and got["laptop"] == [(0.506, 0.435, 0.638, 0.517)]


# ---------- геометрия вырезок ----------

@pytest.mark.parametrize("size", [(464, 848), (1920, 1080), (640, 480)])
@pytest.mark.parametrize("box, zoom", [((0.4, 0.3, 0.5, 0.4), 1.0), ((0.1, 0.1, 0.9, 0.3), 1.0),
                                       ((0.0, 0.0, 1.0, 1.0), 1.0), ((0.40, 0.33, 0.48, 0.38), 6.0),
                                       ((0.95, 0.95, 1.0, 1.0), 1.0)])
def test_crop_region_limits(size, box, zoom):
    (x1, y1, x2, y2), (w, h) = crop_region(size, box, zoom)
    W, H = size
    assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H  # область внутри кадра
    assert max(w, h) <= 672 + 1
    m = w / (x2 - x1)
    if zoom > 1:
        assert m >= zoom - 0.05  # объект увеличен не меньше чем в zoom раз
    elif (x2 - x1) * 336 / min(x2 - x1, y2 - y1) <= 672 and (y2 - y1) * 336 / min(x2 - x1, y2 - y1) <= 672:
        assert min(w, h) >= 335  # короткая сторона >= 336, если это не ломает предел длинной
    # центр зоны остаётся внутри вырезки
    cx, cy = (box[0] + box[2]) / 2 * W, (box[1] + box[3]) / 2 * H
    assert x1 - 1 <= cx <= x2 + 1 and y1 - 1 <= cy <= y2 + 1


def test_crop_region_tiny_object_zoom():
    # замок сейфа ~3x6 px на кадре 464x848: область ~56 px вокруг, увеличение ровно 6, а не 40
    (x1, y1, x2, y2), (w, h) = crop_region((464, 848), (0.4152, 0.3529, 0.4217, 0.3605), zoom=6)
    assert 54 <= x2 - x1 <= 58 and 54 <= y2 - y1 <= 58
    assert w / (x2 - x1) == pytest.approx(6, abs=0.1) and min(w, h) >= 330


def test_crop_region_padding():
    (x1, y1, x2, y2), _ = crop_region((1000, 1000), (0.4, 0.4, 0.5, 0.5))
    assert (x2 - x1) == pytest.approx(160, abs=1) and (y2 - y1) == pytest.approx(160, abs=1)  # +30% с каждой стороны


# ---------- кадры ----------

def test_grab_frames_order_and_pts(synthetic_video):
    times = [9.0, 0.0, 9.0, 3.31, 999.0]
    got = grab_frames(synthetic_video, times)
    assert len(got) == len(times)
    for t, (pts, im) in zip(times[:4], got[:4]):
        assert pts >= t - 1e-6 and pts - t < 1 / 30 + 1e-3
        assert im.size == (464, 848)
    assert got[0][0] == got[2][0]
    assert got[4][0] == pytest.approx(19.0 + 29 / 30, abs=0.05)  # за концом — последний кадр
    # в 9.0 c в кадре красный объект события, в 0.0 — нет
    a = np.asarray(got[0][1])
    assert (a[300:420, :, 0] > 180).any() and not (np.asarray(got[1][1])[300:420, :, 0] > 180).any()


# ---------- клиент с заглушкой транспорта ----------

class FakeServer:
    def __init__(self, reply):
        self.reply, self.bodies = reply, []

    def __call__(self, body):
        self.bodies.append(body)
        return {"choices": [{"message": {"content": self.reply(body)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 7}}


def _keys_in(body):
    return [c["text"].split("key=")[1].split(",")[0] for c in body["messages"][0]["content"]
            if c["type"] == "text" and "key=" in c["text"]]


def _items(n):
    return [CropItem(f"k{i}", 1.0 + i, "safe", (0.2, 0.3, 0.4, 0.5), "Is the safe door open?") for i in range(n)]


def test_ask_batches_cache_and_prompt(synthetic_video, tmp_path):
    srv = FakeServer(lambda b: "```json\n" + json.dumps({k: "yes" for k in _keys_in(b)[::2]}) + "\n```")
    a = CropAsker(url="http://127.0.0.1:9", cache_dir=str(tmp_path / "c"), transport=srv)
    items = _items(MAX_ITEMS + 2)
    out = a.ask(synthetic_video, items, context="Cashier left at 12.0 s.")
    assert [o.key for o in out] == [it.key for it in items]
    assert len(srv.bodies) == 2  # 8 + 2
    b0 = srv.bodies[0]
    content = b0["messages"][0]["content"]
    assert sum(c["type"] == "image_url" for c in content) == MAX_ITEMS
    assert content[0]["text"].startswith("Known facts from the video: Cashier left at 12.0 s.")
    assert "You see" not in content[0]["text"]  # без frames_preamble
    assert "Q1: Is the safe door open?" in content[0]["text"]  # вопрос один раз на пакет
    assert content[1]["text"].startswith("Image 1 — key=k0, area=safe, t=1.0s. Question: Q1")
    assert content[2]["type"] == "image_url" and content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "Return ONLY a JSON object" in content[-1]["text"] and '"k7": "yes"|"no"|"unclear"' in content[-1]["text"]
    assert b0["temperature"] == 0.0 and b0["max_tokens"] == 12 * MAX_ITEMS + 40
    # чётные ключи пакета ответила «yes», нечётные модель пропустила -> unclear
    assert [o.answer for o in out[:4]] == ["yes", "unclear", "yes", "unclear"]
    assert a.stats.calls == 2 and a.stats.cached == 0 and a.stats.prompt_tokens == 200

    # повтор — из диска, новым клиентом (кеш переживает процесс)
    srv2 = FakeServer(lambda b: "{}")
    a2 = CropAsker(url="http://127.0.0.1:9", cache_dir=str(tmp_path / "c"), transport=srv2)
    out2 = a2.ask(synthetic_video, items, context="Cashier left at 12.0 s.")
    assert [o.answer for o in out2] == [o.answer for o in out] and not srv2.bodies and a2.stats.cached == 2
    # другой контекст — другой ключ кеша
    a2.ask(synthetic_video, items[:1], context="other")
    assert len(srv2.bodies) == 1


def test_ask_error_gives_unclear_and_is_not_cached(synthetic_video, tmp_path):
    def boom(body):
        raise TimeoutError("slow")

    a = CropAsker(url="http://127.0.0.1:9", cache_dir=str(tmp_path), transport=boom)
    out = a.ask(synthetic_video, _items(2))
    assert all(o.answer == "unclear" and o.raw.startswith("error:") for o in out)
    assert a.stats.errors == 1 and not list(tmp_path.glob("*.json"))


def test_ask_rejects_duplicate_keys(synthetic_video):
    a = CropAsker(url="http://127.0.0.1:9", transport=FakeServer(lambda b: "{}"))
    with pytest.raises(ValueError):
        a.ask(synthetic_video, _items(1) * 2)


@pytest.mark.parametrize("url", ["https://api.openai.com", "http://8.8.8.8:8080", "http://example.com/v1"])
def test_refuses_non_local(url):
    with pytest.raises(ValueError):
        CropAsker(url=url, transport=lambda b: {})


def test_endpoint_and_key(tmp_path):
    kf = tmp_path / "k"
    kf.write_text("secret-1\n")
    a = CropAsker(url="http://127.0.0.1:18080/v1", key_file=str(kf), transport=lambda b: {})
    assert a.url == "http://127.0.0.1:18080/v1/chat/completions" and a._key == "secret-1"


def test_ground_full_frame_and_region(synthetic_video):
    srv = FakeServer(lambda b: '[{"bbox_2d": [250, 500, 750, 1000], "label": "safe"}]')
    a = CropAsker(url="http://127.0.0.1:9", transport=srv)
    full = a.ground(synthetic_video, 5.0, {"safe": "a safe", "door": "a door"})
    assert full == {"safe": (0.25, 0.5, 0.75, 1.0), "door": None}
    reg = a.ground(synthetic_video, 5.0, {"safe": "a safe"}, region=(0.2, 0.2, 0.6, 0.6))
    x1, y1, x2, y2 = reg["safe"]
    # рамка во вырезке переведена в доли полного кадра
    assert x1 == pytest.approx(0.3, abs=0.01) and y1 == pytest.approx(0.4, abs=0.01)
    assert x2 == pytest.approx(0.5, abs=0.01) and y2 == pytest.approx(0.6, abs=0.01)
    assert len(srv.bodies) == 2 and a.ground(synthetic_video, 5.0, {"safe": "a safe", "door": "a door"}) == full
    assert len(srv.bodies) == 2  # второй раз — из кеша в памяти


# ---------- зоны ----------

LAPTOP, COUNTER, SAFE = (0.55, 0.40, 0.65, 0.48), (0.20, 0.42, 0.30, 0.50), (0.85, 0.15, 0.97, 0.35)
DESK, WHITE = (0.10, 0.38, 0.80, 0.62), (0.0, 0.66, 0.80, 0.99)


def _scan(n=20, persons_at=None):
    frames = []
    for k in range(n):
        t = k * 0.5
        persons = [Det((0.4, 0.3, 0.6, 0.6), 0.9, "person")] if (persons_at is None or not persons_at(t)) else \
            [Det((0.4, 0.3, 0.6, 0.6), 0.9, "person"), Det((0.1, 0.5, 0.3, 0.9), 0.9, "person")]
        if persons_at is not None and 4.0 <= t <= 6.0:
            persons = []
        objs = [Det(LAPTOP, 0.8, "laptop"), Det(COUNTER, 0.7, "tv"), Det(SAFE, 0.8, "tv"),
                Det(WHITE, 0.9, "dining table"), Det((0.0, 0.0, 0.05, 0.05), 0.4, "cup") if k == 0 else
                Det(WHITE, 0.5, "dining table")]
        frames.append(Frame(t, persons, objs))
    return DenseScan(duration=n * 0.5, width=1280, height=720, fps=2.0, frames=frames)


def test_object_clusters():
    scan = _scan()
    tv = object_clusters(scan, "tv")
    assert len(tv) == 2 and all(f == 1.0 for _, f in tv)
    assert object_clusters(scan, "cup") == []  # 1 кадр из 20 — не часть сцены
    assert object_clusters(None, "tv") == []


def _det(scan):
    return {n: object_clusters(scan, n) for n in ("laptop", "tv", "dining table")}


def test_resolve_all_from_vlm():
    vlm = {"safe": [SAFE], "screen": [LAPTOP], "desk": [DESK], "door": [(0.0, 0.0, 0.3, 0.4)], "counter": [COUNTER]}
    z = resolve_zones(vlm, _det(_scan()))
    assert set(z.boxes) == set(ZONES) and all(z.boxes[k] is not None for k in ZONES)
    assert all(z.source[k] == "vlm" for k in ZONES)


def test_resolve_rejects_white_table_huge_and_falls_back_to_detector():
    vlm = {"desk": [WHITE], "door": [(0.0, 0.0, 0.9, 0.9)], "safe": [(0.5, 0.5, 0.5005, 0.9)],
           "screen": [], "counter": []}
    z = resolve_zones(vlm, _det(_scan()))
    assert z.boxes["door"] is None and "huge" in z.diag["rejected"]["door"]
    assert "degenerate" in z.diag["rejected"]["safe"]
    assert "no_equipment_on_it" in z.diag["rejected"]["desk"]
    assert z.source["screen"] == "detector" and z.boxes["screen"] == pytest.approx(LAPTOP)
    # белый стол детектора тоже без техники -> грубый стол вокруг техники
    assert z.source["desk"] == "fallback" and z.boxes["desk"] != pytest.approx(WHITE)
    d = z.boxes["desk"]
    assert d[0] <= LAPTOP[0] and d[2] >= LAPTOP[2]
    # грубый стол мал, машинку детектора в него не взять; сейф — самый крупный tv вне стола
    assert z.source["safe"] == "detector" and z.boxes["safe"] == pytest.approx(SAFE)


def test_resolve_counter_from_desk_tv_when_desk_from_vlm():
    vlm = {"desk": [DESK], "screen": [LAPTOP]}
    z = resolve_zones(vlm, _det(_scan()))
    assert z.source["desk"] == "vlm" and z.boxes["counter"] == pytest.approx(COUNTER)
    assert z.boxes["safe"] == pytest.approx(SAFE) and z.boxes["door"] is None


def test_resolve_screen_disagreeing_with_steady_laptop():
    vlm = {"screen": [(0.40, 0.40, 0.48, 0.47)], "desk": [DESK]}  # на столе, но не там, где ноутбук
    z = resolve_zones(vlm, _det(_scan()))
    assert z.source["screen"] == "detector" and "disagrees_with_laptop" in z.diag["rejected"]["screen"]


def test_resolve_same_box_for_safe_and_counter():
    vlm = {"desk": [DESK], "screen": [LAPTOP], "safe": [COUNTER], "counter": [COUNTER]}
    z = resolve_zones(vlm, {})
    assert z.boxes["counter"] == pytest.approx(COUNTER) and z.boxes["safe"] is None  # на столе — машинка
    vlm = {"desk": [DESK], "screen": [LAPTOP], "safe": [SAFE], "counter": [SAFE]}
    z = resolve_zones(vlm, {})
    assert z.boxes["safe"] == pytest.approx(SAFE) and z.boxes["counter"] is None  # вне стола — сейф


def test_resolve_without_detector_and_equipment_keeps_desk():
    z = resolve_zones({"desk": [DESK]}, {})
    assert z.boxes["desk"] == pytest.approx(DESK) and z.source == {"desk": "vlm"}


def test_pick_time_scan_prefers_empty_lit_frames():
    scan = _scan(persons_at=lambda t: True)  # везде двое, в 4–6 с пусто
    t, info = pick_time("unused.mov", scan)
    assert 4.0 <= t <= 6.0 and info["persons"] == 0
    # в пустом отрезке выключен свет -> берём освещённый кадр
    g = np.full((len(scan.frames), 8, 8), 120, np.uint8)
    for k, f in enumerate(scan.frames):
        if 4.0 <= f.t <= 6.0:
            g[k] = 5
    scan.gray, scan.gray_t = g, [f.t for f in scan.frames]
    t, info = pick_time("unused.mov", scan)
    assert not 4.0 <= t <= 6.0


def test_pick_time_without_scan(synthetic_video):
    t, info = pick_time(synthetic_video)
    assert info["how"] == "median_bg" and not 8.0 <= t < 10.0  # не в момент движущегося объекта
    t, info = pick_time(synthetic_video, people_boxes_at=lambda t: [1] if t < 15 else [])
    assert t >= 15 and info["persons"] == 0


class StubAsker:
    """Grounding-заглушка: full — кандидаты по зонам, safe_retry — ответ повтора сейфа, lock — рамка замка."""

    def __init__(self, full, lock=None, safe_retry=(), door_leaf=None):
        self.full, self.safe_retry, self.calls = full, list(safe_retry), []
        self.small = {"lock": lock, "door_leaf": door_leaf}
        self.last_ground = {}

    def ground_candidates(self, path, t, labels, region=None):
        self.calls.append((t, tuple(labels), region))
        self.last_ground = {"raw": "stub", "error": None}
        if tuple(labels) == ("safe",):
            return {"safe": self.safe_retry}
        by_label = {PROMPT[z][0]: bs for z, bs in self.full.items()}
        return {k: list(by_label.get(k, [])) for k in labels}

    def ground(self, path, t, labels, region=None):
        self.calls.append((t, tuple(labels), region))
        return {k: self.small.get(k) for k in labels}


def test_find_zones_with_scan_and_lock():
    scan = _scan(persons_at=lambda t: True)
    lock = (0.90, 0.25, 0.92, 0.27)
    asker = StubAsker({"safe": [SAFE], "screen": [LAPTOP], "desk": [DESK], "counter": [COUNTER]}, lock=lock)
    z = find_zones("unused.mov", scan, asker)
    assert 4.0 <= z.diag["t"] <= 6.0 and asker.calls[0][0] == z.diag["t"]
    assert all(z.source[k] == "vlm" for k in ("safe", "screen", "desk", "counter"))
    assert z.boxes["door"] is None and z.boxes["safe"] == pytest.approx(SAFE)
    assert len(asker.calls) == 2  # сейф найден сразу — без повтора; второй запрос — замок
    assert z.diag["safe_lock"] == pytest.approx(lock) and asker.calls[1][2] is not None  # замок ищется во вырезке
    # замок вне сейфа отвергается
    z = find_zones("unused.mov", scan, StubAsker({"safe": [SAFE]}, lock=(0.1, 0.1, 0.12, 0.12)))
    assert "safe_lock" not in z.diag


def test_find_zones_safe_retry_and_same_box():
    # VLM назвала машинку и сейфом, и машинкой -> сейф ищется повторным запросом;
    # повтор, совпавший с машинкой, отвергается
    full = {"safe": [COUNTER], "screen": [LAPTOP], "desk": [DESK, WHITE], "counter": [COUNTER]}
    asker = StubAsker(full, safe_retry=[COUNTER, SAFE])
    z = find_zones("unused.mov", _scan(), asker, find_lock=False)
    assert z.boxes["counter"] == pytest.approx(COUNTER) and z.boxes["safe"] == pytest.approx(SAFE)
    assert z.source["safe"] == "vlm" and "same_as_counter" in z.diag["rejected"]["safe"]
    assert z.boxes["desk"] == pytest.approx(DESK)  # белый стол — второй кандидат — не взят
    # повтор ничего не дал -> устойчивый tv детектора вне стола
    z = find_zones("unused.mov", _scan(), StubAsker(full), find_lock=False)
    assert z.source["safe"] == "detector" and z.boxes["safe"] == pytest.approx(SAFE)


def test_resolve_screen_off_desk_and_safe_behind_counter():
    # монитор в коридоре за стеклом (далеко от стола) — не экран кассира
    z = resolve_zones({"desk": [DESK], "screen": [(0.30, 0.10, 0.35, 0.14)], "counter": [COUNTER]}, {})
    assert z.boxes["screen"] is None and "off_desk" in z.diag["rejected"]["screen"]
    # сейф за машинкой перекрыт ею на треть (X-30..33) — это всё ещё отдельный предмет
    behind = (COUNTER[0] - 0.01, COUNTER[1] - 0.06, COUNTER[2] + 0.02, COUNTER[1] + 0.03)
    z = resolve_zones({"desk": [DESK], "counter": [COUNTER], "safe": [behind]}, {})
    assert z.boxes["safe"] == pytest.approx(behind) and z.boxes["counter"] == pytest.approx(COUNTER)


def test_resolve_screen_equals_counter_without_laptop():
    # X-30: нет ноутбука, VLM назвала машинку экраном и сейфом -> это машинка (стоит на столе)
    z = resolve_zones({"desk": [DESK], "screen": [COUNTER], "safe": [COUNTER]}, {})
    assert z.boxes["counter"] == pytest.approx(COUNTER) and z.boxes["screen"] is None and z.boxes["safe"] is None


def test_find_zones_refines_door_to_leaf():
    wall, leaf = (0.0, 0.0, 0.5, 0.5), (0.20, 0.02, 0.30, 0.45)
    full = {"desk": [DESK], "screen": [LAPTOP], "counter": [COUNTER], "safe": [SAFE], "door": [wall]}
    z = find_zones("unused.mov", _scan(), StubAsker(full, door_leaf=leaf), find_lock=False)
    assert z.boxes["door"] == pytest.approx(leaf) and z.diag["door_wall"] == pytest.approx(wall)
    # створка вне стены или почти вся стена — остаётся стена
    z = find_zones("unused.mov", _scan(), StubAsker(full, door_leaf=(0.6, 0.6, 0.7, 0.9)), find_lock=False)
    assert z.boxes["door"] == pytest.approx(wall)
    z = find_zones("unused.mov", _scan(), StubAsker(full), find_lock=False, refine_door=False)
    assert z.boxes["door"] == pytest.approx(wall) and "door_wall" not in z.diag


@pytest.mark.parametrize("v,want", [
    (["cash"], "cash"), (["banknotes", "cup"], "cash+drink"), ("a bundle of money and a mug", "cash+drink"),
    (["none"], "none"), ([], "none"), ("nothing", "none"), (None, "unclear"), ("hmm", "unclear"),
    ({"items": ["handbag"]}, "bag"), (True, "unclear"),
])
def test_norm_choices(v, want):
    assert norm_choices(v, ("cash", "documents", "bag", "food", "drink", "phone", "none")) == want


def test_parse_answers_mixed_yes_no_and_choices():
    ch = ("cash", "food", "drink", "none")
    text = '```json\n{"SAFEp@4": "no", "DESKp@4": ["cash"], "DESKp@8": ["none"]}\n```'
    got = parse_answers(text, ["SAFEp@4", "DESKp@4", "DESKp@8"], {"DESKp@4": ch, "DESKp@8": ch})
    assert got == {"SAFEp@4": "no", "DESKp@4": "cash", "DESKp@8": "none"}


def test_open_question_group_and_note():
    from forte_vision.anchored.ask import parse_notes
    ch = ("cash", "documents", "bag", "food", "drink", "phone", "other", "none")
    text = '{"OBJ0@13": {"what": "bundle of banknotes in plastic", "group": "cash"}, "OBJ1@5": {"what": "stapler", "group": "other"}, "OBJ2@7": {"what": "a coffee mug", "group": "?"}}'
    keys = ["OBJ0@13", "OBJ1@5", "OBJ2@7"]
    got = parse_answers(text, keys, {k: ch for k in keys})
    assert got == {"OBJ0@13": "cash", "OBJ1@5": "other", "OBJ2@7": "drink"}  # группа не распознана -> по «what»
    assert parse_notes(text, keys)["OBJ0@13"] == "bundle of banknotes in plastic"



def test_cash_bag_is_one_word_and_screen_choices():
    desk = ("cash", "documents", "cash_bag", "bag", "food", "drink", "phone", "none")
    assert norm_choices(["cash_bag"], desk) == "cash_bag"
    assert norm_choices(["cash_bag", "phone"], desk) == "cash_bag+phone"
    scr = ("document", "desktop", "lock", "off")
    assert norm_choices(["lock"], scr) == "lock"
    assert norm_choices("a sign-in screen with a clock", scr) == "lock"
    assert norm_choices(["document"], scr) == "document"
    assert norm_choices("lid closed", scr) == "off"


def test_video_id_is_by_content_not_name(tmp_path):
    """Оценка кладёт ролик ссылкой <job_id>.mov: ключ кеша не должен зависеть от имени."""
    from forte_vision.anchored.ask import CropAsker
    a, b, c = tmp_path / "job1.mov", tmp_path / "job2.mov", tmp_path / "other.mov"
    a.write_bytes(b"x" * 5000)
    b.write_bytes(b"x" * 5000)
    c.write_bytes(b"y" * 5000)
    assert CropAsker._video_id(str(a)) == CropAsker._video_id(str(b)) != CropAsker._video_id(str(c))


def test_equipment_named_cash_something_is_not_cash():
    from forte_vision.anchored.ask import norm_choices
    from forte_vision.anchored.decide import OBJ_CHOICES
    assert norm_choices({"what": "cash counting machine", "group": "cash"}, OBJ_CHOICES) == "other"
    assert norm_choices({"what": "black cash register", "group": "cash"}, OBJ_CHOICES) == "other"
    assert norm_choices({"what": "a crumpled banknote", "group": "cash"}, OBJ_CHOICES) == "cash"
    assert norm_choices({"what": "two cans", "group": "drink"}, OBJ_CHOICES) == "drink"


def test_drinkware_is_a_drink_even_if_the_model_says_food():
    from forte_vision.anchored.ask import norm_choices
    from forte_vision.anchored.decide import OBJ_CHOICES
    assert norm_choices({"what": "two cans", "group": "food"}, OBJ_CHOICES) == "drink"
    assert norm_choices({"what": "a sandwich and a cup", "group": "food"}, OBJ_CHOICES) == "food"
    assert norm_choices({"what": "an apple", "group": "food"}, OBJ_CHOICES) == "food"
