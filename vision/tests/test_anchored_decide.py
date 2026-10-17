"""Решалка «по якорям» на синтетике: события людей + ответы модели -> нарушения и таймкоды."""

from forte_vision.anchored import decide
from forte_vision.anchored.types import CropAnswer, Event, People, Track, Zones

Z = Zones(boxes={"screen": (0.5, 0.4, 0.6, 0.5), "desk": (0.2, 0.45, 0.7, 0.6), "safe": (0.75, 0.3, 0.9, 0.5),
                 "door": (0.0, 0.1, 0.2, 0.5), "counter": (0.6, 0.45, 0.7, 0.55)})


def _people(events, present=((0.0, 30.0),)):
    cashier = Track(tid=1, t=[0.0, 30.0], boxes=[(0.4, 0.3, 0.55, 0.55)] * 2, conf=[0.9, 0.9])
    return People(cashier=cashier, cashier_present=list(present), visits=[], visitors=[], desk_dir=1, events=events)


def _answer(probes, yes_tags, desk="none"):
    """yes на вырезки с тегом из yes_tags; на вопрос «что лежит на столе» — desk (например, "cash")."""
    out = {}
    for it in decide.all_items(probes):
        tag = it.key.split("@")[0]
        if it.choices:
            out[it.key] = CropAnswer(it.key, desk)
        else:
            out[it.key] = CropAnswer(it.key, "yes" if tag in yes_tags else "no")
    return out


def test_lock_after_leaving_uses_stand_up_time():
    ev = [Event("cashier_leaves", 17.0, meta={"t_exit": 19.0, "t_stand": 15.0})]
    probes = decide.plan(_people(ev, present=((0.0, 17.0),)), Z, 30.0)
    found = decide.judge(probes, _answer(probes, {"LOCK"}))
    assert [(f.check, f.t_main) for f in found] == [("LOCK", 15.0)]


def test_single_yes_is_not_enough():
    ev = [Event("cashier_leaves", 10.0, meta={"t_exit": 12.0})]
    probes = decide.plan(_people(ev, present=((0.0, 10.0),)), Z, 30.0)
    first_leave = next(it.key for it in decide.all_items(probes) if it.key.startswith("LEAVE@"))
    answers = _answer(probes, set())
    answers[first_leave] = CropAnswer(first_leave, "yes")
    assert decide.judge(probes, answers) == []


def test_door_after_entering_and_mix_on_client_change():
    ev = [Event("cashier_enters", 4.0), Event("client_change", 30.0, t_end=33.0)]
    probes = decide.plan(_people(ev, present=((4.0, 60.0),)), Z, 60.0, door_open=lambda t: True)
    found = decide.judge(probes, _answer(probes, set(), desk="cash"))
    assert [(f.check, f.t_main) for f in found] == [("DOOR", 5.0), ("MIX", 33.0)]


def test_door_closed_soon_after_entering_is_fine():
    ev = [Event("cashier_enters", 4.0)]
    probes = decide.plan(_people(ev, present=((4.0, 60.0),)), Z, 60.0, door_open=lambda t: t < 8.0)
    assert [f.check for f in decide.judge(probes, _answer(probes, set()))] == []


def test_early_stand_up_long_before_leaving_is_not_the_timecode():
    ev = [Event("cashier_leaves", 18.0, meta={"t_exit": 21.0, "t_stand": 12.5})]
    probes = decide.plan(_people(ev, present=((0.0, 18.0),)), Z, 30.0)
    found = decide.judge(probes, _answer(probes, {"LOCK"}))
    assert [(f.check, f.t_main) for f in found] == [("LOCK", 17.5)]


def test_unknown_predicate_not_planned():
    ev = [Event("cashier_leaves", 10.0, meta={"t_exit": 12.0})]
    known = {decide.PREDICATE["LOCK"]}
    probes = decide.plan(_people(ev), Z, 30.0, known_predicates=known)
    assert {p.check for p in probes} == {"LOCK"}


def test_sweep_needs_two_neighbouring_yes_and_skips_cashier_at_safe():
    probes = decide.plan(_people([]), Z, 30.0, known_predicates={decide.PREDICATE["ITEMS"]})
    items = decide.all_items(probes)
    answers = {it.key: CropAnswer(it.key, "none") for it in items}
    answers[items[2].key] = CropAnswer(items[2].key, "bag")
    assert decide.judge(probes, answers) == []
    answers[items[3].key] = CropAnswer(items[3].key, "bag+phone")
    found = decide.judge(probes, answers)
    assert [(f.check, f.t_main) for f in found] == [("ITEMS", items[2].t)]


def test_unclear_never_counts_as_yes():
    ev = [Event("cashier_leaves", 10.0, meta={"t_exit": 12.0})]
    probes = decide.plan(_people(ev, present=((0.0, 10.0),)), Z, 30.0)
    answers = {it.key: CropAnswer(it.key, "unclear") for it in decide.all_items(probes)}
    assert decide.judge(probes, answers) == []


def test_cash_on_desk_is_not_food_and_mix_needs_cash_twice():
    """X-38: пачка денег прошлого клиента — это MIX, а не «еда»; одной вырезки с деньгами мало."""
    ev = [Event("client_change", 30.0, t_end=33.0)]
    probes = decide.plan(_people(ev, present=((0.0, 60.0),)), Z, 60.0)
    assert [f.check for f in decide.judge(probes, _answer(probes, set(), desk="cash"))] == ["MIX"]
    answers = _answer(probes, set(), desk="none")
    mix = [it.key for it in decide.all_items(probes) if it.key.startswith("MIX@")]
    answers[mix[0]] = CropAnswer(mix[0], "cash")
    assert decide.judge(probes, answers) == []


def test_food_and_items_share_one_desk_question():
    probes = decide.plan(_people([]), Z, 20.0)
    desk = [it for it in decide.all_items(probes) if it.key.startswith("DESKp@")]
    assert desk and all(it.choices for it in desk)
    assert len({it.key for it in desk}) == len(desk)  # общий вопрос не задаётся дважды
    answers = _answer(probes, set(), desk="drink+documents")
    assert [f.check for f in decide.judge(probes, answers)] == ["FOOD"]
