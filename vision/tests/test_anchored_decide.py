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
        if it.choices and tag == "LOCK":  # экран: «да» = открыт документ
            out[it.key] = CropAnswer(it.key, "document" if tag in yes_tags else "lock")
        elif it.choices:
            out[it.key] = CropAnswer(it.key, desk)
        else:
            out[it.key] = CropAnswer(it.key, "yes" if tag in yes_tags else "no")
    return out


def test_lock_after_leaving_uses_stand_up_time():
    ev = [Event("cashier_leaves", 17.0, meta={"t_exit": 19.0, "t_stand": 15.0})]
    probes = decide.plan(_people(ev, present=((0.0, 17.0),)), Z, 30.0)
    found = decide.judge(probes, _answer(probes, {"LOCK"}))
    assert [(f.check, f.t_main) for f in found] == [("LOCK", 15.0)]
    # описание для проверяющего — что видно на этом ролике: момент, кадры, зона, подтверждённое
    assert found[0].seen.startswith("после ухода кассира в 17.0 с; кадры ")
    assert found[0].seen.endswith("экран — на экране открыт документ")


def test_single_yes_is_not_enough():
    ev = [Event("cashier_leaves", 10.0, meta={"t_exit": 12.0})]
    probes = decide.plan(_people(ev, present=((0.0, 10.0),)), Z, 30.0)
    first_leave = next(it.key for it in decide.all_items(probes) if it.key.startswith("LEAVE@"))
    answers = _answer(probes, set())
    answers[first_leave] = CropAnswer(first_leave, "yes")
    assert decide.judge(probes, answers) == []


def test_door_after_entering_and_mix_on_client_change():
    ev = [Event("cashier_enters", 4.0), Event("client_change", 30.0, t_end=33.0, meta={"prev_exit": 22.0, "t_seen": 29.0})]
    probes = decide.plan(_people(ev, present=((4.0, 60.0),)), Z, 60.0, door_open=lambda t: True)
    found = decide.judge(probes, _answer(probes, set(), desk="cash"))
    assert [(f.check, f.t_main) for f in found] == [("DOOR", 5.0), ("MIX", 30.0)]
    mix = [it.t for it in decide.all_items(probes) if it.key.startswith("MIX@")]
    assert all(22.0 < t < 29.0 for t in mix)  # до прихода следующего: его деньги не в счёт (X-5)


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
    ev = [Event("client_change", 30.0, t_end=33.0, meta={"prev_exit": 24.0, "t_seen": 30.0})]
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
    answers = _answer(probes, set(), desk="food+documents")
    assert [f.check for f in decide.judge(probes, answers)] == ["FOOD"]
    # напиток на столе — не нарушение (нормы нет; чистый X-25 с банками)
    assert decide.judge(probes, _answer(probes, set(), desk="drink+documents")) == []


def test_lock_screen_after_leaving_is_not_a_violation():
    """X-23/X-29: синий экран входа Windows — не нарушение; белое окно документа (X-28) — нарушение."""
    ev = [Event("cashier_leaves", 10.0, meta={"t_exit": 12.0})]
    probes = decide.plan(_people(ev, present=((0.0, 10.0),)), Z, 30.0)
    lock = [it for it in decide.all_items(probes) if it.key.startswith("LOCK@")]
    assert lock and all(it.choices == decide.SCREEN_CHOICES for it in lock)
    ans = _answer(probes, set())
    for it in lock:
        ans[it.key] = CropAnswer(it.key, "lock")
    assert [f.check for f in decide.judge(probes, ans)] == []
    for it in lock:
        ans[it.key] = CropAnswer(it.key, "desktop+lock")
    assert [f.check for f in decide.judge(probes, ans)] == []
    for it in lock:
        ans[it.key] = CropAnswer(it.key, "document")
    assert [f.check for f in decide.judge(probes, ans)] == ["LOCK"]


def test_second_visitor_across_during_visit_is_queue_violation_timed_when_he_stops():
    """X-18: второй подошёл к столу, пока первый сидит; таймкод — когда дошёл и встал, а не когда появился."""
    walk = [(0.05 + 0.03 * k, 0.2, 0.15 + 0.03 * k, 0.45) for k in range(10)] + [(0.35, 0.2, 0.45, 0.45)] * 15
    late = Track(tid=7, t=[15.0 + 0.4 * k for k in range(25)], boxes=walk, conf=[0.9] * 25)
    ev = [Event("third_person", 15.0, 24.6, {"side": "across", "tid": 7})]
    people = _people(ev, present=((0.0, 30.0),))
    people.visitors = [late]
    probes = decide.plan(people, Z, 30.0)
    found = decide.judge(probes, _answer(probes, set()))
    assert [f.check for f in found] == ["QUE"] and 18.5 <= found[0].t_main <= 19.5
    ev_level = [Event("third_person", 15.0, 24.6, {"side": "level", "tid": 7})]  # рядом с кассиром — не очередь
    p2 = _people(ev_level, present=((0.0, 30.0),))
    p2.visitors = [late]
    assert [f.check for f in decide.judge(decide.plan(p2, Z, 30.0), {})] == []


def test_documents_left_after_leaving_and_safe_left_open_after_sitting_back():
    """X-24: папка лежит весь ролик, кассир ушла -> LEAVE по вопросу с вариантами.
    X-20: сходила к сейфу (встала 4.4), села 6.0, сейф открыт через 1 и 3 с -> SAFE."""
    ev = [Event("cashier_leaves", 8.0, meta={"t_exit": 9.0})]
    probes = decide.plan(_people(ev, present=((0.0, 8.0),)), Z, 15.0)
    ans = _answer(probes, set(), desk="documents")
    assert [f.check for f in decide.judge(probes, ans)] == ["LEAVE"]
    ans = _answer(probes, set(), desk="cash_bag")
    assert [f.check for f in decide.judge(probes, ans)] == ["BAG"]
    ev2 = [Event("cashier_stands", 4.4, t_end=6.0), Event("cashier_sits", 6.0)]
    probes2 = decide.plan(_people(ev2, present=((0.0, 4.4), (6.0, 11.6))), Z, 11.6)
    safe = [it for it in decide.all_items(probes2) if it.key.startswith("SAFEs@")]
    assert [it.t for it in safe] == [7.0, 9.0]
    ans2 = _answer(probes2, set())
    for it in safe:
        ans2[it.key] = CropAnswer(it.key, "yes")
    assert [(f.check, f.t_main) for f in decide.judge(probes2, ans2)] == [("SAFE", 5.5)]
    ans2[safe[1].key] = CropAnswer(safe[1].key, "no")  # закрыла через пару секунд — не нарушение
    assert decide.judge(probes2, ans2) == []


def test_leaving_near_the_end_needs_two_different_frames():
    # кассир вышла за 2 с до конца: поздние моменты прижимаются к последнему кадру — это один кадр, а не два «да»
    ev = [Event("cashier_leaves", 12.0, meta={"t_exit": 12.5})]
    probes = decide.plan(_people(ev, present=((0.0, 12.0),)), Z, 14.7)
    lock = next(p for p in probes if p.check == "LOCK")
    assert len({it.key for it in lock.items}) == len(lock.items)
    last = max(lock.items, key=lambda it: it.t)
    answers = _answer(probes, set())
    answers[last.key] = CropAnswer(last.key, "document")
    assert [f for f in decide.judge(probes, answers) if f.check == "LOCK"] == []


def test_queue_jump_is_not_claimed_when_the_desk_side_is_unknown():
    ev = [Event("third_person", 15.4, 22.0, {"side": "across", "tid": 7})]
    known = _people(ev)
    unknown = People(cashier=known.cashier, cashier_present=known.cashier_present, visits=[], visitors=[],
                     desk_dir=0, events=ev)
    assert any(p.check == "QUE" for p in decide.plan(known, Z, 30.0))
    assert not any(p.check == "QUE" for p in decide.plan(unknown, Z, 30.0))


def test_open_safe_after_sitting_down_only_counts_if_she_stays_seated():
    went = [Event("cashier_stands", 30.0), Event("cashier_sits", 37.2)]
    stays = decide.plan(_people(went), Z, 60.0)
    back_up = decide.plan(_people(went + [Event("cashier_stands", 40.0)]), Z, 60.0)  # встала закрыть сейф (X-19)
    assert any(p.check == "SAFE" and p.reason.startswith("сейф после возвращения") for p in stays)
    assert not any(p.check == "SAFE" and p.reason.startswith("сейф после возвращения") for p in back_up)


def test_counting_on_after_the_client_left_is_a_violation_even_if_it_began_with_the_client():
    # X-34: считала при клиенте и продолжила после его ухода; X-35 (чисто) после ухода уже не считает
    ev = [Event("client_arrives", 5.6), Event("client_leaves", 11.2)]
    probes = decide.plan(_people(ev), Z, 30.0)
    nocl = next(p for p in probes if p.check == "NOCLIENT")
    assert not nocl.veto
    answers = {it.key: CropAnswer(it.key, "yes") for it in decide.all_items(probes) if it.key.startswith("NOCL")}
    assert [f.check for f in decide.judge(probes, answers) if f.check == "NOCLIENT"] == ["NOCLIENT"]
