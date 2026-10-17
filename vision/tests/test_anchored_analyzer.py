"""Анализатор «по якорям»: когда можно выдать вердикт, а когда честно сказать «не обработано»."""

from forte_vision.anchored.analyzer import trustworthy
from forte_vision.anchored.decide import Finding
from forte_vision.anchored.types import CropAnswer, CropItem, Zones

BOX = (0.1, 0.1, 0.3, 0.3)
EV = [CropItem("LOCK@12", 12.0, "screen", BOX, "q"), CropItem("LEAVE@12", 12.0, "desk", BOX, "q")]
SWEEP = CropItem("SAFEp@4", 4.0, "safe", BOX, "q")
OK_ZONES = Zones(boxes={"screen": BOX})
DOWN_ZONES = Zones(boxes={}, diag={"error": "URLError: connection refused"})


def _ans(key, answer="no", err=False):
    return CropAnswer(key, "unclear" if err else answer, "error: URLError: refused" if err else '{"x": "no"}')


def test_found_violation_is_reported_even_if_some_questions_failed():
    found = [Finding("DOOR", 5.0, 4.0, 9.0, [5.0], "пиксели")]
    answers = {it.key: _ans(it.key, err=True) for it in EV}
    assert trustworthy(found, EV, answers, DOWN_ZONES) is True


def test_model_down_never_means_clean():
    answers = {it.key: _ans(it.key, err=True) for it in EV + [SWEEP]}
    assert trustworthy([], EV, answers, OK_ZONES) is False
    assert trustworthy([], [], {}, DOWN_ZONES) is False


def test_event_question_failed_or_not_asked_is_not_clean():
    answers = {EV[0].key: _ans(EV[0].key), EV[1].key: _ans(EV[1].key, err=True)}
    assert trustworthy([], EV, answers, OK_ZONES) is False
    assert trustworthy([], EV, {}, OK_ZONES) is False  # время кончилось до первого вопроса


def test_clean_when_event_questions_answered_even_if_sweep_failed():
    answers = {it.key: _ans(it.key) for it in EV}
    answers[SWEEP.key] = _ans(SWEEP.key, err=True)
    assert trustworthy([], EV, answers, OK_ZONES) is True
    assert trustworthy([], [], {}, OK_ZONES) is True  # спрашивать было не о чем
