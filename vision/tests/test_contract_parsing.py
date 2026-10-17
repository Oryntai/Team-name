import copy

import pytest

from forte_vision.contract import ContractError, validate_observation
from forte_vision.predicates import parse_screen, parse_tri_state

GOOD = {
    "video_id": "v1",
    "duration_sec": 20.0,
    "analysis_status": "complete",
    "coverage_complete": True,
    "candidates": [
        {
            "candidate_id": "c000",
            "start_sec": 8.0,
            "end_sec": 10.0,
            "main_sec": 9.0,
            "source": "motion+vlm",
            "evidence": [{"frame_id": "pts_000009000", "pts_sec": 9.0, "source": "decoded"}],
            "predicates": {"cash_counting_observed": "observed"},
            "visibility": {},
        }
    ],
}


def test_good_passes():
    validate_observation(GOOD)


def _bad(mut):
    o = copy.deepcopy(GOOD)
    mut(o)
    with pytest.raises(ContractError):
        validate_observation(o)


def test_rejects_extra_field():
    _bad(lambda o: o.update(raw_scores={}))
    _bad(lambda o: o["candidates"][0].update(score=0.9))


def test_rejects_bad_times():
    _bad(lambda o: o["candidates"][0].update(main_sec=11.0))
    _bad(lambda o: o["candidates"][0].update(end_sec=25.0))
    _bad(lambda o: o["candidates"][0]["evidence"][0].update(pts_sec=7.0))


def test_rejects_frame_id_conflict():
    def mut(o):
        c2 = copy.deepcopy(o["candidates"][0])
        c2["candidate_id"] = "c001"
        c2["evidence"][0]["pts_sec"] = 9.5
        o["candidates"].append(c2)

    _bad(mut)


def test_rejects_complete_without_coverage_and_bad_tristate():
    _bad(lambda o: o.update(coverage_complete=False))
    _bad(lambda o: o["candidates"][0]["predicates"].update(cash_counting_observed="yes"))
    _bad(lambda o: o["candidates"][0].update(evidence=[]))


def test_parse_tri_state_tolerates_fences_and_garbage():
    names = ["a", "b", "c"]
    txt = 'Sure!\n```json\n{"a": "yes", "b": "No", "c": "maybe"}\n```'
    assert parse_tri_state(txt, names) == {"a": "observed", "b": "not_observed", "c": "unclear"}
    assert parse_tri_state("no json at all", names) == {n: "unclear" for n in names}


def test_parse_screen_clips_and_filters():
    txt = '[{"category": "cash_counting", "start": "12.0s", "end": "30"}, {"category": "unknown", "start": 1, "end": 2}, 5]'
    assert parse_screen(txt, 10.0, 22.0) == [("cash_counting", 12.0, 22.0)]
    assert parse_screen("SELECT 1", 0, 10) == []
