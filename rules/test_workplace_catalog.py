"""Synthetic checks of the workplace catalog: shape, sources and one-predicate firing."""
import json
from pathlib import Path

from forte_decision import Observation, Rule, decide

HERE = Path(__file__).parent


def load(name):
    rows = json.loads((HERE / name).read_text(encoding="utf-8"))
    return [Rule.from_json(json.dumps(row)) for row in rows]


def observation_for(predicates):
    return Observation.from_dict({
        "video_id": "synthetic-workplace-fixture",
        "duration_sec": 10,
        "analysis_status": "complete",
        "coverage_complete": True,
        "candidates": [{
            "candidate_id": "candidate-1", "start_sec": 2, "end_sec": 5, "main_sec": 3,
            "source": "synthetic-test",
            "evidence": [{"frame_id": "frame-3", "pts_sec": 3, "source": "synthetic-test"}],
            "predicates": predicates,
            "visibility": {},
        }],
    })


def test_catalog_ids_unique_across_catalogs_and_sources_are_official():
    rules = load("cash_receipt.json") + load("cash_workplace.json")
    assert len({r.rule_id for r in rules}) == len(rules)
    for rule in load("cash_workplace.json"):
        assert rule.source_url.startswith("https://adilet.zan.kz/rus/docs/")
        assert rule.source_paragraph.strip()
        assert "проверяет человек" in rule.description_template


def test_each_rule_fires_on_its_core_and_not_on_contrary_core():
    for rule in load("cash_workplace.json"):
        positive = decide(observation_for({k: "observed" for k in rule.required}), [rule])
        assert positive.verdict == "violation"
        assert rule.source_paragraph in positive.events[0].description
        negative = decide(observation_for({k: "not_observed" for k in rule.required}), [rule])
        assert negative.verdict == "no_violation" and negative.events == ()


def test_leaving_securely_is_not_a_violation():
    # Clean videos: the cashier leaves after locking the laptop and clearing the desk.
    rule = next(r for r in load("cash_workplace.json") if "WORKSTATION_LEFT" in r.rule_id)
    result = decide(observation_for({
        "cashier_absent_from_room": "observed",
        "workstation_left_unsecured": "not_observed",
    }), [rule])
    assert result.verdict == "no_violation"
