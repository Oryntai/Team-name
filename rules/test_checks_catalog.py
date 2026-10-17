"""Synthetic checks of the checklist catalog: shape, sources and one-predicate firing."""
import json
from pathlib import Path

from forte_decision import Observation, Rule, decide

HERE = Path(__file__).parent


def load(name="cash_checks.json"):
    rows = json.loads((HERE / name).read_text(encoding="utf-8"))
    return [Rule.from_json(json.dumps(row)) for row in rows]


def observation_for(predicates):
    return Observation.from_dict({
        "video_id": "synthetic-checks-fixture",
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


def test_one_rule_per_check_predicate_with_official_source():
    rules = load()
    assert len({r.rule_id for r in rules}) == len(rules) == 13
    assert len({r.required for r in rules}) == len(rules)
    for rule in rules:
        assert len(rule.required) == 1 and not rule.applicability and not rule.exceptions
        assert rule.source_url.startswith("https://adilet.zan.kz/rus/docs/")
        assert rule.source_paragraph.strip()
        assert "проверяет человек" in rule.description_template


def test_rules_without_a_direct_norm_say_so():
    for rule in load():
        if rule.rule_id.startswith("TEAM_"):
            assert any(w in rule.source_paragraph for w in ("нет", "не найдена", "трактовка"))


def test_each_rule_fires_only_on_its_own_predicate():
    rules = load()
    names = [r.required[0] for r in rules]
    for rule in rules:
        preds = {n: ("observed" if n == rule.required[0] else "not_observed") for n in names}
        result = decide(observation_for(preds), rules)
        assert result.verdict == "violation"
        assert [e.rule_id for e in result.events] == [rule.rule_id]
    clean = decide(observation_for({n: "not_observed" for n in names}), rules)
    assert clean.verdict == "no_violation" and clean.events == ()
