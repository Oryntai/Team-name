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
    assert len({r.rule_id for r in rules}) == len(rules) == 14
    assert len({r.required for r in rules}) == len(rules)
    for rule in rules:
        assert len(rule.required) == 1 and not rule.applicability and not rule.exceptions
        assert rule.source_url.startswith("https://adilet.zan.kz/rus/docs/")
        assert rule.source_paragraph.strip()
        assert "проверяет человек" in rule.description_template


def test_rules_without_a_direct_norm_say_so():
    """A norm applied by meaning says so both in the citation and in the text the jury reads."""
    for rule in load():
        if "по смыслу" in rule.source_paragraph:
            assert "по смыслу" in rule.description_template, rule.rule_id
        assert "не найдена" not in rule.source_paragraph, rule.rule_id


def test_fields_fit_the_backend_contract():
    """RuleEvent limits: an overlong field would turn every firing of the rule into failed."""
    for rule in load():
        assert len(rule.type) <= 200 and len(rule.source_paragraph) <= 200, rule.rule_id
        assert len(f"{rule.source_document} — {rule.title}") <= 500, rule.rule_id
        assert ";" not in rule.source_document, rule.rule_id  # the UI splits the basis at the first ";"


def test_no_citations_of_repealed_acts():
    """Law No. 2444 of 1995 and NBK resolution No. 48 lost force in 2026."""
    for rule in load():
        text = json.dumps(rule.to_json() if hasattr(rule, "to_json") else rule.__dict__, ensure_ascii=False)
        assert "2444" not in text and "№ 48" not in text and "Z950002444" not in text, rule.rule_id


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
