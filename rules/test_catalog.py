"""Synthetic assertions of legal rule scope, not real video accuracy tests."""
import json
from pathlib import Path

from forte_decision import Observation, Rule, decide


def load_rules():
    rows = json.loads(Path(__file__).with_name("cash_receipt.json").read_text())
    return [Rule.from_json(json.dumps(row)) for row in rows]


def test_predicate_questions_cover_seed_catalog_once():
    questions = json.loads(Path(__file__).with_name('predicate_questions.json').read_text(encoding='utf-8'))
    assert len({q['name'] for q in questions}) == len(questions)
    rules = load_rules()
    assert {q['name'] for q in questions if q['kind']=='predicate'} == {k for r in rules for k in (*r.applicability,*r.required,*r.exceptions)}
    assert {q['name'] for q in questions if q['kind']=='visibility'} == {k for r in rules for k in r.visibility_required}
    assert all(q['visible_if'].strip() for q in questions)


def observation_for(rule, *, predicates=None, visibility=None):
    return Observation.from_dict({
        "video_id": "synthetic-policy-fixture",
        "duration_sec": 10,
        "analysis_status": "complete",
        "coverage_complete": True,
        "candidates": [{
            "candidate_id": "candidate-1", "start_sec": 2,
            "end_sec": 5, "main_sec": 3,
            "source": "synthetic-test",
            "evidence": [{"frame_id": "frame-3", "pts_sec": 3, "source": "synthetic-test"}],
            "predicates": {
                **{key: "observed" for key in (*rule.applicability, *rule.required)},
                **(predicates or {}),
            },
            "visibility": {
                **{key: "observed" for key in rule.visibility_required},
                **(visibility or {}),
            },
        }],
    })


def test_verified_receipt_context_with_visible_condition_emits_sourced_event():
    for rule in load_rules():
        result = decide(observation_for(rule), [rule])
        assert result.verdict == "violation"
        assert rule.source_url in result.events[0].description
        assert rule.source_paragraph in result.events[0].description


def test_receipt_rule_rejects_explicit_other_operation_but_keeps_uncertain_context_for_review():
    for rule in load_rules():
        for state in ("not_observed", "unclear"):
            result = decide(observation_for(rule, predicates={"cash_receipt_operation_established": state}), [rule])
            if state == 'not_observed':
                assert result.events == ()
            else:
                assert result.verdict == 'violation'
                assert 'Если' in result.events[0].description


def test_review_keeps_strong_core_but_strict_requires_complete_visibility():
    rule = load_rules()[0]
    result = decide(observation_for(rule, visibility={"all_possible_client_positions_visible": "unclear"}), [rule])
    assert result.verdict == 'violation'
    assert result.diagnostics[0]['caveats']
    strict = decide(observation_for(rule, visibility={"all_possible_client_positions_visible": "unclear"}), [rule], mode='strict')
    assert strict.events == ()
    assert strict.status == 'incomplete' and strict.verdict is None


def test_standing_up_is_not_departure_before_service_completion():
    rule = load_rules()[1]
    result = decide(observation_for(rule, predicates={"departure_before_client_service_completed": "unclear"}), [rule])
    assert result.events == ()
