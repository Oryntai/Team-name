"""Synthetic assertions of legal rule scope, not real video accuracy tests."""
import json
from pathlib import Path

from forte_decision import Observation, Rule, decide


def load_rules():
    rows = json.loads(Path(__file__).with_name("cash_receipt.json").read_text())
    return [Rule.from_json(json.dumps(row)) for row in rows]


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


def test_receipt_rule_does_not_apply_to_payment_or_unknown_operation():
    for rule in load_rules():
        for state in ("not_observed", "unclear"):
            result = decide(observation_for(rule, predicates={"cash_receipt_operation_established": state}), [rule])
            assert result.events == ()


def test_offscreen_client_cannot_be_called_absent():
    rule = load_rules()[0]
    result = decide(observation_for(rule, visibility={"all_possible_client_positions_visible": "unclear"}), [rule])
    assert result.events == ()
    assert result.diagnostics[0]["supported"] is False


def test_standing_up_is_not_departure_before_service_completion():
    rule = load_rules()[1]
    result = decide(observation_for(rule, predicates={"departure_before_client_service_completed": "unclear"}), [rule])
    assert result.events == ()
