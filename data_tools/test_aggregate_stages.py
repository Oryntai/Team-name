import json
from pathlib import Path

from data_tools.aggregate_stages import _hash, aggregate_stages, main
from forte_decision.contracts import Rule


def test_temporal_funnel_missing_audit_and_rule_rejection(tmp_path, capsys):
    evaluation = tmp_path / "evaluation"
    (evaluation / "payloads").mkdir(parents=True)
    audits = tmp_path / "audits"
    audits.mkdir()
    rule = Rule(
        rule_id="synthetic",
        type="synthetic",
        title="Synthetic",
        source_document="Synthetic",
        source_paragraph="1",
        source_url="https://example.invalid",
        applicability=(),
        required=("action",),
        exceptions=(),
        visibility_required=(),
        description_template="Synthetic possible condition",
    )
    gold = []
    for video_id, job_id, failed in [
        ("SECRET-A", "job-a", False),
        ("SECRET-B", "job-b", True),
    ]:
        gold.append(
            {
                "video_id": video_id,
                "verdict": "violation",
                "exhaustive": False,
                "events": [{"timecode_sec": 10}],
            }
        )
        payload = (
            {"job_id": job_id, "status": "failed"}
            if failed
            else {
                "job_id": job_id,
                "status": "done",
                "verdict": "no_violation",
                "violations": [],
            }
        )
        (evaluation / "payloads" / _hash(video_id)).write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "job_id": job_id,
                    "payload": payload,
                    "error_code": "service_failed" if failed else None,
                }
            )
        )
    observation = {
        "video_id": "job-a",
        "duration_sec": 20,
        "analysis_status": "complete",
        "coverage_complete": True,
        "candidates": [
            {
                "candidate_id": "c0",
                "start_sec": 9,
                "end_sec": 11,
                "main_sec": 10,
                "evidence": [{"frame_id": "f", "pts_sec": 10, "source": "synthetic"}],
                "predicates": {"action": "not_observed"},
                "visibility": {},
                "source": "synthetic",
            }
        ],
    }
    (audits / _hash("job-a")).write_text(
        json.dumps(
            {
                "observation": observation,
                "diagnostics": {"candidates": {"c0": {"rechecked": True}}},
                "raw_intervals": [[9, 11]],
                "pooled_intervals": [[9, 11]],
            }
        )
    )
    report = aggregate_stages(evaluation, audits, gold, [rule])
    assert report["counts"]["audit_available_positive_points"] == 1
    assert report["counts"]["audit_missing_positive_points"] == 1
    assert report["temporal_coverage_within_1_sec"]["raw"]["coverage_rate"] == 1
    assert (
        report["temporal_coverage_within_1_sec"]["rechecked"]["covered_positive_points"]
        == 1
    )
    assert (
        report["temporal_coverage_within_1_sec"]["rule_supported"][
            "covered_positive_points"
        ]
        == 0
    )
    assert (
        report["adjacent_temporal_changes"]["rechecked_to_rule_supported_coverage_lost"]
        == 1
    )
    assert (
        report["final_binary"]["FN"] == 1
        and report["final_binary"]["failed_positive"] == 1
    )
    assert report["final_point_recall_within_1_sec"]["recall"] == 0
    assert (
        report["rechecked_point_diagnostics"]["nonexclusive_point_counts"][
            "core_explicitly_false_any_rule"
        ]
        == 1
    )
    assert (
        report["audit_quality_positive_files"][
            "invalid_screen_windows_diagnostic_missing_files"
        ]
        == 1
    )
    assert (
        report["audit_quality_positive_files"][
            "invalid_screen_windows_diagnostic_available_files"
        ]
        == 0
    )
    assert "SECRET" not in json.dumps(report)
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps(gold))
    rules_path = tmp_path / "rules.json"
    rules_path.write_text("[" + rule.to_json() + "]")
    assert (
        main(
            [
                "--evaluation-dir",
                str(evaluation),
                "--audit-dir",
                str(audits),
                "--gold",
                str(gold_path),
                "--rules",
                str(rules_path),
            ]
        )
        == 0
    )
    assert "SECRET" not in capsys.readouterr().out


def test_rechecked_flags_separate_core_unknown_from_context_rejection(tmp_path):
    evaluation = tmp_path / "evaluation"
    (evaluation / "payloads").mkdir(parents=True)
    audits = tmp_path / "audits"
    audits.mkdir()
    rule = Rule(
        rule_id="SYNTHETIC-SECRET",
        type="synthetic",
        title="synthetic",
        source_document="synthetic",
        source_paragraph="1",
        source_url="https://example.invalid",
        applicability=("context",),
        required=("core",),
        exceptions=("exception",),
        visibility_required=(),
        description_template="Synthetic possible condition",
    )
    cases = [
        (
            "context-negative",
            {
                "core": "observed",
                "context": "not_observed",
                "exception": "not_observed",
            },
        ),
        (
            "core-unknown",
            {"core": "unclear", "context": "not_observed", "exception": "observed"},
        ),
        (
            "exception-unknown",
            {"core": "observed", "context": "observed", "exception": "unclear"},
        ),
        (
            "exception-present",
            {"core": "observed", "context": "observed", "exception": "observed"},
        ),
    ]
    gold = []
    for index, (job, predicates) in enumerate(cases):
        video = "private-" + job
        gold.append(
            {
                "video_id": video,
                "verdict": "violation",
                "exhaustive": False,
                "events": [{"timecode_sec": 10}],
            }
        )
        (evaluation / "payloads" / _hash(video)).write_text(
            json.dumps(
                {
                    "video_id": video,
                    "job_id": job,
                    "payload": {"job_id": job, "status": "failed"},
                    "error_code": "service_failed",
                }
            )
        )
        candidate = {
            "candidate_id": "SECRET-CANDIDATE",
            "start_sec": 9,
            "end_sec": 11,
            "main_sec": 10,
            "evidence": [{"frame_id": "f", "pts_sec": 10, "source": "synthetic"}],
            "predicates": predicates,
            "visibility": {},
            "source": "synthetic",
        }
        # Duplicate nearby candidate must not multiply a point/reason counter.
        duplicate = {**candidate, "candidate_id": "SECRET-OTHER"}
        observation = {
            "video_id": job,
            "duration_sec": 20,
            "analysis_status": "partial" if index == 0 else "complete",
            "coverage_complete": index != 0,
            "candidates": [candidate, duplicate],
        }
        (audits / _hash(job)).write_text(
            json.dumps(
                {
                    "observation": observation,
                    "diagnostics": {
                        "candidates": {
                            "SECRET-CANDIDATE": {"rechecked": True},
                            "SECRET-OTHER": {"rechecked": True},
                        },
                        "invalid_screen_windows": 2 if index == 0 else 0,
                        "empty_screen_windows": 0,
                    },
                    "raw_intervals": [[9, 11]],
                    "pooled_intervals": [[9, 11]],
                }
            )
        )
    report = aggregate_stages(evaluation, audits, gold, [rule])
    counts = report["rechecked_point_diagnostics"]["nonexclusive_point_counts"]
    assert counts["points_with_nearby_rechecked_candidate"] == 4
    assert counts["core_all_observed_any_rule"] == 3
    assert counts["core_unclear_any_rule"] == 1
    assert counts["applicability_explicit_false_when_core_observed"] == 1
    assert (
        counts["exception_observed_when_core_observed"] == 1
    )  # excludes unknown-core case
    assert counts["exception_unclear_when_core_observed"] == 1
    assert (
        counts["engine_rejected_nearby"] == 3
        and counts["engine_unresolved_nearby"] == 1
    )
    quality = report["audit_quality_positive_files"]
    assert (
        quality["analysis_partial_files"] == 1
        and quality["coverage_incomplete_files"] == 1
    )
    assert (
        quality["invalid_screen_windows_total"] == 2
        and quality["invalid_screen_windows_nonzero_files"] == 1
    )
    assert quality["invalid_screen_windows_diagnostic_available_files"] == 4
    assert (
        report["temporal_coverage_within_1_sec"]["rechecked"]["covered_positive_points"]
        == 4
    )
    assert "SECRET" not in json.dumps(report) and "private-" not in json.dumps(report)
