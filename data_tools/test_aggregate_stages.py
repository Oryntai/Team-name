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
