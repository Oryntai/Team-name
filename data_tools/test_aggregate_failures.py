"""Aggregate synthetic records only, without media or per-video stdout."""

import json

import pytest

from data_tools.aggregate_failures import aggregate, main


def inventory(gold, digests=None):
    digests = digests or [f"{i:064x}" for i in range(len(gold))]
    return {
        "file_count": len(gold),
        "unique_content_count": len(set(digests)),
        "records": [
            {"video_id": row["video_id"], "sha256": digest}
            for row, digest in zip(gold, digests)
        ],
        "duplicate_groups": [],
    }


def positive(video_id, time=1):
    return {
        "video_id": video_id,
        "verdict": "violation",
        "exhaustive": False,
        "events": [{"timecode_sec": time}],
    }


def negative(video_id):
    return {
        "video_id": video_id,
        "verdict": "no_violation",
        "exhaustive": False,
        "events": [],
    }


def prediction(video_id, verdict):
    return {
        "video_id": video_id,
        "status": "complete",
        "verdict": verdict,
        "events": [{"timecode_sec": 1}] if verdict == "violation" else [],
    }


def test_failed_positives_and_negatives_remain_in_denominators():
    gold = [
        positive("tp"),
        negative("fp"),
        positive("fn"),
        negative("tn"),
        positive("failed-positive"),
        negative("failed-negative"),
    ]
    preds = [
        prediction("tp", "violation"),
        prediction("fp", "violation"),
        prediction("fn", "no_violation"),
        prediction("tn", "no_violation"),
        {"video_id": "failed-positive", "status": "failed", "error_code": "timeout"},
        {
            "video_id": "failed-negative",
            "status": "failed",
            "error_code": "SECRET-DESCRIPTION",
        },
    ]
    report = aggregate(preds, gold, inventory(gold))
    assert [
        report["conditional_completed"][key] for key in ("TP", "FP", "FN", "TN")
    ] == [1, 1, 1, 1]
    assert report["conditional_completed"]["recall"] == 0.5
    assert report["end_to_end"]["recall"] == pytest.approx(1 / 3)
    assert report["end_to_end"]["accuracy_including_failed"] == pytest.approx(1 / 3)
    assert report["end_to_end"]["total_positive_misses"] == 2
    assert (
        report["completion"]["failed_positive"]
        == report["completion"]["failed_negative"]
        == 1
    )
    assert report["completion"]["completion_rate"] == pytest.approx(2 / 3)
    assert report["completion"]["failure_error_counts"] == {
        "timeout": 1,
        "unspecified_or_other": 1,
    }
    assert "SECRET" not in json.dumps(report)


def test_missing_positive_not_excluded():
    gold = [positive("missing")]
    report = aggregate([], gold, inventory(gold))
    assert report["completion"]["failed_positive"] == 1
    assert report["completion"]["status_counts"] == {"missing": 1}
    assert report["end_to_end"]["recall"] == 0
    assert report["point_recall_by_tolerance_sec"]["5"]["unmatched_labels"] == 1


def test_duplicate_content_keeps_files_and_conflicting_points():
    gold = [positive("a", 1), positive("b", 2), negative("c")]
    inv = inventory(gold, ["a" * 64, "a" * 64, "b" * 64])
    inv["duplicate_groups"] = [{"annotation_conflict": True}]
    preds = [
        prediction("a", "violation"),
        prediction("b", "no_violation"),
        prediction("c", "no_violation"),
    ]
    report = aggregate(preds, gold, inv)
    assert report["scope"]["selected_file_count"] == 3
    assert report["scope"]["selected_unique_content_count"] == 2
    assert (
        report["conditional_completed"]["TP"] == 1
        and report["conditional_completed"]["FN"] == 1
    )
    assert report["duplicates"]["selected_binary_conflict_groups"] == 0
    assert report["duplicates"]["selected_full_annotation_conflict_groups"] == 1
    assert report["duplicates"]["inventory_declared_annotation_conflict_groups"] == 1
    gold[1] = negative("b")
    report = aggregate(preds, gold, inv)
    assert report["duplicates"]["selected_binary_conflict_groups"] == 1
    assert report["scope"]["selected_file_count"] == 3


def test_exact_matching_independent_of_tolerant_assignment():
    gold = [positive("v")]
    gold[0]["events"] = [{"timecode_sec": 1}, {"timecode_sec": 2}]
    preds = [prediction("v", "violation")]
    preds[0]["events"] = [{"timecode_sec": 0}, {"timecode_sec": 1}]
    report = aggregate(preds, gold, inventory(gold))
    exact = report["point_recall_by_tolerance_sec"]["0"]
    tolerant = report["point_recall_by_tolerance_sec"]["1"]
    assert exact["matched"] == 1 and exact["recall_of_known_labels"] == 0.5
    assert tolerant["matched"] == 2 and tolerant["recall_of_known_labels"] == 1
    assert tolerant["exhaustive_labels"]["precision"] is None


def test_smoke_subset_does_not_silently_claim_full_inventory():
    full = [positive("a"), negative("b")]
    report = aggregate([prediction("a", "violation")], full[:1], inventory(full))
    assert report["scope"]["selected_file_count"] == 1
    assert report["scope"]["inventory_file_count"] == 2
    assert report["scope"]["inventory_files_not_selected"] == 1


def test_stale_extra_predictions_or_bad_inventory_rejected():
    gold = [positive("a")]
    with pytest.raises(ValueError):
        aggregate([prediction("other", "violation")], gold, inventory(gold))
    inv = inventory(gold)
    inv["file_count"] = 2
    with pytest.raises(ValueError):
        aggregate([], gold, inv)


def test_cli_aggregate_has_no_identifiers_paths_or_point_values(tmp_path, capsys):
    gold = [positive("SECRET-VIDEO", 123.456)]
    preds = [
        {"video_id": "SECRET-VIDEO", "status": "failed", "error_code": "SECRET-ACTION"}
    ]
    arguments = []
    for name, value in [
        ("predictions", preds),
        ("gold", gold),
        ("inventory", inventory(gold)),
    ]:
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value))
        arguments.extend(["--" + name, str(path)])
    assert main(arguments) == 0
    captured = capsys.readouterr()
    assert "SECRET" not in captured.out and "123.456" not in captured.out
    assert str(tmp_path) not in captured.out and captured.err == ""
    assert json.loads(captured.out)["completion"]["failed_positive"] == 1
    (tmp_path / "gold.json").write_text("{SECRET")
    assert main(arguments) == 1
    assert capsys.readouterr().out.strip() == '{"status": "error"}'
