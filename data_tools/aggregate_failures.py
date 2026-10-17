"""Aggregate existing local HTTP evaluation outputs; never read media.

python -m data_tools.aggregate_failures --predictions local_data/eval/predictions.json \
  --gold local_data/eval/gold_subset.json --inventory local_data/labels/inventory.json

Optional --output writes a private aggregate JSON. Stdout contains aggregate
counts/metrics only: no paths, video IDs, timestamps, descriptions or evidence.
File-weighted metrics preserve all selected files, including duplicate content.
Conflicting duplicate labels are counted for review and never silently resolved.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from forte_decision.cli import write_result
from forte_decision.contracts import strict_json_loads
from forte_decision.evaluation import evaluate, match_events, ratios

STATUS_KEYS = {"complete", "failed", "incomplete", "partial"}
ERROR_KEYS = {
    "service_failed",
    "timeout",
    "http_error",
    "invalid_result_or_io",
    "analysis_incomplete",
    "deadline_exceeded",
    "analysis_unavailable",
    "invalid_analysis_or_video",
    "analysis_error",
    "model_crashed",
}


def _complete(prediction):
    return (
        prediction is not None
        and prediction.get("status") == "complete"
        and prediction.get("verdict") in ("violation", "no_violation")
    )


def _inventory_index(inventory):
    if not isinstance(inventory, dict) or not isinstance(
        inventory.get("records"), list
    ):
        raise ValueError("invalid inventory")
    result, groups = {}, defaultdict(list)
    for record in inventory["records"]:
        if not isinstance(record, dict):
            raise ValueError("invalid inventory record")
        video_id, digest = record.get("video_id"), record.get("sha256")
        if not isinstance(video_id, str) or not video_id or video_id in result:
            raise ValueError("invalid or duplicate inventory video identity")
        if (
            not isinstance(digest, str)
            or re.fullmatch("[0-9a-fA-F]{64}", digest) is None
        ):
            raise ValueError("invalid inventory digest")
        result[video_id] = record
        groups[digest.lower()].append(video_id)
    if inventory.get("file_count") != len(result) or inventory.get(
        "unique_content_count"
    ) != len(groups):
        raise ValueError("inventory counts disagree with records")
    return result, groups


def aggregate(predictions, gold, inventory):
    """Return only privacy-safe aggregate counts, scopes and metrics."""
    evaluated = evaluate(predictions, gold)
    records, groups = _inventory_index(inventory)
    labels = {row["video_id"]: row for row in gold}
    preds = {row["video_id"]: row for row in predictions}
    if set(labels) - set(records):
        raise ValueError("gold identity missing in inventory")
    if set(preds) - set(labels):
        raise ValueError("prediction outside selected gold scope")
    status, errors = Counter(), Counter()
    failed_positive = failed_negative = failed_unknown = 0
    for video_id, label in labels.items():
        prediction = preds.get(video_id)
        raw_status = prediction.get("status") if prediction else None
        status[
            (
                "missing"
                if prediction is None
                else (
                    raw_status
                    if isinstance(raw_status, str) and raw_status in STATUS_KEYS
                    else "other"
                )
            )
        ] += 1
        if not _complete(prediction):
            verdict = label.get("verdict")
            failed_positive += verdict == "violation"
            failed_negative += verdict == "no_violation"
            failed_unknown += verdict not in ("violation", "no_violation")
            code = prediction.get("error_code") if prediction else None
            errors[
                (
                    code
                    if isinstance(code, str) and code in ERROR_KEYS
                    else (
                        "missing_prediction"
                        if prediction is None
                        else "unspecified_or_other"
                    )
                )
            ] += 1
    video = evaluated["video"]
    positives = sum(row.get("verdict") == "violation" for row in gold)
    negatives = sum(row.get("verdict") == "no_violation" for row in gold)
    known = positives + negatives
    point = {}
    for tolerance in (0, 1, 3, 5):
        matched = unmatched = extra_ignored = 0
        exhaustive_tp = exhaustive_fp = exhaustive_fn = 0
        for video_id, label in labels.items():
            if "events" not in label:
                continue
            prediction = preds.get(video_id)
            predicted_events = prediction["events"] if _complete(prediction) else []
            pairs = match_events(predicted_events, label["events"], tolerance)
            tp = len(pairs)
            fn = len(label["events"]) - tp
            fp = len(predicted_events) - tp
            matched += tp
            unmatched += fn
            if label.get("exhaustive", False):
                exhaustive_tp += tp
                exhaustive_fp += fp
                exhaustive_fn += fn
            else:
                extra_ignored += fp
        point[str(tolerance)] = {
            "known_label_count": matched + unmatched,
            "matched": matched,
            "unmatched_labels": unmatched,
            "recall_of_known_labels": (
                matched / (matched + unmatched) if matched + unmatched else None
            ),
            "ignored_extra_predictions_nonexhaustive": extra_ignored,
            "exhaustive_labels": {
                "TP": exhaustive_tp,
                "FP": exhaustive_fp,
                "FN": exhaustive_fn,
                **ratios(exhaustive_tp, exhaustive_fp, exhaustive_fn),
            },
        }
    duplicate_groups = [ids for ids in groups.values() if len(ids) > 1]
    selected_groups = [[vid for vid in ids if vid in labels] for ids in groups.values()]
    selected_duplicates = [ids for ids in selected_groups if len(ids) > 1]
    binary_conflicts = annotation_conflicts = 0
    for ids in selected_duplicates:
        verdicts = {
            labels[vid].get("verdict")
            for vid in ids
            if labels[vid].get("verdict") is not None
        }
        binary_conflicts += len(verdicts) > 1
        # Preserve timing/type/exhaustiveness distinctions; do not choose a representative.
        signatures = {
            json.dumps(
                {
                    "verdict": labels[vid].get("verdict"),
                    "events": labels[vid].get("events"),
                    "exhaustive": labels[vid].get("exhaustive", False),
                },
                sort_keys=True,
            )
            for vid in ids
        }
        annotation_conflicts += len(signatures) > 1
    selected_sha = {records[vid]["sha256"].lower() for vid in labels}
    declared_conflicts = sum(
        group.get("annotation_conflict") is True
        for group in inventory.get("duplicate_groups", [])
    )
    return {
        "scope": {
            "inventory_file_count": len(records),
            "inventory_unique_content_count": len(groups),
            "selected_file_count": len(labels),
            "selected_unique_content_count": len(selected_sha),
            "inventory_files_not_selected": len(records) - len(labels),
            "gold_positive_count": positives,
            "gold_negative_count": negatives,
            "unknown_binary_gold_count": len(labels) - known,
            "policy": "Metrics weight selected files individually; content duplicates are retained. No unique-content verdict metric or automatic conflict resolution.",
        },
        "completion": {
            "completed": video["completed"],
            "failures": video["failures"],
            "completion_rate": video["completion_rate"],
            "failed_positive": failed_positive,
            "failed_negative": failed_negative,
            "failed_unknown_gold": failed_unknown,
            "status_counts": dict(status),
            "failure_error_counts": dict(errors),
        },
        "conditional_completed": {
            key: video[key]
            for key in ("TP", "FP", "FN", "TN", "precision", "recall", "f1")
        },
        "end_to_end": {
            "TP": video["TP"],
            "FP": video["FP"],
            "TN": video["TN"],
            "decision_false_negatives": video["FN"],
            "failed_positive_misses": failed_positive,
            "total_positive_misses": video["FN"] + failed_positive,
            "recall": video["TP"] / positives if positives else None,
            "accuracy_including_failed": (
                (video["TP"] + video["TN"]) / known if known else None
            ),
            "policy": "Failed positives count as misses; failed negatives are failures, never TN or fabricated FP.",
        },
        "point_recall_by_tolerance_sec": point,
        "duplicates": {
            "inventory_duplicate_groups": len(duplicate_groups),
            "inventory_duplicate_extra_files": len(records) - len(groups),
            "inventory_declared_annotation_conflict_groups": declared_conflicts,
            "selected_duplicate_groups": len(selected_duplicates),
            "selected_binary_conflict_groups": binary_conflicts,
            "selected_full_annotation_conflict_groups": annotation_conflicts,
        },
        "timing_policy": "Tolerance 0 means exact numeric seconds; 1/3/5 use independent maximum-cardinality one-to-one matching. No individual timestamps or errors are emitted.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        read = lambda path: strict_json_loads(path.read_text(encoding="utf-8"))
        report = aggregate(
            read(args.predictions), read(args.gold), read(args.inventory)
        )
        if args.output:
            if args.output.resolve() in {
                path.resolve() for path in (args.predictions, args.gold, args.inventory)
            }:
                raise ValueError("output cannot replace inputs")
            write_result(args.output, json.dumps(report, allow_nan=False))
    except Exception:
        print(json.dumps({"status": "error"}))
        return 1
    print(json.dumps(report, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
