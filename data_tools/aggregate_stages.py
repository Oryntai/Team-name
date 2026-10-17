"""Pure local temporal-coverage funnel from saved service outputs and vision audits.

python -m data_tools.aggregate_stages --evaluation-dir local_data/eval \
  --audit-dir local_data/audit --gold local_data/eval/gold_subset.json \
  --rules rules/cash_receipt.json

Outputs aggregate counts only. Candidate interval coverage is NOT correctness:
no category gold exists. Missing/crashed audits are unavailable, never model
misses. Final point matching is separately one-to-one at +/-1 second.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from backend.contracts import JobDone
from forte_decision.cli import load_rules
from forte_decision.contracts import Observation, number, strict_json_loads
from forte_decision.engine import decide
from forte_decision.evaluation import evaluate, match_events

STAGES = ("raw", "pooled", "selected", "rechecked", "rule_supported")


def read(path):
    return strict_json_loads(Path(path).read_text(encoding="utf-8"))


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest() + ".json"


def _intervals(values):
    result = []
    if not isinstance(values, list):
        raise ValueError("intervals must be arrays")
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("invalid interval")
        start, end = value
        if number(start, "start") < 0 or number(end, "end") < start:
            raise ValueError("invalid interval times")
        result.append((start, end))
    return result


def aggregate_stages(evaluation_dir, audit_dir, gold, rules):
    evaluate([], gold)  # Validate labels, identities, and point timestamps.
    counts = Counter()
    covered = Counter()
    available = Counter()
    transitions = Counter()
    binary = Counter()
    final_matched = final_labels = 0
    for label in gold:
        counts["files"] += 1
        positive = label.get("verdict") == "violation"
        points = label.get("events", []) if positive else []
        if positive:
            counts["positive_files"] += 1
            counts["known_positive_points"] += len(points)
        done = None
        job_id = None
        try:
            wrapper = read(Path(evaluation_dir) / "payloads" / _hash(label["video_id"]))
            if wrapper.get("video_id") != label["video_id"]:
                raise ValueError("payload identity mismatch")
            job_id = wrapper.get("job_id")
            payload = wrapper.get("payload")
            if (
                not wrapper.get("error_code")
                and isinstance(payload, dict)
                and payload.get("status") == "done"
            ):
                done = JobDone.model_validate(payload, strict=True)
                if done.job_id != job_id:
                    raise ValueError("job identity mismatch")
            elif isinstance(payload, dict) and payload.get("status") == "failed":
                counts["service_failed_files"] += 1
            else:
                counts["final_unavailable_files"] += 1
        except FileNotFoundError:
            counts["payload_missing_files"] += 1
        except (ValueError, TypeError, KeyError, OSError, AttributeError):
            counts["payload_invalid_files"] += 1
        if label.get("verdict") in ("violation", "no_violation"):
            if done is None:
                binary["failed_positive" if positive else "failed_negative"] += 1
            else:
                predicted = done.verdict == "violation"
                binary[
                    (
                        "TP"
                        if positive and predicted
                        else "FN" if positive else "FP" if predicted else "TN"
                    )
                ] += 1
        predicted_points = (
            [{"timecode_sec": event.timecode_sec} for event in done.violations]
            if done
            else []
        )
        final_matched += len(match_events(predicted_points, points, 1))
        final_labels += len(points)
        if not positive:
            continue
        if not isinstance(job_id, str) or not job_id:
            counts["audit_missing_positive_files"] += 1
            counts["audit_missing_positive_points"] += len(points)
            continue
        try:
            audit = read(Path(audit_dir) / _hash(job_id))
            observation = Observation.from_dict(audit["observation"])
            if observation.video_id != job_id:
                raise ValueError("audit identity mismatch")
            result = decide(observation, rules, mode="review")
            supported = {
                row["candidate_id"]
                for row in result.diagnostics
                if row.get("supported")
            }
            diagnostics = audit["diagnostics"]["candidates"]
            candidates = observation.candidates
            stages = {
                "raw": _intervals(audit["raw_intervals"]),
                "pooled": _intervals(audit["pooled_intervals"]),
                "selected": [(c.start_sec, c.end_sec) for c in candidates],
                "rechecked": [
                    (c.start_sec, c.end_sec)
                    for c in candidates
                    if diagnostics.get(c.candidate_id, {}).get("rechecked") is True
                ],
                "rule_supported": [
                    (c.start_sec, c.end_sec)
                    for c in candidates
                    if c.candidate_id in supported
                ],
            }
        except FileNotFoundError:
            counts["audit_missing_positive_files"] += 1
            counts["audit_missing_positive_points"] += len(points)
            continue
        except (ValueError, TypeError, KeyError, OSError, AttributeError):
            counts["audit_invalid_positive_files"] += 1
            counts["audit_invalid_positive_points"] += len(points)
            continue
        counts["audit_available_positive_files"] += 1
        counts["audit_available_positive_points"] += len(points)
        for point in points:
            time = point["timecode_sec"]
            hits = {
                stage: any(start - 1 <= time <= end + 1 for start, end in intervals)
                for stage, intervals in stages.items()
            }
            for stage in STAGES:
                available[stage] += 1
                covered[stage] += hits[stage]
            for previous, next_stage in zip(STAGES, STAGES[1:]):
                if hits[previous] and not hits[next_stage]:
                    transitions[previous + "_to_" + next_stage + "_coverage_lost"] += 1
                if not hits[previous] and hits[next_stage]:
                    transitions[
                        previous + "_to_" + next_stage + "_coverage_gained"
                    ] += 1
    return {
        "counts": {
            key: counts[key]
            for key in (
                "files",
                "positive_files",
                "known_positive_points",
                "service_failed_files",
                "final_unavailable_files",
                "payload_missing_files",
                "payload_invalid_files",
                "audit_available_positive_files",
                "audit_available_positive_points",
                "audit_missing_positive_files",
                "audit_missing_positive_points",
                "audit_invalid_positive_files",
                "audit_invalid_positive_points",
            )
        },
        "temporal_coverage_within_1_sec": {
            stage: {
                "available_positive_points": available[stage],
                "covered_positive_points": covered[stage],
                "coverage_rate": (
                    covered[stage] / available[stage] if available[stage] else None
                ),
            }
            for stage in STAGES
        },
        "adjacent_temporal_changes": dict(transitions),
        "final_binary": {
            key: binary[key]
            for key in ("TP", "FP", "FN", "TN", "failed_positive", "failed_negative")
        },
        "final_point_recall_within_1_sec": {
            "matched": final_matched,
            "known_positive_points": final_labels,
            "recall": final_matched / final_labels if final_labels else None,
        },
        "scope": "Coverage means a candidate interval falls within +/-1 second of a known positive point; it does not establish category or legal correctness. Rates condition on available valid audits. Missing/invalid audits are unavailable, not candidate misses. Final matching is separate and one-to-one; failed positives remain misses. Rules rerun locally in review mode; no causal attribution is claimed.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--rules", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    try:
        report = aggregate_stages(
            args.evaluation_dir, args.audit_dir, read(args.gold), load_rules(args.rules)
        )
    except Exception:
        print(json.dumps({"status": "error"}))
        return 1
    print(json.dumps(report, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
