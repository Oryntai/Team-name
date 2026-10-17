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


RECHECK_FLAGS = (
    "points_with_nearby_rechecked_candidate",
    "core_all_observed_any_rule",
    "core_unclear_any_rule",
    "core_explicitly_false_any_rule",
    "applicability_explicit_false_when_core_observed",
    "exception_observed_when_core_observed",
    "exception_unclear_when_core_observed",
    "engine_supported_nearby",
    "engine_rejected_nearby",
    "engine_unresolved_nearby",
)


def _point_recheck_flags(
    point, candidates, candidate_diagnostics, engine_diagnostics, rules
):
    """Describe values/outcomes; never infer an event category or a new verdict."""
    nearby = {
        candidate.candidate_id: candidate
        for candidate in candidates
        if candidate.start_sec - 1 <= point <= candidate.end_sec + 1
        and candidate_diagnostics.get(candidate.candidate_id, {}).get("rechecked")
        is True
    }
    flags = set()
    if nearby:
        flags.add("points_with_nearby_rechecked_candidate")
    catalog = {rule.rule_id: rule for rule in rules}
    for row in engine_diagnostics:
        candidate = nearby.get(row.get("candidate_id"))
        if candidate is None or row.get("outcome") == "not_targeted":
            continue
        outcome = row.get("outcome")
        if outcome in ("supported", "rejected", "unresolved"):
            flags.add("engine_" + outcome + "_nearby")
        rule = catalog.get(row.get("rule_id"))
        if rule is None:
            continue
        core = [candidate.predicates.get(name, "unclear") for name in rule.required]
        if "unclear" in core:
            flags.add("core_unclear_any_rule")
        if "not_observed" in core:
            flags.add("core_explicitly_false_any_rule")
        if not all(value == "observed" for value in core):
            continue
        flags.add("core_all_observed_any_rule")
        if any(
            candidate.predicates.get(name, "unclear") == "not_observed"
            for name in rule.applicability
        ):
            flags.add("applicability_explicit_false_when_core_observed")
        exceptions = [
            candidate.predicates.get(name, "unclear") for name in rule.exceptions
        ]
        if "observed" in exceptions:
            flags.add("exception_observed_when_core_observed")
        if "unclear" in exceptions:
            flags.add("exception_unclear_when_core_observed")
    return flags


def aggregate_stages(evaluation_dir, audit_dir, gold, rules):
    evaluate([], gold)  # Validate labels, identities, and point timestamps.
    counts = Counter()
    covered = Counter()
    available = Counter()
    transitions = Counter()
    binary = Counter()
    recheck_flags = Counter()
    audit_quality = Counter()
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
        audit_quality["analysis_" + observation.analysis_status + "_files"] += 1
        audit_quality[
            (
                "coverage_complete_files"
                if observation.coverage_complete
                else "coverage_incomplete_files"
            )
        ] += 1
        for key in ("invalid_screen_windows", "empty_screen_windows"):
            value = audit["diagnostics"].get(key)
            if type(value) is int and value >= 0:
                audit_quality[key + "_diagnostic_available_files"] += 1
                audit_quality[key + "_total"] += value
                audit_quality[key + "_nonzero_files"] += value > 0
            else:
                audit_quality[key + "_diagnostic_missing_files"] += 1
        for point in points:
            time = point["timecode_sec"]
            recheck_flags.update(
                _point_recheck_flags(
                    time,
                    candidates,
                    diagnostics,
                    result.diagnostics,
                    rules,
                )
            )
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
        "rechecked_point_diagnostics": {
            "available_positive_points": counts["audit_available_positive_points"],
            "nonexclusive_point_counts": {
                key: recheck_flags[key] for key in RECHECK_FLAGS
            },
            "scope": "Each generic flag counts a known positive point at most once across nearby rechecked candidates and actually evaluated rules; flags are nonexclusive and may coexist. Core/context/exception values describe candidate confirmation, not VLM false negatives or legal/category correctness. Context/exception flags require all core predicates observed for that candidate/rule. Actual outcomes come from engine diagnostics; not_targeted rules are excluded.",
        },
        "audit_quality_positive_files": {
            **{
                key: audit_quality[key]
                for key in (
                    "analysis_complete_files",
                    "analysis_partial_files",
                    "analysis_failed_files",
                    "coverage_complete_files",
                    "coverage_incomplete_files",
                )
            },
            **{
                name + suffix: audit_quality[name + suffix]
                for name in ("invalid_screen_windows", "empty_screen_windows")
                for suffix in (
                    "_diagnostic_available_files",
                    "_diagnostic_missing_files",
                    "_total",
                    "_nonzero_files",
                )
            },
            "scope": "Valid available audits for positive files only; absent numeric window diagnostics remain missing, not zero. Audit status/coverage is separate from candidate-value and final-response outcomes.",
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
