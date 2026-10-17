"""Conservative deterministic rule evaluation; no inference or probabilities."""

from dataclasses import dataclass, asdict
import hashlib
import json
from .contracts import Observation, Rule, EvidenceRef, number


@dataclass(frozen=True)
class Event:
    event_id: str
    video_id: str
    rule_id: str
    type: str
    timecode_sec: float
    start_sec: float
    end_sec: float
    description: str
    evidence: tuple[EvidenceRef, ...]
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class DecisionResult:
    video_id: str
    status: str
    verdict: str | None
    events: tuple[Event, ...]
    diagnostics: tuple[dict, ...]

    def to_json(self):
        return json.dumps(asdict(self), allow_nan=False, sort_keys=True)


def decide(
    observation: Observation,
    rules: list[Rule],
    *,
    max_anchor_delta_sec: float = 1.0,
) -> DecisionResult:
    """Apply rules and a tunable interval/anchor duplicate heuristic."""
    number(max_anchor_delta_sec, "max_anchor_delta_sec")
    if max_anchor_delta_sec < 0:
        raise ValueError("max_anchor_delta_sec must be nonnegative")
    # Revalidate mutable nested mappings at trust boundary.
    if not rules:
        raise ValueError("rule catalog must not be empty")
    rules = [Rule.from_json(rule.to_json()) for rule in rules]
    observation = Observation.from_json(observation.to_json())
    if len({r.rule_id for r in rules}) != len(rules):
        raise ValueError("duplicate rule ID")
    diagnostics, accepted = [], []
    for rule in sorted(rules, key=lambda r: r.rule_id):
        eligible = []
        for c in sorted(
            observation.candidates,
            key=lambda c: (c.start_sec, c.end_sec, c.candidate_id),
        ):
            checks = [
                (key, c.predicates.get(key, "unclear"), "observed")
                for key in (*rule.applicability, *rule.required)
            ]
            checks += [
                (key, c.predicates.get(key, "unclear"), "not_observed")
                for key in rule.exceptions
            ]
            checks += [
                ("visibility:" + key, c.visibility.get(key, "unclear"), "observed")
                for key in rule.visibility_required
            ]
            reasons = [
                f"{key}={actual}; requires {wanted}"
                for key, actual, wanted in checks
                if actual != wanted
            ]
            if not c.evidence:
                reasons.append("missing evidence")
            diagnostics.append(
                {
                    "rule_id": rule.rule_id,
                    "candidate_id": c.candidate_id,
                    "supported": not reasons,
                    "reasons": reasons,
                }
            )
            if not reasons:
                eligible.append(c)
        groups = []
        # Sort by anchor so extending a group cannot bridge through a chain of
        # nearby anchors beyond the configured total anchor span.
        for candidate in sorted(
            eligible, key=lambda c: (c.main_sec, c.start_sec, c.end_sec, c.candidate_id)
        ):
            destination = None
            for group in groups:
                anchors = [member.main_sec for member in group] + [candidate.main_sec]
                anchor_span = max(anchors) - min(anchors)
                overlaps = any(
                    candidate.start_sec <= member.end_sec
                    and member.start_sec <= candidate.end_sec
                    for member in group
                )
                if overlaps and anchor_span <= max_anchor_delta_sec:
                    destination = group
                    break
            if destination is None:
                groups.append([candidate])
            else:
                destination.append(candidate)
        for group in groups:
            # Anchor and refs always originate in accepted input. Deterministic tie breaks.
            anchor = min(group, key=lambda c: (c.main_sec, c.candidate_id))
            refs = {r.frame_id: r for c in group for r in c.evidence}
            evidence = tuple(
                sorted(
                    refs.values(),
                    key=lambda r: (
                        abs(r.pts_sec - anchor.main_sec),
                        r.pts_sec,
                        r.frame_id,
                    ),
                )[:3]
            )
            ids = tuple(sorted(c.candidate_id for c in group))
            identity = json.dumps(
                [observation.video_id, rule.rule_id, ids], separators=(",", ":")
            )
            accepted.append(
                Event(
                    hashlib.sha256(identity.encode()).hexdigest()[:24],
                    observation.video_id,
                    rule.rule_id,
                    rule.type,
                    anchor.main_sec,
                    min(c.start_sec for c in group),
                    max(c.end_sec for c in group),
                    f"{rule.description_template} Source: {rule.source_document}, {rule.source_paragraph}. {rule.source_url}",
                    evidence,
                    ids,
                )
            )
    complete = (
        observation.analysis_status == "complete" and observation.coverage_complete
    )
    events = tuple(
        sorted(accepted, key=lambda e: (e.timecode_sec, e.rule_id, e.event_id))
    )
    return DecisionResult(
        observation.video_id,
        "complete" if complete else "incomplete",
        ("violation" if events else "no_violation") if complete else None,
        events,
        tuple(diagnostics),
    )
