"""Conservative deterministic rule evaluation; no inference or probabilities."""

from dataclasses import dataclass, asdict
import hashlib
import json
from .contracts import Observation, Candidate, Rule, EvidenceRef, number, text


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
    mode: str = "review",
) -> DecisionResult:
    """Validate video-wide metadata and evaluate in review (or strict audit) mode."""
    observation = Observation.from_json(observation.to_json())
    return evaluate_candidates(
        observation.video_id,
        observation.candidates,
        rules,
        analysis_complete=(
            observation.analysis_status == "complete" and observation.coverage_complete
        ),
        max_anchor_delta_sec=max_anchor_delta_sec,
        mode=mode,
    )


def evaluate_candidates(
    video_id: str,
    candidates: tuple[Candidate, ...],
    rules: list[Rule],
    *,
    analysis_complete: bool,
    max_anchor_delta_sec: float = 1.0,
    mode: str = "review",
) -> DecisionResult:
    """Shared internal evaluator; caller owns actual video duration validation.

    Review mode permits contextual/visibility uncertainty with strong observed
    core predicates. Unknown core or exceptions remain unresolved, never negative.
    """
    text(video_id, "video_id")
    if type(analysis_complete) is not bool:
        raise ValueError("analysis_complete must be bool")
    if mode not in ("review", "strict"):
        raise ValueError("invalid decision mode")
    candidates = tuple(
        Candidate(
            candidate_id=c.candidate_id,
            start_sec=c.start_sec,
            end_sec=c.end_sec,
            main_sec=c.main_sec,
            evidence=tuple(EvidenceRef(**asdict(r)) for r in c.evidence),
            predicates=dict(c.predicates),
            visibility=dict(c.visibility),
            source=c.source,
        )
        for c in candidates
    )
    if len({c.candidate_id for c in candidates}) != len(candidates):
        raise ValueError("duplicate candidate ID")
    refs = {}
    for candidate in candidates:
        for ref in candidate.evidence:
            if ref.frame_id in refs and refs[ref.frame_id] != ref:
                raise ValueError("conflicting frame identity")
            refs[ref.frame_id] = ref
    number(max_anchor_delta_sec, "max_anchor_delta_sec")
    if max_anchor_delta_sec < 0:
        raise ValueError("max_anchor_delta_sec must be nonnegative")
    # Revalidate mutable nested mappings at trust boundary.
    if not rules:
        raise ValueError("rule catalog must not be empty")
    rules = [Rule.from_json(rule.to_json()) for rule in rules]
    if len({r.rule_id for r in rules}) != len(rules):
        raise ValueError("duplicate rule ID")
    diagnostics, accepted = [], []
    unresolved = False
    known_predicates = {
        name
        for rule in rules
        for name in (
            *rule.applicability,
            *rule.required,
            *rule.exceptions,
            *rule.visibility_required,
        )
    }
    for candidate in candidates:
        if candidate.visibility.get("evidence_usable") == "not_observed":
            unresolved = True
            diagnostics.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "outcome": "unresolved",
                    "supported": False,
                    "reasons": ["all supplied object evidence is unusable"],
                }
            )
        unknown_positive = any(
            value == "observed" and name not in known_predicates
            for name, value in candidate.predicates.items()
        )
        if unknown_positive:
            unresolved = True
            diagnostics.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "outcome": "unresolved",
                    "supported": False,
                    "reasons": ["observed predicate has no catalog coverage"],
                }
            )
    core_names = {name for rule in rules for name in rule.required}
    for rule in sorted(rules, key=lambda r: r.rule_id):
        eligible = []
        for c in sorted(
            candidates,
            key=lambda c: (c.start_sec, c.end_sec, c.candidate_id),
        ):
            if c.visibility.get("evidence_usable") == "not_observed":
                continue
            mentioned_core = core_names.intersection(c.predicates)
            if mentioned_core and not set(rule.required).intersection(c.predicates):
                diagnostics.append(
                    {
                        "rule_id": rule.rule_id,
                        "candidate_id": c.candidate_id,
                        "supported": False,
                        "outcome": "not_targeted",
                        "reasons": ["candidate names another rule's core predicates"],
                        "caveats": [],
                    }
                )
                continue
            core = [(name, c.predicates.get(name, "unclear")) for name in rule.required]
            context = [
                (name, c.predicates.get(name, "unclear")) for name in rule.applicability
            ]
            exceptions = [
                (name, c.predicates.get(name, "unclear")) for name in rule.exceptions
            ]
            visibility = [
                (name, c.visibility.get(name, "unclear"))
                for name in rule.visibility_required
            ]
            rejected = any(
                value == "not_observed" for _, value in (*core, *context)
            ) or any(value == "observed" for _, value in exceptions)
            blockers = [
                f"{name}={value}" for name, value in core if value != "observed"
            ]
            blockers += [
                f"exception:{name}={value}"
                for name, value in exceptions
                if value != "not_observed"
            ]
            caveats = [
                f"context:{name}={value}"
                for name, value in context
                if value != "observed"
            ]
            caveats += [
                f"visibility:{name}={value}"
                for name, value in visibility
                if value != "observed"
            ]
            if c.visibility.get("evidence_usable") == "unclear":
                caveats.append("evidence_usable=unclear")
            if mode == "strict":
                # Missing generic support remains soft even in audit mode;
                # named visibility requirements remain the strict audit gate.
                blockers += [
                    item for item in caveats if item != "evidence_usable=unclear"
                ]
            if not c.evidence:
                blockers.append("missing evidence")
            supported = not rejected and not blockers
            outcome = (
                "supported" if supported else "rejected" if rejected else "unresolved"
            )
            reasons = blockers + (
                ["explicit contrary core/context or observed exception"]
                if rejected
                else []
            )
            diagnostics.append(
                {
                    "rule_id": rule.rule_id,
                    "candidate_id": c.candidate_id,
                    "supported": supported,
                    "outcome": outcome,
                    "reasons": reasons,
                    "caveats": caveats if supported else [],
                }
            )
            if outcome == "unresolved":
                unresolved = True
            if supported:
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
            identity = json.dumps([video_id, rule.rule_id, ids], separators=(",", ":"))
            accepted.append(
                Event(
                    hashlib.sha256(identity.encode()).hexdigest()[:24],
                    video_id,
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
    complete = analysis_complete and (bool(accepted) or not unresolved)
    events = tuple(
        sorted(accepted, key=lambda e: (e.timecode_sec, e.rule_id, e.event_id))
    )
    return DecisionResult(
        video_id,
        "complete" if complete else "incomplete",
        ("violation" if events else "no_violation") if complete else None,
        events,
        tuple(diagnostics),
    )
