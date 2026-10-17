"""Strict, local-only observation and rule interchange contracts."""

from dataclasses import dataclass, field, asdict
from typing import Literal
import json
import math

State = Literal["observed", "not_observed", "unclear"]
STATES = {"observed", "not_observed", "unclear"}


def strict_json_loads(value):
    """Reject ambiguous duplicate keys and non-standard NaN/Infinity JSON."""

    def object_pairs(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = item
        return result

    def invalid_constant(value):
        raise ValueError("non-finite JSON constant")

    return json.loads(
        value, object_pairs_hook=object_pairs, parse_constant=invalid_constant
    )


def number(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number, not bool")
    return value


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def states(mapping, name):
    if not isinstance(mapping, dict):
        raise ValueError(f"{name} must be an object")
    for key, value in mapping.items():
        text(key, name)
        if not isinstance(value, str) or value not in STATES:
            raise ValueError(f"{name}.{key}: invalid state")


def names(values, name):
    if (
        not isinstance(values, (list, tuple))
        or any(not isinstance(x, str) or not x.strip() for x in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} must contain unique nonempty strings")


@dataclass(frozen=True)
class EvidenceRef:
    frame_id: str
    pts_sec: float
    source: str

    def __post_init__(self):
        text(self.frame_id, "frame_id")
        text(self.source, "source")
        if number(self.pts_sec, "pts_sec") < 0:
            raise ValueError("negative PTS")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    start_sec: float
    end_sec: float
    main_sec: float
    evidence: tuple[EvidenceRef, ...]
    predicates: dict[str, State] = field(default_factory=dict)
    visibility: dict[str, State] = field(default_factory=dict)
    source: str = "local_observer"

    def __post_init__(self):
        text(self.candidate_id, "candidate_id")
        text(self.source, "source")
        for key in ("start_sec", "end_sec", "main_sec"):
            number(getattr(self, key), key)
        if not 0 <= self.start_sec <= self.main_sec <= self.end_sec:
            raise ValueError("invalid candidate interval/anchor")
        states(self.predicates, "predicates")
        states(self.visibility, "visibility")
        if not isinstance(self.evidence, (list, tuple)) or any(
            not isinstance(x, EvidenceRef) for x in self.evidence
        ):
            raise ValueError("invalid evidence")
        if any(not self.start_sec <= x.pts_sec <= self.end_sec for x in self.evidence):
            raise ValueError("evidence outside candidate interval")
        if len({x.frame_id for x in self.evidence}) != len(self.evidence):
            raise ValueError("duplicate frame ID in candidate")


@dataclass(frozen=True)
class Observation:
    video_id: str
    duration_sec: float
    analysis_status: Literal["complete", "partial", "failed"]
    coverage_complete: bool
    candidates: tuple[Candidate, ...]

    def __post_init__(self):
        text(self.video_id, "video_id")
        if number(self.duration_sec, "duration_sec") <= 0:
            raise ValueError("duration must be positive")
        if self.analysis_status not in ("complete", "partial", "failed"):
            raise ValueError("invalid analysis_status")
        if type(self.coverage_complete) is not bool:
            raise ValueError("coverage_complete must be bool")
        if not isinstance(self.candidates, (list, tuple)) or any(
            not isinstance(c, Candidate) for c in self.candidates
        ):
            raise ValueError("invalid candidates")
        if len({c.candidate_id for c in self.candidates}) != len(self.candidates):
            raise ValueError("duplicate candidate ID")
        refs = {}
        for c in self.candidates:
            if c.end_sec > self.duration_sec:
                raise ValueError("candidate outside video")
            for ref in c.evidence:
                if ref.frame_id in refs and refs[ref.frame_id] != ref:
                    raise ValueError("conflicting frame identity")
                refs[ref.frame_id] = ref

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("observation must be a JSON object")
        data = dict(data)
        if not isinstance(data.get("candidates"), list):
            raise ValueError("candidates must be a JSON array")
        if any(not isinstance(c, dict) for c in data["candidates"]):
            raise ValueError("candidate must be a JSON object")
        for candidate in data["candidates"]:
            if not isinstance(candidate.get("evidence"), list):
                raise ValueError("candidate evidence must be a JSON array")
            if any(not isinstance(ref, dict) for ref in candidate["evidence"]):
                raise ValueError("evidence reference must be a JSON object")
        data["candidates"] = tuple(
            Candidate(
                **{**c, "evidence": tuple(EvidenceRef(**r) for r in c["evidence"])}
            )
            for c in data["candidates"]
        )
        return cls(**data)

    @classmethod
    def from_json(cls, value):
        return cls.from_dict(strict_json_loads(value))

    def to_json(self):
        return json.dumps(asdict(self), allow_nan=False, sort_keys=True)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    type: str
    title: str
    source_document: str
    source_paragraph: str
    source_url: str
    applicability: tuple[str, ...]
    required: tuple[str, ...]
    exceptions: tuple[str, ...]
    visibility_required: tuple[str, ...]
    description_template: str

    def __post_init__(self):
        for key in (
            "rule_id",
            "type",
            "title",
            "source_document",
            "source_paragraph",
            "source_url",
            "description_template",
        ):
            text(getattr(self, key), key)
        for key in ("applicability", "required", "exceptions", "visibility_required"):
            names(getattr(self, key), key)
            object.__setattr__(self, key, tuple(getattr(self, key)))
        if not self.required:
            raise ValueError("rule needs violation predicates")
        if set(self.exceptions) & (set(self.required) | set(self.applicability)):
            raise ValueError("contradictory rule predicates")
        # A literal description avoids arbitrary template evaluation and unsupported facts.
        if "{" in self.description_template or "}" in self.description_template:
            raise ValueError("description must be literal text")

    @classmethod
    def from_json(cls, value):
        return cls(**strict_json_loads(value))

    def to_json(self):
        return json.dumps(asdict(self), allow_nan=False, sort_keys=True)
