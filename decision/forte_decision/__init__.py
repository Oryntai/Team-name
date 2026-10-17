"""Public exports are lazy so the evaluation CLI never imports the engine."""

from .contracts import Observation, Candidate, EvidenceRef, Rule

__all__ = [
    "Observation",
    "Candidate",
    "EvidenceRef",
    "Rule",
    "decide",
    "DecisionResult",
    "Event",
]


def __getattr__(name):
    if name in ("decide", "DecisionResult", "Event"):
        from . import engine

        return getattr(engine, name)
    raise AttributeError(name)
