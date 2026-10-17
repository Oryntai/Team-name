"""Stable external and internal contracts. See contracts/README.md."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Protocol, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from backend.video_io import VideoIO

Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JobAccepted(StrictModel):
    job_id: str


class IssuedApiKey(StrictModel):
    api_key: str


class JobProcessing(JobAccepted):
    status: Literal["processing"] = "processing"


class JobFailed(JobAccepted):
    status: Literal["failed"] = "failed"


class EvidenceFrame(StrictModel):
    timestamp_sec: Seconds
    image_base64: str = Field(min_length=1)


class Violation(StrictModel):
    type: str = Field(min_length=1, max_length=200)
    timecode_sec: Seconds
    description: str = Field(min_length=1, max_length=8000)
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)


class JobDone(JobAccepted):
    status: Literal["done"] = "done"
    verdict: Literal["violation", "no_violation"]
    violations: list[Violation]

    @model_validator(mode="after")
    def consistent_verdict(self) -> "JobDone":
        if (self.verdict == "violation") != bool(self.violations):
            raise ValueError("verdict must agree with the violations list")
        return self


JobResponse = JobProcessing | JobFailed | JobDone


class AnalysisContext(StrictModel):
    job_id: str
    video_path: Path
    deadline_at: float = Field(allow_inf_nan=False, description="Absolute UTC Unix seconds, including upload and queue time")
    profile: str = "standard"
    analysis_budget_sec: float | None = Field(
        default=None, gt=0, allow_inf_nan=False,
        description="Seconds the analyzer may spend when other jobs are queued; None means up to deadline_at")


class EventSpan(StrictModel):
    t_start: Seconds
    t_end: Seconds
    t_main: Seconds

    @model_validator(mode="after")
    def ordered_times(self) -> "EventSpan":
        if not self.t_start <= self.t_main <= self.t_end:
            raise ValueError("Require t_start <= t_main <= t_end")
        return self


class ObservedEvent(EventSpan):
    frame_pts: list[Seconds] = Field(min_length=1, max_length=256)
    observations: dict[str, Literal["observed", "not_observed", "unclear"]]
    visibility: dict[str, Literal["clear", "partial", "unusable"]] = Field(default_factory=dict)
    sources: list[Literal["vlm", "motion", "detector"]] = Field(min_length=1)


class ObservationBatch(StrictModel):
    complete: bool
    events: list[ObservedEvent]
    coverage: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = Field(
        default=None, description="Share of the video the analyzer covered; required for a partial clean result")
    diagnostics: dict[str, str | Annotated[float, Field(allow_inf_nan=False)] | int | bool | None] = Field(default_factory=dict)


class RuleEvent(EventSpan):
    type: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=6000)
    rule_id: str = Field(min_length=1, max_length=200)
    rule_title: str = Field(min_length=1, max_length=500)
    rule_section: str = Field(min_length=1, max_length=200)
    rule_url: str = Field(pattern=r"^https?://[^\s]+$", max_length=2000)
    frame_pts: list[Seconds] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def evidence_in_span(self) -> "RuleEvent":
        if any(not self.t_start <= t <= self.t_end for t in self.frame_pts):
            raise ValueError("Evidence timestamps must lie in the event span")
        return self


class Analyzer(Protocol):
    def analyze(self, context: AnalysisContext, video: "VideoIO") -> ObservationBatch: ...


class RuleEngine(Protocol):
    def evaluate(self, observations: ObservationBatch, context: AnalysisContext) -> list[RuleEvent]: ...


class AnalysisUnavailable(RuntimeError):
    """No real analyzer configured. This must produce failed, never a clean verdict."""


class IncompleteAnalysis(RuntimeError):
    """Analyzer did not cover the required input before its deadline."""
