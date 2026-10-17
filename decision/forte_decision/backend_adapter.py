"""Production rules factory for the canonical backend contracts.

No media is loaded here. Backend VideoIO/worker owns duration, PTS and JPEG
verification. Object visibility is support only, never proof of named predicates.
"""

import hashlib
import json
import os
from pathlib import Path
import time

from backend.contracts import (
    AnalysisContext,
    IncompleteAnalysis,
    ObservationBatch,
    RuleEvent,
)

from .cli import load_rules
from .contracts import Candidate, EvidenceRef, strict_json_loads
from .engine import evaluate_candidates


def _identity(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()[:24]


class BackendRules:
    def __init__(self, catalog_paths=None, *, mode="review", max_anchor_delta_sec=1.0):
        if catalog_paths is None:
            configured = os.environ.get("FORTE_RULE_CATALOGS")
            if configured is None:
                rules_dir = Path(__file__).resolve().parents[2] / "rules"
                catalog_paths = [rules_dir / "cash_checks.json"]
            else:
                catalog_paths = strict_json_loads(configured)
                if (
                    not isinstance(catalog_paths, list)
                    or not catalog_paths
                    or any(
                        not isinstance(path, str) or not path for path in catalog_paths
                    )
                ):
                    raise ValueError(
                        "FORTE_RULE_CATALOGS must be a nonempty JSON array of local paths"
                    )
        self.rules = load_rules(catalog_paths)
        self.mode = mode
        self.max_anchor_delta_sec = max_anchor_delta_sec
        # Validate configuration at model startup, before processing any jobs.
        evaluate_candidates(
            "configuration-check",
            (),
            self.rules,
            analysis_complete=True,
            mode=mode,
            max_anchor_delta_sec=max_anchor_delta_sec,
        )

    def evaluate(
        self, observations: ObservationBatch, context: AnalysisContext
    ) -> list[RuleEvent]:
        if not isinstance(observations, ObservationBatch) or not isinstance(
            context, AnalysisContext
        ):
            raise ValueError("expected canonical ObservationBatch and AnalysisContext")
        observations = ObservationBatch.model_validate(observations.model_dump())
        context = AnalysisContext.model_validate(context.model_dump())
        if time.time() >= context.deadline_at:
            raise TimeoutError("analysis deadline exceeded")
        if not observations.complete:
            raise IncompleteAnalysis("analysis coverage incomplete")
        candidates = {}
        notes: dict[str, str] = {}  # candidate_id -> что видно на этом видео (от анализатора)
        named_visibility = {
            name for rule in self.rules for name in rule.visibility_required
        }
        for event in observations.events:
            # IDs depend on actual supplied job/PTS, never invented video duration.
            evidence = tuple(
                EvidenceRef(
                    frame_id=_identity([context.job_id, float(pts)]),
                    pts_sec=pts,
                    source="backend.frame_pts",
                )
                for pts in sorted(set(event.frame_pts))
            )
            visibility = {}
            for name in named_visibility:
                direct = event.observations.get(name)
                prefixed = event.observations.get("visibility:" + name)
                if direct is not None and prefixed is not None and direct != prefixed:
                    raise ValueError("conflicting explicit visibility predicates")
                value = direct if direct is not None else prefixed
                if value is not None:
                    visibility[name] = value
            # Object-level visibility only establishes evidence usability support,
            # never semantic coverage (e.g. all possible client positions).
            if not event.visibility:
                visibility["evidence_usable"] = "unclear"
            elif any(
                value in ("clear", "partial") for value in event.visibility.values()
            ):
                visibility["evidence_usable"] = "observed"
            else:
                visibility["evidence_usable"] = "not_observed"
            predicates = {
                (
                    name.removeprefix("visibility:")
                    if name.startswith("visibility:")
                    else name
                ): value
                for name, value in event.observations.items()
            }
            identity = {
                "job_id": context.job_id,
                "start": event.t_start,
                "end": event.t_end,
                "main": event.t_main,
                "pts": sorted(set(event.frame_pts)),
                "predicates": predicates,
                "object_visibility": event.visibility,
                "sources": sorted(set(event.sources)),
            }
            candidate_id = _identity(identity)
            if event.note:
                notes[candidate_id] = event.note
            candidates[candidate_id] = Candidate(
                candidate_id=candidate_id,
                start_sec=event.t_start,
                end_sec=event.t_end,
                main_sec=event.t_main,
                evidence=evidence,
                predicates=predicates,
                visibility=visibility,
                source="+".join(sorted(set(event.sources))),
            )
        result = evaluate_candidates(
            context.job_id,
            tuple(candidates.values()),
            self.rules,
            analysis_complete=True,
            mode=self.mode,
            max_anchor_delta_sec=self.max_anchor_delta_sec,
        )
        if result.status != "complete":
            raise IncompleteAnalysis("unresolved observations prevent a clean negative")
        catalog = {rule.rule_id: rule for rule in self.rules}
        output = []
        for event in result.events:
            rule = catalog[event.rule_id]
            output.append(
                RuleEvent(
                    t_start=event.start_sec,
                    t_end=event.end_sec,
                    t_main=event.timecode_sec,
                    type=event.type,
                    description=_with_seen(rule.description_template,
                                           [notes[c] for c in event.candidate_ids if c in notes]),
                    rule_id=rule.rule_id,
                    rule_title=f"{rule.source_document} — {rule.title}",
                    rule_section=rule.source_paragraph,
                    rule_url=rule.source_url,
                    frame_pts=[ref.pts_sec for ref in event.evidence],
                )
            )
        return output


def _with_seen(template: str, notes: list[str]) -> str:
    """Шаблон правила + что именно видно на этом видео (момент, кадры, зона, ответ модели)."""
    seen = "; ".join(dict.fromkeys(n.strip() for n in notes if n.strip()))
    return f"{template}\nЧто видно: {seen}." if seen else template


def create_rules():
    """Backend factory spec: forte_decision.backend_adapter:create_rules."""
    return BackendRules()
