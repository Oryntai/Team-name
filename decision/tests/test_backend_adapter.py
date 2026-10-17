"""Synthetic canonical backend tests; no user media or cloud inference."""

import json
from pathlib import Path
import time

import pytest

pytest.importorskip("backend.contracts")
from backend.contracts import (
    AnalysisContext,
    IncompleteAnalysis,
    ObservationBatch,
    ObservedEvent,
)
from forte_decision.backend_adapter import BackendRules, create_rules


@pytest.fixture(autouse=True)
def _seed_receipt_catalog(monkeypatch):
    # These tests pin the semantics of the seed cash-receipt catalog (applicability, visibility).
    catalog = Path(__file__).resolve().parents[2] / "rules" / "cash_receipt.json"
    monkeypatch.setenv("FORTE_RULE_CATALOGS", json.dumps([str(catalog)]))


@pytest.fixture
def context(tmp_path):
    return AnalysisContext(
        job_id="synthetic-job",
        video_path=tmp_path / "never-opened.mov",
        deadline_at=time.time() + 60,
    )


def batch(**observations):
    return ObservationBatch(
        complete=True,
        events=[
            ObservedEvent(
                t_start=1,
                t_end=5,
                t_main=3,
                frame_pts=[1, 3, 5],
                observations=observations,
                visibility={"cashier": "partial", "client": "clear"},
                sources=["vlm"],
            )
        ],
    )


def test_factory_strong_departure_core_unknown_visibility(context):
    rules = create_rules()
    observed = batch(
        cashier_left_workstation="observed",
        departure_before_client_service_completed="observed",
    )
    events = rules.evaluate(observed, context)
    assert (
        len(events) == 1 and events[0].rule_id == "KZ_NBK_231_P22_LEAVE_DURING_SERVICE"
    )
    catalog = {rule.rule_id: rule for rule in rules.rules}
    assert events[0].description == catalog[events[0].rule_id].description_template
    assert events[0].rule_url == catalog[events[0].rule_id].source_url
    assert events[0].frame_pts == [3, 1, 5]
    assert not context.video_path.exists()  # Duration/media never inferred/read.


@pytest.mark.parametrize(
    "observations",
    [
        {
            "cashier_left_workstation": "unclear",
            "departure_before_client_service_completed": "observed",
        },
        {"unexpected_new_action": "observed"},
        {"cashier_left_workstation": "unclear"},
        {},
    ],
)
def test_unresolved_never_negative(context, observations):
    with pytest.raises(IncompleteAnalysis):
        create_rules().evaluate(batch(**observations), context)


@pytest.mark.parametrize(
    "observations",
    [
        {
            "cashier_left_workstation": "not_observed",
            "departure_before_client_service_completed": "observed",
        },
        {
            "cashier_left_workstation": "observed",
            "departure_before_client_service_completed": "observed",
            "cash_receipt_operation_established": "not_observed",
        },
    ],
)
def test_explicit_contrary_core_or_context_negative(context, observations):
    assert create_rules().evaluate(batch(**observations), context) == []


def test_empty_complete_and_incomplete_batches(context):
    assert (
        create_rules().evaluate(ObservationBatch(complete=True, events=[]), context)
        == []
    )
    with pytest.raises(IncompleteAnalysis):
        create_rules().evaluate(ObservationBatch(complete=False, events=[]), context)


def test_known_positive_survives_other_unresolved(context):
    positive = batch(
        cashier_left_workstation="observed",
        departure_before_client_service_completed="observed",
    )
    unknown = batch(uncovered_action="observed").events[0]
    combined = ObservationBatch(complete=True, events=[*positive.events, unknown])
    assert len(create_rules().evaluate(combined, context)) == 1


def test_object_clear_does_not_prove_all_positions(context):
    # Strict mode cannot silently convert object clear to named observed predicate.
    observed = batch(
        cashier_left_workstation="observed",
        departure_before_client_service_completed="observed",
        bank_cashier_role_established="observed",
        cash_receipt_operation_established="observed",
    )
    observed.events[0].visibility = {
        "workstation_and_exit_visible": "clear",
        "service_sequence_around_departure_visible": "clear",
    }
    with pytest.raises(IncompleteAnalysis):
        BackendRules(mode="strict").evaluate(observed, context)
    observed.events[0].observations.update(
        workstation_and_exit_visible="observed",
        service_sequence_around_departure_visible="observed",
    )
    assert len(BackendRules(mode="strict").evaluate(observed, context)) == 1


def test_factory_config_duplicate_and_bad_types(context, monkeypatch):
    with pytest.raises(ValueError):
        create_rules().evaluate({"complete": True, "events": []}, context)
    monkeypatch.setenv("FORTE_RULE_CATALOGS", "[]")
    with pytest.raises(ValueError):
        create_rules()


def test_canonical_dedup_and_pts_stable(context):
    rules = create_rules()
    observed = batch(
        cashier_left_workstation="observed",
        departure_before_client_service_completed="observed",
    )
    single = rules.evaluate(observed, context)
    assert (
        rules.evaluate(
            ObservationBatch(complete=True, events=observed.events * 2), context
        )
        == single
    )
    observed.events[0].frame_pts = list(reversed(observed.events[0].frame_pts))
    assert rules.evaluate(observed, context) == single


def test_backend_guard_rejects_evidence_outside_span(context):
    observed = batch(
        cashier_left_workstation="observed",
        departure_before_client_service_completed="observed",
    )
    observed.events[0].frame_pts = [0]
    with pytest.raises(ValueError):
        create_rules().evaluate(observed, context)


@pytest.mark.parametrize("profile", ["positive", "clean", "unknown"])
def test_real_worker_synthetic_media(tmp_path, profile):
    """Exercise real local VideoIO/extraction with synthetic analyzer observations."""
    pytest.importorskip("PIL")
    from backend.tests.video_fixtures import ffmpeg_tools, make_video
    from backend.worker import _analyze_job

    ffmpeg, ffprobe = ffmpeg_tools()
    path = make_video(tmp_path / "synthetic.mov", ffmpeg)

    class SyntheticAnalyzer:
        def analyze(self, context, video):
            if profile == "clean":
                return ObservationBatch(complete=True, events=[])
            metadata = video.probe()
            pts = metadata.frame_pts[0]
            predicates = (
                {
                    "cashier_left_workstation": "observed",
                    "departure_before_client_service_completed": "observed",
                }
                if profile == "positive"
                else {"cashier_left_workstation": "unclear"}
            )
            return ObservationBatch(
                complete=True,
                events=[
                    ObservedEvent(
                        t_start=pts,
                        t_end=pts,
                        t_main=pts,
                        frame_pts=[pts],
                        observations=predicates,
                        visibility={"cashier": "partial"},
                        sources=["vlm"],
                    )
                ],
            )

    job = {
        "job_id": "synthetic-worker",
        "video_path": str(path),
        "deadline_at": time.time() + 60,
        "profile": profile,
    }
    config = {"ffmpeg_bin": ffmpeg, "ffprobe_bin": ffprobe}
    if profile == "unknown":
        with pytest.raises(IncompleteAnalysis):
            _analyze_job(job, config, SyntheticAnalyzer(), create_rules())
    else:
        result = _analyze_job(job, config, SyntheticAnalyzer(), create_rules())
        assert result["status"] == "done"
        assert result["verdict"] == (
            "violation" if profile == "positive" else "no_violation"
        )
        if profile == "positive":
            import base64
            import io
            from PIL import Image

            event = result["violations"][0]
            assert (
                event["description"].count("https://adilet.zan.kz/rus/docs/V1900019680")
                == 1
            )
            for frame in event["evidence_frames"]:
                with Image.open(
                    io.BytesIO(base64.b64decode(frame["image_base64"]))
                ) as image:
                    image.load()
                    assert image.format == "JPEG"


@pytest.mark.parametrize(
    "object_visibility",
    [
        {"cashier": "unusable"},
        {"cashier": "unusable", "client": "unusable"},
    ],
)
def test_all_unusable_is_unresolved(context, object_visibility):
    observed = batch(
        cashier_left_workstation="observed",
        departure_before_client_service_completed="observed",
    )
    observed.events[0].visibility = object_visibility
    with pytest.raises(IncompleteAnalysis):
        create_rules().evaluate(observed, context)
    observed.events[0].observations["cashier_left_workstation"] = "not_observed"
    with pytest.raises(IncompleteAnalysis):
        create_rules().evaluate(observed, context)


@pytest.mark.parametrize(
    "object_visibility",
    [
        {"cashier": "partial"},
        {"cashier": "clear", "client": "unusable"},
        {},
    ],
)
def test_soft_usable_or_unknown_support_keeps_strong_core(context, object_visibility):
    observed = batch(
        cashier_left_workstation="observed",
        departure_before_client_service_completed="observed",
    )
    observed.events[0].visibility = object_visibility
    assert len(create_rules().evaluate(observed, context)) == 1


def test_partial_batch_returns_proven_violation_but_never_negative(context):
    positive = batch(
        cashier_left_workstation="observed",
        departure_before_client_service_completed="observed",
    )
    partial = ObservationBatch(complete=False, events=positive.events)
    assert len(create_rules().evaluate(partial, context)) == 1
    contrary = batch(cashier_left_workstation="not_observed")
    with pytest.raises(IncompleteAnalysis):
        create_rules().evaluate(ObservationBatch(complete=False, events=contrary.events), context)

