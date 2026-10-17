"""Real upload/queue/spawn/video/rules/evaluator integration on synthetic media.

The forwarding transport only connects httpx to the real ASGI app; it invents no
HTTP payload. The analyzer is test-only and returns generic synthetic predicates.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import threading

import httpx
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from backend.app import create_app
from backend.config import Settings
from backend.contracts import ObservationBatch, ObservedEvent
from backend.tests.video_fixtures import ffmpeg_tools, make_video
from backend.worker import Supervisor
from data_tools.evaluate_service import run_evaluation


class SyntheticIntegrationAnalyzer:
    """Importable by the actual spawned model process; no real model or dataset."""

    def analyze(self, context, video):
        if context.profile == "integration-clean":
            return ObservationBatch(complete=True, events=[])
        metadata = video.probe()
        main = metadata.frame_pts[2]
        predicates = {
            "cashier_left_workstation": (
                "unclear" if context.profile == "integration-unknown" else "observed"
            ),
            "departure_before_client_service_completed": "observed",
        }
        return ObservationBatch(
            complete=True,
            events=[
                ObservedEvent(
                    t_start=metadata.frame_pts[0],
                    t_end=metadata.frame_pts[-1],
                    t_main=main,
                    frame_pts=[metadata.frame_pts[0], main, metadata.frame_pts[-1]],
                    observations=predicates,
                    visibility={"cashier": "partial"},
                    sources=["vlm"],
                )
            ],
        )


def create_analyzer():
    return SyntheticIntegrationAnalyzer()


@pytest.mark.parametrize("profile", ["positive", "clean", "unknown"])
def test_real_service_http_evaluator_and_production_rules(
    tmp_path, monkeypatch, profile
):
    ffmpeg, ffprobe = ffmpeg_tools()
    # Isolate configuration from any user runtime catalogs/environment.
    catalog = Path(__file__).resolve().parents[1] / "rules" / "cash_receipt.json"
    monkeypatch.setenv("FORTE_RULE_CATALOGS", json.dumps([str(catalog)]))
    settings = Settings(
        _env_file=None,
        environment="development",
        data_dir=tmp_path / "service",
        api_key="synthetic-integration-key",
        deadline_seconds=45,
        ffmpeg_bin=ffmpeg,
        ffprobe_bin=ffprobe,
        analyzer_factory="data_tools.test_service_integration:create_analyzer",
        rules_factory="forte_decision.backend_adapter:create_rules",
        profile="integration-" + profile,
    )
    videos = tmp_path / "videos"
    videos.mkdir()
    make_video(videos / "synthetic.mov", ffmpeg)
    gold = [
        {
            "video_id": "synthetic.mov",
            "verdict": "no_violation" if profile == "clean" else "violation",
            "exhaustive": False,
            "events": [] if profile == "clean" else [{"timecode_sec": 0.2}],
        }
    ]
    app = create_app(settings)
    stop = threading.Event()
    worker_errors = []
    requests = []
    with TestClient(app) as service, Supervisor(settings) as supervisor:

        def drive_worker():
            try:
                while not stop.is_set():
                    supervisor.step()
                    stop.wait(0.01)
            except BaseException as error:
                worker_errors.append(type(error).__name__)
                stop.set()

        worker = threading.Thread(target=drive_worker, daemon=True)
        worker.start()

        def forward(request):
            # Forward actual multipart upload and polling through backend routes.
            requests.append((request.method, request.url.path))
            response = service.request(
                request.method,
                request.url.path,
                content=request.read(),
                headers=dict(request.headers),
            )
            return httpx.Response(
                response.status_code,
                headers=dict(response.headers),
                content=response.content,
            )

        try:
            report = run_evaluation(
                base_url="http://127.0.0.1:8000",
                videos=videos,
                gold=gold,
                output_dir=tmp_path / "evaluation",
                api_key="synthetic-integration-key",
                deadline_sec=40,
                poll_interval=0.02,
                transport=httpx.MockTransport(forward),
                durations={"synthetic.mov": 1.0},
            )
        finally:
            stop.set()
            worker.join(timeout=5)
        assert not worker.is_alive() and not worker_errors
    assert requests[0] == ("POST", "/api/v1/analyze")
    assert any(
        method == "GET" and path.startswith("/api/v1/jobs/")
        for method, path in requests
    )
    payload_files = list((tmp_path / "evaluation" / "payloads").glob("*.json"))
    assert len(payload_files) == 1
    response = json.loads(payload_files[0].read_text())["payload"]
    predictions = json.loads((tmp_path / "evaluation" / "predictions.json").read_text())
    if profile == "unknown":
        assert response["status"] == "failed"
        assert predictions[0]["status"] == "failed"
        assert report["video"]["failures"] == 1
        assert report["video"]["TN"] == 0
        assert report["service"]["error_counts"] == {"service_failed": 1}
        assert report["service"]["end_to_end_recall"] == 0
        assert report["service"]["accuracy_including_failed"] == 0
        assert report["events"]["1"]["all_known_labels"]["unmatched_labels"] == 1
    elif profile == "clean":
        assert response["status"] == "done" and response["verdict"] == "no_violation"
        assert response["violations"] == []
        assert report["video"]["TN"] == 1 and report["video"]["failures"] == 0
        assert report["service"]["accuracy_including_failed"] == 1
    else:
        assert response["status"] == "done" and response["verdict"] == "violation"
        assert report["video"]["TP"] == 1 and report["video"]["failures"] == 0
        assert report["service"]["end_to_end_recall"] == 1
        assert report["events"]["1"]["all_known_labels"]["recall_of_known_labels"] == 1
        assert len(response["violations"]) == 1
        event = response["violations"][0]
        assert (
            event["description"].count("https://adilet.zan.kz/rus/docs/V1900019680")
            == 1
        )
        assert 1 <= len(event["evidence_frames"]) <= 3
        for frame in event["evidence_frames"]:
            assert 0 <= frame["timestamp_sec"] <= 1
            with Image.open(
                io.BytesIO(base64.b64decode(frame["image_base64"], validate=True))
            ) as image:
                image.load()
                assert image.format == "JPEG" and image.size == (96, 64)
