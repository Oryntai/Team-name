"""Upload -> durable queue -> separate process -> public polling, synthetic ML only."""

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import time

from fastapi.testclient import TestClient
from PIL import Image

from backend.app import create_app
from backend.config import Settings
from backend.tests.video_fixtures import ffmpeg_tools, make_video
from backend.worker import Supervisor


def settings_for(tmp_path: Path, profile: str = "clean") -> Settings:
    ffmpeg, ffprobe = ffmpeg_tools()
    return Settings(
        _env_file=None, data_dir=tmp_path / "data", api_key="synthetic-test-key",
        ffmpeg_bin=ffmpeg, ffprobe_bin=ffprobe, deadline_seconds=60,
        analyzer_factory="backend.tests.test_worker_store:create_synthetic_analyzer",
        rules_factory="backend.tests.test_worker_store:create_synthetic_rules",
        profile=profile, max_pending_jobs=5,
    )


def finish_jobs(client, supervisor, job_ids, timeout=60):
    pending = set(job_ids)
    finished = {}
    until = time.monotonic() + timeout
    while pending and time.monotonic() < until:
        supervisor.step()
        for job_id in list(pending):
            response = client.get(f"/ui-api/jobs/{job_id}")
            assert response.status_code == 200
            payload = response.json()
            if payload["status"] != "processing":
                finished[job_id] = (payload, time.perf_counter())
                pending.remove(job_id)
        time.sleep(0.02)
    assert not pending, "Synthetic queue did not terminate"
    return finished


def test_five_concurrent_uploads_include_queue_time(tmp_path):
    settings = settings_for(tmp_path)
    content = make_video(tmp_path / "synthetic.mov", settings.ffmpeg_bin).read_bytes()
    app = create_app(settings)
    with TestClient(app) as client:
        def upload(index):
            started = time.perf_counter()
            response = client.post("/ui-api/analyze", files={"file": (f"synthetic-{index}.mov", content, "video/quicktime")})
            assert response.status_code == 200
            return response.json()["job_id"], started

        with ThreadPoolExecutor(max_workers=5) as pool:
            jobs = dict(pool.map(upload, range(5)))
        assert len(jobs) == 5
        assert client.post("/ui-api/analyze", files={"file": ("extra.mov", content)}).status_code == 503
        with Supervisor(settings) as supervisor:
            results = finish_jobs(client, supervisor, jobs)
        durations = []
        for job_id, (payload, ended) in results.items():
            assert payload == {"job_id": job_id, "status": "done", "verdict": "no_violation", "violations": []}
            durations.append(round(ended - jobs[job_id], 3))
            assert ended - jobs[job_id] < settings.deadline_seconds
        print("SYNTHETIC five-upload elapsed seconds including queue: " + json.dumps(sorted(durations)))


def test_api_evidence_and_corrupt_mov(tmp_path):
    settings = settings_for(tmp_path, "multi")
    content = make_video(tmp_path / "synthetic.mov", settings.ffmpeg_bin).read_bytes()
    app = create_app(settings)
    with TestClient(app) as client:
        valid = client.post("/api/v1/analyze", headers={"X-API-Key": "synthetic-test-key"},
                            files={"file": ("synthetic.mov", content)}).json()["job_id"]
        corrupt = client.post("/ui-api/analyze", files={"file": ("broken.mov", b"not a video")}).json()["job_id"]
        with Supervisor(settings) as supervisor:
            results = finish_jobs(client, supervisor, [valid, corrupt])
        result = results[valid][0]
        assert set(result) == {"job_id", "status", "verdict", "violations"}
        assert result["verdict"] == "violation" and len(result["violations"]) == 2
        for event in result["violations"]:
            assert 1 <= len(event["evidence_frames"]) <= 3
            assert "https://" in event["description"]
            for frame in event["evidence_frames"]:
                assert not frame["image_base64"].startswith("data:")
                image_bytes = base64.b64decode(frame["image_base64"], validate=True)
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.load()
                    assert image.format == "JPEG"
                    assert image.size == (96, 64)
        assert results[corrupt][0] == {"job_id": corrupt, "status": "failed"}


def test_default_adapter_never_invents_clean_result(tmp_path):
    ffmpeg, ffprobe = ffmpeg_tools()
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", ffmpeg_bin=ffmpeg, ffprobe_bin=ffprobe)
    content = make_video(tmp_path / "synthetic.mov", ffmpeg).read_bytes()
    app = create_app(settings)
    with TestClient(app) as client:
        job_id = client.post("/ui-api/analyze", files={"file": ("synthetic.mov", content)}).json()["job_id"]
        with Supervisor(settings) as supervisor:
            result = finish_jobs(client, supervisor, [job_id])[job_id][0]
        assert result == {"job_id": job_id, "status": "failed"}
        assert app.state.store.get(job_id)["error_code"] == "analysis_unavailable"
