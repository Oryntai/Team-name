import asyncio
import time

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app import create_app
from backend.config import Settings
from backend.contracts import ObservationBatch


@pytest.fixture
def api(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", api_key="test-only-key", max_pending_jobs=2)
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, app.state.store, settings


def test_key_required_before_upload_and_poll(api):
    client, _, _ = api
    for headers in ({}, {"X-API-Key": "wrong"}):
        assert client.post("/api/v1/analyze", content=b"broken", headers=headers).status_code == 401
        assert client.get("/api/v1/jobs/no-such-job", headers=headers).status_code == 401
    assert client.get("/api/v1/jobs/absent", headers={"X-API-Key": "test-only-key"}).status_code == 404


def test_upload_is_persisted_before_accepted_and_ui_is_public(api):
    client, store, settings = api
    response = client.post("/ui-api/analyze", files={"file": ("../../input.MOV", b"synthetic-invalid-video", "video/quicktime")})
    assert response.status_code == 200
    assert set(response.json()) == {"job_id"}
    job_id = response.json()["job_id"]
    job = store.get(job_id)
    assert job["status"] == "queued"
    assert (settings.upload_dir / (job_id + ".mov")).read_bytes() == b"synthetic-invalid-video"
    assert job["deadline_at"] - job["arrival_at"] == 300
    assert client.get(f"/ui-api/jobs/{job_id}").json() == {"job_id": job_id, "status": "processing"}
    assert client.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": "test-only-key"}).json()["status"] == "processing"


def test_rejection_does_not_leave_files(api):
    client, _, settings = api
    assert client.post("/ui-api/analyze", files={"file": ("bad.exe", b"123")}).status_code == 415
    assert client.post("/ui-api/analyze", files={"file": ("empty.mov", b"")}).status_code == 400
    for _ in range(2):
        assert client.post("/ui-api/analyze", files={"file": ("test.mov", b"123")}).status_code == 200
    assert client.post("/ui-api/analyze", files={"file": ("extra.mov", b"123")}).status_code == 503
    assert len(list(settings.upload_dir.iterdir())) == 2


def test_size_limit_checked_before_body(api):
    client, _, settings = api
    response = client.post("/ui-api/analyze", headers={"Content-Length": str(settings.max_upload_bytes + 2 * 1024 * 1024)})
    assert response.status_code == 413


def test_polling_expires_job_without_worker_and_hides_error(api):
    client, store, settings = api
    now = time.time()
    store.enqueue("expired", settings.upload_dir / "expired.mov", now - 301, now - 1, "standard", 2)
    assert client.get("/ui-api/jobs/expired").json() == {"job_id": "expired", "status": "failed"}


def test_production_rejects_development_plugins_and_missing_key(tmp_path):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="production", data_dir=tmp_path, api_key="long-enough-test-key")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="production", data_dir=tmp_path, analyzer_factory="ml.plugin:create_analyzer", rules_factory="rules.plugin:create_rules")


def test_invalid_settings_do_not_print_secret():
    secret = "sensitive-test-key-never-in-traceback"
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None, environment="production", api_key=secret)
    assert secret not in str(error.value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="production", api_key="replace-with-a-long-random-secret",
                 analyzer_factory="ml.plugin:create_analyzer", rules_factory="rules.plugin:create_rules")


def test_nonfinite_diagnostics_are_rejected_before_storage():
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            ObservationBatch(complete=True, events=[], diagnostics={"score": value})


def test_health_is_honest_without_worker_or_ml(api):
    client, _, _ = api
    assert client.get("/health/live").json() == {"status": "ok"}
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["plugins_configured"] is False


def test_streamed_body_limit_without_content_length(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path, max_upload_mb=29)
    app = create_app(settings)

    async def request():
        chunks = iter([
            b'--test\r\nContent-Disposition: form-data; name="file"; filename="large.mov"\r\n\r\n',
            *([b"x" * (1024 * 1024)] * 32), b"\r\n--test--\r\n",
        ])
        responses = []

        async def receive():
            chunk = next(chunks, None)
            return {"type": "http.request", "body": chunk or b"", "more_body": chunk is not None}

        async def send(message):
            responses.append(message)

        await app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                   "method": "POST", "scheme": "http", "path": "/ui-api/analyze", "raw_path": b"/ui-api/analyze",
                   "query_string": b"", "root_path": "", "server": ("test", 80), "client": ("test", 1234),
                   "headers": [(b"content-type", b"multipart/form-data; boundary=test")]}, receive, send)
        return next(message["status"] for message in responses if message["type"] == "http.response.start")

    assert asyncio.run(request()) == 413


def test_slow_upload_uses_request_arrival_deadline(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path, deadline_seconds=0.03)
    app = create_app(settings)

    async def request():
        responses = []

        async def receive():
            await asyncio.sleep(1)
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            responses.append(message)

        await app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                   "method": "POST", "scheme": "http", "path": "/ui-api/analyze", "raw_path": b"/ui-api/analyze",
                   "query_string": b"", "root_path": "", "server": ("test", 80), "client": ("test", 1234),
                   "headers": [(b"content-type", b"multipart/form-data; boundary=test")]}, receive, send)
        return next(message["status"] for message in responses if message["type"] == "http.response.start")

    assert asyncio.run(request()) == 408



def test_public_key_is_one_stable_key_that_opens_jury_routes(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path, api_key="test-only-key")
    with TestClient(create_app(settings)) as client:
        first = client.get("/ui-api/key").json()["api_key"]
        assert first.startswith("fk_") and len(first) > 40
        assert {client.get("/ui-api/key").json()["api_key"] for _ in range(3)} == {first}
        assert client.get("/api/v1/jobs/absent", headers={"X-API-Key": first}).status_code == 404
        assert client.get("/api/v1/jobs/absent", headers={"X-API-Key": first + "x"}).status_code == 401
        assert client.get("/api/v1/jobs/absent", headers={"X-API-Key": "test-only-key"}).status_code == 404
    # A restarted service hands out the same key.
    with TestClient(create_app(settings)) as client:
        assert client.get("/ui-api/key").json()["api_key"] == first


def test_public_key_can_be_switched_off(tmp_path):
    on = Settings(_env_file=None, data_dir=tmp_path, api_key="test-only-key")
    with TestClient(create_app(on)) as client:
        public = client.get("/ui-api/key").json()["api_key"]
    off = Settings(_env_file=None, data_dir=tmp_path, api_key="test-only-key", public_api_keys=False)
    with TestClient(create_app(off)) as client:
        assert client.get("/ui-api/key").status_code == 404
        assert client.get("/api/v1/jobs/absent", headers={"X-API-Key": public}).status_code == 401
        assert client.get("/api/v1/jobs/absent", headers={"X-API-Key": "test-only-key"}).status_code == 404
    unconfigured = Settings(_env_file=None, data_dir=tmp_path / "other", public_api_keys=False)
    with TestClient(create_app(unconfigured)) as client:
        assert client.get("/api/v1/jobs/absent", headers={"X-API-Key": "x"}).status_code == 503


def test_other_formats_are_converted_to_mov(api, tmp_path):
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    client, _, settings = api
    for ext in ("mp4", "webm"):
        src = tmp_path / f"clip.{ext}"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1",
                        *(["-c:v", "libvpx"] if ext == "webm" else ["-c:v", "libx264", "-pix_fmt", "yuv420p"]),
                        str(src)], check=True)
        r = client.post("/ui-api/analyze", files={"file": (src.name, src.read_bytes())})
        assert r.status_code == 200, r.text
        job = r.json()["job_id"]
        converted = settings.upload_dir / f"{job}.mov"
        assert converted.exists() and converted.stat().st_size > 0
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=format_name", "-of", "csv=p=0",
                                str(converted)], capture_output=True, text=True, check=True).stdout
        assert "mov" in probe
    assert not list(settings.upload_dir.glob("*.part"))


def test_double_slash_after_base_url_still_reaches_the_api(api):
    client, _, _ = api
    r = client.post("http://testserver//api/v1/analyze", headers={"X-API-Key": "test-only-key"}, files={"file": ("a.mov", b"123")})
    assert r.status_code == 200, r.text
    job = r.json()["job_id"]
    assert client.get(f"http://testserver//api/v1/jobs/{job}", headers={"X-API-Key": "test-only-key"}).json()["status"] == "processing"
    assert client.get("http://testserver//api/v1/jobs/nope").status_code == 401  # ключ по-прежнему обязателен
