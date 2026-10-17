"""HTTP service. Start independently from the one model-owning worker."""

import asyncio
import hmac
import shutil
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from starlette.staticfiles import StaticFiles

from backend.config import Settings
from backend.contracts import JobAccepted, JobResponse
from backend.service import JobService
from backend.store import Store


class RequestEnvelope:
    """Authenticate before multipart parsing; bound streamed bytes and arrival time."""

    def __init__(self, app, settings: Settings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        arrival = time.time()
        scope.setdefault("state", {})["arrival_at"] = arrival
        path = scope["path"]
        headers = {name.lower(): value for name, value in scope["headers"]}
        if path.startswith("/api/v1/"):
            expected = self.settings.api_key.get_secret_value().encode("utf-8")
            provided = headers.get(b"x-api-key", b"")
            if not expected:
                return await JSONResponse({"detail": "Jury API key is not configured"}, 503)(scope, receive, send)
            if not hmac.compare_digest(expected, provided):
                return await JSONResponse({"detail": "Invalid API key"}, 401)(scope, receive, send)

        upload = scope["method"] == "POST" and path.rstrip("/") in ("/api/v1/analyze", "/ui-api/analyze")
        if not upload:
            return await self.app(scope, receive, send)
        # Allow multipart headers/boundaries in addition to the exact file limit.
        body_limit = self.settings.max_upload_bytes + 1024 * 1024
        length = headers.get(b"content-length")
        if length is not None:
            try:
                size = int(length)
                if size < 0:
                    raise ValueError
            except ValueError:
                return await JSONResponse({"detail": "Invalid Content-Length"}, 400)(scope, receive, send)
            if size > body_limit:
                return await JSONResponse({"detail": "Upload exceeds size limit"}, 413)(scope, receive, send)
        received = 0
        deadline = arrival + self.settings.deadline_seconds

        async def limited_receive():
            nonlocal received
            remaining = deadline - time.time()
            if remaining <= 0:
                raise HTTPException(408, "Upload deadline exceeded")
            try:
                message = await asyncio.wait_for(receive(), remaining)
            except TimeoutError:
                raise HTTPException(408, "Upload deadline exceeded") from None
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > body_limit:
                    raise HTTPException(413, "Upload exceeds size limit")
            return message

        await self.app(scope, limited_receive, send)


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings()
    store = Store(config.db_path, max_attempts=config.max_attempts)
    service = JobService(config, store)

    @asynccontextmanager
    async def lifespan(application):
        config.upload_dir.mkdir(parents=True, exist_ok=True)
        store.initialize()
        yield

    application = FastAPI(title="ForteBank Video Analysis", version="0.1.0", lifespan=lifespan)
    application.state.settings = config
    application.state.store = store
    application.add_middleware(RequestEnvelope, settings=config)
    key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def jury_key(key: Annotated[str | None, Depends(key_header)]):
        expected = config.api_key.get_secret_value()
        if not expected:
            raise HTTPException(503, "Jury API key is not configured")
        if key is None or not hmac.compare_digest(key.encode("utf-8"), expected.encode("utf-8")):
            raise HTTPException(401, "Invalid API key")

    @application.post("/api/v1/analyze", response_model=JobAccepted, dependencies=[Depends(jury_key)])
    @application.post("/ui-api/analyze", response_model=JobAccepted)
    async def analyze(request: Request, file: Annotated[UploadFile, File()]):
        return await service.upload(file, request.state.arrival_at)

    @application.get("/api/v1/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(jury_key)])
    @application.get("/ui-api/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str):
        return service.poll(job_id)

    @application.get("/health/live", include_in_schema=False)
    def live():
        return {"status": "ok"}

    @application.get("/health/ready", include_in_schema=False)
    def ready():
        try:
            worker = store.health()
            db_ok = True
        except Exception:
            worker, db_ok = {}, False
        binaries_ok = all(shutil.which(binary) is not None for binary in (config.ffmpeg_bin, config.ffprobe_bin))
        heartbeat_at = worker.get("heartbeat_at")
        worker_ok = isinstance(heartbeat_at, (float, int)) and time.time() - heartbeat_at < 10
        model_ready = worker.get("model_ready") is True
        checks = {"database": db_ok, "video_tools": binaries_ok, "plugins_configured": config.plugins_configured,
                  "worker": worker_ok, "model_ready": model_ready}
        is_ready = all(checks.values())
        return JSONResponse({"status": "ready" if is_ready else "not_ready", "checks": checks}, 200 if is_ready else 503)

    if (config.frontend_dist / "index.html").is_file():
        application.mount("/", StaticFiles(directory=config.frontend_dist, html=True), name="frontend")
    return application


app = create_app()
