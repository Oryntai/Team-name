"""Shared upload/polling logic used by both the jury and the public UI."""

import os
import time
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.config import Settings
from backend.contracts import JobAccepted, JobDone, JobFailed, JobProcessing
from backend.store import QueueFull, Store


def _sync_directory(path: Path) -> None:
    # Windows has no directory fsync; the file itself is flushed on both platforms.
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class JobService:
    def __init__(self, settings: Settings, store: Store):
        self.settings = settings
        self.store = store

    async def upload(self, file: UploadFile, arrival_at: float) -> JobAccepted:
        deadline_at = arrival_at + self.settings.deadline_seconds
        job_id = uuid.uuid4().hex
        destination = self.settings.upload_dir / (job_id + ".mov")
        temporary = destination.with_suffix(".part")
        enqueued = False
        try:
            if Path(file.filename or "").suffix.lower() != ".mov":
                raise HTTPException(415, "Expected a .mov video")
            total = 0
            with temporary.open("xb") as output:
                while chunk := await file.read(1024 * 1024):
                    if time.time() >= deadline_at:
                        raise HTTPException(408, "Upload deadline exceeded")
                    total += len(chunk)
                    if total > self.settings.max_upload_bytes:
                        raise HTTPException(413, "Video exceeds configured size limit")
                    await run_in_threadpool(output.write, chunk)
                if total == 0:
                    raise HTTPException(400, "Empty upload")
                await run_in_threadpool(output.flush)
                await run_in_threadpool(os.fsync, output.fileno())
            if time.time() >= deadline_at:
                raise HTTPException(408, "Upload deadline exceeded")
            await run_in_threadpool(os.replace, temporary, destination)
            await run_in_threadpool(_sync_directory, destination.parent)
            try:
                await run_in_threadpool(
                    self.store.enqueue, job_id, destination, arrival_at, deadline_at,
                    self.settings.profile, self.settings.max_pending_jobs,
                )
            except QueueFull:
                raise HTTPException(503, "Analysis queue is full", headers={"Retry-After": "5"}) from None
            enqueued = True
            return JobAccepted(job_id=job_id)
        finally:
            await file.close()
            if not enqueued:
                temporary.unlink(missing_ok=True)
                destination.unlink(missing_ok=True)

    def poll(self, job_id: str):
        # Expire even when the worker is down: stale jobs may never appear done.
        self.store.expire()
        job = self.store.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        if job["status"] == "done":
            return JobDone.model_validate(job["result"])
        if job["status"] == "failed":
            return JobFailed(job_id=job_id)
        return JobProcessing(job_id=job_id)
