"""Shared upload/polling logic used by both the jury and the public UI."""

import hashlib
import os
import subprocess
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


# Сервис анализирует .mov (как требует ТЗ). Видео других форматов из интерфейса переводим в .mov при загрузке:
# контейнеры семейства MP4 — перепаковкой без перекодирования, остальные — в H.264. Звук не берём.
REMUX = {".mp4", ".m4v", ".3gp"}
ENCODE = {".avi", ".mkv", ".webm", ".wmv", ".mpg", ".mpeg", ".flv", ".ts"}


def _to_mov(ffmpeg: str, source: Path, target: Path, suffix: str, timeout: float) -> bool:
    """source (любой поддерживаемый контейнер) -> target (.mov, только видеопоток). True — получилось."""
    tmp = target.with_suffix(".conv.mov")
    attempts = []
    if suffix in REMUX:
        attempts.append(["-c:v", "copy"])
    attempts.append(["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"])
    started = time.monotonic()
    for codec in attempts:
        left = timeout - (time.monotonic() - started)
        if left <= 1:
            break
        try:
            done = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(source), "-map", "0:v:0",
                                   *codec, "-an", "-sn", "-dn", "-f", "mov", str(tmp)],
                                  capture_output=True, timeout=left)
        except (OSError, subprocess.TimeoutExpired):
            done = None
        if done is not None and done.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            os.replace(tmp, target)
            return True
        tmp.unlink(missing_ok=True)
    return False


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
            suffix = Path(file.filename or "").suffix.lower()
            if suffix != ".mov" and suffix not in REMUX | ENCODE:
                raise HTTPException(415, "Expected a .mov video (.mp4, .avi, .mkv, .webm are converted to .mov)")
            total = 0
            digest = hashlib.sha256()
            with temporary.open("xb") as output:
                while chunk := await file.read(1024 * 1024):
                    if time.time() >= deadline_at:
                        raise HTTPException(408, "Upload deadline exceeded")
                    total += len(chunk)
                    digest.update(chunk)
                    if total > self.settings.max_upload_bytes:
                        raise HTTPException(413, "Video exceeds configured size limit")
                    await run_in_threadpool(output.write, chunk)
                if total == 0:
                    raise HTTPException(400, "Empty upload")
                await run_in_threadpool(output.flush)
                await run_in_threadpool(os.fsync, output.fileno())
            if time.time() >= deadline_at:
                raise HTTPException(408, "Upload deadline exceeded")
            if suffix == ".mov":
                await run_in_threadpool(os.replace, temporary, destination)
            else:
                # перевод в .mov входит в общий срок задания; на сам анализ оставляем не меньше минуты
                budget = max(5.0, deadline_at - time.time() - 60.0)
                ok = await run_in_threadpool(_to_mov, self.settings.ffmpeg_bin, temporary, destination, suffix, budget)
                temporary.unlink(missing_ok=True)
                if not ok:
                    raise HTTPException(415, "Could not convert the video to .mov")
            await run_in_threadpool(_sync_directory, destination.parent)
            try:
                await run_in_threadpool(
                    self.store.enqueue, job_id, destination, arrival_at, deadline_at,
                    self.settings.profile, self.settings.max_pending_jobs,
                    original_name=Path(file.filename or "").name[:200] or None,
                    video_bytes=total, video_sha256=digest.hexdigest(),
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
