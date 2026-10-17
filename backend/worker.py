"""One durable queue consumer and one supervised, persistent model process.

The HTTP application never imports or constructs ML factories. A hung model cannot
hold the queue past a job's absolute deadline: the supervisor kills its process tree.
"""

from __future__ import annotations

import base64
import bisect
import importlib
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from backend.contracts import (
    AnalysisContext, AnalysisUnavailable, EvidenceFrame, IncompleteAnalysis,
    JobDone, ObservationBatch, RuleEvent, Violation,
)
from backend.store import Store


class WorkerAlreadyRunning(RuntimeError):
    pass


class SingletonLock:
    """An OS lock, automatically released after a crash; a PID file is insufficient."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0, os.SEEK_END)
                if self.handle.tell() == 0:
                    self.handle.write(b"0")
                    self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            self.handle.close()
            self.handle = None
            raise WorkerAlreadyRunning("A worker already owns this data directory") from error
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


class _WindowsJob:
    """Kill the model and all descendants even if the supervisor itself is killed."""

    def __init__(self, pid: int):
        self.handle = None
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                        ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IOCounts(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IOCounts),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                   ctypes.c_void_p, wintypes.DWORD]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        self.kernel = kernel
        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "Cannot create model process guard")
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        process = None
        try:
            if not kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise OSError(ctypes.get_last_error(), "Cannot configure model process guard")
            process = kernel.OpenProcess(0x0100 | 0x0001, False, pid)
            if not process or not kernel.AssignProcessToJobObject(self.handle, process):
                raise OSError(ctypes.get_last_error(), "Cannot attach model process guard")
        except BaseException:
            self.close()
            raise
        finally:
            if process:
                kernel.CloseHandle(process)

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _factory(spec: str):
    module, separator, name = spec.partition(":")
    if not separator or not module or not name or "." in name:
        raise ValueError("Factory must have module:function format")
    factory = getattr(importlib.import_module(module), name)
    return factory()


def _require_time(deadline_at: float) -> None:
    if time.time() >= deadline_at:
        raise TimeoutError("Job deadline exceeded")


# A jury burst arrives at once and every deadline counts from upload, so one sequential worker
# would spend the whole window on the first video and let the tail expire. Each claimed job gets
# its share of the time left to the last queued deadline; the analyzer then plans fewer, wider windows.
QUEUE_TAIL_RESERVE_SEC = 20.0  # the last job's analyzer keeps this much for rules and JPEG
QUEUE_JOB_OVERHEAD_SEC = 8.0   # decoding probe, rules and evidence JPEG outside the analysis budget
QUEUE_MIN_BUDGET_SEC = 30.0


def _queue_budget(deadline_at: float, queued_deadlines: list[float], now: float) -> float | None:
    if not queued_deadlines:
        return None
    share = ((max(queued_deadlines) - now - QUEUE_TAIL_RESERVE_SEC) / (len(queued_deadlines) + 1)
             - QUEUE_JOB_OVERHEAD_SEC)
    return max(QUEUE_MIN_BUDGET_SEC, min(deadline_at - now, share))


def _require_frame_pts(pts: float, available: tuple[float, ...]) -> None:
    index = bisect.bisect_left(available, pts)
    candidates = available[max(0, index - 1):index + 1]
    if not candidates or min(abs(value - pts) for value in candidates) > 1e-6:
        raise ValueError("An evidence timestamp was not returned by VideoIO")


def _analyze_job(job: dict, config: dict, analyzer: Any, rules: Any,
                 diagnostics: dict | None = None) -> dict:
    from backend.video_io import VideoIO

    context = AnalysisContext(job_id=job["job_id"], video_path=job["video_path"],
                              deadline_at=job["deadline_at"], profile=job["profile"],
                              analysis_budget_sec=job.get("analysis_budget_sec"))
    _require_time(context.deadline_at)
    video = VideoIO(context.video_path, ffmpeg_bin=config["ffmpeg_bin"],
                    ffprobe_bin=config["ffprobe_bin"], deadline_at=context.deadline_at)
    metadata = video.validate()
    batch = ObservationBatch.model_validate(analyzer.analyze(context, video))
    if diagnostics is not None:
        diagnostics.update(batch.diagnostics)
    _require_time(context.deadline_at)
    for observed in batch.events:
        if observed.t_end > metadata.duration_sec + 1e-6:
            raise ValueError("Observed event extends beyond video duration")
        for pts in observed.frame_pts:
            _require_frame_pts(pts, metadata.frame_pts)
            if not observed.t_start - 1e-6 <= pts <= observed.t_end + 1e-6:
                raise ValueError("Observed frame outside event span")
    evaluated = rules.evaluate(batch, context)
    if not isinstance(evaluated, list):
        raise ValueError("Rules must return a list of RuleEvent")
    # Partial coverage keeps the violations already proven on the analysed part (FAQ 11), but an
    # incomplete analysis without any of them stays failed: it is never reported as no_violation.
    if not batch.complete and not evaluated:
        raise IncompleteAnalysis("Analysis coverage incomplete")
    partial_note = (" Анализ завершён не по всему видео: нарушение найдено на проверенной части, "
                    "остальные фрагменты не проверены." if not batch.complete else "")
    if diagnostics is not None and not batch.complete:
        diagnostics["partial_result"] = True
    violations = []
    for item in evaluated:
        _require_time(context.deadline_at)
        event = RuleEvent.model_validate(item)
        if event.t_end > metadata.duration_sec + 1e-6:
            raise ValueError("Rule event extends beyond video duration")
        frames = []
        seen = set()
        for pts in event.frame_pts:
            _require_frame_pts(pts, metadata.frame_pts)
            frame = video.frame_at(pts)
            if not event.t_start - 1e-6 <= frame.timestamp_sec <= event.t_end + 1e-6:
                raise ValueError("Extracted evidence is outside event span")
            if abs(frame.timestamp_sec - pts) > 1e-6:
                raise ValueError("Decoder returned a different evidence timestamp")
            if frame.timestamp_sec in seen:
                continue
            seen.add(frame.timestamp_sec)
            frames.append(EvidenceFrame(timestamp_sec=frame.timestamp_sec,
                                        image_base64=base64.b64encode(frame.jpeg_bytes).decode("ascii")))
        main = min(frames, key=lambda frame: abs(frame.timestamp_sec - event.t_main)).timestamp_sec
        # The note stays in the observation part: the UI parses everything after the basis marker.
        description = (f"{event.description}{partial_note}\n"
                       f"Нормативное основание: {event.rule_title}; "
                       f"{event.rule_section} (ID: {event.rule_id}). {event.rule_url}")
        violations.append(Violation(type=event.type, timecode_sec=main,
                                    description=description, evidence_frames=frames))
    _require_time(context.deadline_at)
    return JobDone(job_id=context.job_id, verdict="violation" if violations else "no_violation",
                   violations=violations).model_dump(mode="json")


def _error_code(error: BaseException) -> str:
    if isinstance(error, TimeoutError):
        return "deadline_exceeded"
    if isinstance(error, AnalysisUnavailable):
        return "analysis_unavailable"
    if isinstance(error, IncompleteAnalysis):
        return "analysis_incomplete"
    if isinstance(error, ValueError):
        return "invalid_analysis_or_video"
    return "analysis_error"


def _model_main(connection, config: dict, parent_pid: int) -> None:
    if os.name != "nt":
        os.setsid()
        if sys.platform.startswith("linux"):
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            # A supervisor SIGKILL must not leave an old GPU owner alive.
            if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
                raise OSError(ctypes.get_errno(), "Cannot configure model parent-death guard")
            if os.getppid() != parent_pid:
                os.kill(os.getpid(), signal.SIGKILL)

        def monitor_parent():
            while os.getppid() == parent_pid:
                time.sleep(0.5)
            os.killpg(os.getpgrp(), signal.SIGKILL)

        threading.Thread(target=monitor_parent, daemon=True).start()
    try:
        analyzer = _factory(config["analyzer_factory"])
        rules = _factory(config["rules_factory"])
    except BaseException as error:
        connection.send({"kind": "startup_failed", "diagnostics": {"error_type": type(error).__name__}})
        connection.close()
        return
    connection.send({"kind": "ready"})
    try:
        while True:
            job = connection.recv()
            if job is None:
                break
            diagnostics = {}
            try:
                result = _analyze_job(job, config, analyzer, rules, diagnostics)
                message = {"kind": "result", "result": result}
            except BaseException as error:
                message = {"kind": "failed", "error_code": _error_code(error)}
                diagnostics["error_type"] = type(error).__name__
            message.update(job_id=job["job_id"], attempt_token=job["attempt_token"], diagnostics=diagnostics)
            connection.send(message)
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


class Supervisor:
    """Use as a context manager; step() does one bounded queue/IPC polling cycle."""

    def __init__(self, settings):
        self.settings = settings
        self.store = Store(settings.db_path, max_attempts=settings.max_attempts)
        self.lock = SingletonLock(Path(settings.data_dir) / "worker.lock")
        self.process = None
        self.connection = None
        self.guard = None
        self.active = None
        self.entered = False
        self.last_heartbeat = 0.0
        self.last_purge = 0.0
        self.model_ready = False
        self.config = {name: str(getattr(settings, name)) for name in (
            "ffmpeg_bin", "ffprobe_bin", "analyzer_factory", "rules_factory")}

    def __enter__(self):
        if self.entered:
            raise RuntimeError("Supervisor already entered")
        self.lock.__enter__()
        try:
            self.store.initialize()
            self.store.recover()
            self.store.heartbeat(model_ready=False)
            self.entered = True
            self._start_model()
        except BaseException:
            self.lock.__exit__(None, None, None)
            self.entered = False
            raise
        return self

    def __exit__(self, *args):
        self.close()

    def _start_model(self):
        self.model_ready = False
        self.store.heartbeat(model_ready=False)
        context = multiprocessing.get_context("spawn")
        self.connection, child_connection = context.Pipe(duplex=True)
        self.process = context.Process(target=_model_main,
                                       args=(child_connection, self.config, os.getpid()),
                                       name="forte-model-owner")
        self.process.start()
        child_connection.close()
        try:
            self.guard = _WindowsJob(self.process.pid)
        except BaseException:
            self._stop_model()
            raise

    def _stop_model(self):
        process = self.process
        if self.guard is not None:
            self.guard.close()
            self.guard = None
        if process is not None:
            if os.name != "nt":
                # Descendants can survive after the model leader has exited.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    # On native macOS a vanished process group can report EPERM
                    # after its leader crashes. Still attempt group cleanup above,
                    # but do not let that dead leader block queue recovery. A live
                    # model, or the same failure on another OS, remains an error.
                    if sys.platform != "darwin" or process.is_alive():
                        raise
            if process.is_alive():
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=5, check=False,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if process.is_alive():
                    process.kill()
            process.join(timeout=2)
            if process.is_alive():
                raise RuntimeError("Model process did not stop")
            process.close()
        if self.connection is not None:
            self.connection.close()
        self.process = None
        self.connection = None
        self.model_ready = False
        if self.entered:
            self.store.heartbeat(model_ready=False)

    def _receive(self) -> None:
        if self.connection is None:
            return
        try:
            if not self.connection.poll():
                return
            message = self.connection.recv()
        except (EOFError, OSError):
            message = {"kind": "crashed"}
        kind = message.get("kind")
        diagnostics = message.get("diagnostics", {})
        if kind == "ready":
            self.model_ready = True
            self.store.heartbeat(model_ready=True)
            return
        if self.active is not None:
            job_id = self.active["job_id"]
            token = self.active["attempt_token"]
            if kind == "startup_failed":
                self.store.fail(job_id, token, "model_startup_failed", diagnostics)
            elif message.get("job_id") == job_id and message.get("attempt_token") == token:
                if kind == "result":
                    try:
                        self.store.complete(job_id, token, message["result"], diagnostics)
                    except (ValueError, KeyError) as error:
                        self.store.fail(job_id, token, "invalid_result", {"error_type": type(error).__name__})
                else:
                    self.store.fail(job_id, token, message.get("error_code", "model_crashed"), diagnostics)
            else:
                self.store.fail(job_id, token, "model_crashed", {"error_type": "ModelProcessExit"})
                kind = "crashed"
            self.active = None
        if kind in ("startup_failed", "crashed"):
            self._stop_model()

    def purge_uploads(self) -> int:
        """Delete uploaded videos of finished jobs. Only files inside upload_dir are touched; a file that is
        still locked (Windows) is retried on the next sweep."""
        if self.settings.keep_uploads:
            return 0
        upload_dir = Path(self.settings.upload_dir).resolve()
        active = self.active["job_id"] if self.active is not None else None
        deleted = 0
        for job in self.store.finished_videos():
            if job["job_id"] == active:
                continue
            path = Path(job["video_path"]).resolve()
            if path.parent != upload_dir:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue
            self.store.mark_video_deleted(job["job_id"])
            deleted += 1
        return deleted

    def step(self) -> None:
        if not self.entered:
            raise RuntimeError("Use Supervisor as a context manager")
        now = time.time()
        self.store.expire()
        if now - self.last_heartbeat >= 1:
            self.store.heartbeat(model_ready=self.model_ready)
            self.last_heartbeat = now
        now = time.time()
        if self.active is not None:
            current = self.store.get(self.active["job_id"])
            if now >= self.active["deadline_at"] or current["status"] != "running":
                self._stop_model()
                self.active = None
        self._receive()
        if now - self.last_purge >= 1:
            self.purge_uploads()
            self.last_purge = now
        if self.active is None:
            if self.process is not None and not self.process.is_alive():
                self._stop_model()
            job = self.store.claim()
            if job is not None:
                job["analysis_budget_sec"] = _queue_budget(job["deadline_at"], self.store.queued_deadlines(),
                                                           time.time())
                self.active = job
                try:
                    if self.process is None:
                        self._start_model()
                    self.connection.send(job)
                except (OSError, RuntimeError, BrokenPipeError) as error:
                    self.store.fail(job["job_id"], job["attempt_token"], "model_startup_failed",
                                    {"error_type": type(error).__name__})
                    self._stop_model()
                    self.active = None

    def close(self):
        try:
            self._stop_model()
            self.active = None
        finally:
            if self.entered:
                self.lock.__exit__(None, None, None)
                self.entered = False


def run_worker(settings, stop_event=None) -> None:
    stop_event = stop_event or threading.Event()
    with Supervisor(settings) as supervisor:
        while not stop_event.is_set():
            supervisor.step()
            stop_event.wait(settings.worker_poll_sec)


def main() -> None:
    from backend.config import Settings
    settings = Settings()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *args: stop.set())
    run_worker(settings, stop)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
