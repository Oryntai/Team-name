"""Synthetic adapters exercise infrastructure only; they never claim ML accuracy."""

import base64
import io
import multiprocessing
import os
import shutil
import sqlite3
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from backend.config import Settings
from backend.contracts import ObservationBatch, ObservedEvent, RuleEvent
from backend.store import QueueFull, Store
from backend.worker import SingletonLock, Supervisor, WorkerAlreadyRunning, run_worker


def _enqueue(store, name, path, *, deadline=60, profile="clean", max_pending=10):
    now = time.time()
    return store.enqueue(name, path, now, now + deadline, profile, max_pending)


@pytest.fixture
def store(tmp_path):
    result = Store(tmp_path / "jobs.sqlite3")
    result.initialize()
    return result


def test_store_wal_capacity_and_expiry(store, tmp_path):
    with sqlite3.connect(store.db_path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    job = _enqueue(store, "first", tmp_path / "first.mov", max_pending=1)
    with pytest.raises(QueueFull):
        _enqueue(store, "second", tmp_path / "second.mov", max_pending=1)
    assert store.expire(job["deadline_at"] + 1) == 1
    assert store.get("first")["error_code"] == "deadline_exceeded"
    _enqueue(store, "second", tmp_path / "second.mov", max_pending=1)


def test_existing_database_migrates_private_diagnostics_column(tmp_path):
    path = tmp_path / "older.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY, video_path TEXT, arrival_at REAL, deadline_at REAL,
            profile TEXT, status TEXT, attempts INTEGER DEFAULT 0, attempt_token TEXT,
            started_at REAL, finished_at REAL, result_json TEXT, error_code TEXT)""")
    store = Store(path)
    store.initialize()
    store.initialize()  # Migration is safe when both API and worker initialize.
    _enqueue(store, "migrated", tmp_path / "test.mov")
    job = store.claim()
    assert store.fail(job["job_id"], job["attempt_token"], "test_error", {"error_type": "SyntheticError"})
    assert store.get(job["job_id"])["diagnostics"] == {"error_type": "SyntheticError"}


def test_atomic_claiming_and_stale_attempt_cannot_overwrite(store, tmp_path):
    for index in range(5):
        _enqueue(store, f"job-{index}", tmp_path / "synthetic.mov")
    with ThreadPoolExecutor(max_workers=8) as pool:
        claimed = list(pool.map(lambda _: store.claim(), range(8)))
    jobs = [job for job in claimed if job is not None]
    assert len(jobs) == len({job["job_id"] for job in jobs}) == 5
    assert store.recover() == 5
    new = store.claim()
    old = next(job for job in jobs if job["job_id"] == new["job_id"])
    done = {"job_id": new["job_id"], "status": "done", "verdict": "no_violation", "violations": []}
    assert new["attempt_token"] != old["attempt_token"]
    assert new["attempts"] == 2
    assert not store.complete(old["job_id"], old["attempt_token"], done)
    assert not store.fail(old["job_id"], old["attempt_token"], "late_failure", {"stale": True})
    assert store.get(new["job_id"])["diagnostics"] == {}
    assert store.complete(new["job_id"], new["attempt_token"], done, {"sampled_frames": 5})
    assert not store.complete(old["job_id"], old["attempt_token"], done, {"stale": True})
    assert store.get(new["job_id"])["result"] == done
    assert store.get(new["job_id"])["diagnostics"] == {"sampled_frames": 5}


def test_attempt_limit_and_deadline_reject_late_success(store, tmp_path):
    _enqueue(store, "repeated", tmp_path / "test.mov")
    for attempt in range(3):
        assert store.claim()["attempts"] == attempt + 1
        assert store.recover() == (1 if attempt < 2 else 0)
    assert store.get("repeated")["error_code"] == "attempt_limit"
    _enqueue(store, "late", tmp_path / "test.mov")
    job = store.claim()
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("UPDATE jobs SET deadline_at=? WHERE job_id='late'", (time.time() - 1,))
    assert not store.complete("late", job["attempt_token"], {
        "job_id": "late", "status": "done", "verdict": "no_violation", "violations": [],
    })
    store.expire()
    assert store.get("late")["status"] == "failed"


def test_completion_waiting_on_writer_cannot_cross_deadline(store, tmp_path):
    _enqueue(store, "blocked-completion", tmp_path / "test.mov", deadline=0.5)
    job = store.claim()
    result = {"job_id": job["job_id"], "status": "done", "verdict": "no_violation", "violations": []}
    with sqlite3.connect(store.db_path, isolation_level=None) as writer:
        writer.execute("BEGIN IMMEDIATE")
        with ThreadPoolExecutor(max_workers=1) as pool:
            completion = pool.submit(store.complete, job["job_id"], job["attempt_token"], result)
            time.sleep(0.7)
            writer.commit()
            assert completion.result(timeout=3) is False
    store.expire()
    assert store.get(job["job_id"])["status"] == "failed"


def test_os_singleton_lock_and_heartbeat(store, tmp_path):
    with SingletonLock(tmp_path / "worker.lock"):
        with pytest.raises(WorkerAlreadyRunning):
            with SingletonLock(tmp_path / "worker.lock"):
                pass
    with SingletonLock(tmp_path / "worker.lock"):
        pass
    store.heartbeat(model_ready=True)
    assert store.health()["model_ready"] is True
    assert store.health()["heartbeat_at"] > time.time() - 2


class SyntheticAnalyzer:
    def __init__(self):
        self.calls = 0

    def analyze(self, context, video):
        self.calls += 1
        # A persisted PID/call count verifies a single long-lived model instance.
        marker = context.video_path.parent / f"{context.job_id}.test-marker"
        previous_attempt = marker.exists()
        marker.write_text(f"{os.getpid()}:{self.calls}", encoding="ascii")
        if context.profile == "hang" or (context.profile == "hang_once" and not previous_attempt):
            time.sleep(600)
        if context.profile == "crash":
            os._exit(31)
        if context.profile == "incomplete":
            return ObservationBatch(complete=False, events=[], diagnostics={"frames_seen": 3})
        if context.profile == "clean":
            return ObservationBatch(complete=True, events=[], diagnostics={"sampled_frames": 5, "synthetic_fixture": True})
        pts = video.validate().frame_pts
        events = [ObservedEvent(t_start=pts[0], t_end=pts[2], t_main=pts[1],
                                frame_pts=list(pts[:3]), observations={"synthetic": "observed"},
                                sources=["detector"])]
        if context.profile == "multi":
            events.append(ObservedEvent(t_start=pts[2], t_end=pts[4], t_main=pts[3],
                                        frame_pts=list(pts[2:5]), observations={"synthetic": "observed"},
                                        sources=["detector"]))
        return ObservationBatch(complete=True, events=events)


class SyntheticRules:
    def evaluate(self, observations, context):
        return [RuleEvent(t_start=event.t_start, t_end=event.t_end, t_main=event.t_main,
                          type="synthetic_fixture", description="Synthetic infrastructure test event.",
                          rule_id="TEST-ONLY", rule_title="Synthetic test reference", rule_section="Fixture 1",
                          rule_url="https://example.com/synthetic-test-only", frame_pts=event.frame_pts)
                for event in observations.events]


def create_synthetic_analyzer():
    return SyntheticAnalyzer()


def create_synthetic_rules():
    return SyntheticRules()


@pytest.fixture(scope="module")
def ffmpeg_tools():
    root = Path(__file__).resolve().parents[2]
    local_bins = list((root / ".tools").glob("*/bin"))
    found = []
    for name in ("ffmpeg", "ffprobe"):
        candidate = os.environ.get(f"FORTE_{name.upper()}_BIN") or shutil.which(name)
        if not candidate:
            executable = name + (".exe" if os.name == "nt" else "")
            candidate = next((str(folder / executable) for folder in local_bins
                              if (folder / executable).exists()), None)
        if not candidate:
            pytest.skip("FFmpeg/ffprobe required for real decoding integration checks")
        found.append(candidate)
    return found


@pytest.fixture
def worker_settings(tmp_path, ffmpeg_tools):
    return Settings(_env_file=None, data_dir=tmp_path / "data", worker_poll_sec=0.02,
                    ffmpeg_bin=ffmpeg_tools[0], ffprobe_bin=ffmpeg_tools[1],
                    analyzer_factory="backend.tests.test_worker_store:create_synthetic_analyzer",
                    rules_factory="backend.tests.test_worker_store:create_synthetic_rules")


@pytest.fixture
def synthetic_video(tmp_path, ffmpeg_tools):
    path = tmp_path / "synthetic.mov"
    subprocess.run([ffmpeg_tools[0], "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc2=size=64x48:rate=5:duration=1", "-an", "-c:v", "mpeg4", str(path)],
                   check=True, capture_output=True, timeout=30)
    return path


def _drain(supervisor, ids, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        supervisor.step()
        jobs = [supervisor.store.get(job_id) for job_id in ids]
        if all(job["status"] in ("done", "failed") for job in jobs):
            return jobs
        time.sleep(0.02)
    pytest.fail("Worker integration test exceeded its local test deadline")


def test_five_queued_jobs_both_verdicts_multiple_events_and_real_jpegs(worker_settings, synthetic_video):
    with Supervisor(worker_settings) as supervisor:
        profiles = ["clean", "multi", "violation", "clean", "multi"]
        with ThreadPoolExecutor(max_workers=5) as pool:
            jobs = list(pool.map(lambda item: _enqueue(
                supervisor.store, f"parallel-{item[0]}", synthetic_video, profile=item[1]), enumerate(profiles)))
        results = _drain(supervisor, [job["job_id"] for job in jobs])
        assert all(job["status"] == "done" for job in results), results
        assert results[0]["result"]["verdict"] == "no_violation"
        assert results[0]["result"]["violations"] == []
        assert results[0]["diagnostics"] == {"sampled_frames": 5, "synthetic_fixture": True}
        from fastapi.testclient import TestClient
        from backend.app import create_app
        with TestClient(create_app(worker_settings)) as client:
            response = client.get(f"/ui-api/jobs/{results[0]['job_id']}")
            assert response.json() == results[0]["result"]
            assert "diagnostics" not in response.json()
        assert len(results[1]["result"]["violations"]) == 2
        for event in results[1]["result"]["violations"]:
            assert "Synthetic test reference" in event["description"]
            assert "https://example.com/synthetic-test-only" in event["description"]
            assert 1 <= len(event["evidence_frames"]) <= 3
            assert event["timecode_sec"] in [frame["timestamp_sec"] for frame in event["evidence_frames"]]
            for frame in event["evidence_frames"]:
                image = Image.open(io.BytesIO(base64.b64decode(frame["image_base64"], validate=True)))
                assert image.format == "JPEG"
                assert image.size == (64, 48)
        markers = [(synthetic_video.parent / f"{job['job_id']}.test-marker").read_text().split(":")
                   for job in results]
        assert len({marker[0] for marker in markers}) == 1
        assert {int(marker[1]) for marker in markers} == {1, 2, 3, 4, 5}
        assert all(job["finished_at"] - job["arrival_at"] < 60 for job in results)


def test_corrupt_video_and_incomplete_analysis_fail(worker_settings, synthetic_video):
    broken = synthetic_video.parent / "broken.mov"
    broken.write_bytes(b"not a movie despite extension")
    with Supervisor(worker_settings) as supervisor:
        _enqueue(supervisor.store, "broken", broken)
        _enqueue(supervisor.store, "incomplete", synthetic_video, profile="incomplete")
        results = _drain(supervisor, ["broken", "incomplete"])
        assert [job["status"] for job in results] == ["failed", "failed"]
        assert results[1]["error_code"] == "analysis_incomplete"
        assert results[1]["diagnostics"] == {"frames_seen": 3, "error_type": "IncompleteAnalysis"}
        assert all(job["result"] is None for job in results)


def test_timeout_includes_queue_and_worker_restarts_model(worker_settings, synthetic_video):
    with Supervisor(worker_settings) as supervisor:
        _enqueue(supervisor.store, "hang", synthetic_video, profile="hang", deadline=3)
        _enqueue(supervisor.store, "queue-expired", synthetic_video, deadline=1)
        _enqueue(supervisor.store, "after-timeout", synthetic_video, deadline=30)
        jobs = _drain(supervisor, ["hang", "queue-expired", "after-timeout"])
        assert jobs[0]["status"] == jobs[1]["status"] == "failed"
        assert jobs[0]["error_code"] == jobs[1]["error_code"] == "deadline_exceeded"
        assert jobs[1]["attempts"] == 0
        assert jobs[2]["status"] == "done"
        assert jobs[0]["finished_at"] - jobs[0]["arrival_at"] < 3.5


def test_recover_after_worker_restart(worker_settings, synthetic_video):
    with Supervisor(worker_settings) as first:
        _enqueue(first.store, "restart", synthetic_video)
        first.step()
        assert first.store.get("restart")["status"] == "running"
    with Supervisor(worker_settings) as second:
        job = _drain(second, ["restart"])[0]
        assert job["status"] == "done"
        assert job["attempts"] == 2


def test_model_crash_does_not_stop_following_jobs(worker_settings, synthetic_video):
    with Supervisor(worker_settings) as supervisor:
        _enqueue(supervisor.store, "crashed", synthetic_video, profile="crash")
        _enqueue(supervisor.store, "survivor", synthetic_video)
        jobs = _drain(supervisor, ["crashed", "survivor"])
        assert jobs[0]["status"] == "failed"
        assert jobs[1]["status"] == "done"


def test_abrupt_supervisor_process_death_is_recovered(worker_settings, synthetic_video):
    store = Store(worker_settings.db_path)
    store.initialize()
    _enqueue(store, "supervisor-crash", synthetic_video, profile="hang_once", deadline=40)
    worker = multiprocessing.get_context("spawn").Process(target=run_worker, args=(worker_settings,))
    worker.start()
    marker = synthetic_video.parent / "supervisor-crash.test-marker"
    try:
        until = time.monotonic() + 15
        while time.monotonic() < until and not marker.exists():
            time.sleep(0.02)
        assert marker.exists(), "Synthetic model did not start"
        old_model_pid = marker.read_text().split(":")[0]
        worker.kill()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert store.get("supervisor-crash")["status"] == "running"
        with Supervisor(worker_settings) as restarted:
            job = _drain(restarted, ["supervisor-crash"])[0]
            assert job["status"] == "done"
            assert job["attempts"] == 2
            assert marker.read_text().split(":")[0] != old_model_pid
    finally:
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=5)
        worker.close()


def test_default_adapter_cannot_produce_a_clean_verdict(worker_settings, synthetic_video):
    settings = worker_settings.model_copy(update={
        "analyzer_factory": "backend.adapters.unavailable:create_analyzer",
        "rules_factory": "backend.adapters.unavailable:create_rules",
    })
    with Supervisor(settings) as supervisor:
        _enqueue(supervisor.store, "missing-ml", synthetic_video)
        job = _drain(supervisor, ["missing-ml"])[0]
        assert job["status"] == "failed"
        assert job["error_code"] == "analysis_unavailable"
        assert job["result"] is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group cleanup only")
@pytest.mark.parametrize("platform,alive,recovers", [
    ("darwin", False, True),
    ("darwin", True, False),
    ("linux", False, False),
])
def test_group_permission_failure_only_tolerated_for_dead_macos_model(
    monkeypatch, platform, alive, recovers,
):
    """A dead Darwin group must not stall jobs; live/Linux EPERM stays visible."""
    import backend.worker as worker_module

    class Process:
        pid = 123456
        joined = False
        closed = False

        def is_alive(self):
            return alive

        def join(self, timeout):
            assert timeout == 2
            self.joined = True

        def close(self):
            self.closed = True

    class Connection:
        closed = False

        def close(self):
            self.closed = True

    supervisor = Supervisor.__new__(Supervisor)
    process = Process()
    connection = Connection()
    supervisor.process = process
    supervisor.connection = connection
    supervisor.guard = None
    supervisor.entered = False
    supervisor.model_ready = True
    attempts = []

    def denied(pid, signum):
        attempts.append((pid, signum))
        raise PermissionError("synthetic group permission failure")

    with monkeypatch.context() as patch:
        patch.setattr(worker_module.sys, "platform", platform)
        patch.setattr(worker_module.os, "killpg", denied)
        if recovers:
            supervisor._stop_model()
        else:
            with pytest.raises(PermissionError):
                supervisor._stop_model()
    assert attempts == [(process.pid, worker_module.signal.SIGKILL)]
    if recovers:
        assert process.joined and process.closed and connection.closed
        assert supervisor.process is None and supervisor.connection is None
        assert supervisor.model_ready is False
    else:
        assert not process.closed and not connection.closed
