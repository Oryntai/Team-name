"""Durable SQLite queue; all state transitions use transactions or compare-and-swap."""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class QueueFull(RuntimeError):
    """The configured number of pending jobs has already been accepted."""


class Store:
    def __init__(self, db_path: Path, max_attempts: int = 3):
        self.db_path = Path(db_path)
        self.max_attempts = max_attempts

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA synchronous=FULL")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    video_path TEXT NOT NULL,
                    arrival_at REAL NOT NULL,
                    deadline_at REAL NOT NULL,
                    profile TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('queued','running','done','failed')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    attempt_token TEXT,
                    started_at REAL,
                    finished_at REAL,
                    result_json TEXT,
                    diagnostics_json TEXT,
                    error_code TEXT
                );
                CREATE INDEX IF NOT EXISTS jobs_pending ON jobs(status, arrival_at);
                CREATE TABLE IF NOT EXISTS service_state (
                    key TEXT PRIMARY KEY,
                    value REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS public_api_key (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    api_key TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
            """)
            connection.execute("BEGIN IMMEDIATE")
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(jobs)")}
            if "diagnostics_json" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN diagnostics_json TEXT")
            connection.commit()

    def public_api_key(self, new_key: str | None = None) -> str | None:
        """The one shared public key. With new_key, creates it on first use; later calls return the same value."""
        with self._connect() as connection:
            row = connection.execute("SELECT api_key FROM public_api_key WHERE id=1").fetchone()
            if row is not None or new_key is None:
                return row["api_key"] if row else None
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("INSERT OR IGNORE INTO public_api_key(id, api_key, created_at) VALUES (1, ?, ?)",
                               (new_key, time.time()))
            connection.commit()
            return connection.execute("SELECT api_key FROM public_api_key WHERE id=1").fetchone()["api_key"]

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        result = dict(row)
        encoded = result.pop("result_json")
        result["result"] = json.loads(encoded) if encoded else None
        diagnostics = result.pop("diagnostics_json")
        result["diagnostics"] = json.loads(diagnostics) if diagnostics else {}
        return result

    @staticmethod
    def _expire(connection: sqlite3.Connection, now: float) -> int:
        return connection.execute(
            "UPDATE jobs SET status='failed', finished_at=?, error_code='deadline_exceeded', "
            "diagnostics_json=? WHERE status IN ('queued','running') AND deadline_at<=?",
            (now, '{"error_type":"DeadlineExceeded"}', now),
        ).rowcount

    def enqueue(self, job_id: str, video_path: Path, arrival_at: float,
                deadline_at: float, profile: str, max_pending: int) -> dict:
        if (not math.isfinite(arrival_at) or not math.isfinite(deadline_at)
                or deadline_at <= arrival_at or max_pending < 1):
            raise ValueError("Invalid queue limits or timing")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = time.time()
            self._expire(connection, now)
            pending = connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')"
            ).fetchone()[0]
            if pending >= max_pending:
                connection.rollback()
                raise QueueFull("Queue is full")
            expired = deadline_at <= now
            connection.execute(
                "INSERT INTO jobs(job_id,video_path,arrival_at,deadline_at,profile,status,"
                "finished_at,error_code) VALUES(?,?,?,?,?,?,?,?)",
                (job_id, str(Path(video_path).resolve()), arrival_at, deadline_at, profile,
                 "failed" if expired else "queued", now if expired else None,
                 "deadline_exceeded" if expired else None),
            )
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            connection.commit()
            return self._row(row)

    def get(self, job_id: str) -> dict | None:
        with self._connect() as connection:
            return self._row(connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,),
            ).fetchone())

    def expire(self, now: float | None = None) -> int:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            count = self._expire(connection, time.time() if now is None else now)
            connection.commit()
            return count

    def claim(self) -> dict | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = time.time()
            self._expire(connection, now)
            row = connection.execute(
                "SELECT job_id FROM jobs WHERE status='queued' ORDER BY arrival_at,job_id LIMIT 1"
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            token = uuid.uuid4().hex
            connection.execute(
                "UPDATE jobs SET status='running', attempts=attempts+1, attempt_token=?, "
                "started_at=?, error_code=NULL WHERE job_id=? AND status='queued'",
                (token, now, row["job_id"]),
            )
            claimed = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (row["job_id"],),
            ).fetchone()
            connection.commit()
            return self._row(claimed)

    def complete(self, job_id: str, attempt_token: str, result: dict,
                 diagnostics: dict | None = None) -> bool:
        # Validate even if a caller bypasses the worker's serializer.
        from backend.contracts import JobDone
        parsed = JobDone.model_validate(result)
        if parsed.job_id != job_id:
            raise ValueError("Result belongs to a different job")
        encoded = parsed.model_dump_json()
        diagnostic_data = json.dumps(diagnostics or {}, ensure_ascii=False, allow_nan=False)
        with self._connect() as connection:
            # Obtain the write lock before measuring time. A competing writer
            # must not make a pre-lock timestamp authorize an overdue result.
            connection.execute("BEGIN IMMEDIATE")
            now = time.time()
            completed = bool(connection.execute(
                "UPDATE jobs SET status='done', result_json=?, diagnostics_json=?, finished_at=?, error_code=NULL "
                "WHERE job_id=? AND status='running' AND attempt_token=? AND deadline_at>?",
                (encoded, diagnostic_data, now, job_id, attempt_token, now),
            ).rowcount)
            connection.commit()
            return completed

    def fail(self, job_id: str, attempt_token: str, error_code: str,
             diagnostics: dict | None = None) -> bool:
        diagnostic_data = json.dumps(diagnostics or {}, ensure_ascii=False, allow_nan=False)
        with self._connect() as connection:
            return bool(connection.execute(
                "UPDATE jobs SET status='failed', finished_at=?, error_code=?, diagnostics_json=? "
                "WHERE job_id=? AND status='running' AND attempt_token=?",
                (time.time(), error_code, diagnostic_data, job_id, attempt_token),
            ).rowcount)

    def recover(self) -> int:
        """Call only while owning the process singleton lock, after an old worker exits."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = time.time()
            self._expire(connection, now)
            connection.execute(
                "UPDATE jobs SET status='failed', finished_at=?, error_code='attempt_limit' "
                "WHERE status='running' AND attempts>=?", (now, self.max_attempts),
            )
            count = connection.execute(
                "UPDATE jobs SET status='queued', attempt_token=NULL, started_at=NULL, "
                "error_code=NULL WHERE status='running' AND attempts<?", (self.max_attempts,),
            ).rowcount
            connection.commit()
            return count

    def heartbeat(self, model_ready: bool | None = None) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO service_state(key,value) VALUES('worker_heartbeat',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (time.time(),),
            )
            if model_ready is not None:
                connection.execute(
                    "INSERT INTO service_state(key,value) VALUES('model_ready',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (int(model_ready),),
                )
            connection.commit()

    def health(self) -> dict:
        with self._connect() as connection:
            counts = {row["status"]: row["n"] for row in connection.execute(
                "SELECT status,COUNT(*) AS n FROM jobs GROUP BY status"
            )}
            beat = connection.execute(
                "SELECT value FROM service_state WHERE key='worker_heartbeat'"
            ).fetchone()
            ready = connection.execute(
                "SELECT value FROM service_state WHERE key='model_ready'"
            ).fetchone()
        last = beat["value"] if beat else None
        return {"database": "ok", "jobs": counts, "heartbeat_at": last,
                "model_ready": bool(ready["value"]) if ready else False,
                "worker_heartbeat_at": last,
                "worker_heartbeat_age_sec": max(0, time.time() - last) if last else None}
