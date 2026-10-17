#!/usr/bin/env python3
"""Measure concurrent uploads through terminal API status, without logging frames.

Uses only Python's standard library. FORTE_API_KEY must be an environment variable.
This is a client-side measurement including transfer, queue, analysis and polling.
It never follows redirects or sends data anywhere except --base-url.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import http.client
import json
import math
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any
import urllib.parse
import uuid


MAX_RESPONSE_BYTES = 64 * 1024 * 1024


class ClientFailure(Exception):
    """A safe error category that contains no response content or credentials."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class ApiClient:
    def __init__(self, base_url: str, api_key: str):
        url = urllib.parse.urlsplit(base_url)
        if url.scheme not in ("http", "https") or not url.hostname:
            raise ValueError("--base-url must be an http(s) URL")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("--base-url must not contain credentials, a query or fragment")
        self.url = url
        self.api_key = api_key

    def request(self, route: str, deadline: float, video: Path | None = None) -> tuple[int, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        connection_type = http.client.HTTPSConnection if self.url.scheme == "https" else http.client.HTTPConnection
        connection = connection_type(self.url.hostname, self.url.port, timeout=remaining)
        path = self.url.path.rstrip("/") + route
        try:
            if video is None:
                connection.request("GET", path, headers={"X-API-Key": self.api_key})
            else:
                boundary = "forte-load-" + uuid.uuid4().hex
                # The multipart filename is deliberately generic; local paths stay local.
                opening = (
                    f"--{boundary}\r\n"
                    'Content-Disposition: form-data; name="file"; filename="video.mov"\r\n'
                    "Content-Type: video/quicktime\r\n\r\n"
                ).encode("ascii")
                closing = f"\r\n--{boundary}--\r\n".encode("ascii")
                connection.putrequest("POST", path)
                connection.putheader("X-API-Key", self.api_key)
                connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
                connection.putheader("Content-Length", str(len(opening) + video.stat().st_size + len(closing)))
                connection.endheaders()
                connection.send(opening)
                with video.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError
                        if connection.sock is not None:
                            connection.sock.settimeout(remaining)
                        connection.send(chunk)
                connection.send(closing)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            if connection.sock is not None:
                connection.sock.settimeout(remaining)
            response = connection.getresponse()
            # Error bodies may contain diagnostics; never include them in the report.
            if response.status < 200 or response.status >= 300:
                return response.status, None
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ClientFailure("response_too_large")
            try:
                payload = json.loads(raw)
            except (ValueError, UnicodeError):
                raise ClientFailure("invalid_json") from None
            if not isinstance(payload, dict):
                raise ClientFailure("invalid_response_shape")
            return response.status, payload
        finally:
            connection.close()


def run_one(
    index: int,
    video: Path,
    client: ApiClient,
    barrier: threading.Barrier,
    timeout: float,
    interval: float,
    budget: float,
) -> dict[str, Any]:
    barrier.wait()
    started = time.monotonic()
    deadline = started + timeout
    report: dict[str, Any] = {
        "request_index": index,
        "video_name": video.name,
        "video_bytes": video.stat().st_size,
        "started_at": utc_now(),
        "status": "client_error",
        "poll_count": 0,
    }
    stage = "upload"
    try:
        status, accepted = client.request("/api/v1/analyze", deadline, video)
        report["upload_seconds"] = round(time.monotonic() - started, 4)
        report["upload_http_status"] = status
        if status != 200 or not isinstance(accepted, dict):
            raise ClientFailure("upload_http_error")
        job_id = accepted.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ClientFailure("missing_job_id")
        report["job_id"] = job_id
        stage = "poll"
        job_route = "/api/v1/jobs/" + urllib.parse.quote(job_id, safe="")
        while time.monotonic() < deadline:
            status, result = client.request(job_route, deadline)
            report["poll_count"] += 1
            report["last_poll_http_status"] = status
            if status != 200 or not isinstance(result, dict):
                raise ClientFailure("poll_http_error")
            if result.get("job_id") != job_id:
                raise ClientFailure("job_id_mismatch")
            state = result.get("status")
            if state not in ("processing", "done", "failed"):
                raise ClientFailure("invalid_job_status")
            report["last_server_status"] = state
            if state == "failed":
                report["status"] = "failed"
                break
            if state == "done":
                verdict = result.get("verdict")
                violations = result.get("violations")
                if verdict not in ("violation", "no_violation") or not isinstance(violations, list):
                    raise ClientFailure("invalid_done_result")
                if (verdict == "violation") != bool(violations):
                    raise ClientFailure("inconsistent_verdict")
                report.update(status="done", verdict=verdict, violation_count=len(violations))
                break
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(interval, remaining))
        else:
            raise TimeoutError
    except (TimeoutError, socket.timeout):
        report.update(status="timeout", error="client_deadline_exceeded", error_stage=stage)
    except ClientFailure as error:
        report.update(status="client_error", error=error.args[0], error_stage=stage)
    except Exception as error:
        # Exception messages can contain URLs or payloads; persist the type only.
        report.update(status="client_error", error=type(error).__name__, error_stage=stage)
    elapsed = time.monotonic() - started
    report["finished_at"] = utc_now()
    report["client_elapsed_seconds"] = round(elapsed, 4)
    report["done_within_budget"] = report["status"] == "done" and elapsed <= budget
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--video", action="append", required=True, type=Path, help="Local .mov path; repeat for different videos")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of simultaneous jobs; video paths are cycled")
    parser.add_argument("--timeout", type=float, default=310, help="Client timeout per job, including upload and polling")
    parser.add_argument("--budget-seconds", type=float, default=300, help="Maximum client-observed time for a passing result")
    parser.add_argument("--poll-interval", type=float, default=1, help="Seconds between polls")
    parser.add_argument("--output", type=Path, default=Path("data/smoke-load-results.json"))
    args = parser.parse_args(argv)
    if not 1 <= args.concurrency <= 32:
        parser.error("--concurrency must be between 1 and 32")
    durations = (args.poll_interval, args.timeout, args.budget_seconds)
    if not all(math.isfinite(value) for value in durations) or not (
        0 < args.poll_interval <= args.timeout and 0 < args.budget_seconds <= args.timeout
    ):
        parser.error("Timeout, budget and poll interval must be positive; timeout must cover the budget and poll interval")
    for path in args.video:
        if path.suffix.lower() != ".mov" or not path.is_file() or path.stat().st_size == 0:
            parser.error("Each --video must point to an existing nonempty .mov file")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    key = os.environ.get("FORTE_API_KEY", "")
    if not key:
        raise SystemExit("Set FORTE_API_KEY in the environment; this script does not read .env")
    try:
        client = ApiClient(args.base_url, key)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    started_at = utc_now()
    barrier = threading.Barrier(args.concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(run_one, index + 1, args.video[index % len(args.video)], client, barrier,
                        args.timeout, args.poll_interval, args.budget_seconds)
            for index in range(args.concurrency)
        ]
        results = [future.result() for future in futures]
    counts = {state: sum(item["status"] == state for item in results)
              for state in ("done", "failed", "timeout", "client_error")}
    summary = {
        "counts": counts,
        "all_done_within_budget": all(item["done_within_budget"] for item in results),
        "max_client_elapsed_seconds": max(item["client_elapsed_seconds"] for item in results),
    }
    report = {
        "schema_version": 1,
        "started_at": started_at,
        "finished_at": utc_now(),
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "client_timeout_seconds": args.timeout,
        "budget_seconds": args.budget_seconds,
        "poll_interval_seconds": args.poll_interval,
        "timing_scope": "Client upload start through terminal polling response; includes transfer, queue, analysis and polling delay",
        "summary": summary,
        "jobs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(summary, ensure_ascii=False))
    print(f"Report: {args.output}")
    return 0 if summary["all_done_within_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
