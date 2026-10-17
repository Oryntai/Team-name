"""Evaluate the real local HTTP pipeline without logging video semantics.

python -m data_tools.evaluate_service --base-url http://127.0.0.1:8000 \
  --videos /local/videos --gold /local/gold.json --output-dir local_data/eval \
  --api-key-env FORTE_API_KEY --limit 2

Loopback only; no redirects or environment proxies. Each video's upload and
polling share a monotonic deadline (maximum 300 seconds). Full payloads and
prediction records stay in private local files; stdout contains aggregate metrics
only. Requires httpx, Pillow, backend and forte_decision. No retry/resume.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import io
import ipaddress
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import quote, urlsplit

import httpx
from PIL import Image

from backend.contracts import JobAccepted, JobDone, JobFailed, JobProcessing
from forte_decision.cli import write_result
from forte_decision.contracts import number, strict_json_loads
from forte_decision.evaluation import evaluate
from data_tools.inventory import _key


def validate_base_url(value):
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("invalid service URL")
    if (
        parts.username
        or parts.password
        or parts.query
        or parts.fragment
        or parts.path not in ("", "/")
    ):
        raise ValueError("service URL must be a plain origin")
    host = parts.hostname.lower()
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback:
        raise ValueError("service must use loopback")
    _ = parts.port  # Validate malformed or out-of-range port.
    return value.rstrip("/")


def validate_done(payload, job_id, duration=None):
    done = JobDone.model_validate(payload, strict=True)
    if done.job_id != job_id:
        raise ValueError("job identity mismatch")
    if duration is not None and number(duration, "duration") <= 0:
        raise ValueError("invalid duration")
    for event in done.violations:
        if duration is not None and event.timecode_sec > duration:
            raise ValueError("event exceeds video duration")
        for frame in event.evidence_frames:
            if duration is not None and frame.timestamp_sec > duration:
                raise ValueError("evidence exceeds video duration")
            data = base64.b64decode(frame.image_base64, validate=True)
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "JPEG":
                    raise ValueError("evidence is not JPEG")
                image.load()
    return done


def _private_directory(path):
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def _percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def _video_index(root):
    root = root.resolve()
    index = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".mov":
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("video symlink escapes local root")
        video_id = _key(path.relative_to(root).as_posix())
        if video_id in index:
            raise ValueError("ambiguous normalized video identity")
        index[video_id] = resolved
    return index


def run_evaluation(
    *,
    base_url,
    videos,
    gold,
    output_dir,
    api_key,
    limit=None,
    deadline_sec=300.0,
    poll_interval=0.5,
    durations=None,
    transport=None,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Run sequentially; injectable transport/clock are for synthetic HTTP tests."""
    base_url = validate_base_url(base_url)
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("API key required")
    number(deadline_sec, "deadline_sec")
    number(poll_interval, "poll_interval")
    if not 0 < deadline_sec <= 300 or poll_interval <= 0:
        raise ValueError("invalid timing configuration")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    # Reuse schema validation without marking any gold video as a clean prediction.
    evaluate([], gold)
    selected = sorted(gold, key=lambda row: row["video_id"])[:limit]
    index = _video_index(Path(videos))
    if any(row["video_id"] not in index for row in selected):
        raise ValueError("gold video missing from local root")
    durations = {} if durations is None else durations
    if not isinstance(durations, dict):
        raise ValueError("durations must be an object")
    for row in selected:
        if (
            row["video_id"] in durations
            and number(durations[row["video_id"]], "duration") <= 0
        ):
            raise ValueError("invalid duration")
    output_dir = Path(output_dir)
    _private_directory(output_dir)
    payload_dir = output_dir / "payloads"
    _private_directory(payload_dir)
    predictions, elapsed, errors = [], [], Counter()
    with httpx.Client(
        base_url=base_url,
        headers={"X-API-Key": api_key},
        transport=transport,
        trust_env=False,
        follow_redirects=False,
    ) as client:
        for label in selected:
            video_id = label["video_id"]
            started = clock()
            job_id, last_payload, error_code = None, None, None
            prediction = {"video_id": video_id, "status": "failed", "events": []}

            def remaining():
                value = deadline_sec - (clock() - started)
                if value <= 0:
                    raise TimeoutError("deadline exceeded")
                return value

            try:
                with index[video_id].open("rb") as stream:
                    response = client.post(
                        "/api/v1/analyze",
                        files={
                            "file": (index[video_id].name, stream, "video/quicktime")
                        },
                        timeout=remaining(),
                    )
                remaining()
                response.raise_for_status()
                last_payload = strict_json_loads(response.text)
                accepted = JobAccepted.model_validate(last_payload, strict=True)
                if not accepted.job_id:
                    raise ValueError("empty job identity")
                job_id = accepted.job_id
                while True:
                    response = client.get(
                        "/api/v1/jobs/" + quote(job_id, safe=""), timeout=remaining()
                    )
                    remaining()
                    response.raise_for_status()
                    last_payload = strict_json_loads(response.text)
                    if not isinstance(last_payload, dict):
                        raise ValueError("invalid job response")
                    status = last_payload.get("status")
                    if status == "processing":
                        processing = JobProcessing.model_validate(
                            last_payload, strict=True
                        )
                        if processing.job_id != job_id:
                            raise ValueError("job identity mismatch")
                        sleep(min(poll_interval, remaining()))
                        continue
                    if status == "failed":
                        failed = JobFailed.model_validate(last_payload, strict=True)
                        if failed.job_id != job_id:
                            raise ValueError("job identity mismatch")
                        error_code = "service_failed"
                        break
                    if status != "done":
                        raise ValueError("invalid job status")
                    done = validate_done(last_payload, job_id, durations.get(video_id))
                    remaining()
                    prediction = {
                        "video_id": video_id,
                        "status": "complete",
                        "verdict": done.verdict,
                        "events": [
                            {"timecode_sec": event.timecode_sec, "type": event.type}
                            for event in done.violations
                        ],
                    }
                    break
            except (TimeoutError, httpx.TimeoutException):
                error_code = "timeout"
            except httpx.HTTPError:
                error_code = "http_error"
            except (
                ValueError,
                TypeError,
                KeyError,
                OSError,
                Image.DecompressionBombError,
            ):
                error_code = "invalid_result_or_io"
            duration = max(0.0, clock() - started)
            elapsed.append(duration)
            if error_code:
                errors[error_code] += 1
                prediction["error_code"] = error_code
            predictions.append(prediction)
            name = hashlib.sha256(video_id.encode()).hexdigest() + ".json"
            write_result(
                payload_dir / name,
                json.dumps(
                    {
                        "video_id": video_id,
                        "job_id": job_id,
                        "payload": last_payload,
                        "error_code": error_code,
                        "elapsed_sec": duration,
                    },
                    allow_nan=False,
                ),
            )
    metrics = evaluate(predictions, selected)
    positives = sum(row.get("verdict") == "violation" for row in selected)
    labelled = sum(
        row.get("verdict") in ("violation", "no_violation") for row in selected
    )
    video = metrics["video"]
    metrics["service"] = {
        "attempted": len(selected),
        "error_counts": dict(errors),
        "end_to_end_recall": video["TP"] / positives if positives else None,
        "accuracy_including_failed": (
            (video["TP"] + video["TN"]) / labelled if labelled else None
        ),
        "elapsed_p50_sec": _percentile(elapsed, 0.5),
        "elapsed_p95_sec": _percentile(elapsed, 0.95),
        "duration_bounds_checked_count": sum(
            row["video_id"] in durations for row in selected
        ),
    }
    # These files contain per-video semantics; never echo them into agent logs.
    write_result(
        output_dir / "predictions.json",
        json.dumps(predictions, ensure_ascii=False, allow_nan=False),
    )
    write_result(
        output_dir / "gold_subset.json",
        json.dumps(selected, ensure_ascii=False, allow_nan=False),
    )
    write_result(
        output_dir / "report.json",
        json.dumps(metrics, ensure_ascii=False, allow_nan=False),
    )
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key-env", default="FORTE_API_KEY")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--deadline-sec", type=float, default=300)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument(
        "--durations",
        type=Path,
        help="Optional local JSON video_id -> duration_sec metadata map",
    )
    args = parser.parse_args(argv)
    try:
        gold = strict_json_loads(args.gold.read_text(encoding="utf-8"))
        durations = (
            strict_json_loads(args.durations.read_text(encoding="utf-8"))
            if args.durations
            else None
        )
        report = run_evaluation(
            base_url=args.base_url,
            videos=args.videos,
            gold=gold,
            output_dir=args.output_dir,
            api_key=os.environ.get(args.api_key_env),
            limit=args.limit,
            deadline_sec=args.deadline_sec,
            poll_interval=args.poll_interval,
            durations=durations,
        )
    except Exception:
        # Never expose exception text, response bodies, paths, API keys or scenes.
        print(json.dumps({"status": "error"}))
        return 1
    # Per-event timing errors remain in the private report, not stdout.
    aggregate_report = json.loads(json.dumps(report, allow_nan=False))
    for scopes in aggregate_report.get("events", {}).values():
        scopes.get("all_known_labels", {}).pop("time_errors_sec", None)
    print(json.dumps(aggregate_report, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
