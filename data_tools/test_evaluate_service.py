"""Synthetic byte uploads and mocked HTTP only; no competition videos."""

import base64
import io
import json
from pathlib import Path

import httpx
from PIL import Image
import pytest

from data_tools.evaluate_service import (
    main,
    run_evaluation,
    validate_base_url,
    validate_done,
)


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, amount):
        self.value += amount


def jpeg():
    output = io.BytesIO()
    Image.new("RGB", (4, 4), (25, 25, 25)).save(output, format="JPEG")
    return base64.b64encode(output.getvalue()).decode()


def done(job="job", positive=True):
    return {
        "job_id": job,
        "status": "done",
        "verdict": "violation" if positive else "no_violation",
        "violations": (
            [
                {
                    "type": "synthetic",
                    "timecode_sec": 3,
                    "description": "SYNTHETIC SECRET SCENE",
                    "evidence_frames": [{"timestamp_sec": 3, "image_base64": jpeg()}],
                }
            ]
            if positive
            else []
        ),
    }


def setup(tmp_path, count=1):
    root = tmp_path / "videos"
    root.mkdir()
    gold = []
    for i in range(count):
        (root / f"{i}.mov").write_bytes(b"SYNTHETIC-NOT-VIDEO")
        gold.append(
            {
                "video_id": f"{i}.mov",
                "verdict": "violation",
                "exhaustive": False,
                "events": [{"timecode_sec": 3}],
            }
        )
    return root, gold


def run(tmp_path, handler, count=1, **kwargs):
    root, gold = setup(tmp_path, count)
    clock = kwargs.pop("clock", Clock())
    return run_evaluation(
        base_url="http://127.0.0.1:8000",
        videos=root,
        gold=gold,
        output_dir=tmp_path / "eval",
        api_key="SECRET-KEY",
        transport=httpx.MockTransport(handler),
        clock=clock,
        sleep=clock.sleep,
        **kwargs,
    )


def test_upload_poll_decode_and_private_results(tmp_path):
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        assert request.headers["X-API-Key"] == "SECRET-KEY"
        if request.method == "POST":
            assert b"SYNTHETIC-NOT-VIDEO" in request.read()
            return httpx.Response(200, json={"job_id": "job"})
        return httpx.Response(200, json=done())

    report = run(tmp_path, handler, durations={"0.mov": 10})
    assert seen == [("POST", "/api/v1/analyze"), ("GET", "/api/v1/jobs/job")]
    assert report["video"]["TP"] == 1 and report["service"]["end_to_end_recall"] == 1
    assert report["events"]["1"]["all_known_labels"]["recall_of_known_labels"] == 1
    assert report["events"]["1"]["exhaustive_labels"]["precision"] is None
    predictions = json.loads((tmp_path / "eval/predictions.json").read_text())
    assert predictions[0]["status"] == "complete"
    assert "description" not in predictions[0]["events"][0]
    assert len(list((tmp_path / "eval/payloads").glob("*.json"))) == 1
    if __import__("os").name != "nt":
        assert (tmp_path / "eval").stat().st_mode & 0o077 == 0
        for file in (tmp_path / "eval").rglob("*.json"):
            assert file.stat().st_mode & 0o077 == 0


def test_processing_and_monotonic_upload_deadline(tmp_path):
    clock = Clock()

    def handler(request):
        if request.method == "POST":
            clock.value += 2
            return httpx.Response(200, json={"job_id": "job"})
        raise AssertionError("upload already consumed deadline")

    report = run(tmp_path, handler, clock=clock, deadline_sec=1)
    assert report["service"]["error_counts"] == {"timeout": 1}
    assert report["video"]["failures"] == 1 and report["video"]["TN"] == 0
    assert report["service"]["end_to_end_recall"] == 0
    assert report["service"]["accuracy_including_failed"] == 0


def test_poll_processing_timeout(tmp_path):
    def handler(request):
        return httpx.Response(
            200,
            json=(
                {"job_id": "job"}
                if request.method == "POST"
                else {"job_id": "job", "status": "processing"}
            ),
        )

    report = run(tmp_path, handler, deadline_sec=1, poll_interval=0.4)
    assert report["service"]["error_counts"] == {"timeout": 1}
    assert report["service"]["elapsed_p50_sec"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"job_id": "job", "status": "failed"},
        {
            "job_id": "wrong",
            "status": "done",
            "verdict": "no_violation",
            "violations": [],
        },
        {
            "job_id": "job",
            "status": "done",
            "verdict": "no_violation",
            "violations": [],
            "unexpected": True,
        },
        {"job_id": "job", "status": "invalid"},
    ],
)
def test_failed_and_invalid_never_negative(tmp_path, payload):
    def handler(request):
        return httpx.Response(
            200, json={"job_id": "job"} if request.method == "POST" else payload
        )

    report = run(tmp_path, handler)
    assert report["video"]["failures"] == 1 and report["video"]["TN"] == 0
    assert report["video"]["completed"] == 0
    assert report["events"]["1"]["all_known_labels"]["unmatched_labels"] == 1


@pytest.mark.parametrize(
    "host",
    [
        "https://example.org",
        "http://192.168.1.1",
        "http://127.0.0.1@example.org",
        "http://localhost/path",
        "http://localhost?token=SECRET",
    ],
)
def test_remote_and_non_origin_rejected(host):
    with pytest.raises(ValueError):
        validate_base_url(host)


@pytest.mark.parametrize(
    "host", ["http://127.0.0.1:8000", "http://localhost:8000", "http://[::1]:8000"]
)
def test_loopback_hosts(host):
    assert validate_base_url(host) == host


def test_bad_jpeg_and_timestamp_invalid(tmp_path):
    payload = done()
    payload["violations"][0]["evidence_frames"][0]["image_base64"] = base64.b64encode(
        b"\xff\xd8\xff\xd9"
    ).decode()
    with pytest.raises(OSError):
        validate_done(payload, "job")
    with pytest.raises(ValueError):
        validate_done(done(), "job", duration=2)
    bad = done()
    bad["violations"][0]["timecode_sec"] = True
    with pytest.raises(ValueError):
        validate_done(bad, "job")


def test_limit_filters_gold_and_metrics_subset(tmp_path):
    def handler(request):
        return httpx.Response(
            200, json={"job_id": "job"} if request.method == "POST" else done()
        )

    report = run(tmp_path, handler, count=3, limit=1)
    assert report["video"]["total"] == 1 and report["service"]["attempted"] == 1
    assert len(json.loads((tmp_path / "eval/gold_subset.json").read_text())) == 1


def test_http_error_and_redirect_not_followed(tmp_path):
    requests = []

    def handler(request):
        requests.append(request.url.host)
        return httpx.Response(
            307, headers={"location": "https://remote.invalid/SECRET"}
        )

    report = run(tmp_path, handler)
    assert requests == ["127.0.0.1"]
    assert report["service"]["error_counts"] == {"http_error": 1}


def test_cli_exception_does_not_leak(tmp_path, capsys, monkeypatch):
    root, gold = setup(tmp_path)
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(gold))
    monkeypatch.setenv("FORTE_API_KEY", "SECRET-KEY")
    code = main(
        [
            "--base-url",
            "http://example.org/SECRET",
            "--videos",
            str(root),
            "--gold",
            str(path),
            "--output-dir",
            str(tmp_path / "eval"),
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert captured.out.strip() == '{"status": "error"}' and captured.err == ""


def test_mixed_confusion_end_to_end_and_latency(tmp_path):
    root, gold = setup(tmp_path, 5)
    for i in (1, 3):
        gold[i].update(verdict="no_violation", events=[])
    clock = Clock()
    current = [-1]
    results = [
        done(positive=True),
        done(positive=True),
        done(positive=False),
        done(positive=False),
        {"job_id": "job", "status": "failed"},
    ]

    def handler(request):
        if request.method == "POST":
            current[0] += 1
            clock.value += current[0] + 1
            return httpx.Response(200, json={"job_id": "job"})
        return httpx.Response(200, json=results[current[0]])

    report = run_evaluation(
        base_url="http://localhost:8000",
        videos=root,
        gold=gold,
        output_dir=tmp_path / "eval",
        api_key="key",
        transport=httpx.MockTransport(handler),
        clock=clock,
        sleep=clock.sleep,
    )
    video = report["video"]
    assert [video[key] for key in ("TP", "FP", "FN", "TN")] == [1, 1, 1, 1]
    assert video["completion_rate"] == 0.8 and video["failures"] == 1
    assert report["service"]["end_to_end_recall"] == pytest.approx(1 / 3)
    assert report["service"]["accuracy_including_failed"] == 0.4
    assert report["service"]["elapsed_p50_sec"] == 3
    assert report["service"]["elapsed_p95_sec"] == pytest.approx(4.8)


def test_cli_stdout_only_aggregate_metrics(tmp_path, capsys, monkeypatch):
    root, gold = setup(tmp_path)
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(gold))
    monkeypatch.setenv("FORTE_API_KEY", "SECRET-KEY")

    def fake_run(**kwargs):
        return {
            "video": {"TP": 1},
            "events": {
                "1": {
                    "all_known_labels": {
                        "time_errors_sec": [0.2, 0.3],
                        "mean_time_error_sec": 0.25,
                    }
                }
            },
        }

    monkeypatch.setattr("data_tools.evaluate_service.run_evaluation", fake_run)
    assert (
        main(
            [
                "--videos",
                str(root),
                "--gold",
                str(path),
                "--output-dir",
                str(tmp_path / "eval"),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "time_errors_sec" not in captured.out
    assert "SECRET" not in captured.out and captured.err == ""
    assert json.loads(captured.out)["video"]["TP"] == 1
