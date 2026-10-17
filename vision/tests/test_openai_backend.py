"""OpenAI-совместимый бэкенд против поддельного локального сервера: формат запроса, ключ, запрет внешних адресов."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image

from forte_vision.vlm import OpenAICompatBackend, make_backend


class _Handler(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.append((self.path, self.headers.get("Authorization"), body))
        out = json.dumps({"choices": [{"message": {"content": " []\n"}}],
                          "usage": {"prompt_tokens": 321, "completion_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    _Handler.seen.clear()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_request_format_and_key(server):
    b = OpenAICompatBackend(server, "qwen3vl-8b", api_key="test-key-123")
    frames = [(1.0, Image.new("RGB", (32, 48), (90, 90, 90))), (2.5, Image.new("RGB", (32, 48), (200, 0, 0)))]
    a = b.ask(frames, "Return []", max_tokens=7)
    assert a.text == "[]" and a.prompt_tokens == 321 and a.gen_tokens == 2
    path, auth, body = _Handler.seen[0]
    assert path == "/v1/chat/completions" and auth == "Bearer test-key-123"
    content = body["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image_url", "image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "1.0s, 2.5s" in content[-1]["text"]
    assert body["max_tokens"] == 7 and body["temperature"] == 0.0
    assert b.stats.calls == 1


@pytest.mark.parametrize("url", ["https://api.openai.com", "http://8.8.8.8:8080", "http://example.com:8080"])
def test_refuses_non_local_servers(url):
    with pytest.raises(ValueError):
        OpenAICompatBackend(url)


@pytest.mark.parametrize("url", ["http://127.0.0.1:8080", "http://192.168.1.50:8080", "http://10.0.0.7:8080", "http://localhost:8080"])
def test_accepts_local_and_private(url):
    OpenAICompatBackend(url)


def test_make_backend_reads_key_from_file(server, tmp_path, monkeypatch):
    key = tmp_path / "key.txt"
    key.write_text("file-key\n", encoding="utf-8")
    monkeypatch.setenv("FORTE_VLM_URL", server)
    monkeypatch.setenv("FORTE_VLM_KEY_FILE", str(key))
    b = make_backend("openai", "local-qwen")
    b.ask([(0.0, Image.new("RGB", (8, 8)))], "x")
    assert _Handler.seen[-1][1] == "Bearer file-key"


def test_fallback_switches_and_retries_after_cooldown():
    from forte_vision.vlm import FallbackBackend, StubBackend

    now = [0.0]
    calls = {"n": 0}

    class Flaky:
        model_id, revision = "gpu", "server"

        def __init__(self):
            from forte_vision.vlm import VLMStats

            self.stats = VLMStats()

        def ask(self, frames, question, max_tokens=128):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionRefusedError("tunnel down")
            from forte_vision.vlm import VLMAnswer

            return VLMAnswer("gpu")

    local = StubBackend(lambda times, q: "mac", model_id="mlx")
    b = FallbackBackend(Flaky(), local, cooldown_sec=60, clock=lambda: now[0])
    frames = [(0.0, Image.new("RGB", (8, 8)))]
    assert b.ask(frames, "q").text == "mac"  # сбой основного -> запасной в том же запросе
    assert b.ask(frames, "q").text == "mac" and calls["n"] == 1  # в cooldown основной не дёргаем
    now[0] = 61.0
    assert b.ask(frames, "q").text == "gpu"  # после cooldown основной снова в деле
    assert b.routing == {"primary_calls": 1, "fallback_calls": 2, "primary_failures": 1, "primary_unhealthy": 0}
    assert b.stats.calls == 3 and "fallback:mlx" in b.model_id


def test_fallback_on_dead_local_server_port(tmp_path, monkeypatch):
    """Порт, на котором никто не слушает (туннель ноута упал), — быстрый переход на запасной."""
    import socket

    from forte_vision.vlm import FallbackBackend, StubBackend

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    b = FallbackBackend(OpenAICompatBackend(f"http://127.0.0.1:{port}", timeout=5),
                        StubBackend(lambda t, q: "mac", model_id="mlx"))
    assert b.ask([(0.0, Image.new("RGB", (8, 8)))], "q").text == "mac"
    assert b.routing["primary_failures"] == 1


def test_fallback_sticks_to_local_for_the_rest_of_the_job():
    """Within a job one primary failure keeps the local model: a hung server would cost a timeout per window."""
    from forte_vision.vlm import FallbackBackend, StubBackend, VLMAnswer, VLMStats

    calls = {"n": 0}
    now = [0.0]

    class Flaky:
        model_id, revision = "gpu", "0"
        stats = VLMStats()

        def ask(self, frames, question, max_tokens=128):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("server hung")
            return VLMAnswer("gpu")

    b = FallbackBackend(Flaky(), StubBackend(lambda t, q: "mac", model_id="mlx"), cooldown_sec=1,
                        clock=lambda: now[0])
    frames = [(0.0, Image.new("RGB", (8, 8)))]
    b.begin_job(None)
    assert b.ask(frames, "q").text == "mac"
    now[0] = 100.0  # cooldown long gone, but still the same job
    assert b.ask(frames, "q").text == "mac" and calls["n"] == 1
    b.begin_job(None)  # next job tries the server again
    assert b.ask(frames, "q").text == "gpu"


def test_unhealthy_server_at_job_start_goes_straight_to_local():
    from forte_vision.vlm import FallbackBackend, StubBackend, VLMStats

    class Dead:
        model_id, revision = "gpu", "0"
        stats = VLMStats()

        def healthy(self):
            return False

        def ask(self, frames, question, max_tokens=128):
            raise AssertionError("an unhealthy server must not be asked")

    b = FallbackBackend(Dead(), StubBackend(lambda t, q: "mac", model_id="mlx"))
    b.begin_job(None)
    assert b.ask([(0.0, Image.new("RGB", (8, 8)))], "q").text == "mac"
    assert b.routing["primary_unhealthy"] == 1 and b.routing["fallback_calls"] == 1


def test_health_check_against_local_server_and_dead_port(server):
    import socket

    assert OpenAICompatBackend(server).healthy()
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    assert not OpenAICompatBackend(f"http://127.0.0.1:{port}").healthy(timeout=1)


def test_call_timeout_never_outlives_the_job_budget(server):
    """No VLM call starts when the job budget is gone; the fallback does not start one either."""
    import time

    from forte_vision.vlm import FallbackBackend, StubBackend, start_job

    frames = [(0.0, Image.new("RGB", (8, 8)))]
    b = OpenAICompatBackend(server, timeout=60)
    start_job(b, time.monotonic() + 30)
    assert 25 < b._call_timeout() <= 30
    start_job(b, time.monotonic() + 1)
    with pytest.raises(TimeoutError):
        b.ask(frames, "q")
    assert _Handler.seen == []
    fb = FallbackBackend(OpenAICompatBackend(server), StubBackend(lambda t, q: "mac", model_id="mlx"))
    start_job(fb, time.monotonic() + 1)
    with pytest.raises(TimeoutError):
        fb.ask(frames, "q")


def test_call_estimate_follows_the_backend_that_answers():
    from forte_vision.vlm import DEFAULT_CALL_SEC, FALLBACK_CALL_SEC, FallbackBackend, StubBackend, expected_call_sec

    gpu, mac = StubBackend(lambda t, q: "gpu"), StubBackend(lambda t, q: "mac")
    b = FallbackBackend(gpu, mac)
    assert expected_call_sec(b) == DEFAULT_CALL_SEC
    gpu.stats.calls, gpu.stats.wall_sec = 10, 80.0
    assert expected_call_sec(b) == 10.0
    b.down_until = float("inf")
    assert expected_call_sec(b) == FALLBACK_CALL_SEC

