"""Бэкенды VLM с одинаковым интерфейсом: MLX на маке, transformers на CUDA, заглушка для тестов.

Выше этого слоя никто не знает, на чём крутится модель. Всё локально: загрузка весов идёт только
из локального кеша (HF_HUB_OFFLINE / local_files_only), сеть при обработке не нужна.
"""

from __future__ import annotations

import http.client
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image


@dataclass
class VLMAnswer:
    text: str
    prompt_tokens: int = 0
    gen_tokens: int = 0
    wall_sec: float = 0.0


@dataclass
class VLMStats:
    calls: int = 0
    prompt_tokens: int = 0
    gen_tokens: int = 0
    wall_sec: float = 0.0

    def add(self, a: VLMAnswer) -> None:
        self.calls += 1
        self.prompt_tokens += a.prompt_tokens
        self.gen_tokens += a.gen_tokens
        self.wall_sec += a.wall_sec


class VLMBackend(Protocol):
    model_id: str
    revision: str
    stats: VLMStats

    def ask(self, frames: Sequence[tuple[float, Image.Image]], question: str, max_tokens: int = 128) -> VLMAnswer: ...


def frames_preamble(frames: Sequence[tuple[float, Image.Image]]) -> str:
    """Реальные PTS кадров в тексте запроса — модель отвечает в тех же секундах."""
    ts = ", ".join(f"{t:.1f}s" for t, _ in frames)
    return f"You see {len(frames)} frames from one fixed ceiling camera over a bank cash desk, in time order. Timestamps: {ts}.\n"


def _local_snapshot(model_id: str) -> tuple[str, str]:
    """Путь к локальным весам и их ревизия; в сеть не ходим."""
    if os.path.isdir(model_id):
        return model_id, os.path.basename(os.path.normpath(model_id))
    from huggingface_hub import snapshot_download

    path = snapshot_download(model_id, local_files_only=True)
    return path, os.path.basename(path)


class MLXBackend:
    """mlx-vlm (MIT). Картинки отдаём файлами во временном каталоге, который сразу удаляем."""

    def __init__(self, model_id: str = "mlx-community/Qwen3-VL-8B-Instruct-4bit"):
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template

        path, self.revision = _local_snapshot(model_id)
        self.model_id = model_id
        self.model, self.processor = load(path)
        self._generate, self._template = generate, apply_chat_template
        self.stats = VLMStats()

    def ask(self, frames, question, max_tokens=128):
        with tempfile.TemporaryDirectory(prefix="fv_") as d:
            paths = []
            for i, (_, im) in enumerate(frames):
                p = os.path.join(d, f"{i:04d}.jpg")
                im.convert("RGB").save(p, quality=90)
                paths.append(p)
            prompt = self._template(
                self.processor, self.model.config, frames_preamble(frames) + question, num_images=len(paths)
            )
            t0 = time.perf_counter()
            r = self._generate(
                self.model, self.processor, prompt, image=paths, max_tokens=max_tokens, temperature=0.0, verbose=False
            )
        a = VLMAnswer(r.text.strip(), r.prompt_tokens, r.generation_tokens, time.perf_counter() - t0)
        self.stats.add(a)
        return a


class TransformersBackend:
    """transformers на CUDA для Linux-хоста с NVIDIA. На маке не проверялся — проверить на GPU-хосте."""

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct", device: str = "cuda"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        path, self.revision = _local_snapshot(model_id)
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            path, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device
        )
        self.stats = VLMStats()

    def ask(self, frames, question, max_tokens=128):
        import torch

        content = [{"type": "image", "image": im.convert("RGB")} for _, im in frames]
        content.append({"type": "text", "text": frames_preamble(frames) + question})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        n_in = inputs["input_ids"].shape[1]
        text = self.processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
        a = VLMAnswer(text.strip(), int(n_in), int(out.shape[1] - n_in), time.perf_counter() - t0)
        self.stats.add(a)
        return a


class OpenAICompatBackend:
    """Локальный сервер с OpenAI-совместимым API: llama.cpp llama-server (MIT) или vLLM (Apache-2.0)
    на своей машине в локальной сети. Облачные API этим классом не используются: base_url обязан
    указывать на localhost или адрес из частной сети."""

    def __init__(self, base_url: str, model_id: str = "local", api_key: str | None = None, timeout: float = 120.0):
        import ipaddress
        import urllib.parse

        host = urllib.parse.urlparse(base_url).hostname or ""
        try:
            private = host == "localhost" or ipaddress.ip_address(host).is_private or ipaddress.ip_address(host).is_loopback
        except ValueError:
            private = host.endswith(".local") or host == "localhost"
        if not private:
            raise ValueError(f"VLM-сервер должен быть локальным (localhost / частная сеть), а не {host!r}")
        self.base_url = base_url.rstrip("/")
        self.model_id, self.revision = model_id, "server"
        self.api_key, self.timeout = api_key, timeout
        self.stats = VLMStats()

    def ask(self, frames, question, max_tokens=128):
        import base64
        import io
        import json as _json
        import urllib.request

        content = []
        for _, im in frames:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=90)
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}})
        content.append({"type": "text", "text": frames_preamble(frames) + question})
        body = _json.dumps({"model": self.model_id, "messages": [{"role": "user", "content": content}],
                            "max_tokens": max_tokens, "temperature": 0.0}).encode()
        req = urllib.request.Request(self.base_url + "/v1/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", "Bearer " + self.api_key)
        t0 = time.perf_counter()
        # прокси окружения не используем: кадры не должны уходить никуда, кроме указанного локального сервера
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=self.timeout) as r:
            data = _json.loads(r.read())
        usage = data.get("usage") or {}
        a = VLMAnswer(data["choices"][0]["message"]["content"].strip(), int(usage.get("prompt_tokens", 0)),
                      int(usage.get("completion_tokens", 0)), time.perf_counter() - t0)
        self.stats.add(a)
        return a


@dataclass
class StubBackend:
    """Детерминированная заглушка для тестов контракта: ответ — функция от (времена кадров, вопрос)."""

    respond: Callable[[list[float], str], str]
    model_id: str = "stub"
    revision: str = "0"
    stats: VLMStats = field(default_factory=VLMStats)

    def ask(self, frames, question, max_tokens=128):
        a = VLMAnswer(self.respond([t for t, _ in frames], question))
        self.stats.add(a)
        return a


class FallbackBackend:
    """Основной бэкенд (сервер модели на GPU-машине) и запасной локальный на этой же машине.

    Сбой связи или кривой ответ основного — запрос сразу уходит запасному, а основной не
    трогаем cooldown_sec секунд, чтобы не терять время на каждом окне. Потом пробуем снова.
    """

    _FAILURES = (OSError, http.client.HTTPException, ValueError, KeyError, IndexError)

    def __init__(self, primary: VLMBackend, secondary: VLMBackend, cooldown_sec: float = 60.0,
                 clock: Callable[[], float] = time.monotonic):
        self.primary, self.secondary = primary, secondary
        self.cooldown_sec, self.clock = cooldown_sec, clock
        self.down_until = 0.0
        self.model_id = f"{primary.model_id}|fallback:{secondary.model_id}"
        self.revision = f"{primary.revision}|{secondary.revision}"
        self.stats = VLMStats()
        self.routing = {"primary_calls": 0, "fallback_calls": 0, "primary_failures": 0}

    def ask(self, frames, question, max_tokens=128):
        if self.clock() >= self.down_until:
            try:
                a = self.primary.ask(frames, question, max_tokens)
            except self._FAILURES:
                self.routing["primary_failures"] += 1
                self.down_until = self.clock() + self.cooldown_sec
            else:
                self.routing["primary_calls"] += 1
                self.stats.add(a)
                return a
        a = self.secondary.ask(frames, question, max_tokens)
        self.routing["fallback_calls"] += 1
        self.stats.add(a)
        return a


def _openai_from_env(model_id: str | None) -> OpenAICompatBackend:
    # FORTE_VLM_URL — адрес локального сервера; ключ — из файла FORTE_VLM_KEY_FILE (не из аргументов)
    url = os.environ.get("FORTE_VLM_URL")
    if not url:
        raise ValueError("для бэкенда openai задайте FORTE_VLM_URL (локальный llama-server / vLLM)")
    key_file = os.environ.get("FORTE_VLM_KEY_FILE")
    key = open(key_file, encoding="utf-8").read().strip() if key_file else None
    timeout = float(os.environ.get("FORTE_VLM_TIMEOUT", "60"))
    return OpenAICompatBackend(url, model_id or "local", key, timeout=timeout)


def make_backend(kind: str, model_id: str | None = None) -> VLMBackend:
    if kind == "mlx":
        return MLXBackend(model_id) if model_id else MLXBackend()
    if kind == "transformers":
        return TransformersBackend(model_id) if model_id else TransformersBackend()
    if kind == "openai":
        return _openai_from_env(model_id)
    if kind == "openai+mlx":
        # сервер модели на GPU-машине; если он недоступен — локальная MLX-модель (FORTE_VLM_FALLBACK_MODEL)
        fallback = os.environ.get("FORTE_VLM_FALLBACK_MODEL", "mlx-community/Qwen3-VL-8B-Instruct-4bit")
        return FallbackBackend(_openai_from_env(model_id), MLXBackend(fallback),
                               cooldown_sec=float(os.environ.get("FORTE_VLM_RETRY_SEC", "60")))
    raise ValueError(f"неизвестный бэкенд VLM: {kind}")
