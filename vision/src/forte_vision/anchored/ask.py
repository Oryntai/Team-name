"""Пакетные вопросы VLM по увеличенным вырезкам зон и grounding зон на полном кадре.

Свой HTTP-клиент к локальному OpenAI-совместимому серверу (llama-server / vLLM), без frames_preamble
из forte_vision.vlm: в пакете не кадры подряд, а вырезки разных зон, и каждая подписана своим ключом.
Ответ — один JSON {ключ: yes|no|unclear}; разбор терпит обёртки ```json, прозу и пропуски (-> unclear).
Кадры берутся одним проходом декодера на запрос: первый кадр с PTS >= t. Всё в долях кадра.
Ответы кешируются на диске (ключ: модель + имя и размер файла + вырезки + контекст + версия промпта).
"""

from __future__ import annotations

import base64
import hashlib
import io
import ipaddress
import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .types import Box, CropAnswer, CropItem

PROMPT_VERSION = "ask-v2"  # v2: каждый вопрос в пакете звучит один раз (Q1, Q2…), картинки ссылаются на него
MAX_ITEMS = 8  # вырезок на запрос: больше — модель начинает путать картинки
PAD = 0.30  # поле вокруг зоны с каждой стороны, доля размера зоны
PAD_MAX = 0.08  # но не больше этой доли кадра: у большой зоны (стол, стена с дверью) поле съедает увеличение
MIN_SHORT, MAX_LONG = 336, 672  # короткая сторона вырезки не меньше, длинная не больше (px)
ANSWERS = ("yes", "no", "unclear")

_YES = {"yes", "y", "true", "да", "1"}
_NO = {"no", "n", "false", "нет", "0", "none"}


# ---------- адрес и транспорт ----------

def _endpoint(url: str) -> str:
    """Принимаем и базу (http://h:p), и .../v1, и полный путь .../v1/chat/completions."""
    u = url.rstrip("/")
    if u.endswith("/chat/completions"):
        return u
    return u + ("/chat/completions" if u.endswith("/v1") else "/v1/chat/completions")


def _check_local(url: str) -> None:
    """Кадры не уходят наружу: только localhost или частная сеть (как в vlm.OpenAICompatBackend)."""
    host = urllib.parse.urlparse(url).hostname or ""
    try:
        ip = ipaddress.ip_address(host)
        ok = ip.is_private or ip.is_loopback
    except ValueError:
        ok = host == "localhost" or host.endswith(".local")
    if not ok:
        raise ValueError(f"VLM-сервер должен быть локальным (localhost / частная сеть), а не {host!r}")


# ---------- кадры и вырезки ----------

def grab_frames(path: str, times: Sequence[float]) -> list[tuple[float, Image.Image]]:
    """Один проход декодера: на каждое время (в порядке запроса) — первый кадр с PTS >= t, за концом — последний."""
    import av

    out: list[tuple[float, Image.Image] | None] = [None] * len(times)
    if not times:
        return []
    order = sorted(range(len(times)), key=lambda i: times[i])
    t_min = times[order[0]]
    for use_seek in (t_min > 1.0, False):
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            if use_seek:
                try:
                    c.seek(int(max(0.0, t_min - 1.0) / s.time_base), stream=s, backward=True)
                except Exception:  # noqa: BLE001 — контейнер без индекса: читаем с начала
                    continue
            k, last, first, imgs = 0, None, None, {}
            for f in c.decode(s):
                if f.pts is None:
                    continue
                t = float(f.pts * s.time_base)
                first = t if first is None else first
                last = (t, f)
                while k < len(order) and t + 1e-6 >= times[order[k]]:
                    if t not in imgs:
                        imgs[t] = f.to_image()
                    out[order[k]] = (t, imgs[t])
                    k += 1
                if k == len(order):
                    break
            if use_seek and first is not None and first > t_min + 1e-6:
                continue  # seek проскочил нужное время — читаем с начала
            if k < len(order) and last is not None:
                im = last[1].to_image()
                for i in order[k:]:
                    out[i] = (last[0], im)
            break
    if any(o is None for o in out):
        raise ValueError(f"в файле нет видеокадров: {path}")
    return out  # type: ignore[return-value]


def crop_region(size: tuple[int, int], box: Box, zoom: float = 1.0, pad: float = PAD,
                min_short: int = MIN_SHORT, max_long: int = MAX_LONG
                ) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """Область вырезки (px, внутри кадра) и итоговый размер картинки.

    Область = зона + pad с каждой стороны (не больше PAD_MAX кадра). Увеличение m (px картинки на px
    кадра) = max(короткая сторона >= min_short, zoom), но длинная сторона не больше max_long. Если zoom>1
    не влезает в max_long, область сужается вокруг центра зоны: объект в центре и увеличен ровно в zoom раз.
    """
    W, H = size
    x1, y1, x2, y2 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rw = min(W, max(8.0, (x2 - x1) + 2 * min(pad * (x2 - x1), PAD_MAX * W)))
    rh = min(H, max(8.0, (y2 - y1) + 2 * min(pad * (y2 - y1), PAD_MAX * H)))
    if zoom > 1:
        # крошечный объект (замок 5 px): область не меньше min_short/zoom, чтобы увеличение было ровно zoom,
        # а не 40x размытого пятна; слишком большая — сужаем вокруг центра (увеличение важнее поля)
        rw, rh = min(W, max(rw, min_short / zoom)), min(H, max(rh, min_short / zoom))
        if max(rw, rh) * zoom > max_long:
            rw, rh = min(rw, max_long / zoom), min(rh, max_long / zoom)
    # сдвигаем область внутрь кадра, а не обрезаем: объект не уезжает к краю вырезки
    rx1 = min(max(0.0, cx - rw / 2), W - rw)
    ry1 = min(max(0.0, cy - rh / 2), H - rh)
    reg = (int(round(rx1)), int(round(ry1)), int(round(rx1 + rw)), int(round(ry1 + rh)))
    iw, ih = max(1, reg[2] - reg[0]), max(1, reg[3] - reg[1])
    m = min(max(min_short / min(iw, ih), zoom), max_long / max(iw, ih))
    return reg, (max(1, round(iw * m)), max(1, round(ih * m)))


def crop_image(im: Image.Image, box: Box, zoom: float = 1.0) -> Image.Image:
    reg, out = crop_region(im.size, box, zoom)
    return im.crop(reg).resize(out, Image.LANCZOS)


def _jpeg_url(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------- разбор ответов ----------

def norm_answer(v) -> str:
    """yes | no | unclear из чего угодно: true/false, 'Yes.', 'no, the door is closed', {'answer': 'yes'}."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, dict):
        v = next((v[k] for k in ("answer", "value", "a") if k in v), None)
    if v is None or isinstance(v, (list, float)):
        return "unclear"
    if isinstance(v, int):
        return {1: "yes", 0: "no"}.get(v, "unclear")
    words = re.findall(r"[a-zа-яё0-9]+", str(v).lower())
    if not words:
        return "unclear"
    w = words[0]
    if w == "not" and len(words) > 1 and words[1] in ("visible", "sure", "clear", "known"):
        return "unclear"
    return "yes" if w in _YES else "no" if w in _NO else "unclear"


# синонимы для вопроса «что лежит»: модель не всегда отвечает словами из списка
_CHOICE_SYN = {
    "cash": ("cash", "money", "banknote", "banknotes", "bill", "bills", "notes", "bundle", "coins"),
    "documents": ("documents", "document", "papers", "paper", "forms", "form", "folder", "passport", "receipt"),
    "bag": ("bag", "handbag", "backpack", "purse", "coat", "jacket", "clothes", "scarf"),
    "food": ("food", "snack", "sandwich", "plate", "lunch", "fruit", "apple", "cookie", "cookies", "candy"),
    "drink": ("drink", "cup", "mug", "glass", "bottle", "coffee", "tea", "water", "can"),
    "phone": ("phone", "smartphone", "mobile", "cellphone"),
    "cash_bag": ("cash_bag", "sack", "collection"),
    # экран
    "document": ("document", "page", "table", "form", "window", "program", "spreadsheet", "text"),
    "desktop": ("desktop", "icons", "taskbar"),
    "lock": ("lock", "locked", "login", "signin", "sign", "password", "clock"),
    "off": ("off", "black", "closed", "dark", "lid"),
}


# Оборудование рабочего места: модель зовёт его «cash counting machine», «cash register» и сама ставит группу
# cash, а это не деньги на столе (ложные «ушла, оставив деньги» на чистых X-27, X-29)
_EQUIPMENT = {"machine", "counter", "register", "detector", "device", "laptop", "computer", "monitor", "keyboard",
              "mouse", "printer", "scanner", "terminal", "tablet", "calculator", "lamp", "chair", "armrest", "cable"}


def norm_choices(v, choices: Sequence[str]) -> str:
    """Ответ на вопрос «что из этого видно» -> "cash+food" | "none" | "unclear" (подмножество choices)."""
    if isinstance(v, dict):
        if "other" in choices and _EQUIPMENT & set(re.findall(r"[a-z]+", str(v.get("what") or "").lower())):
            return "other"
        g = next((v[k] for k in ("group", "class", "category", "answer", "value", "items", "objects") if k in v), None)
        if g is None or norm_choices(g, choices) == "unclear":  # группа не распознана — по словам «what»
            g = v.get("what", g)
        v = g
    if v is None or isinstance(v, (bool, int, float)):
        return "unclear"
    # cash_bag — одно слово: не дробить на cash + bag
    words = re.findall(r"[a-z_]+", " ".join(map(str, v)).lower() if isinstance(v, list) else str(v).lower())
    if not words:
        return "none" if isinstance(v, list) else "unclear"
    got = set()
    for c in choices:
        syn = _CHOICE_SYN.get(c, (c,))
        if any(w in syn or w == c for w in words):
            got.add(c)
    got.discard("none")
    if got:
        return "+".join(sorted(got))
    if any(w in ("none", "nothing", "no", "empty") for w in words):
        return "none"
    return "unclear"


def parse_notes(text: str, keys: Sequence[str]) -> dict[str, str]:
    """open-вопросы: {ключ: "what"} — что это, словами модели (для описания и разбора ошибок)."""
    out: dict[str, str] = {}
    by_norm = {_norm_key(k): k for k in keys}
    for obj in _json_objects(text or ""):
        for k, v in obj.items():
            key = k if k in keys else by_norm.get(_norm_key(k))
            if key and isinstance(v, dict) and isinstance(v.get("what"), str):
                out.setdefault(key, v["what"].strip()[:60])
    return out


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9а-яё]+", "", str(s).lower())


def _json_objects(text: str) -> list[dict]:
    """Все JSON-объекты в тексте: сначала самый внешний {...}, потом мелкие; одинарные кавычки чиним."""
    text = re.sub(r"```(?:json)?", "", text)
    cands = []
    a, b = text.find("{"), text.rfind("}")
    if a >= 0 and b > a:
        cands.append(text[a:b + 1])
    cands += re.findall(r"\{[^{}]*\}", text)
    out = []
    for c in cands:
        for variant in (c, c.replace("'", '"'), re.sub(r",\s*}", "}", c.replace("'", '"'))):
            try:
                obj = json.loads(variant)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
            break
    return out


def parse_answers(text: str, keys: Sequence[str], choices: dict[str, Sequence[str]] | None = None) -> dict[str, str]:
    """Ответ модели -> {ключ: yes|no|unclear} для КАЖДОГО ключа; не найденный ключ = unclear.
    Для ключей из choices ответ — список вариантов -> "cash+food" | "none" | "unclear".

    Ключи сопоставляются точно, затем без регистра/знаков, затем как «Image k»/«k» (номер картинки).
    Позиционно (по порядку значений) не сопоставляем: пропуск одного ответа сдвинул бы все остальные.
    """
    choices = choices or {}
    norm = (lambda k, v: norm_choices(v, choices[k]) if k in choices else norm_answer(v))
    res = {k: "unclear" for k in keys}
    found: dict[str, object] = {}
    for obj in _json_objects(text or ""):
        for k, v in obj.items():
            found.setdefault(str(k), v)
    by_norm = {_norm_key(k): k for k in keys}
    for k, v in found.items():
        key = k if k in res else by_norm.get(_norm_key(k))
        if key is None:
            m = re.fullmatch(r"(?:image|img|crop|picture)?\s*#?\s*(\d+)", k.strip().lower())
            if m and 1 <= int(m.group(1)) <= len(keys):
                key = keys[int(m.group(1)) - 1]
        if key is not None and res[key] == "unclear":
            res[key] = norm(key, v)
    # без JSON: строки вида «key: yes»
    for k in keys:
        if k not in found and res[k] == "unclear":
            m = re.search(r"[\"']?" + re.escape(k) + r"[\"']?\s*[:=\-–—]\s*[\"']?([A-Za-zА-Яа-я]+)", text or "")
            if m:
                res[k] = norm(k, m.group(1))
    return res


def parse_boxes(text: str, labels: Sequence[str], size: tuple[int, int] | None = None) -> dict[str, list[Box]]:
    """Grounding Qwen3-VL: [{"bbox_2d": [x1,y1,x2,y2], "label": ...}] в сетке 0..1000 -> доли кадра.

    Если координата > 1000, считаем, что модель ответила в пикселях присланной картинки (size).
    Метка сопоставляется точно, без регистра, затем по вхождению.
    """
    out: dict[str, list[Box]] = {lab: [] for lab in labels}
    norm = {_norm_key(lab): lab for lab in labels}
    for obj in re.findall(r"\{[^{}]*\}", re.sub(r"```(?:json)?", "", text or "")):
        try:
            it = json.loads(obj)
        except json.JSONDecodeError:
            continue
        b = it.get("bbox_2d") or it.get("bbox")
        if not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)):
            continue
        # модель иногда повторяет «метка: описание» целиком — берём часть до двоеточия;
        # иначе длинная метка с «...with the laptop...» ушла бы в laptop
        raw = _norm_key(str(it.get("label", "")).split(":")[0])
        lab = norm.get(raw) or next((lab for n, lab in sorted(norm.items(), key=lambda kv: -len(kv[0]))
                                     if n and raw.startswith(n)), None) \
            or next((lab for n, lab in norm.items() if n and (n in raw or raw and raw in n)), None)
        if lab is None:
            continue
        sx, sy = (size if size and max(b) > 1000 else (1000, 1000))
        x1, x2 = sorted((min(1.0, max(0.0, b[0] / sx)), min(1.0, max(0.0, b[2] / sx))))
        y1, y2 = sorted((min(1.0, max(0.0, b[1] / sy)), min(1.0, max(0.0, b[3] / sy))))
        if x2 - x1 > 1e-3 and y2 - y1 > 1e-3:
            out[lab].append((x1, y1, x2, y2))
    return out


# ---------- клиент ----------

@dataclass
class AskStats:
    calls: int = 0  # реальные запросы к серверу
    cached: int = 0  # ответы из кеша
    items: int = 0  # вырезок спрошено (с кешем)
    errors: int = 0
    wall_sec: float = 0.0
    prompt_tokens: int = 0
    gen_tokens: int = 0
    log: list = field(default_factory=list)  # (вид, вырезок, с, prompt_tokens, gen_tokens) по каждому запросу


class CropAsker:
    """Вопросы да/нет по вырезкам и grounding зон. transport(body)->response — подмена сервера в тестах."""

    def __init__(self, url: str | None = None, key_file: str | None = None, model: str = "local",
                 cache_dir: str | None = None, timeout: float = 60,
                 transport: Callable[[dict], dict] | None = None):
        url = url or os.environ.get("FORTE_VLM_URL") or "http://127.0.0.1:18080"
        _check_local(url)
        self.url, self.model, self.timeout = _endpoint(url), model, timeout
        key_file = key_file or os.environ.get("FORTE_VLM_KEY_FILE")
        self._key = Path(key_file).read_text(encoding="utf-8").strip() if key_file else None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, dict] = {}
        self._transport = transport or self._http
        self.stats = AskStats()
        self.last_ground: dict = {}

    # --- транспорт и кеш ---
    def _http(self, body: dict) -> dict:
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        if self._key:
            req.add_header("Authorization", "Bearer " + self._key)
        # прокси окружения не используем: кадры не должны уходить никуда, кроме локального сервера
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    @staticmethod
    def _video_id(path: str) -> list:
        """Ролик по содержимому, а не по имени: оценка и сервис кладут его как <job_id>.mov, и ключ по имени
        не совпадал между прогонами — кеш ответов не срабатывал ни разу. Размер + sha1 первого и последнего МБ."""
        p = Path(path)
        if not p.exists():
            return [p.name, -1]
        size = p.stat().st_size
        h = hashlib.sha1()
        with open(p, "rb") as f:
            h.update(f.read(1 << 20))
            if size > 2 << 20:
                f.seek(-(1 << 20), 2)
                h.update(f.read(1 << 20))
        return [size, h.hexdigest()[:16]]

    def _cache_get(self, key: str) -> dict | None:
        if key in self._mem:
            return self._mem[key]
        if self.cache_dir and (f := self.cache_dir / f"{key}.json").exists():
            try:
                self._mem[key] = json.loads(f.read_text(encoding="utf-8"))
                return self._mem[key]
            except (OSError, json.JSONDecodeError):
                return None
        return None

    def _cache_put(self, key: str, rec: dict) -> None:
        self._mem[key] = rec
        if self.cache_dir:
            tmp = self.cache_dir / f"{key}.tmp"
            tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.cache_dir / f"{key}.json")

    def _call(self, kind: str, content: list, max_tokens: int, n_items: int) -> tuple[str, str | None]:
        """Один запрос; возвращает (текст, ошибка). Ошибки не кешируются."""
        body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                "max_tokens": int(max_tokens), "temperature": 0.0}
        t0 = time.perf_counter()
        try:
            data = self._transport(body)
            text = str(data["choices"][0]["message"]["content"] or "").strip()
            err = None
        except Exception as e:  # noqa: BLE001 — сеть/таймаут/битый ответ: вызывающий получит unclear
            text, err = "", f"{type(e).__name__}: {e}"
            data = {}
        wall = time.perf_counter() - t0
        usage = data.get("usage") or {}
        pt, gt = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        self.stats.calls += 1
        self.stats.wall_sec += wall
        self.stats.prompt_tokens += pt
        self.stats.gen_tokens += gt
        self.stats.errors += err is not None
        self.stats.log.append((kind, n_items, round(wall, 2), pt, gt))
        return text, err

    # --- вопросы по вырезкам ---
    @staticmethod
    def build_content(items: Sequence[CropItem], images: Sequence[Image.Image], context: str = "") -> list:
        keys = [it.key for it in items]
        head = ""
        if context.strip():
            head += f"Known facts from the video: {context.strip()}\n\n"
        # вопрос звучит один раз на пакет: модель тратит время в основном на чтение запроса (prefill),
        # а длинный вопрос «что лежит на столе» в пакете из 8 вырезок повторялся бы 8 раз
        qs = list(dict.fromkeys(it.question for it in items))
        qid = {q: i for i, q in enumerate(qs, 1)}
        head += (f"You will see {len(items)} images. Each one is an enlarged crop of one area of the scene "
                 "from a fixed ceiling CCTV camera in a bank cash room, at one moment. "
                 "Each image has its own question (Q1, Q2, ... below); answer it about that image only.\n"
                 + "\n".join(f"Q{i}: {q}" for i, q in enumerate(qs, 1)))
        content: list = [{"type": "text", "text": head}]
        for k, (it, im) in enumerate(zip(items, images), 1):
            text = f"Image {k} — key={it.key}, area={it.zone}, t={it.t:.1f}s. Question: Q{qid[it.question]}"
            content.append({"type": "text", "text": text})
            content.append({"type": "image_url", "image_url": {"url": _jpeg_url(im)}})
        def one(it):
            if it.choices and it.open:
                return f'"{it.key}": {{"what": "<what it is, 1-4 words>", "group": ' + "|".join(f'"{c}"' for c in it.choices) + "}"
            if it.choices:
                return f'"{it.key}": [' + "|".join(f'"{c}"' for c in it.choices) + ", ...]"
            return f'"{it.key}": "yes"|"no"|"unclear"'
        schema = ", ".join(one(it) for it in items)
        tail = ("Return ONLY a JSON object, no other text: {" + schema + "}. "
                'Use "unclear" if the image does not show enough to answer.')
        if any(it.choices and not it.open for it in items):
            tail += (' For a "which" question answer a list of every listed word that applies, '
                     'or ["none"] if none of them is there.')
        if any(it.open for it in items):
            tail += (' For a "what is it" question first name the object in your own words, then pick the closest '
                     'group ("other" if none fits, "none" if there is no object).')
        content.append({"type": "text", "text": tail})
        return content

    def _chunk_key(self, vid: list, items: Sequence[CropItem], context: str) -> str:
        rec = [PROMPT_VERSION, self.model, vid, context.strip(),
               [[it.key, round(it.t, 3), it.zone, [round(v, 4) for v in it.box], it.question, round(it.zoom, 3)]
                + ([list(it.choices)] if it.choices else []) + (["open"] if it.open else []) for it in items]]
        return hashlib.sha1(json.dumps(rec, ensure_ascii=False).encode()).hexdigest()

    def ask(self, video_path: str, items: Sequence[CropItem], context: str = "") -> list[CropAnswer]:
        """Ответ на каждую вырезку в порядке items. Не более MAX_ITEMS вырезок на запрос; ключи уникальны."""
        items = list(items)
        if len({it.key for it in items}) != len(items):
            raise ValueError("ключи CropItem должны быть уникальны")
        if not items:
            return []
        vid = self._video_id(video_path)
        chunks = [items[i:i + MAX_ITEMS] for i in range(0, len(items), MAX_ITEMS)]
        keys = [self._chunk_key(vid, ch, context) for ch in chunks]
        todo = [i for i, k in enumerate(keys) if self._cache_get(k) is None]
        frames = {}
        if todo:  # один проход декодера на весь вызов
            need = [it.t for i in todo for it in chunks[i]]
            frames = dict(zip(need, grab_frames(video_path, need)))
        answers: dict[str, CropAnswer] = {}
        for i, (ch, key) in enumerate(zip(chunks, keys)):
            self.stats.items += len(ch)
            rec = self._cache_get(key)
            if rec is not None:
                self.stats.cached += 1
                text, err = rec["raw"], None
            else:
                images = [crop_image(frames[it.t][1], it.box, it.zoom) for it in ch]
                # открытый ответ {"what": до 6 слов, "group": ...} длиннее выбора: без запаса он обрезался
                max_tokens = max(12 * len(ch) + 40, sum(len(it.key) // 3 + 8 + 12 * bool(it.choices) + 20 * bool(it.open)
                                                        for it in ch) + 40)
                text, err = self._call("ask", self.build_content(ch, images, context), max_tokens, len(ch))
                if err is None:
                    self._cache_put(key, {"raw": text, "keys": [it.key for it in ch]})
            parsed = parse_answers(text, [it.key for it in ch], {it.key: it.choices for it in ch if it.choices})
            notes = parse_notes(text, [it.key for it in ch if it.open]) if any(it.open for it in ch) else {}
            for it in ch:
                answers[it.key] = CropAnswer(it.key, parsed[it.key], text if err is None else f"error: {err}",
                                             note=notes.get(it.key, ""))
        return [answers[it.key] for it in items]

    # --- grounding ---
    @staticmethod
    def ground_prompt(labels: dict[str, str]) -> str:
        cats = "\n".join(f"- {k}: {v}" for k, v in labels.items())
        return ("Locate each of these objects in the image (label: description):\n" + cats + "\n"
                'Report bbox coordinates in JSON format: [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]. '
                "Use the label before the colon exactly as written. Skip objects that are not visible.")

    def ground_candidates(self, path: str, t: float, labels: dict[str, str],
                          region: Box | None = None) -> dict[str, list[Box]]:
        """Все рамки по каждой метке на кадре t (доли кадра). region — искать внутри увеличенной вырезки."""
        key = hashlib.sha1(json.dumps([PROMPT_VERSION, "ground", self.model, self._video_id(path), round(t, 3),
                                       sorted(labels.items()), region and [round(v, 4) for v in region]],
                                      ensure_ascii=False).encode()).hexdigest()
        rec = self._cache_get(key)
        if rec is not None:
            self.stats.cached += 1
        else:
            (_, im), = grab_frames(path, [t])
            if region is not None:
                (rx1, ry1, rx2, ry2), out = crop_region(im.size, region, pad=0.0)
                W, H = im.size
                im = im.crop((rx1, ry1, rx2, ry2)).resize(out, Image.LANCZOS)
                frame_reg = (rx1 / W, ry1 / H, rx2 / W, ry2 / H)
            else:
                frame_reg = (0.0, 0.0, 1.0, 1.0)
            content = [{"type": "image_url", "image_url": {"url": _jpeg_url(im)}},
                       {"type": "text", "text": self.ground_prompt(labels)}]
            text, err = self._call("ground", content, 60 * len(labels) + 60, 1)
            rec = {"raw": text, "size": list(im.size), "region": list(frame_reg), "error": err}
            if err is None:
                self._cache_put(key, rec)
        self.last_ground = rec  # для диагностики: сырой ответ и ошибка
        rx1, ry1, rx2, ry2 = rec["region"]
        found = parse_boxes(rec["raw"], list(labels), tuple(rec["size"]))
        return {lab: [(rx1 + b[0] * (rx2 - rx1), ry1 + b[1] * (ry2 - ry1),
                       rx1 + b[2] * (rx2 - rx1), ry1 + b[3] * (ry2 - ry1)) for b in boxes]
                for lab, boxes in found.items()}

    def ground(self, path: str, t: float, labels: dict[str, str], region: Box | None = None) -> dict[str, Box | None]:
        """Первая рамка каждой метки (доли кадра) или None. labels: {метка: описание по-английски}."""
        return {lab: (bs[0] if bs else None) for lab, bs in self.ground_candidates(path, t, labels, region).items()}
