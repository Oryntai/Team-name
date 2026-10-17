"""Черновая разметка рамок через локальный Qwen3-VL (grounding) -> COCO + картинки для проверки глазами.

    python autolabel.py --frames frames/ --url http://127.0.0.1:8080 --key-file .vlm_key --out labels/

Один запрос на кадр со всеми классами из classes.py. Ответ модели — JSON со списком
{"bbox_2d": [x1, y1, x2, y2], "label": "..."}; Qwen3-VL отдаёт координаты в относительной сетке 0..1000
(--coords rel1000), для моделей с пиксельными координатами — --coords abs.
Сырые ответы кешируются в out/raw/: повторный запуск продолжает с места остановки.
Это черновик для обучения, а не эталон: рамки из overlays/ просматриваются человеком, явные ошибки
удаляются в любом редакторе COCO (CVAT, Label Studio — локально) до обучения.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

from classes import BY_PROMPT, CLASSES, NAMES, select

COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
          "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324"]


def prompt_for(classes) -> str:
    cats = "\n".join(f"- {c.name}: {c.prompt}" for c in classes)
    return ("Locate every instance of the following categories (label: description):\n"
            f"{cats}\n"
            "Small objects matter: keys in a lock, banknotes in hands, documents on the desk. "
            "Use the label before the colon exactly as written. "
            'Report bbox coordinates in JSON format: [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]. '
            "If none are visible, return [].")


def ask(url: str, key: str | None, image: Image.Image, text: str, timeout: float, max_tokens: int) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=92)
    body = json.dumps({
        "model": "local",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}},
            {"type": "text", "text": text}]}],
        "max_tokens": max_tokens, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def parse(text: str) -> list[dict]:
    """Достаёт JSON-список из ответа (с ```json-обёрткой или без); мусор — пустой список."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [i for i in items if isinstance(i, dict) and isinstance(i.get("bbox_2d"), list) and len(i["bbox_2d"]) == 4]


def to_label(raw: str) -> str | None:
    raw = (raw or "").strip().strip('"').lower()
    if raw in NAMES:
        return raw
    for prompt, name in BY_PROMPT.items():
        if raw == prompt.lower() or raw in prompt.lower() or prompt.lower() in raw:
            return name
    # модель иногда пересказывает описание своими словами: берём класс с наибольшим пересечением слов
    words = set(re.findall(r"[a-z]+", raw)) - {"a", "or", "the", "of", "with", "its", "and", "showing"}
    best = max(BY_PROMPT.items(), key=lambda kv: len(words & set(re.findall(r"[a-z]+", kv[0].lower()))))
    return best[1] if len(words & set(re.findall(r"[a-z]+", best[0].lower()))) >= 2 else None


def to_xywh(box, w: int, h: int, coords: str) -> list[float] | None:
    x1, y1, x2, y2 = (float(v) for v in box)
    if coords == "rel1000":
        x1, x2, y1, y2 = x1 * w / 1000, x2 * w / 1000, y1 * h / 1000, y2 * h / 1000
    x1, x2 = sorted((max(0.0, min(w, x1)), max(0.0, min(w, x2))))
    y1, y2 = sorted((max(0.0, min(h, y1)), max(0.0, min(h, y2))))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)]


SMALL = {"key", "cash", "document"}
ANCHORS = {"safe_open", "safe_closed", "money_counter", "cash_bag", "screen_unlocked", "screen_locked"}


def iou(a: list[float], b: list[float]) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-6)


def zoom_regions(found, w: int, h: int, pad: float = 1.0, min_side: int = 200, limit: int = 3):
    """Области вокруг крупных предметов (сейф, счётчик, ноутбук), расширенные на pad размеров рамки."""
    regions: list[tuple[int, int, int, int]] = []
    for name, (x, y, bw, bh) in sorted(found, key=lambda f: -f[1][2] * f[1][3]):
        if name not in ANCHORS:
            continue
        cx, cy = x + bw / 2, y + bh / 2
        side_w, side_h = max(min_side, bw * (1 + 2 * pad)), max(min_side, bh * (1 + 2 * pad))
        r = (int(max(0, cx - side_w / 2)), int(max(0, cy - side_h / 2)),
             int(min(w, cx + side_w / 2)), int(min(h, cy + side_h / 2)))
        rb = [r[0], r[1], r[2] - r[0], r[3] - r[1]]
        if all(iou(rb, [o[0], o[1], o[2] - o[0], o[3] - o[1]]) < 0.5 for o in regions):
            regions.append(r)
        if len(regions) >= limit:
            break
    return regions


def overlay(im: Image.Image, anns: list[tuple[str, list[float]]]) -> Image.Image:
    im = im.convert("RGB")
    d = ImageDraw.Draw(im)
    for name, (x, y, bw, bh) in anns:
        c = COLORS[NAMES.index(name) % len(COLORS)]
        d.rectangle([x, y, x + bw, y + bh], outline=c, width=3)
        d.text((x + 3, y + 2), name, fill=c)
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", type=Path, required=True, help="папка make_frames.py (с manifest.csv)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8080", help="OpenAI-совместимый llama-server")
    ap.add_argument("--key-file", type=Path, help="файл с ключом llama-server (ключ не передаётся аргументом)")
    ap.add_argument("--classes", default="", help="подмножество классов через запятую")
    ap.add_argument("--coords", choices=["rel1000", "abs"], default="rel1000")
    ap.add_argument("--limit", type=int, default=0, help="только первые N кадров — пробный прогон")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--no-zoom", dest="zoom", action="store_false",
                    help="без второго прохода по увеличенным вырезкам (быстрее, но мелочь теряется)")
    args = ap.parse_args()

    classes = select(args.classes)
    text = prompt_for(classes)
    key = args.key_file.read_text(encoding="utf-8").strip() if args.key_file else None
    with (args.frames / "manifest.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    raw_dir, ov_dir = args.out / "raw", args.out / "overlays"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ov_dir.mkdir(parents=True, exist_ok=True)

    wanted = {c.name for c in classes}
    small = tuple(c for c in classes if c.name in SMALL)
    zoom_text = prompt_for(small) if small and args.zoom else None

    def cached(cache: Path, image: Image.Image, question: str) -> str | None:
        if cache.exists():
            return cache.read_text(encoding="utf-8")
        try:
            reply = ask(args.url, key, image, question, args.timeout, args.max_tokens)
        except Exception as e:  # сеть/таймаут: при повторном запуске спросится снова
            print(f"{cache.name}: ошибка запроса: {e}")
            return None
        cache.write_text(reply, encoding="utf-8")
        return reply

    def boxes(reply: str, w: int, h: int, allowed: set[str]) -> list[tuple[str, list[float]]]:
        nonlocal unknown
        out = []
        for item in parse(reply):
            name = to_label(str(item.get("label", "")))
            if name is None or name not in allowed:
                unknown += 1
                continue
            box = to_xywh(item["bbox_2d"], w, h, args.coords)
            if box is not None:
                out.append((name, box))
        return out

    images, annotations, counts, unknown, failed = [], [], {n: 0 for n in NAMES}, 0, 0
    started = time.time()
    for i, row in enumerate(rows, 1):
        im = Image.open(args.frames / row["file"])
        reply = cached(raw_dir / (row["file"] + ".txt"), im, text)
        if reply is None:
            failed += 1
            continue
        found = boxes(reply, im.width, im.height, wanted)
        # Второй проход: мелочь (ключ, деньги, документ) ищется на увеличенной вырезке вокруг крупных
        # предметов, на целом кадре 464x848 модель её не видит.
        if zoom_text:
            for k, (x0, y0, x1, y1) in enumerate(zoom_regions(found, im.width, im.height)):
                crop = im.crop((x0, y0, x1, y1))
                scale = max(1.0, 640 / max(crop.size))
                crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.BICUBIC)
                zr = cached(raw_dir / f"{row['file']}.zoom{k}.txt", crop, zoom_text)
                if zr is None:
                    continue
                for name, (bx, by, bw, bh) in boxes(zr, crop.width, crop.height, {c.name for c in small}):
                    b = [round(x0 + bx / scale, 1), round(y0 + by / scale, 1), round(bw / scale, 1), round(bh / scale, 1)]
                    if all(n != name or iou(b, o) < 0.5 for n, o in found):
                        found.append((name, b))
        image_id = len(images) + 1
        images.append({"id": image_id, "file_name": row["file"], "width": im.width, "height": im.height})
        for name, box in found:
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": NAMES.index(name) + 1,
                                "bbox": box, "area": round(box[2] * box[3], 1), "iscrowd": 0})
            counts[name] += 1
        overlay(im, found).save(ov_dir / row["file"], quality=85)
        if i % 25 == 0 or i == len(rows):
            rate = (time.time() - started) / i
            print(f"{i}/{len(rows)} кадров, {rate:.1f} с/кадр, осталось ~{rate * (len(rows) - i) / 60:.0f} мин")

    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": k + 1, "name": c.name, "supercategory": "cash_desk"} for k, c in enumerate(CLASSES)]}
    (args.out / "autolabel.coco.json").write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    print(f"\nкадров размечено: {len(images)}, рамок: {len(annotations)}, ошибок запроса: {failed}, "
          f"меток вне списка: {unknown}")
    for name in NAMES:
        print(f"  {name:16s} {counts[name]}")
    print(f"проверка глазами: {ov_dir}")


if __name__ == "__main__":
    main()
