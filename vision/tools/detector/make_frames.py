"""Кадры из роликов для обучения детектора + манифест с группами и ракурсами.

    python make_frames.py --videos DIR --labels team_labels.csv --out frames/

Что берётся: весь ролик редко (--fps), а вокруг размеченной секунды нарушения — часто (--event-fps
в окне ±--event-window). Почти одинаковые соседние кадры отбрасываются: статичная сцена не должна
заполнить датасет одним и тем же столом.

Группа (столбец group) — ролик, его «чистый близнец» и побайтные дубли (Х-4 = Х-47, Х-5 = Х-48): при
делении на train/valid/test группа целиком уходит в одну часть, иначе модель видит ответ.
Ракурс (столбец view) — кластер по средней картинке ролика: текстовые пометки ракурса в разборе
противоречат друг другу, поэтому считаем по самим кадрам.
Имена «X-1» (латиница) и «Х-11» (кириллица) приводятся к одному виду X-<номер>.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

import numpy as np
from PIL import Image

from forte_vision.video import grab_at, probe


def norm_id(name: str) -> str:
    """'Х-11.mov', 'X-11', 'х-11' -> 'X-11'."""
    m = re.search(r"(\d+)", Path(name).stem)
    if not m:
        raise ValueError(f"не понял номер ролика: {name}")
    return f"X-{int(m.group(1))}"


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_labels(path: Path) -> dict[str, dict]:
    """team_labels.csv: id, official_verdict, official_sec, team_sec, type_code, clean_pair, ..."""
    out = {}
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            vid = norm_id(row["id"])
            sec = row.get("official_sec") or row.get("team_sec") or ""
            out[vid] = {
                "violation": row.get("official_verdict", "").strip() == "violation",
                "sec": float(sec) if sec.strip() else None,
                "type": (row.get("type_code") or "").strip(),
                "pairs": [norm_id(p) for p in (row.get("clean_pair") or "").split("|") if p.strip()],
            }
    return out


class Groups:
    """Объединение роликов в группы (близнецы + дубли)."""

    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def sample_times(duration: float, fps: float, event: float | None, event_fps: float, window: float) -> list[float]:
    times = set(np.round(np.arange(0.0, duration, 1.0 / fps), 2).tolist())
    if event is not None:
        lo, hi = max(0.0, event - window), min(duration, event + window)
        times |= set(np.round(np.arange(lo, hi, 1.0 / event_fps), 2).tolist())
    return sorted(times)


def thumb(im: Image.Image, size=(24, 42)) -> np.ndarray:
    return np.asarray(im.convert("L").resize(size, Image.BILINEAR), dtype=np.float32) / 255.0


def cluster_views(means: dict[str, np.ndarray], threshold: float) -> dict[str, int]:
    """Средняя связь по корреляции средних кадров: ракурс = номер кластера.

    На открытом наборе даёт 11 постановок камеры, и разбиение одинаковое при пороге 0.95–0.97.
    """
    ids = sorted(means, key=lambda v: int(v.split("-")[1]))
    z = []
    for vid in ids:
        x = means[vid].ravel()
        z.append((x - x.mean()) / (x.std() + 1e-6))
    corr = np.array([[float(np.mean(a * b)) for b in z] for a in z])
    clusters = [[i] for i in range(len(ids))]
    while len(clusters) > 1:
        score, a, b = max((corr[np.ix_(clusters[a], clusters[b])].mean(), a, b)
                          for a in range(len(clusters)) for b in range(a + 1, len(clusters)))
        if score < threshold:
            break
        clusters[a] += clusters.pop(b)
    clusters.sort(key=min)
    return {ids[i]: n for n, c in enumerate(clusters) for i in c}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=0.5, help="частота по всему ролику")
    ap.add_argument("--event-fps", type=float, default=2.0, help="частота около размеченной секунды")
    ap.add_argument("--event-window", type=float, default=4.0, help="± секунд вокруг нарушения")
    ap.add_argument("--min-diff", type=float, default=0.015, help="порог отличия от прошлого кадра (0 — не отбрасывать)")
    ap.add_argument("--view-threshold", type=float, default=0.96, help="корреляция средних кадров для одного ракурса")
    ap.add_argument("--only", default="", help="номера роликов через запятую — для пробного прогона")
    args = ap.parse_args()

    labels = read_labels(args.labels)
    videos = sorted(p for p in args.videos.iterdir() if p.suffix.lower() == ".mov")
    if args.only:
        keep = {norm_id(x) for x in args.only.split(",")}
        videos = [p for p in videos if norm_id(p.name) in keep]
    if not videos:
        raise SystemExit(f"нет .mov в {args.videos}")
    args.out.mkdir(parents=True, exist_ok=True)

    groups = Groups()
    by_hash: dict[str, str] = {}
    dup_of: dict[str, str] = {}
    for p in videos:
        vid = norm_id(p.name)
        groups.find(vid)
        for twin in labels.get(vid, {}).get("pairs", []):
            groups.union(vid, twin)
        h = file_hash(p)
        if h in by_hash:  # побайтный дубль: та же картинка, кадры не повторяем
            groups.union(vid, by_hash[h])
            dup_of[vid] = by_hash[h]
            print(f"{vid}: дубль {by_hash[h]}, кадры не берём")
        else:
            by_hash[h] = vid

    rows, means = [], {}
    for p in videos:
        vid = norm_id(p.name)
        if vid in dup_of:
            continue
        lab = labels.get(vid, {})
        info = probe(str(p))
        times = sample_times(info.duration_sec, args.fps, lab.get("sec"), args.event_fps, args.event_window)
        kept, prev, acc = 0, None, None
        for t, im in grab_at(str(p), times):
            th = thumb(im)
            acc = th if acc is None else acc + th
            near = lab.get("sec") is not None and abs(t - lab["sec"]) <= args.event_window
            if prev is not None and not near and float(np.mean(np.abs(th - prev))) < args.min_diff:
                continue
            prev = th
            name = f"{vid}_{t:07.2f}.jpg"
            im.convert("RGB").save(args.out / name, quality=92)
            rows.append({"file": name, "video": vid, "t": f"{t:.3f}", "width": im.width, "height": im.height,
                         "violation": int(bool(lab.get("violation"))), "type": lab.get("type", ""),
                         "near_event": int(near)})
            kept += 1
        means[vid] = acc / max(1, len(times))
        print(f"{vid}: {kept} кадров из {len(times)} ({info.duration_sec:.1f} с)")

    views = cluster_views(means, args.view_threshold)
    for vid, orig in dup_of.items():  # дубль получает ракурс оригинала
        views[vid] = views[orig]
    for r in rows:
        r["group"] = groups.find(r["video"])
        r["view"] = views[r["video"]]
    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by_view: dict[int, list[str]] = {}
    for vid, v in sorted(views.items()):
        by_view.setdefault(v, []).append(vid)
    print(f"\nкадров: {len(rows)}; групп: {len({r['group'] for r in rows})}; ракурсов: {len(by_view)}")
    for v, vids in sorted(by_view.items()):
        print(f"  ракурс {v}: {', '.join(vids)}")


if __name__ == "__main__":
    main()
