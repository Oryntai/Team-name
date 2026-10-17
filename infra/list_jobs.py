"""Что загружали в сервис и что он ответил — без хранения самих видео.

    python infra/list_jobs.py --data-dir local_data/service
    python infra/list_jobs.py --data-dir local_data/service --since 3h --frames local_data/review

Видео удаляется сразу после обработки; в SQLite остаются имя файла, размер, sha256, ответ и
кадры-доказательства. --frames сохраняет эти кадры JPEG-файлами (<время>_<имя>_<job>/<секунда>.jpg),
чтобы глазами посмотреть, что было на присланных роликах в моменты найденных нарушений.
Только стандартная библиотека; база открывается только на чтение и сервис не мешает.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path


def parse_since(value: str | None) -> float | None:
    if not value:
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([mhd])", value.strip())
    if not m:
        raise SystemExit("--since: число и m/h/d, например 30m или 3h")
    return time.time() - float(m.group(1)) * {"m": 60, "h": 3600, "d": 86400}[m.group(2)]


def safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name)[:60] or "video"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", type=Path, default=Path("data"), help="FORTE_DATA_DIR сервиса")
    ap.add_argument("--since", help="только задания за последние 30m / 3h / 2d")
    ap.add_argument("--frames", type=Path, help="папка для кадров-доказательств")
    args = ap.parse_args()

    db = args.data_dir / "jobs.sqlite3"
    if not db.exists():
        raise SystemExit(f"нет базы {db}")
    try:
        con = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
        columns = {r[1] for r in con.execute("PRAGMA table_info(jobs)")}
    except sqlite3.OperationalError:
        # WAL-база без открытых соединений: режим ro не может создать -shm. Скрипт всё равно только читает.
        con = sqlite3.connect(db)
        columns = {r[1] for r in con.execute("PRAGMA table_info(jobs)")}
    con.row_factory = sqlite3.Row
    since = parse_since(args.since)
    rows = con.execute("SELECT * FROM jobs WHERE arrival_at >= ? ORDER BY arrival_at",
                       (since or 0,)).fetchall()

    print(f"{'пришло':16s} {'файл':28s} {'МБ':>5s} {'sha256':10s} {'статус':7s} {'вердикт':13s} "
          f"{'сек':>5s} {'видео':7s} события")
    saved = 0
    for r in rows:
        name = (r["original_name"] if "original_name" in columns else None) or "—"
        size = r["video_bytes"] if "video_bytes" in columns else None
        sha = (r["video_sha256"] if "video_sha256" in columns else None) or ""
        deleted = r["video_deleted_at"] if "video_deleted_at" in columns else None
        on_disk = Path(r["video_path"]).exists()
        result = json.loads(r["result_json"]) if r["result_json"] else {}
        events = result.get("violations", [])
        took = (r["finished_at"] - r["arrival_at"]) if r["finished_at"] else None
        print(f"{datetime.fromtimestamp(r['arrival_at']):%m-%d %H:%M:%S} {name[:28]:28s} "
              f"{(size or 0) / 1e6:5.1f} {sha[:10]:10s} {r['status']:7s} "
              f"{result.get('verdict', r['error_code'] or '—'):13s} {took if took is None else round(took):>5} "
              f"{'лежит' if on_disk else ('удалено' if deleted else 'нет')!s:7s} "
              + "; ".join(f"{e.get('type', '?')}@{e.get('timecode_sec', '?')}s" for e in events))
        if args.frames and events:
            folder = args.frames / f"{datetime.fromtimestamp(r['arrival_at']):%m%d-%H%M%S}_{safe(name)}_{r['job_id'][:8]}"
            folder.mkdir(parents=True, exist_ok=True)
            for e in events:
                for f in e.get("evidence_frames", []):
                    out = folder / f"{float(f['timestamp_sec']):07.2f}s_{safe(str(e.get('type', 'event')))}.jpg"
                    out.write_bytes(base64.b64decode(f["image_base64"]))
                    saved += 1
            (folder / "result.json").write_text(json.dumps(
                {k: v for k, v in result.items() if k != "violations"} | {"violations": [
                    {k: v for k, v in e.items() if k != "evidence_frames"} for e in events]},
                ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nзаданий: {len(rows)}" + (f"; кадров сохранено: {saved} в {args.frames}" if args.frames else ""))


if __name__ == "__main__":
    main()
