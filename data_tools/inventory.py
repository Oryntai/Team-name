"""Inventory local video bytes, retaining duplicate label conflicts for review."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


def _key(value: str) -> str:
    # Normalize Unicode encoding, not visually similar Latin/Cyrillic letters.
    return unicodedata.normalize("NFC", value.replace("\\", "/"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_time(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def inventory(root: Path, annotations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Annotations optionally contain file, label (0/1), and timestamp_sec.

    Each annotation is retained as supplied. Different labels/timestamps for
    identical bytes are flagged; they never silently create multiple events.
    Scenario groups and dataset splits require a later human decision.
    """
    if not root.is_dir():
        raise ValueError("video directory does not exist")
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".mov")
    if not paths:
        raise ValueError("no .mov files found")
    lookup: dict[str, Path] = {}
    for path in paths:
        key = _key(path.relative_to(root).as_posix())
        if key in lookup:
            raise ValueError(f"Unicode-normalized filename collision: {key}")
        lookup[key] = path
    labels: dict[str, dict[str, Any]] = {}
    for annotation in annotations or []:
        if not isinstance(annotation, dict) or not isinstance(annotation.get("file"), str):
            raise ValueError("each annotation needs a string file")
        key = _key(annotation["file"])
        if key not in lookup:
            raise ValueError(f"annotation has no matching file: {key}")
        if key in labels:
            raise ValueError(f"duplicate annotation row: {key}")
        label = annotation.get("label")
        if type(label) is not int or label not in (0, 1):
            raise ValueError("label must be integer 0 or 1")
        timestamp = annotation.get("timestamp_sec")
        if timestamp is not None:
            _finite_time(timestamp, "timestamp_sec")
        labels[key] = dict(annotation)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records = []
    for key, path in lookup.items():
        digest = _sha256(path)
        row = {
            "video_id": key,
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
            "duplicate_group": digest,
            "scenario_group": None,
            "split": "unassigned",
            "annotation": labels.get(key),
            "annotation_complete": False,
        }
        records.append(row)
        groups[digest].append(row)
    duplicate_groups = []
    for digest, rows in groups.items():
        if len(rows) < 2:
            continue
        annotations_in_group = [r["annotation"] for r in rows if r["annotation"] is not None]
        signatures = {(a["label"], a.get("timestamp_sec")) for a in annotations_in_group}
        duplicate_groups.append({
            "sha256": digest,
            "video_ids": [r["video_id"] for r in rows],
            "annotation_conflict": len(signatures) > 1,
            "review_required": len(signatures) > 1,
        })
    return {
        "schema_version": 1,
        "file_count": len(records),
        "unique_content_count": len(groups),
        "annotated_file_count": len(labels),
        "records": records,
        "duplicate_groups": duplicate_groups,
        "split_ready": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path)
    parser.add_argument("--annotations", type=Path, help="Optional JSON array with file/label/timestamp_sec")
    parser.add_argument("--output", type=Path, required=True, help="Keep output in ignored local_data/")
    args = parser.parse_args()
    annotations = json.loads(args.annotations.read_text()) if args.annotations else None
    if annotations is not None and not isinstance(annotations, list):
        parser.error("annotations must be a JSON array")
    result = inventory(args.videos, annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("file_count", "unique_content_count", "annotated_file_count", "split_ready")}))


if __name__ == "__main__":
    main()
