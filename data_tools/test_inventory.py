import math
from pathlib import Path

import pytest

from data_tools.inventory import inventory


def test_duplicate_timestamp_conflict_is_not_two_events(tmp_path: Path):
    (tmp_path / "a.mov").write_bytes(b"same synthetic bytes")
    (tmp_path / "b.mov").write_bytes(b"same synthetic bytes")
    result = inventory(tmp_path, [
        {"file": "a.mov", "label": 1, "timestamp_sec": 3},
        {"file": "b.mov", "label": 1, "timestamp_sec": 5},
    ])
    assert result["unique_content_count"] == 1
    assert result["duplicate_groups"][0]["annotation_conflict"] is True
    assert all(r["split"] == "unassigned" for r in result["records"])
    assert not result["split_ready"]


def test_unlabelled_duplicates_and_uppercase_extension(tmp_path: Path):
    (tmp_path / "a.MOV").write_bytes(b"same")
    (tmp_path / "b.mov").write_bytes(b"same")
    result = inventory(tmp_path)
    assert result["file_count"] == 2
    assert not result["duplicate_groups"][0]["annotation_conflict"]
    assert result["annotated_file_count"] == 0


def test_latin_and_cyrillic_names_are_not_merged(tmp_path: Path):
    (tmp_path / "X-1.mov").write_bytes(b"a")
    (tmp_path / "Х-1.mov").write_bytes(b"b")
    result = inventory(tmp_path)
    assert result["unique_content_count"] == 2


@pytest.mark.parametrize("timestamp", [True, -1, math.nan, math.inf, "2"])
def test_reject_invalid_timestamp(tmp_path: Path, timestamp):
    (tmp_path / "a.mov").write_bytes(b"a")
    with pytest.raises(ValueError):
        inventory(tmp_path, [{"file": "a.mov", "label": 1, "timestamp_sec": timestamp}])


def test_missing_file_annotation_is_not_silently_ignored(tmp_path: Path):
    (tmp_path / "a.mov").write_bytes(b"a")
    with pytest.raises(ValueError, match="no matching"):
        inventory(tmp_path, [{"file": "missing.mov", "label": 0}])
