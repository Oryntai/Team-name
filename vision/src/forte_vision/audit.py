"""Opt-in local diagnostics. Never log or transmit observation contents."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


def save_audit(video_id, observation, diagnostics, raw_intervals, pooled_intervals):
    root = os.environ.get("FORTE_LOCAL_AUDIT_DIR")
    if not root:
        return
    directory = Path(root).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    target = directory / (hashlib.sha256(video_id.encode()).hexdigest() + ".json")
    payload = {
        "observation": observation,
        "diagnostics": diagnostics,
        "raw_intervals": raw_intervals,
        "pooled_intervals": pooled_intervals,
    }
    fd, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
