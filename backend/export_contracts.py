"""Regenerate static contracts: python -m backend.export_contracts."""

import base64
import io
import json
from pathlib import Path

from PIL import Image
from pydantic import TypeAdapter

from backend.app import create_app
from backend.config import Settings
from backend.contracts import AnalysisContext, JobResponse, ObservationBatch, RuleEvent


def main():
    destination = Path(__file__).resolve().parent.parent / "contracts"
    destination.mkdir(exist_ok=True)
    schemas = {
        "job-response.schema.json": TypeAdapter(JobResponse).json_schema(),
        "analysis-context.schema.json": AnalysisContext.model_json_schema(),
        "observations.schema.json": ObservationBatch.model_json_schema(),
        "rule-event.schema.json": RuleEvent.model_json_schema(),
        "openapi.json": create_app(Settings(_env_file=None)).openapi(),
    }
    for name, schema in schemas.items():
        (destination / name).write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (60, 90, 120)).save(buffer, format="JPEG")
    examples = {
        "accepted": {"job_id": "example-job"},
        "processing": {"job_id": "example-job", "status": "processing"},
        "failed": {"job_id": "example-job", "status": "failed"},
        "no_violation": {"job_id": "example-job", "status": "done", "verdict": "no_violation", "violations": []},
        "violation": {
            "job_id": "example-job", "status": "done", "verdict": "violation",
            "violations": [{"type": "Synthetic contract example", "timecode_sec": 1.2,
                            "description": "Synthetic observation, not a real finding. Example document, section 1. https://example.org/rules",
                            "evidence_frames": [{"timestamp_sec": 1.2, "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii")}]}],
        },
    }
    for name in ("processing", "failed", "no_violation", "violation"):
        TypeAdapter(JobResponse).validate_python(examples[name])
    (destination / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
