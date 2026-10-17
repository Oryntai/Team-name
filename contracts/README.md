# Service contracts

Contract version: 1. Backend implements `backend/contracts.py`; generated JSON schemas and examples are in this directory. The jury's response schema is fixed. Internal observations are never returned as additional jury fields.

## Frontend integration

- Development backend: `http://localhost:8000`; frontend: `http://localhost:5173`.
- Proxy `/ui-api` from the frontend development server to port 8000. Production uses one origin.
- `POST /ui-api/analyze`: multipart field `file`, `.mov` file. Returns `{"job_id":"..."}`.
- `GET /ui-api/jobs/{job_id}`: poll every 1–2 seconds until `done` or `failed`.
- Public UI requests do not contain a jury key. Never put `FORTE_API_KEY` in browser code or a `VITE_*` variable.
- Jury routes are `/api/v1/analyze` and `/api/v1/jobs/{job_id}`, with the same payloads and mandatory `X-API-Key`.
- Queued and running jobs both appear as `processing`. A failed job exposes only its ID and status. A completed clean analysis has `verdict: no_violation` and `violations: []`.
- Every violation has `type`, `timecode_sec`, `description`, and 1–3 `evidence_frames`. Frames are JPEG base64 without a data URL prefix. Add `data:image/jpeg;base64,` only when constructing the image element.
- `description` includes the observation, document title, section, and normative URL.
- Upload failures use HTTP 400/413/415; a full queue returns 503; missing jobs return 404. Analysis failure is a successful polling response with `status: failed`.

## ML integration

Implement a no-argument factory `your_package.module:create_analyzer` returning an object with:

```python
def analyze(self, context: AnalysisContext, video: VideoIO) -> ObservationBatch:
    ...
```

Set `FORTE_ANALYZER_FACTORY=your_package.module:create_analyzer`. Factory construction and model loading happen only in the worker's model process, never in the HTTP server. Download all weights and dependencies before offline execution. Return `complete=True` only after the required analysis is complete; raise on failures. An empty event list does not authorize skipping the video.

`context` contains `job_id`, the generated absolute `video_path`, absolute Unix `deadline_at`, and `profile` (`standard` by default). Check the deadline during inference. The supervisor also enforces it externally.

Use the supplied `video` object for metadata, sampled frames, and evidence timestamps. Do not create a second decoder. All times are seconds relative to the beginning of the video, derived from real frame PTS. The shared decoder handles orientation. `frame_pts` must reference frames from this decoder.

Exact decoder signatures and sampling limits are in [video_io.md](video_io.md). `sample_frames` stops at its configured cap; that alone does not establish complete coverage. `examples.json` contains a generated synthetic JPEG for contract illustration, not an analysis finding. Regenerate schemas and examples with `python -m backend.export_contracts`.

Each observed event contains `t_start`, `t_end`, `t_main`, `frame_pts`, predicate states (`observed`, `not_observed`, `unclear`), per-object visibility (`clear`, `partial`, `unusable`), and candidate `sources` (`vlm`, `motion`, `detector`). Do not identify faces or analyze audio.

## Rules integration

Implement a no-argument factory `your_package.module:create_rules` with:

```python
def evaluate(self, observations: ObservationBatch, context: AnalysisContext) -> list[RuleEvent]:
    ...
```

Set `FORTE_RULES_FACTORY=your_package.module:create_rules`. Return only supported events with verified normative references. Each `RuleEvent` includes its time span, type, factual description, `rule_id`, `rule_title`, `rule_section`, `rule_url`, and 1–3 `frame_pts`. The backend validates these, extracts real JPEGs, and uses real returned frame timestamps in the final response. The backend never invents a rule or an ML finding.

## Development adapter

The default development adapter deliberately reports analysis unavailable. It allows frontend teams to exercise upload and polling without pretending ML exists. Synthetic outcome adapters are confined to tests. Production refuses to start without explicitly configured analyzer and rules factories, and refuses the development adapter.
