# Forte decision subsystem

Local-only Python package. No models, media loading, HTTP server, downloads, or cloud calls. Examples are **synthetic demonstration policy, not law**. Real legal rules require human review before use. No real-data accuracy is claimed.

From `decision/`:

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
python -m forte_decision.evaluation evaluate examples/predictions.json examples/annotations.json
python -m forte_decision.evaluation validate-splits manifest.json
```

```python
import json
from forte_decision import Observation, Rule, decide
observation = Observation.from_json(open('examples/observation.json').read())
rule = Rule(**json.load(open('examples/synthetic_rule.json')))
result = decide(observation, [rule])
print(result.to_json())
```

## Observer → decision contract (Claude/backend handoff)

`Observation` JSON has `video_id` (nonempty string), `duration_sec` (finite positive number), `analysis_status` (`complete`, `partial`, `failed`), `coverage_complete` (strict bool), and `candidates` (array). Complete coverage is an upstream assertion about processing, never inferred from an empty candidate array.

Each candidate has stable `candidate_id`, `start_sec`, `end_sec`, `main_sec`, `source`, `evidence` array, and optional `predicates` and `visibility` objects. Every atomic map value is exactly `observed`, `not_observed`, or `unclear`; missing keys mean unclear. `source` identifies the local observer/version; `visibility` keys name rule-specific visibility requirements. Do not replace tri-state observations with scores; scores are not calibrated probabilities and are not accepted in this schema. Unknown JSON fields fail constructor parsing.

Evidence references contain only `frame_id`, `pts_sec`, and `source`. IDs must be stable within one video; a repeated ID must have identical PTS/source. No media bytes or external frame URLs belong in this contract. All timestamps are finite numeric values; bools are rejected. Require `0 <= start <= main <= end <= duration`; each evidence PTS lies inside its candidate interval. Candidate IDs must be unique. Malformed input raises an exception and must become upstream analysis failure, never a clean negative.

Rule JSON fields: `rule_id`, `type`, `title`, `source_document`, `source_paragraph`, `source_url`, `applicability`, `required`, `exceptions`, `visibility_required`, `description_template`. Four predicate lists contain unique string names. Description is literal reviewed text (no executable/interpolated template). All applicability and required predicates plus visibility requirements must be observed. Every exception must explicitly be not_observed. Missing/unclear blocks a positive. At least one real input evidence reference is required. The engine records per-candidate reasons in diagnostics.

## Result and backend boundary

`DecisionResult` has `video_id`, `status` (`complete`/`incomplete`), `verdict` (`violation`/`no_violation` for complete only, otherwise null), `events`, and `diagnostics`. Incomplete results may retain provisional supported events, but must never publish a final binary decision. Backend should map incomplete to the official API's **failed** response. This internal result is not the official API schema.

An event has `event_id`, `video_id`, `rule_id`, `type`, `timecode_sec`, `start_sec`, `end_sec`, `description`, `evidence`, and `candidate_ids`. Description includes source document, paragraph and URL. Duplicate consolidation requires overlapping/touching intervals for the same video/rule and total group anchor span at most `max_anchor_delta_sec` (default 1.0). Disjoint intervals and distant anchors stay separate; chains cannot bridge past the total span threshold. Configure with `decide(observation, rules, max_anchor_delta_sec=1.0)`; this keyword-only finite nonnegative numeric value rejects bool/NaN. Zero permits only identical anchors. Anchor is an existing earliest candidate main timestamp; choose at most three original evidence references nearest that anchor. Sorting and IDs are stable under input reordering. IDs may change when new candidates join a group. All merged candidate IDs preserve provenance. The interval plus anchor threshold is a deliberate, tunable duplicate heuristic, not a learned event identity model. Tune on held-out labelled data; distinct events within the threshold can still merge.

Backend owns local decoding, exact frame extraction, JPEG encoding, and verification that frame contents and PTS match the requested video/reference. `adapter.to_public_payload(result, evidence_map, job_id="job-123")` implements the official completed response:

```json
{"job_id":"job-123","status":"done","verdict":"violation","violations":[{"type":"synthetic_marker","timecode_sec":3,"description":"Reviewed description and source citation","evidence_frames":[{"timestamp_sec":3,"image_base64":"<backend JPEG base64>"}]}]}
```

The serializer requires explicit backend-supplied `VerifiedJPEG` values keyed by frame ID. It checks video/ID/PTS consistency, verdict/event consistency, nonempty job ID and 1–3 unique evidence refs, then base64-encodes only supplied bytes. `VerifiedJPEG` is a backend assertion: marker checks do **not** verify a decodable image. Backend must decode and verify the complete JPEG before constructing it. Missing evidence or incomplete result fails closed. A clean result has `verdict:"no_violation"` and `violations:[]`. An empty rule catalog raises configuration error, and rule instances are revalidated at the decision boundary.

## Independent evaluation

`evaluation.py` does not import the decision engine. Predictions are arrays of `{video_id,status,verdict,events:[{timecode_sec,type?,rule_id?}]}`; labels are arrays of `{video_id,verdict?,exhaustive:bool,events?:[...]}`. Missing verdict and absent events mean those labels are unavailable. `exhaustive:true` concerns event annotations only: it is necessary to count unmatched event predictions as false positives. Every supplied binary verdict is known gold, including `no_violation` when event timing is incomplete or omitted. Positive non-exhaustive labels remain usable; extra events are reported as ignored, not false positives.

Video TP/FP/FN/TN and precision/recall/F1 are conditional on completed predictions with known usable labels. `completed`, `total`, `completion_rate`, `failures`, `missing_predictions`, `partial_predictions`, `failed_predictions` and `invalid_predictions` make this scope explicit. Total is the union of prediction and annotation video IDs. Missing/partial/failed predictions never count as TN; their labelled positive events remain unmatched. Report completion and failures beside video metrics. Missing binary labels are separately counted. A supplied negative binary label counts toward TN/FP regardless of event exhaustiveness. Undefined ratios are JSON null. Negative ground truth with positive events is rejected; a positive binary label can legitimately lack event annotation.

At each 1/3/5-second tolerance, events use maximum-cardinality, minimum-total-time-error one-to-one temporal matching. Predictions cannot be reused; match types do not influence temporal matching. Two explicitly named scopes prevent optimistic mixed precision:

- `exhaustive_labels`: TP/FP/FN and precision/recall/F1 use only videos with explicit exhaustive event annotations, including unmatched labels on failed analyses.
- `all_known_labels`: matched/unmatched labels, `recall_of_known_labels`, ignored extras, timing errors and type/rule accuracy use all supplied event labels. No precision or F1 is reported for this scope because unmatched predictions on non-exhaustive videos have unknown truth.

This evaluates timing and classification separately; it is not type-constrained detection AP.

Split manifests are arrays of `{video_id,sha256,group_id,split}`. Video IDs must be unique, SHA must contain exactly 64 hexadecimal characters, and group ID must be a nonempty string. Both SHA and group must remain within one of train/dev/test; validator rejects crossing. Multiple distinct video IDs may share SHA within one split. Group all related source videos together and compute SHA locally. Never split frames at random. This validator trusts supplied metadata and does not open videos.

The parent metadata inventory leaves `scenario_group:null` until manual grouping. Before split validation, review related videos, assign nonempty scenario groups, map each reviewed `scenario_group` to `group_id`, and add the intended split; do not silently substitute file paths or frame IDs for unknown groups. Example validator row: `{"video_id":"v1","sha256":"<64 hex characters>","group_id":"reviewed-session-1","split":"train"}`.
