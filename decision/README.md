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

## Standalone local JSON contract (synthetic/offline harness)

`Observation` JSON has `video_id` (nonempty string), `duration_sec` (finite positive number), `analysis_status` (`complete`, `partial`, `failed`), `coverage_complete` (strict bool), and `candidates` (array). Complete coverage is an upstream assertion about processing, never inferred from an empty candidate array.

Each candidate has stable `candidate_id`, `start_sec`, `end_sec`, `main_sec`, `source`, `evidence` array, and optional `predicates` and `visibility` objects. Every atomic map value is exactly `observed`, `not_observed`, or `unclear`; missing keys mean unclear. `source` identifies the local observer/version; `visibility` keys name rule-specific visibility requirements. Do not replace tri-state observations with scores; scores are not calibrated probabilities and are not accepted in this schema. Unknown JSON fields fail constructor parsing.

Evidence references contain only `frame_id`, `pts_sec`, and `source`. IDs must be stable within one video; a repeated ID must have identical PTS/source. No media bytes or external frame URLs belong in this contract. All timestamps are finite numeric values; bools are rejected. Require `0 <= start <= main <= end <= duration`; each evidence PTS lies inside its candidate interval. Candidate IDs must be unique. Malformed input raises an exception and must become upstream analysis failure, never a clean negative.

Rule JSON fields: `rule_id`, `type`, `title`, `source_document`, `source_paragraph`, `source_url`, `applicability`, `required`, `exceptions`, `visibility_required`, `description_template`. Four predicate lists contain unique string names. Description is literal reviewed text (no executable/interpolated template). Default `mode="review"` requires all core `required` predicates observed and every exception explicitly not_observed. Explicit not_observed core/applicability or observed exception rejects that rule. Unknown core/exception remains unresolved. With strong core, unclear applicability or visibility permits a **possible** violation; diagnostics retain conditional-context and visibility caveats. Descriptions must use human-reviewed conditional wording. `mode="strict"` additionally requires observed applicability/visibility; otherwise those candidates remain unresolved. At least one real input evidence reference is required.

Candidates mentioning named core predicates target their corresponding rules; other rules with no mentioned core are not targeted. If no core is named, unknown/missing core stays unresolved. Observed predicates not covered by the catalog also mark unresolved. Complete analysis with known supported events may return violation despite other unresolved candidates. With no supported events, any unresolved candidate makes the result incomplete, never no_violation. Empty complete batches and explicitly rejected covered candidates can produce no_violation. The engine does not measure calibration or real-data accuracy.

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

## Local integration harness

```sh
forte-decide --observation /local/observation.json \
  --rules /local/catalog-a.json --rules /local/catalog-b.json \
  --output /local/decision.json --max-anchor-delta-sec 1.0
# Or: python -m forte_decision.cli ...
```

Install editable again after adding the console entry point. Each catalog is a JSON rule array or an object containing only `rules` with that array. Multiple catalogs combine; duplicate rule IDs and an empty combined catalog fail closed. Parsing rejects duplicate JSON keys and nonstandard NaN/Infinity constants. Output atomically replaces the explicit destination with private file permissions; it cannot replace observation/catalog input paths. Destination parent must already exist.

This CLI writes **internal results only**. It never opens media or accepts an evidence manifest: backend must verify local JPEG extraction and call the public adapter. Stdout contains only `status` and `event_count`; parse/load failures print a generic error without input, rule text or exception messages. Exit codes: 0 complete, 1 invalid input/configuration/I/O, 2 incomplete analysis (internal result still saved), argparse usage errors also use 2. The saved file contains full diagnostics and source descriptions.

**Real observations, decision JSON, descriptions, frames and videos must remain local. Do not print them into cloud agent/tool logs or send them to cloud AI.** Run the harness locally and report only aggregate status/counts. Do not use the introductory `print(result.to_json())` synthetic demonstration on real data. The harness and evaluator make no cloud requests, but whoever invokes them owns the logging boundary.


## Canonical production backend adapter

Set backend `rules_factory="forte_decision.backend_adapter:create_rules"`. The zero-argument factory loads only repository `rules/cash_receipt.json` (two seed rules). Optional `FORTE_RULE_CATALOGS` is a JSON array of local catalog paths; packaging outside this repository must supply it. Factory validates rules/configuration at startup. For tests or explicit construction: `BackendRules(catalog_paths=[...], mode="review", max_anchor_delta_sec=1.0)`.

`evaluate(observations: backend.contracts.ObservationBatch, context: backend.contracts.AnalysisContext) -> list[backend.contracts.RuleEvent]` uses the existing canonical contract directly. It rejects alternate dictionaries/wire formats. The standalone JSON Observation is **not** another production boundary. `evaluate_candidates` is the shared internal logic, with no duration parameter: actual video duration and decoded PTS validation belong to backend VideoIO/worker. No duration is inferred from event end. Evidence IDs derive deterministically from job ID and actual supplied PTS; duplicate canonical candidates deduplicate.

Backend observations map supplies tri-state named atomic predicates. For named rule visibility requirements, explicit observations under the exact name or `visibility:<name>` are accepted. Conflicting forms fail. Object `visibility` values (`clear`, `partial`, `unusable`) never establish all-positions visibility or any named visibility predicate. They map only to internal support `evidence_usable`: observed if any supplied object is clear/partial, not_observed if the nonempty map is entirely unusable, unclear if the map is empty. All-unusable candidates remain unresolved (even with explicit false core), preventing both positive output and a clean negative from unusable evidence. Partial/clear and empty/unknown object visibility do not hard-gate strong core; unknown support is retained as a diagnostic caveat. Other supported candidates can still establish a positive in a complete batch. No frame/image content is opened by the adapter.

Incomplete batches or unresolved-only results raise backend `IncompleteAnalysis`, which must produce official failed. A complete empty batch returns `[]`. Known positives can return despite unresolved other candidates. RuleEvent description equals the literal reviewed `rule.description_template`; source document/title, paragraph, URL and rule ID travel in separate canonical fields so backend appends the citation exactly once. Internal DecisionResult diagnostics contain caveats; backend RuleEvent has no dedicated caveat field, so production catalog descriptions must retain possible/conditional wording.

Local CLI also accepts `--mode review` (default) or `--mode strict`. Never send canonical observations, local results or their descriptions to cloud logs.
