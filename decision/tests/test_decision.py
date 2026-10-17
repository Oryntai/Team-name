import json
from pathlib import Path
from dataclasses import replace
import pytest
from forte_decision import Observation, Rule, decide
from forte_decision.adapter import VerifiedJPEG, to_public_payload
from forte_decision.evaluation import evaluate, match_events, validate_splits

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def observation():
    return Observation.from_json((ROOT / "examples/observation.json").read_text())


@pytest.fixture
def rule():
    return Rule(**json.loads((ROOT / "examples/synthetic_rule.json").read_text()))


def test_roundtrip(observation):
    assert Observation.from_json(observation.to_json()) == observation


@pytest.mark.parametrize(
    "change", ["absent", "exception_observed", "exception_unclear", "visibility"]
)
def test_conservative(observation, rule, change):
    c = observation.candidates[0]
    predicates, visibility = dict(c.predicates), dict(c.visibility)
    if change == "absent":
        predicates.pop("demo_marker_present")
    elif change == "visibility":
        visibility.clear()
    else:
        predicates["demo_permission_present"] = change.split("_")[1]
    result = decide(
        replace(
            observation,
            candidates=(replace(c, predicates=predicates, visibility=visibility),),
        ),
        [rule],
    )
    if change == "visibility":
        assert result.verdict == "violation" and result.diagnostics[0]["caveats"]
    elif change == "exception_observed":
        assert result.verdict == "no_violation"
    else:
        assert result.status == "incomplete" and result.verdict is None


@pytest.mark.parametrize(
    "status,coverage", [("partial", True), ("failed", True), ("complete", False)]
)
def test_incomplete(observation, rule, status, coverage):
    result = decide(
        replace(observation, analysis_status=status, coverage_complete=coverage), [rule]
    )
    assert result.status == "incomplete" and result.verdict is None
    with pytest.raises(ValueError):
        to_public_payload(result, {}, job_id="job")


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1, 11])
def test_bad_time(observation, value):
    data = json.loads(observation.to_json())
    data["candidates"][0]["main_sec"] = value
    with pytest.raises(ValueError):
        Observation.from_dict(data)


def test_merge_separate_provenance(observation, rule):
    c = observation.candidates[0]
    duplicate = replace(c, candidate_id="c2")
    separate = replace(
        c,
        candidate_id="c3",
        start_sec=6,
        end_sec=8,
        main_sec=7,
        evidence=(replace(c.evidence[0], frame_id="f7", pts_sec=7),),
    )
    first = decide(replace(observation, candidates=(separate, duplicate, c)), [rule])
    second = decide(replace(observation, candidates=(c, duplicate, separate)), [rule])
    assert first == second and len(first.events) == 2
    assert first.events[0].candidate_ids == ("c1", "c2")
    assert first.events[0].evidence == c.evidence
    assert first.events[0].timecode_sec == c.main_sec
    assert "Example 1" in first.events[0].description
    with pytest.raises(ValueError):
        replace(
            observation,
            candidates=(
                c,
                replace(duplicate, evidence=(replace(c.evidence[0], pts_sec=3.1),)),
            ),
        )


def test_adapter(observation, rule):
    result = decide(observation, [rule])
    with pytest.raises(ValueError, match="missing evidence"):
        to_public_payload(result, {}, job_id="job")
    image = VerifiedJPEG("wrong", "f3", 3, b"\xff\xd8\xff\xd9")
    with pytest.raises(ValueError):
        to_public_payload(result, {"f3": image}, job_id="job")
    assert (
        len(
            to_public_payload(
                result,
                {"f3": replace(image, video_id=observation.video_id)},
                job_id="job",
            )["violations"]
        )
        == 1
    )


def test_matching_one_to_one_and_optimal():
    assert len(match_events([{"timecode_sec": 3}] * 2, [{"timecode_sec": 3}], 1)) == 1
    # Greedy nearest-first can strand the second label; augmentation recovers both.
    pairs = match_events(
        [{"timecode_sec": 1}, {"timecode_sec": 3}],
        [{"timecode_sec": 2}, {"timecode_sec": 0}],
        1,
    )
    assert len(pairs) == 2


def test_metrics():
    predictions = [
        {
            "video_id": str(i),
            "status": "complete",
            "verdict": v,
            "events": ([{"timecode_sec": 1}] if v == "violation" else []),
        }
        for i, v in enumerate(
            ["violation", "violation", "no_violation", "no_violation"]
        )
    ]
    labels = [
        {"video_id": str(i), "verdict": v, "exhaustive": True, "events": []}
        for i, v in enumerate(
            ["violation", "no_violation", "violation", "no_violation"]
        )
    ]
    result = evaluate(predictions, labels)["video"]
    assert [result[k] for k in ("TP", "FP", "FN", "TN")] == [1, 1, 1, 1]
    assert result["precision"] == result["recall"] == result["f1"] == 0.5


def test_failures_partial_labels_and_events():
    p = [
        {"video_id": "a", "status": "partial", "verdict": "no_violation"},
        {
            "video_id": "b",
            "status": "complete",
            "verdict": "violation",
            "events": [{"timecode_sec": 3}, {"timecode_sec": 3}],
        },
    ]
    l = [
        {
            "video_id": "a",
            "verdict": "violation",
            "exhaustive": True,
            "events": [{"timecode_sec": 2}],
        },
        {
            "video_id": "b",
            "verdict": "violation",
            "exhaustive": False,
            "events": [{"timecode_sec": 3, "type": "x"}],
        },
    ]
    result = evaluate(p, l)
    assert result["video"]["failures"] == 1 and result["video"]["TN"] == 0
    s = result["events"]["1"]["all_known_labels"]
    assert (
        s["matched"] == 1
        and s["unmatched_labels"] == 1
        and s["ignored_extra_predictions"] == 1
    )
    assert s["type_accuracy"] == 0 and s["recall_of_known_labels"] == 0.5


def test_split_leakage():
    assert validate_splits(
        [{"video_id": "v1", "sha256": "a" * 64, "group_id": "g", "split": "train"}]
    )
    for field in ("sha256", "group_id"):
        rows = [
            {"video_id": "v1", "sha256": "a" * 64, "group_id": "g", "split": "train"},
            {"video_id": "v2", "sha256": "b" * 64, "group_id": "h", "split": "test"},
        ]
        rows[1][field] = rows[0][field]
        with pytest.raises(ValueError):
            validate_splits(rows)


def test_evaluation_does_not_import_engine():
    import subprocess, sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import forte_decision.evaluation; assert 'forte_decision.engine' not in sys.modules",
        ],
        check=True,
    )


def test_rule_roundtrip(rule):
    assert Rule.from_json(rule.to_json()) == rule


def test_evidence_cap_and_existing_anchor(observation, rule):
    c = observation.candidates[0]
    refs = tuple(
        replace(c.evidence[0], frame_id=f"f{i}", pts_sec=2 + i * 0.4) for i in range(6)
    )
    result = decide(
        replace(observation, candidates=(replace(c, evidence=refs),)), [rule]
    )
    assert len(result.events[0].evidence) == 3
    assert all(r in refs for r in result.events[0].evidence)
    assert result.events[0].timecode_sec == c.main_sec


@pytest.mark.parametrize("data", [None, [], "text", 3, True])
def test_observation_requires_object(data):
    with pytest.raises(ValueError, match="object"):
        Observation.from_dict(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("duration_sec", 0),
        ("duration_sec", True),
        ("coverage_complete", 1),
        ("candidates", {}),
        ("candidates", [None]),
    ],
)
def test_observation_boundary(observation, field, value):
    data = json.loads(observation.to_json())
    data[field] = value
    with pytest.raises(ValueError):
        Observation.from_dict(data)


@pytest.mark.parametrize("value", [[], {}, True, 1, None])
def test_predicate_state_shape(observation, value):
    data = json.loads(observation.to_json())
    data["candidates"][0]["predicates"]["bad"] = value
    with pytest.raises(ValueError):
        Observation.from_dict(data)


def test_empty_and_mutated_catalog(observation, rule):
    with pytest.raises(ValueError, match="catalog"):
        decide(observation, [])
    # Frozen dataclasses do not replace trust-boundary validation.
    object.__setattr__(rule, "required", [])
    with pytest.raises(ValueError, match="violation predicates"):
        decide(observation, [rule])


def test_official_adapter_contract(observation, rule):
    result = decide(observation, [rule])
    jpeg = b"\xff\xd8\xff\xd9"  # Marker-only synthetic bytes; no actual image claim.
    evidence = {"f3": VerifiedJPEG(observation.video_id, "f3", 3, jpeg)}
    payload = to_public_payload(result, evidence, job_id="job-1")
    assert payload == {
        "job_id": "job-1",
        "status": "done",
        "verdict": "violation",
        "violations": [
            {
                "type": result.events[0].type,
                "timecode_sec": 3,
                "description": result.events[0].description,
                "evidence_frames": [{"timestamp_sec": 3, "image_base64": "/9j/2Q=="}],
            }
        ],
    }
    with pytest.raises(ValueError):
        to_public_payload(
            replace(result, verdict="no_violation"), evidence, job_id="job"
        )
    with pytest.raises(ValueError):
        to_public_payload(
            replace(result, events=(replace(result.events[0], video_id="other"),)),
            evidence,
            job_id="job",
        )
    with pytest.raises(TypeError):
        to_public_payload(result, evidence)
    with pytest.raises(ValueError):
        to_public_payload(result, evidence, job_id="")
    clean = replace(result, verdict="no_violation", events=())
    assert to_public_payload(clean, {}, job_id="job")["violations"] == []


def test_mixed_event_metric_scopes():
    predictions = [
        {
            "video_id": "full",
            "status": "complete",
            "verdict": "violation",
            "events": [{"timecode_sec": 1}, {"timecode_sec": 9}],
        },
        {
            "video_id": "partial-labels",
            "status": "complete",
            "verdict": "violation",
            "events": [{"timecode_sec": 1}] * 9,
        },
        {"video_id": "outage", "status": "failed"},
    ]
    labels = [
        {
            "video_id": "full",
            "verdict": "violation",
            "exhaustive": True,
            "events": [{"timecode_sec": 1}, {"timecode_sec": 5}],
        },
        {
            "video_id": "partial-labels",
            "verdict": "violation",
            "exhaustive": False,
            "events": [{"timecode_sec": 1}],
        },
        {
            "video_id": "outage",
            "verdict": "violation",
            "exhaustive": True,
            "events": [{"timecode_sec": 1}],
        },
    ]
    result = evaluate(predictions, labels)
    full = result["events"]["1"]["exhaustive_labels"]
    known = result["events"]["1"]["all_known_labels"]
    assert (full["TP"], full["FP"], full["FN"]) == (1, 1, 2)
    assert full["precision"] == 0.5 and full["recall"] == pytest.approx(1 / 3)
    assert full["f1"] == pytest.approx(0.4)
    assert known["matched"] == 2 and known["recall_of_known_labels"] == 0.5
    assert known["ignored_extra_predictions"] == 8
    assert "precision" not in known and "f1" not in known
    assert result["video"]["completed"] == 2 and result["video"]["total"] == 3
    assert result["video"]["completion_rate"] == pytest.approx(2 / 3)
    assert result["video"]["failed_predictions"] == 1


def test_annotation_contradiction_and_binary_positive():
    with pytest.raises(ValueError, match="contradicts"):
        evaluate(
            [],
            [
                {
                    "video_id": "a",
                    "verdict": "no_violation",
                    "events": [{"timecode_sec": 1}],
                }
            ],
        )
    assert (
        evaluate([], [{"video_id": "a", "verdict": "violation"}])["video"]["failures"]
        == 1
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"video_id": ""},
        {"sha256": "bad"},
        {"sha256": "g" * 64},
        {"sha256": None},
        {"group_id": None},
        {"group_id": ""},
        {"split": "validation"},
    ],
)
def test_split_schema(mutation):
    row = {"video_id": "v", "sha256": "a" * 64, "group_id": "g", "split": "train"}
    with pytest.raises(ValueError):
        validate_splits([{**row, **mutation}])


def test_duplicate_video_and_same_split_hash():
    row = {"video_id": "v", "sha256": "a" * 64, "group_id": "g", "split": "train"}
    with pytest.raises(ValueError, match="duplicate video_id"):
        validate_splits([row, row])
    assert validate_splits([row, {**row, "video_id": "other"}])


def test_binary_gold_does_not_require_exhaustive_events():
    predictions = [
        {
            "video_id": "fp",
            "status": "complete",
            "verdict": "violation",
            "events": [{"timecode_sec": 1}],
        },
        {
            "video_id": "tn",
            "status": "complete",
            "verdict": "no_violation",
            "events": [],
        },
    ]
    labels = [
        {
            "video_id": "fp",
            "verdict": "no_violation",
            "exhaustive": False,
            "events": [],
        },
        {"video_id": "tn", "verdict": "no_violation"},
    ]
    result = evaluate(predictions, labels)
    assert result["video"]["FP"] == 1 and result["video"]["TN"] == 1
    assert result["events"]["1"]["exhaustive_labels"]["FP"] == 0
    assert result["events"]["1"]["all_known_labels"]["ignored_extra_predictions"] == 1


@pytest.mark.parametrize("threshold", [-1, True, float("nan"), float("inf")])
def test_invalid_merge_threshold(observation, rule, threshold):
    with pytest.raises(ValueError):
        decide(observation, [rule], max_anchor_delta_sec=threshold)


def test_overlap_requires_close_anchors(observation, rule):
    original = observation.candidates[0]
    first = replace(original, start_sec=1, end_sec=9, main_sec=3)
    second = replace(first, candidate_id="c2", main_sec=8)
    assert (
        len(decide(replace(observation, candidates=(first, second)), [rule]).events)
        == 2
    )


def test_anchor_span_blocks_transitive_bridge(observation, rule):
    original = observation.candidates[0]
    candidates = tuple(
        replace(original, candidate_id=f"c{i}", start_sec=1, end_sec=9, main_sec=anchor)
        for i, anchor in enumerate([3, 3.8, 4.6])
    )
    first = decide(replace(observation, candidates=candidates), [rule])
    second = decide(
        replace(observation, candidates=tuple(reversed(candidates))), [rule]
    )
    assert first == second and len(first.events) == 2
    assert first.events[0].candidate_ids == ("c0", "c1")
    assert (
        len(
            decide(
                replace(observation, candidates=candidates),
                [rule],
                max_anchor_delta_sec=2,
            ).events
        )
        == 1
    )


def test_review_unknown_context_is_possible_with_caveat(observation, rule):
    c = observation.candidates[0]
    predicates = dict(c.predicates)
    predicates.pop("in_demo_zone")
    result = decide(
        replace(observation, candidates=(replace(c, predicates=predicates),)), [rule]
    )
    assert result.verdict == "violation"
    assert result.diagnostics[0]["caveats"]
    strict = decide(
        replace(observation, candidates=(replace(c, predicates=predicates),)),
        [rule],
        mode="strict",
    )
    assert strict.status == "incomplete"


def test_review_exception_unknown_blocks(observation, rule):
    c = observation.candidates[0]
    predicates = dict(c.predicates)
    predicates.pop("demo_permission_present")
    result = decide(
        replace(observation, candidates=(replace(c, predicates=predicates),)), [rule]
    )
    assert result.status == "incomplete"
