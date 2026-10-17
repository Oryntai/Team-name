"""Сквозной смоук контракта: синтетическое видео + заглушка VLM -> валидный Observation."""

import json

from conftest import EVENT
from forte_vision import analyze_video, validate_observation
from forte_vision.predicates import PREDICATES, VISIBILITY
from forte_vision.vlm import StubBackend


def _respond(times, question):
    if "Categories:" in question:  # скрининг окна
        if any(EVENT[0] <= t < EVENT[1] for t in times):
            return json.dumps([{"category": "cash_counting", "start": EVENT[0], "end": EVENT[1]}])
        return "[]"
    answers = {n: "unclear" for n in [*PREDICATES, *VISIBILITY]}
    if any(EVENT[0] <= t <= EVENT[1] for t in times):
        answers["cash_counting_observed"] = "yes"
    return "```json\n" + json.dumps(answers) + "\n```"


def test_full_pipeline_with_stub(synthetic_video):
    backend = StubBackend(_respond)
    obs, diag = analyze_video(synthetic_video, backend, video_id="syn", profile="fast")
    validate_observation(obs)
    assert obs["analysis_status"] == "complete" and obs["coverage_complete"]
    hits = [c for c in obs["candidates"] if c["start_sec"] <= 9.0 <= c["end_sec"]]
    assert hits, "событие 8–10 с должно стать кандидатом"
    c = hits[0]
    assert "vlm" in c["source"] and "motion" in c["source"]
    assert c["predicates"]["cash_counting_observed"] == "observed"
    assert set(diag) >= {"model_id", "stage_timings", "vlm", "candidates"}
    assert diag["vlm"]["calls"] >= 2


def test_long_vlm_interval_is_split(synthetic_video):
    """Длинный ответ скрининга не должен поглощать короткие пики движения в один огромный кандидат."""

    def respond(times, question):
        if "Categories:" in question:
            return json.dumps([{"category": "other_suspicious", "start": 0, "end": 19.5}])
        return "{}"

    obs, _ = analyze_video(synthetic_video, StubBackend(respond), video_id="syn", profile="full")
    validate_observation(obs)
    lengths = [c["end_sec"] - c["start_sec"] for c in obs["candidates"]]
    assert lengths and max(lengths) <= 6.0 + 0.5
    assert any(c["start_sec"] <= 9.0 <= c["end_sec"] and "motion" in c["source"] for c in obs["candidates"])


def test_deadline_gives_honest_partial(synthetic_video):
    obs, diag = analyze_video(synthetic_video, StubBackend(_respond), video_id="syn", deadline_sec=0.0)
    validate_observation(obs)
    assert obs["coverage_complete"] is False
    assert obs["analysis_status"] == "partial"
    assert all(c["predicates"] == {} for c in obs["candidates"])
