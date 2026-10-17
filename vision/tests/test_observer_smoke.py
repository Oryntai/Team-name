"""Сквозной смоук контракта: синтетическое видео + заглушка VLM -> валидный Observation."""

import json

from conftest import EVENT
from forte_vision import analyze_video, validate_observation
from forte_vision.predicates import LEGACY_PREDICATES as PREDICATES, LEGACY_VISIBILITY as VISIBILITY
from forte_vision.vlm import StubBackend


def _respond(times, question):
    if "Categories:" in question:  # скрининг окна
        if any(EVENT[0] <= t < EVENT[1] for t in times):
            return json.dumps(
                [{"category": "cash_counting", "start": EVENT[0], "end": EVENT[1]}]
            )
        return "[]"
    answers = {n: "unclear" for n in [*PREDICATES, *VISIBILITY]}
    if any(EVENT[0] <= t <= EVENT[1] for t in times):
        answers["cash_counting_observed"] = "yes"
    return "```json\n" + json.dumps(answers) + "\n```"


def test_full_pipeline_with_stub(synthetic_video):
    backend = StubBackend(_respond)
    obs, diag = analyze_video(synthetic_video, backend, video_id="syn", profile="legacy_fast")
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
            return json.dumps(
                [{"category": "other_suspicious", "start": 0, "end": 19.5}]
            )
        return "{}"

    obs, _ = analyze_video(
        synthetic_video, StubBackend(respond), video_id="syn", profile="legacy_full"
    )
    validate_observation(obs)
    lengths = [c["end_sec"] - c["start_sec"] for c in obs["candidates"]]
    assert lengths and max(lengths) <= 6.0 + 0.5
    assert any(
        c["start_sec"] <= 9.0 <= c["end_sec"] and "motion" in c["source"]
        for c in obs["candidates"]
    )


def test_deadline_gives_honest_partial(synthetic_video):
    obs, diag = analyze_video(
        synthetic_video, StubBackend(_respond), video_id="syn", deadline_sec=0.0, profile="legacy_fast")
    validate_observation(obs)
    assert obs["coverage_complete"] is False
    assert obs["analysis_status"] == "partial"
    assert all(c["predicates"] == {} for c in obs["candidates"])


class StaticSyntheticSource:
    """In-memory static images; no actual recording or media file is opened."""

    def __init__(self, empty_screen=False):
        self.empty_screen = empty_screen

    def probe(self):
        from forte_vision.video import VideoInfo

        return VideoInfo(2.0, 5.0, 16, 16, 10)

    def sample(self, fps, size=None, start=0, end=None):
        from PIL import Image

        if self.empty_screen and fps == 0.5:
            return
        for i in range(int(2 * fps)):
            time = i / fps
            if time >= start and (end is None or time < end):
                yield time, Image.new("RGB", size or (16, 16), (80, 80, 80))

    def grab_at(self, times, size=None):
        from PIL import Image

        return [
            (time, Image.new("RGB", size or (16, 16), (80, 80, 80)))
            for time in sorted(set(times))
        ]


def test_malformed_screen_cannot_be_complete_empty_negative(monkeypatch):
    monkeypatch.delenv("FORTE_LOCAL_AUDIT_DIR", raising=False)
    obs, diag = analyze_video(
        "unused-synthetic",
        StubBackend(lambda times, question: "BROKEN RESPONSE"),
        video_id="synthetic",
        source=StaticSyntheticSource(), profile="legacy_fast")
    assert obs["candidates"] == []
    assert obs["analysis_status"] == "partial" and obs["coverage_complete"] is False
    assert diag["invalid_screen_windows"] == 1 and diag["valid_screen_windows"] == 0
    assert diag["pool_selected_count"] == diag["pool_dropped_count"] == 0


def test_valid_empty_screen_is_complete(monkeypatch):
    monkeypatch.delenv("FORTE_LOCAL_AUDIT_DIR", raising=False)
    obs, diag = analyze_video(
        "unused-synthetic",
        StubBackend(lambda times, question: "[]"),
        video_id="synthetic",
        source=StaticSyntheticSource(), profile="legacy_fast")
    assert obs["candidates"] == []
    assert obs["analysis_status"] == "complete" and obs["coverage_complete"] is True
    assert diag["invalid_screen_windows"] == 0 and diag["valid_screen_windows"] == 1


def test_empty_screen_window_is_incomplete_without_model_call(monkeypatch):
    monkeypatch.delenv("FORTE_LOCAL_AUDIT_DIR", raising=False)
    backend = StubBackend(lambda times, question: "[]")
    obs, diag = analyze_video(
        "unused-synthetic",
        backend,
        video_id="synthetic",
        source=StaticSyntheticSource(empty_screen=True), profile="legacy_fast")
    assert obs["analysis_status"] == "partial" and obs["coverage_complete"] is False
    assert diag["empty_screen_windows"] == 1 and backend.stats.calls == 0


class _FakeDetector:
    """Detector double: finds a cup on every frame or nothing at all."""

    size = "fake"
    device = "cpu"
    threshold = 0.4

    def __init__(self, items):
        self.items = items

    def detect(self, frames, batch=8):
        from forte_vision.scene import FrameDets

        return [FrameDets(t, [], [(name, (0.0, 0.0, 10.0, 10.0)) for name in self.items]) for t, _ in frames]


def _drink_everywhere(times, question):
    if "Categories:" in question:
        return json.dumps([{"category": "food_or_drink", "start": EVENT[0], "end": EVENT[1]}])
    answers = {n: "no" for n in [*PREDICATES, *VISIBILITY]}
    answers["food_or_drink_at_desk"] = "yes"
    return json.dumps(answers)


def test_object_predicate_needs_detector_confirmation(synthetic_video):
    obs, diag = analyze_video(synthetic_video, StubBackend(_drink_everywhere), video_id="syn",
                              profile="legacy_fast", detector=_FakeDetector([]))
    validate_observation(obs)
    assert obs["candidates"]
    assert all(c["predicates"]["food_or_drink_at_desk"] == "not_observed" for c in obs["candidates"])
    assert any("food_or_drink_at_desk" in d["detector_veto"] for d in diag["candidates"].values())

    obs, _ = analyze_video(synthetic_video, StubBackend(_drink_everywhere), video_id="syn",
                           profile="legacy_fast", detector=_FakeDetector(["cup"]))
    assert any(c["predicates"]["food_or_drink_at_desk"] == "observed" for c in obs["candidates"])


def test_recheck_budget_drops_low_priority_candidates_instead_of_partial(synthetic_video):
    import time

    def slow(times, question):
        if "Categories:" in question:
            return json.dumps([{"category": "other_suspicious", "start": 0, "end": 19.5}])
        time.sleep(0.4)
        return json.dumps({n: "no" for n in [*PREDICATES, *VISIBILITY]})

    obs, diag = analyze_video(synthetic_video, StubBackend(slow), video_id="syn", profile="legacy_full",
                              deadline_sec=1.5)
    validate_observation(obs)
    assert obs["coverage_complete"] is True
    assert obs["analysis_status"] == "complete"
    assert diag["budget_dropped_count"] > 0
    assert obs["candidates"] and all(c["predicates"] for c in obs["candidates"])


class _PeopleDetector(_FakeDetector):
    def __init__(self, n):
        super().__init__([])
        self.n = n

    def detect(self, frames, batch=8):
        from forte_vision.scene import FrameDets

        return [FrameDets(t, [(0.0, 0.0, 5.0, 5.0)] * self.n, []) for t, _ in frames]


def _outsider_everywhere(times, question):
    if "Categories:" in question:
        return json.dumps([{"category": "person_enters_cash_area", "start": EVENT[0], "end": EVENT[1]}])
    answers = {n: "no" for n in [*PREDICATES, *VISIBILITY]}
    answers["outsider_in_cash_room"] = "yes"
    return json.dumps(answers)




def test_outsider_needs_a_third_person_on_frames(synthetic_video):
    for people, expected in ((2, "not_observed"), (3, "observed")):
        obs, _ = analyze_video(synthetic_video, StubBackend(_outsider_everywhere), video_id="syn",
                               profile="legacy_fast", detector=_PeopleDetector(people))
        assert any(c["predicates"]["outsider_in_cash_room"] == expected for c in obs["candidates"])
        assert all(c["predicates"]["outsider_in_cash_room"] in ("not_observed", expected) for c in obs["candidates"])
