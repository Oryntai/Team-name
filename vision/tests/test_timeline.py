"""Checklist on synthetic timelines: each check fires on its bad sequence and stays silent on the good one."""

import json

from forte_vision.timeline import CHECKS, EVENTS, parse_timeline, run_checklist


def checks(steps, duration=30.0):
    return [v.check for v in run_checklist(steps, duration)]


def test_doc_needs_id_before_cash():
    assert checks([(5, "client_arrives"), (9, "cash_given_by_client")]) == ["DOC"]
    assert "DOC" not in checks([(5, "client_arrives"), (7, "id_document_shown"), (9, "cash_given_by_client")])


def test_lock_and_leave_on_departure():
    bad = [(6, "screen_working"), (8, "cashier_stands_up"), (10, "cashier_leaves"), (11, "cash_on_desk")]
    found = run_checklist(bad, 20.0)
    assert {v.check for v in found} == {"LOCK", "LEAVE"}
    assert all(v.main_sec == 8 for v in found)  # timecode = when she stands up
    good = [(5, "put_in_safe"), (6, "screen_locked"), (8, "cashier_stands_up"), (10, "cashier_leaves")]
    assert checks(good, 20.0) == []


def test_safe_open_must_persist_and_never_close():
    assert checks([(2, "put_in_safe"), (3, "safe_open"), (12, "safe_open")]) == ["SAFE"]
    assert checks([(2, "put_in_safe"), (3, "safe_open"), (6, "safe_closed"), (12, "safe_open")]) == []


def test_door_left_open_vs_closed_after_entry():
    assert checks([(1, "cashier_enters"), (2, "door_open"), (10, "door_open")]) == ["DOOR"]
    assert checks([(1, "cashier_enters"), (2, "door_open"), (3, "door_closed")]) == []
    # glass door "open" all the time and a client walking in is not the cashier leaving the door open
    assert checks([(0.4, "door_open"), (3, "outsider_enters"), (12, "door_open")]) == []


def test_single_mention_of_food_or_items_is_ignored():
    assert checks([(13, "food_drink")]) == []
    assert checks([(11, "food_drink"), (15, "food_drink")]) == ["FOOD"]


def test_mix_and_noclient():
    mix = [(5, "client_arrives"), (6, "id_document_shown"), (7, "cash_given_by_client"), (12, "client_leaves"),
           (14, "cash_on_desk"), (16, "client_arrives")]
    assert "MIX" in checks(mix)
    noclient = [(3, "client_arrives"), (4, "id_document_shown"), (5, "cash_given_by_client"),
                (7, "client_leaves"), (12, "counting_machine")]
    assert "NOCLIENT" in checks(noclient)
    present = [(3, "client_arrives"), (4, "id_document_shown"), (5, "cash_given_by_client"),
               (6, "counting_machine"), (12, "client_leaves")]
    assert "NOCLIENT" not in checks(present)


def test_bag_to_safe_without_registration():
    assert "REG" in checks([(40, "bag_opened"), (47, "safe_open"), (49, "put_in_safe")], 60.0)
    ok = [(40, "bag_opened"), (45, "counting_machine"), (60, "computer_work"), (70, "put_in_safe")]
    assert "REG" not in checks(ok, 90.0)


def test_every_check_has_a_predicate_and_parser_drops_unknown_items():
    assert len(CHECKS) == 13 and len(set(CHECKS.values())) == 13
    steps, ok, dropped = parse_timeline(
        json.dumps([{"event": "door_open", "t": 3}, {"event": "dragon", "t": 4}, {"event": "food_drink", "t": "5s"}]),
        0.0, 10.0)
    assert ok and dropped == 1 and steps == [(3.0, "door_open"), (5.0, "food_drink")]
    assert parse_timeline("not json", 0, 10) == ([], False, 0)
    assert set(e for _, e in steps) <= set(EVENTS)


def test_verifier_semantics():
    from forte_vision.timeline import VERIFY, confirmed, parse_verdict

    assert set(VERIFY) == set(CHECKS)
    assert parse_verdict("Yes.") == "yes" and parse_verdict(" no ") == "no" and parse_verdict("maybe") == "unclear"
    assert confirmed("FOOD", "yes") and not confirmed("FOOD", "no") and confirmed("FOOD", "unclear")
    # DOC asks whether the document WAS shown: "no" confirms the violation
    assert confirmed("DOC", "no") and not confirmed("DOC", "yes")


def test_observer_drops_violation_the_verifier_rejects(synthetic_video):
    from forte_vision import analyze_video
    from forte_vision.vlm import StubBackend

    def respond(times, question):
        if "Return ONLY a JSON list" in question:
            return json.dumps([{"event": "food_drink", "t": times[0]}, {"event": "food_drink", "t": times[-1]}])
        return "no"

    obs, diag = analyze_video(synthetic_video, StubBackend(respond), video_id="syn", profile="fast")
    assert obs["analysis_status"] == "complete" and obs["candidates"] == []
    assert diag["rejected_by_verifier"] and all(v["answer"] == "no" for v in diag["verifier"])


def test_recall_pass_adds_violation_when_timeline_invented_the_good_step(synthetic_video):
    from forte_vision import analyze_video
    from forte_vision.vlm import StubBackend

    def respond(times, question):
        if "Return ONLY a JSON list" in question:
            # timeline claims the screen was locked before leaving -> checklist stays silent on LOCK
            if times[0] < 5:
                return json.dumps([{"event": "screen_locked", "t": times[0]},
                                   {"event": "cashier_stands_up", "t": times[1]},
                                   {"event": "cashier_leaves", "t": times[2]}])
            return "[]"
        return "yes" if "laptop screen" in question else "no"

    obs, diag = analyze_video(synthetic_video, StubBackend(respond), video_id="syn", profile="fast")
    assert [c for c in obs["candidates"] if c["predicates"]["left_with_computer_unlocked"] == "observed"]
    assert any(v["check"] == "LOCK" for v in diag["recovered_by_verifier"])
    assert any(v["mode"] == "recall" and v["check"] == "LOCK" and v["answer"] == "yes" for v in diag["verifier"])


def test_recall_triggers_only_for_silent_checks_with_their_event():
    from forte_vision.timeline import recall_triggers

    steps = [(3, "cashier_stands_up"), (5, "cashier_leaves"), (9, "safe_open")]
    assert dict(recall_triggers(steps, set())) == {"LOCK": 3, "LEAVE": 3, "SAFE": 9}
    assert "LOCK" not in dict(recall_triggers(steps, {"LOCK"}))


def test_step_at_zero_seconds_counts_as_found():
    # bag opened in the very first frame, no bag_received: REG must still be checked
    assert "REG" in checks([(0.0, "bag_opened"), (5, "safe_open"), (6, "put_in_safe")])
    # standing up at 0.0 is the departure timecode, not "missing"
    found = run_checklist([(0.0, "cashier_stands_up"), (2, "screen_working"), (4, "cashier_leaves")], 20.0)
    assert [(v.check, v.main_sec) for v in found] == [("LOCK", 0.0)]


def test_every_departure_is_checked():
    steps = [(2, "screen_locked"), (3, "cashier_leaves"), (8, "cashier_enters"),
             (40, "screen_working"), (42, "cashier_stands_up"), (44, "cashier_leaves"), (45, "cash_on_desk")]
    found = {v.check: v.main_sec for v in run_checklist(steps, 60.0)}
    assert found == {"LOCK": 42, "LEAVE": 42}


def test_safe_and_door_reopened_after_closing():
    assert checks([(3, "safe_open"), (6, "safe_closed"), (20, "safe_open"), (28, "safe_open")], 40.0) == ["SAFE"]
    assert checks([(3, "safe_open"), (6, "safe_closed"), (20, "safe_open"), (22, "safe_closed")], 40.0) == []
    door = [(1, "cashier_enters"), (2, "door_open"), (3, "door_closed"), (15, "door_open"), (25, "door_open")]
    assert checks(door, 40.0) == ["DOOR"]


def test_gaps_drop_only_violations_that_depend_on_them():
    steps = [(2, "food_drink"), (4, "food_drink"), (9, "cash_given_by_client")]
    assert set(checks(steps)) == {"FOOD", "DOC"}
    # the ID document may have been shown in the uncovered window: DOC is not proven, FOOD still is
    assert [v.check for v in run_checklist(steps, 30.0, gaps=[(5.0, 8.0)])] == ["FOOD"]
    assert set(v.check for v in run_checklist(steps, 30.0, gaps=[(20.0, 30.0)])) == {"FOOD", "DOC"}


def test_checks_match_vision_predicates():
    from forte_vision.predicates import PREDICATES

    assert set(CHECKS.values()) == set(PREDICATES)


def test_partial_timeline_keeps_violations_seen_in_covered_windows(synthetic_video):
    from forte_vision import analyze_video
    from forte_vision.vlm import StubBackend

    def respond(times, question):
        if "Return ONLY a JSON list" in question:
            if times[0] < 8:
                return "the model rambled instead of JSON"
            return json.dumps([{"event": "items_on_desk", "t": times[0]}, {"event": "items_on_desk", "t": times[-1]}])
        return "yes"

    obs, diag = analyze_video(synthetic_video, StubBackend(respond), video_id="syn", profile="fast")
    assert obs["analysis_status"] == "partial" and obs["coverage_complete"] is False
    assert diag["timeline_gaps"]
    assert any(c["predicates"]["personal_items_on_desk"] == "observed" for c in obs["candidates"])


def test_crop_reports_fallback_and_fit_size():
    from PIL import Image

    from forte_vision.observer import _crop_frames, _fit_size

    frames = [(1.0, Image.new("RGB", (100, 80)))]
    out, cropped = _crop_frames(frames, [(10, 10, 12, 12)], margin=0.0)
    assert not cropped and out is frames
    out, cropped = _crop_frames(frames, [(10, 10, 60, 60)])
    assert cropped and min(out[0][1].size) >= 448
    assert _fit_size(464, 848, 960) is None
    assert _fit_size(1920, 1080, 960) == (960, 540)
