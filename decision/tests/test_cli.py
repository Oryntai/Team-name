import itertools
import json
from pathlib import Path
import subprocess
import sys

import pytest

from forte_decision.cli import main
from forte_decision.contracts import strict_json_loads
from forte_decision.evaluation import evaluate, match_events

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def inputs(tmp_path):
    observation = tmp_path / "observation.json"
    observation.write_text((EXAMPLES / "observation.json").read_text())
    rule = json.loads((EXAMPLES / "synthetic_rule.json").read_text())
    catalog = tmp_path / "rules.json"
    catalog.write_text(json.dumps([rule]))
    output = tmp_path / "result.json"
    return observation, catalog, output


def test_cli_multiple_catalogs_and_duplicate_ids(tmp_path, capsys):
    observation, catalog, output = inputs(tmp_path)
    second = tmp_path / "second.json"
    rule = json.loads(catalog.read_text())[0]
    second.write_text(json.dumps({"rules": [{**rule, "rule_id": "SYNTHETIC-002"}]}))
    arguments = [
        "--observation",
        str(observation),
        "--rules",
        str(catalog),
        "--rules",
        str(second),
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "complete",
        "event_count": 2,
    }
    assert len(json.loads(output.read_text())["events"]) == 2
    second.write_text(catalog.read_text())
    before = output.read_text()
    assert main(arguments) == 1
    assert json.loads(capsys.readouterr().out) == {"status": "error", "event_count": 0}
    assert output.read_text() == before


def test_cli_module_and_incomplete(tmp_path):
    observation, catalog, output = inputs(tmp_path)
    data = json.loads(observation.read_text())
    data["coverage_complete"] = False
    observation.write_text(json.dumps(data))
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "forte_decision.cli",
            "--observation",
            str(observation),
            "--rules",
            str(catalog),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 2
    assert json.loads(process.stdout) == {"status": "incomplete", "event_count": 1}
    result = json.loads(output.read_text())
    assert result["verdict"] is None and result["status"] == "incomplete"


@pytest.mark.parametrize(
    "bad_json", ["{SECRET-SCENE", '{"video_id":"SECRET-SCENE"}', "null"]
)
def test_cli_does_not_echo_sensitive_errors(tmp_path, capsys, bad_json):
    observation, catalog, output = inputs(tmp_path)
    observation.write_text(bad_json)
    assert (
        main(
            [
                "--observation",
                str(observation),
                "--rules",
                str(catalog),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == '{"status": "error", "event_count": 0}'
    assert captured.err == ""
    assert "SECRET" not in captured.out
    assert not output.exists()


def test_cli_cannot_overwrite_input(tmp_path, capsys):
    observation, catalog, _ = inputs(tmp_path)
    before = observation.read_text()
    assert (
        main(
            [
                "--observation",
                str(observation),
                "--rules",
                str(catalog),
                "--output",
                str(observation),
            ]
        )
        == 1
    )
    assert observation.read_text() == before
    assert json.loads(capsys.readouterr().out)["status"] == "error"


@pytest.mark.parametrize("value", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_strict_json(value):
    with pytest.raises(ValueError):
        strict_json_loads(value)


@pytest.mark.parametrize("value", [True, [], {}, "", 4])
def test_invalid_event_class_labels(value):
    with pytest.raises(ValueError):
        evaluate(
            [], [{"video_id": "x", "events": [{"timecode_sec": 1, "type": value}]}]
        )


def test_assignment_matches_brute_force_optimum():
    # Exhaustively compare small synthetic timelines to independent enumeration.
    for predicted_times in itertools.product((0, 1, 3), repeat=3):
        for label_times in itertools.product((0, 2, 3), repeat=2):
            predicted = [{"timecode_sec": t} for t in predicted_times]
            labels = [{"timecode_sec": t} for t in label_times]
            possibilities = []
            for assignment in itertools.product((-1, 0, 1), repeat=3):
                used = [j for j in assignment if j >= 0]
                if len(used) != len(set(used)):
                    continue
                errors = [
                    abs(predicted_times[i] - label_times[j])
                    for i, j in enumerate(assignment)
                    if j >= 0
                ]
                if all(error <= 1 for error in errors):
                    possibilities.append((-len(errors), sum(errors)))
            pairs = match_events(predicted, labels, 1)
            assert (-len(pairs), sum(error for _, _, error in pairs)) == min(
                possibilities
            )


def test_cli_review_unknown_is_incomplete_and_strict_soft_context(tmp_path, capsys):
    observation, catalog, output = inputs(tmp_path)
    data = json.loads(observation.read_text())
    data["candidates"][0]["predicates"]["demo_marker_present"] = "unclear"
    observation.write_text(json.dumps(data))
    arguments = [
        "--observation",
        str(observation),
        "--rules",
        str(catalog),
        "--output",
        str(output),
    ]
    assert main(arguments) == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "incomplete",
        "event_count": 0,
    }
    data["candidates"][0]["predicates"]["demo_marker_present"] = "observed"
    data["candidates"][0]["visibility"] = {}
    observation.write_text(json.dumps(data))
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "complete",
        "event_count": 1,
    }
    assert main([*arguments, "--mode", "strict"]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "incomplete",
        "event_count": 0,
    }
