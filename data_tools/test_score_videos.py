"""Независимый счёт: синтетические каталоги результатов, своя разметка и свой каталог правил."""

import ast
import json
from pathlib import Path

import pytest

from data_tools import score_videos as sv

LABELS = """id,official_verdict,official_sec,team_sec,type_code,team_confidence,clean_pair,team_note
X-1,violation,22,22,T01_no_client_id,high,,
Х-3,clean,,,,high,,
Х-4,violation,57,57,T02_cash_mixing,high,,
Х-31,clean,,,,high,,
Х-33,violation,6,6,T10_door_left_open,high,,
Х-35,clean,,,,high,,
Х-46,violation,11,11,T08_valuables_left_on_desk,high,,
Х-47,violation,68,68,T02_cash_mixing,high,,
"""
CATALOG = [
    {"rule_id": "R_DOC", "type": "Без документа", "required": ["cash_accepted_without_id_document"]},
    {"rule_id": "R_MIX", "type": "Смешаны деньги", "required": ["client_cash_mixed_or_left_for_next_client"]},
    {"rule_id": "R_DOOR", "type": "Дверь открыта", "required": ["cash_room_door_left_open"]},
    {"rule_id": "R_FOOD", "type": "Еда", "required": ["food_or_drink_at_desk"]},
    {"rule_id": "R_SAFE", "type": "Сейф", "required": ["safe_left_open"]},
    {"rule_id": "R_KEYS", "type": "Ключи в замке", "required": ["keys_in_safe_lock_new"]},
]


def _api(*viols, status="done"):
    if status != "done":
        return {"job_id": "j", "status": status}
    vs = [{"type": t, "timecode_sec": s, "description": f"... (ID: {rid}). http://x",
           "evidence_frames": [{"timestamp_sec": s, "image_base64": "/9j/"}]} for t, s, rid in viols]
    return {"job_id": "j", "status": "done", "verdict": "violation" if vs else "no_violation", "violations": vs}


def _write(d: Path, vid, api, elapsed=10.0, raw=False):
    d.mkdir(exist_ok=True)
    rec = api if raw else {"video": vid, "elapsed_sec": elapsed, "status": api["status"], "result": api}
    (d / f"{vid}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def env(tmp_path):
    (tmp_path / "labels.csv").write_text(LABELS, encoding="utf-8")
    (tmp_path / "cat.json").write_text(json.dumps(CATALOG, ensure_ascii=False), encoding="utf-8")
    codes = sv.load_catalog_codes([tmp_path / "cat.json"])
    return tmp_path, sv.load_labels(tmp_path / "labels.csv"), codes


def _run(d, labels, codes, **kw):
    return sv.score(sv.load_results(d, codes), labels, **kw)


def test_ids_and_splits():
    assert sv.norm_id("Х-33.mov") == sv.norm_id("x33") == sv.norm_id("33") == sv.norm_id("X-33.json") == "X-33"
    assert sv.norm_id("job_X 24") == "job_X 24"  # нераспознанное не трогаем
    assert sv.split_of("X-33") == "holdout" and sv.split_of("X-31") == "quarantine" and sv.split_of("X-1") == "dev"


def test_catalog_codes_and_accepted(env):
    _, labels, codes = env
    title2code, rid2code = codes
    assert title2code["Дверь открыта"] == "DOOR" and title2code["Ключи в замке"] == "KEYS"  # по ключевому слову
    assert rid2code["R_FOOD"] == "FOOD"
    assert labels["X-46"].accepted == {"LEAVE", "FOOD"} and labels["X-3"].accepted == set()
    # неизвестное название, но ID правила в описании — код по ID; без ID — «?название»
    assert sv.violation_code("Новое название", "x (ID: R_SAFE). u", codes) == "SAFE"
    assert sv.violation_code("Новое название", "", codes).startswith("?")


def test_metrics_and_baselines(env):
    tmp, labels, codes = env
    d = tmp / "a"
    _write(d, "X-1", _api(("Без документа", 20.5, "R_DOC")))                 # верно, ±3 с
    _write(d, "X-3", _api(("Без документа", 7.9, "R_DOC")))                  # ложная тревога
    _write(d, "X-4", _api(("Сейф", 56.0, "R_SAFE")))                         # вердикт да, тип нет, время да
    _write(d, "X-33", _api(status="failed"))                                 # failed = неверный вердикт
    _write(d, "X-35", _api())                                                # чисто, верно
    _write(d, "X-46", _api(("Еда", 18.0, "R_FOOD"), ("Сейф", 11.5, "R_SAFE")))  # тип да (не вовремя), время у чужого
    rows = _run(d, labels, codes)
    by = {r["video"]: r for r in rows}
    assert by["X-31"]["status"] == "missing" and "X-47" not in by           # двойник по умолчанию не в счёте
    assert by["X-1"]["verdict_ok"] and by["X-1"]["type_ok"] and by["X-1"]["tc_ok"]
    assert by["X-4"]["verdict_ok"] and not by["X-4"]["type_ok"] and by["X-4"]["tc_any"] and not by["X-4"]["tc_ok"]
    assert by["X-46"]["type_ok"] and not by["X-46"]["tc_ok"] and by["X-46"]["tc_any"] and by["X-46"]["extra"] == 1
    assert not by["X-33"]["verdict_ok"] and not by["X-33"]["done"]

    s = sv.by_split(rows)
    assert set(s) == {"dev", "holdout", "all"}                                # quarantine без результатов — нет строки
    a = s["all"]
    assert (a["n"], a["missing"], a["n_viol"], a["n_clean"]) == (6, 1, 4, 2)
    assert (a["done"], a["failed"], a["verdict_ok"], a["recall"], a["false_alarm"]) == (5, 1, 4, 3, 1)
    assert (a["type_ok"], a["tc_ok"], a["tc_any"], a["extra"]) == (2, 1, 3, 3)
    assert a["base_violation"] == {"verdict_ok": 4, "recall": 4, "false_alarm": 2, "type_ok": 0, "tc_ok": 0}
    assert a["base_clean"]["verdict_ok"] == 2 and a["base_clean"]["recall"] == 0
    assert a["elapsed_median"] == 10.0 and s["holdout"]["n"] == 3


def test_dupes_reuse_twin_result(env):
    tmp, labels, codes = env
    d = tmp / "a"
    _write(d, "X-4", _api(("Смешаны деньги", 67.0, "R_MIX")))
    rows = {r["video"]: r for r in _run(d, labels, codes, include_dupes=True)}
    assert rows["X-47"]["reused"] == "X-4" and rows["X-47"]["tc_ok"]          # 67 у X-47 (68) в допуске
    assert rows["X-4"]["type_ok"] and not rows["X-4"]["tc_ok"]                 # а у X-4 эталон 57


def test_raw_api_dir_jsonl_and_override(env, tmp_path):
    tmp, labels, codes = env
    d = tmp / "raw"
    _write(d, "Х-33", _api(("Дверь открыта", 6.5, "R_DOOR")), raw=True)       # имя файла кириллицей
    (d / "broken.json").write_text("{", encoding="utf-8")
    assert {r["video"]: r for r in _run(d, labels, codes)}["X-33"]["tc_ok"]

    jl = tmp / "r.jsonl"
    recs = [{"video": "X-33", "tf": "orig", "status": "done", "verdict": "violation", "elapsed": 5,
             "violations": [{"type": "Дверь открыта", "t": 6.0}]},
            {"video": "X-33", "tf": "mirror", "status": "done", "verdict": "no_violation", "violations": []}]
    jl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")
    assert sv.load_results(jl, codes)["X-33"].verdict == "violation"          # по умолчанию orig
    assert sv.load_results(jl, codes, tf="mirror")["X-33"].verdict == "no_violation"

    ov = tmp / "acc.csv"
    ov.write_text("type_code,accepted\nT10,SAFE|DOOR\n", encoding="utf-8")
    assert sv.load_labels(tmp / "labels.csv", ov)["X-33"].accepted == {"SAFE", "DOOR"}
    ov.write_text("video,accepted_checks\nХ-33,SAFE\n", encoding="utf-8")
    assert sv.load_labels(tmp / "labels.csv", ov)["X-33"].accepted == {"SAFE"}


def test_compare_and_cli(env, capsys):
    tmp, labels, codes = env
    a, b = tmp / "a", tmp / "b"
    _write(a, "X-1", _api())
    _write(b, "X-1", _api(("Без документа", 22.0, "R_DOC")))
    _write(a, "X-35", _api(("Сейф", 3.0, "R_SAFE")))
    _write(b, "X-35", _api(("Сейф", 3.0, "R_SAFE")))
    diffs = sv.compare(_run(a, labels, codes), _run(b, labels, codes), labels)
    assert [(d["video"], d["how"]) for d in diffs] == [("X-1", "лучше")]

    out = tmp / "s.json"
    assert sv.main([str(b), "--labels", str(tmp / "labels.csv"), "--catalog", str(tmp / "cat.json"),
                    "--json", str(out), "--csv", str(tmp / "rows.csv")]) == 0
    assert json.loads(out.read_text())["all"]["verdict_ok"] == 1
    assert "X-35" in (tmp / "rows.csv").read_text()
    assert sv.main([str(a), "--compare", str(b), "--labels", str(tmp / "labels.csv"),
                    "--catalog", str(tmp / "cat.json")]) == 0
    assert "лучше 1" in capsys.readouterr().out


def test_default_catalog_maps_every_rule():
    title2code, _ = sv.load_catalog_codes()
    cash = json.loads((sv.ROOT / "rules/cash_checks.json").read_text(encoding="utf-8"))
    assert all(not title2code[r["type"]].startswith("?") for r in cash)


def test_scorer_does_not_import_solution_code():
    tree = ast.parse(Path(sv.__file__).read_text(encoding="utf-8"))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not {m.split(".")[0] for m in mods if m} & {"forte_vision", "backend", "forte_decision", "decision"}
