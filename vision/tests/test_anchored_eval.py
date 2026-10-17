"""Офлайн-стенд vision/tools/anchored_eval.py: id роликов, искажения ffmpeg, путь воркера на игрушечном анализаторе.

Видео — синтетика ffmpeg (lavfi), без роликов кейса, GPU и сервера модели.
"""

import importlib.util
import json
import re
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("anchored_eval", ROOT / "vision/tools/anchored_eval.py")
ae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ae)


def _ff():
    from backend.tests.video_fixtures import ffmpeg_tools

    return ffmpeg_tools()


def test_ids_groups_and_cyrillic_names(tmp_path):
    for name in ("X-1.mov", "Х-4.mov", "Х-31.mov", "Х-33.mov", "Х-47.mov", "notes.mov"):
        (tmp_path / name).write_bytes(b"")
    vids = ae.list_videos(tmp_path)
    assert list(vids) == ["X-1", "X-4", "X-31", "X-33", "X-47"] and vids["X-33"].name == "Х-33.mov"
    assert ae.resolve_ids(["all"], vids) == ["X-1", "X-4", "X-31", "X-33"]  # копия Х-47 только явно
    assert ae.resolve_ids(["dev"], vids) == ["X-1", "X-4"]
    assert ae.resolve_ids(["holdout", "quarantine"], vids) == ["X-33", "X-31"]
    assert ae.resolve_ids(["33,Х-47", "x33"], vids) == ["X-33", "X-47"]
    with pytest.raises(SystemExit):
        ae.resolve_ids(["X-99"], vids)


def _dims(ffprobe, path):
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"], s["r_frame_rate"]


def test_transforms_keep_fractions_and_even_sizes(tmp_path, monkeypatch):
    ffmpeg, ffprobe = _ff()
    monkeypatch.setitem(ae.FF, "ffmpeg_bin", ffmpeg)
    src = tmp_path / "src.mov"
    subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=120x212:rate=30:duration=0.4", "-pix_fmt", "yuv420p", str(src)], check=True)
    got = {tf: _dims(ffprobe, ae.transformed(src, "X-1", tf, tmp_path)) for tf in ae.TF}
    assert got["mirror"][:2] == got["persp"][:2] == (120, 212)
    assert got["landscape"][:2] == (212, 120)
    for tf in ("rot12", "zoomin"):  # размер исходника с точностью до чётного округления
        w, h, _ = got[tf]
        assert w % 2 == h % 2 == 0 and abs(w - 120) <= 2 and abs(h - 212) <= 2
    assert got["hires25"][1] == 1920 and got["hires25"][2] == "25/1"
    f = tmp_path / "tf/mirror/X-1.mov"
    m = f.stat().st_mtime_ns
    ae.transformed(src, "X-1", "mirror", tmp_path)  # кеш: второй раз не перекодирует
    assert f.stat().st_mtime_ns == m and ae.transformed(src, "X-1", None, tmp_path) == src


class Toy:
    """Режимы по очереди: ok — сейф открыт в середине ролика, boom — исключение, slow — не уложился в дедлайн."""

    def __init__(self, modes):
        self.modes = modes
        self.last_debug = {"calls": 0}

    def analyze(self, context, video):
        from backend.contracts import ObservationBatch, ObservedEvent

        self.last_debug["calls"] += 1
        mode = self.modes.pop(0)
        if mode == "boom":
            raise RuntimeError("boom")
        if mode == "slow":
            time.sleep(30)
        pts = video.probe().frame_pts
        m = pts[len(pts) // 2]
        ev = ObservedEvent(t_start=m, t_end=m, t_main=m, frame_pts=[m],
                           observations={"safe_left_open": "observed"}, sources=["detector"])
        return ObservationBatch(complete=True, events=[ev],
                                diagnostics={"analyzer": "toy", "name": Path(context.video_path).name})


def test_worker_path_failures_deadline_and_resume(tmp_path, monkeypatch, capsys):
    pytest.importorskip("backend.contracts")
    pytest.importorskip("forte_decision.backend_adapter")
    from backend.tests.video_fixtures import make_video
    from forte_decision.backend_adapter import create_rules

    ffmpeg, ffprobe = _ff()
    monkeypatch.setitem(ae.FF, "ffmpeg_bin", ffmpeg)
    monkeypatch.setitem(ae.FF, "ffprobe_bin", ffprobe)
    # каталог этого репозитория явно: при PYTHONPATH=. forte_decision может прийти из другой копии со своим rules/
    monkeypatch.setenv("FORTE_RULE_CATALOGS", json.dumps([str(ROOT / "rules/cash_checks.json")]))
    video = make_video(tmp_path / "Х-1.mov", ffmpeg, duration=1.0, rate=10)
    videos = {"X-1": video, "X-2": video, "X-3": video}
    modes, made = ["ok", "boom", "slow"], []

    def factory():
        made.append(1)
        return Toy(modes)

    out, cache = tmp_path / "res", tmp_path / "cache"
    kw = dict(cache=cache, deadline=2.0, meta={"factory": "toy"})
    recs = ae.evaluate(["X-1", "X-2", "X-3"], videos, factory, create_rules, out, **kw)
    ok, boom, slow = (json.loads((out / f"X-{i}.json").read_text(encoding="utf-8")) for i in (1, 2, 3))
    assert [r["status"] for r in recs] == ["done", "failed", "failed"]

    res = ok["result"]
    assert res["status"] == "done" and res["verdict"] == "violation"
    v = res["violations"][0]
    assert v["type"] == "Сейф оставлен открытым" and abs(v["timecode_sec"] - 0.5) < 0.06
    assert v["evidence_frames"][0]["image_base64"].startswith("/9j/")  # JPEG, как в ответе API
    assert ok["diagnostics"]["analyzer"] == "toy" and re.fullmatch(r"[0-9a-f]{32}\.mov", ok["diagnostics"]["name"])
    assert ok["debug"] == {"calls": 1} and ok["analyze_sec"] <= ok["elapsed_sec"] and ok["file"] == "Х-1.mov"

    assert boom["result"] == {"job_id": boom["job_id"], "status": "failed"}  # ровно JobFailed сервиса
    assert boom["error_code"] == "analysis_error" and not boom["killed"]
    assert slow["error_code"] == "deadline_exceeded" and slow["killed"] and slow["elapsed_sec"] < 10
    assert len(made) == 2  # после убийства анализатор создан заново, как процесс модели
    assert not any((cache / "jobs").iterdir())  # ссылки <job_id>.mov убраны
    assert "X-3" in capsys.readouterr().out  # строка на каждый ролик

    assert ae.evaluate(["X-1"], videos, factory, create_rules, out, **kw) == []  # готовое пропускается
    with pytest.raises(SystemExit):  # другой анализатор в тот же каталог — отказ, а не смесь
        ae.evaluate(["X-1"], videos, factory, create_rules, out, cache=cache, meta={"factory": "other"})
    from data_tools import score_videos as sv

    loaded = sv.load_results(out, sv.load_catalog_codes())
    assert loaded["X-1"].viols[0][0] == "SAFE" and loaded["X-2"].status == "failed"
