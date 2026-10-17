"""Офлайн-прогон любого анализатора по роликам открытого набора тем же путём, что у воркера сервиса.

backend.worker._analyze_job: VideoIO.validate -> analyzer.analyze -> правила каталога -> JPEG доказательств.
На выходе по файлу <out>/X-N.json на ролик: в поле result ровно тот JSON, что отдал бы
GET /api/v1/jobs/{id}, рядом время, диагностика анализатора и окружение. HTTP, очереди и хранилища нет.
Дедлайн как у сервиса: от старта ролика; вышли за него — failed (супервизор убил бы процесс модели,
здесь его заменяет SIGALRM, после чего анализатор создаётся заново, как при перезапуске процесса).

  cd anchored && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:decision:vision/src python vision/tools/anchored_eval.py \
      --factory forte_vision.anchored.analyzer:create_analyzer --videos holdout --out RESULTS [--transform mirror]
  python -m data_tools.score_videos RESULTS

--videos: X-33,X-37 | 33 37 | holdout | dev | quarantine | all (Х-47/Х-48 только явно: это копии Х-4/Х-5).
Повторный запуск пропускает готовые ролики (--force — пересчитать). Имя файла анализатор не видит:
ролик подкладывается жёсткой ссылкой <job_id>.mov, как в каталоге загрузок сервиса.
Если у объекта анализатора есть атрибут last_debug (JSON-сериализуемый), он сохраняется в поле debug.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # корень репозитория: vision/tools -> ../..
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_tools import score_videos as sv  # noqa: E402  (независимый счёт: разбиение и коды типов)

VIDEO_DIR = ROOT.parent / "data/raw/open/videos"
FF = {"ffmpeg_bin": os.environ.get("FORTE_FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg",
      "ffprobe_bin": os.environ.get("FORTE_FFPROBE_BIN") or shutil.which("ffprobe") or "ffprobe"}

# Синтетическая смена ракурса (как angle/run_angle.py), всё в долях кадра: поворот камеры на месте —
# гомография, её 2D-искажение передаёт точно; перенос камеры (параллакс) — нет.
E = lambda x: f"trunc({x}/2)*2"  # чётный размер для yuv420p
TF = {
    "mirror": "hflip",
    "persp": "perspective=x0=W*0.12:y0=H*0.03:x1=W*0.95:y1=0:x2=0:y2=H:x3=W*0.88:y3=H*0.97:interpolation=linear",
    "rot12": f"scale={E('iw*1.25')}:{E('ih*1.25')},rotate=12*PI/180:ow={E('iw/1.25')}:oh={E('ih/1.25')}:c=black",
    "landscape": "transpose=1",
    "zoomin": f"crop={E('iw*0.75')}:{E('ih*0.75')}:iw*0.20:ih*0.12,scale={E('iw/0.75')}:{E('ih/0.75')}",
    "hires25": "scale='if(gt(iw,ih),1920,-2)':'if(gt(iw,ih),-2,1920)':flags=bicubic,fps=25",
}
ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-an"]
EXTRA = {"hires25": ["-maxrate", "2400k", "-bufsize", "4800k"]}
GROUPS = ("all", "dev", "holdout", "quarantine")


def list_videos(video_dir: str | Path = VIDEO_DIR) -> dict[str, Path]:
    """'X-N' -> файл; имена бывают с кириллической Х и с латинской X."""
    found = {sv.norm_id(p.stem): p for p in Path(video_dir).glob("*.mov")}
    return {k: found[k] for k in sorted(found, key=sv._num) if re.fullmatch(r"X-\d+", k)}


def resolve_ids(specs: list[str], available: dict[str, Path]) -> list[str]:
    out: list[str] = []
    for tok in re.split(r"[,\s]+", " ".join(specs).strip()):
        if not tok:
            continue
        if tok.lower() in GROUPS:
            ids = [v for v in available if v not in sv.DUPES and tok.lower() in ("all", sv.split_of(v))]
        else:
            ids = [sv.norm_id(tok)]
        for v in ids:
            if v not in available:
                raise SystemExit(f"нет ролика {v} (смотрел {len(available)} файлов)")
            if v not in out:
                out.append(v)
    return out


def transformed(src: Path, vid: str, tf: str | None, cache: Path) -> Path:
    """Искажённая копия ролика (кешируется в cache/tf/<tf>/X-N.mov); без tf — исходный файл."""
    if not tf:
        return src
    out = cache / "tf" / tf / f"{vid}.mov"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".part.mov")
        subprocess.run([FF["ffmpeg_bin"], "-v", "error", "-nostdin", "-y", "-i", str(src), "-vf", TF[tf], *ENC,
                        *EXTRA.get(tf, []), "-movflags", "+faststart", str(tmp)], check=True, timeout=600)
        tmp.replace(out)
    return out


class Killed(TimeoutError):
    """Вышли за дедлайн внутри анализатора: в сервисе процесс модели был бы убит."""


def _on_alarm(signum, frame):
    raise Killed("deadline exceeded inside analyzer")


class _Timed:
    """Обёртка анализатора: замеряет сам analyze() отдельно от validate и выдачи JPEG."""

    def __init__(self, inner):
        self.inner, self.sec = inner, None

    def analyze(self, context, video):
        t0 = time.perf_counter()
        try:
            return self.inner.analyze(context, video)
        finally:
            self.sec = time.perf_counter() - t0


def _jsonable(x):
    try:
        json.dumps(x)
        return x
    except (TypeError, ValueError):
        return repr(x)[:2000]


def run_one(path: Path, analyzer, rules, *, work: Path, deadline: float = 300.0, profile: str = "standard",
            grace: float = 0.5) -> dict:
    """Один ролик путём воркера. Возвращает status/result/ошибку/время/диагностику (без полей ролика)."""
    from backend.worker import _analyze_job, _error_code

    job_id = uuid.uuid4().hex
    work.mkdir(parents=True, exist_ok=True)
    link = work / f"{job_id}.mov"
    try:
        os.link(path, link)
    except OSError:
        shutil.copy2(path, link)
    timed, diag = _Timed(analyzer), {}
    t0 = time.time()
    p0 = time.perf_counter()
    job = {"job_id": job_id, "video_path": str(link), "deadline_at": t0 + deadline, "profile": profile}
    alarm = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    old = signal.signal(signal.SIGALRM, _on_alarm) if alarm else None
    if alarm:
        signal.setitimer(signal.ITIMER_REAL, deadline + grace)
    rec = {"started_at": datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds"), "job_id": job_id}
    try:
        result = _analyze_job(job, FF, timed, rules, diag)
        rec.update(status="done", error_code=None, killed=False)
    except Exception as e:  # как воркер: любая ошибка -> failed без вердикта (Ctrl-C пролетает)
        result = {"job_id": job_id, "status": "failed"}
        rec.update(status="failed", error_code=_error_code(e), error_type=type(e).__name__, error=repr(e)[:500],
                   killed=isinstance(e, Killed))
    finally:
        if alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
        link.unlink(missing_ok=True)
    rec["elapsed_sec"] = round(time.perf_counter() - p0, 3)
    rec["analyze_sec"] = None if timed.sec is None else round(timed.sec, 3)
    rec["diagnostics"] = diag
    if getattr(analyzer, "last_debug", None) is not None:
        rec["debug"] = _jsonable(analyzer.last_debug)
    rec["result"] = result
    return rec


def _code_rev() -> str:
    try:
        rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True,
                               timeout=10).stdout.strip()
        return rev + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return ""


def _env() -> dict:
    """FORTE_* и офлайн-флаги без ключей и путей к ключам."""
    keep = ("FORTE_", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    return {k: v for k, v in sorted(os.environ.items())
            if k.startswith(keep) and not any(s in k for s in ("KEY", "TOKEN", "SECRET", "PASS"))}


def absolutize_catalogs() -> None:
    """Относительные пути FORTE_RULE_CATALOGS — от корня репозитория (сервис работает из него)."""
    raw = os.environ.get("FORTE_RULE_CATALOGS")
    if raw:
        paths = json.loads(raw)
        os.environ["FORTE_RULE_CATALOGS"] = json.dumps(
            [str(ROOT / p) if not Path(p).is_absolute() and (ROOT / p).exists() else str(Path(p).resolve())
             for p in paths])


def line(vid: str, rec: dict, gold, codes) -> str:
    res = rec["result"]
    if rec["status"] != "done":
        ans = f"FAILED {rec.get('error_code')} ({rec.get('error_type')})"
    elif res["verdict"] == "violation":
        ans = "V " + " ".join(f"{sv.violation_code(v['type'], v['description'], codes)}@{v['timecode_sec']:.1f}"
                              for v in res["violations"])
    else:
        ans = "—"
    g = ("" if gold is None else f"{gold.type_code}@{gold.sec:g}" if gold.viol and gold.sec is not None
         else "violation" if gold.viol else "чистый")
    an = f", analyze {rec['analyze_sec']:.1f}" if rec.get("analyze_sec") is not None else ""
    return f"{vid:6} {sv.split_of(vid):10} эталон {g:10} | {ans:40} | {rec['elapsed_sec']:.1f} с{an}"


def evaluate(ids, videos: dict[str, Path], make_analyzer, make_rules, out: Path, *, cache: Path,
             transform: str | None = None, deadline: float = 300.0, profile: str = "standard",
             force: bool = False, meta: dict | None = None, labels=None, codes=None) -> list[dict]:
    """Прогон по списку id; make_* — вызываемые без аргументов (как фабрики сервиса). Пишет out/X-N.json."""
    out.mkdir(parents=True, exist_ok=True)
    meta = {"transform": transform, "deadline_sec": deadline, "profile": profile, **(meta or {})}
    todo = []
    for vid in ids:
        f = out / f"{vid}.json"
        if f.exists() and not force:
            old = json.loads(f.read_text(encoding="utf-8"))
            if (old.get("factory"), old.get("transform")) != (meta.get("factory"), transform):
                raise SystemExit(f"{f}: там прогон {old.get('factory')} / {old.get('transform')}, "
                                 f"а сейчас {meta.get('factory')} / {transform}: другой --out или --force")
            print(f"{vid:6} уже есть, пропуск", flush=True)
            continue
        todo.append(vid)
    if not todo:
        return []
    codes = codes or sv.load_catalog_codes()
    labels = labels or {}
    t = time.perf_counter()
    analyzer, rules = make_analyzer(), make_rules()
    print(f"анализатор готов за {time.perf_counter() - t:.1f} с; роликов {len(todo)} -> {out}", flush=True)
    recs = []
    for vid in todo:
        src = videos[vid]
        path = transformed(src, vid, transform, cache)
        rec = {"video": vid, "file": src.name, "split": sv.split_of(vid), **meta}
        rec.update(run_one(path, analyzer, rules, work=cache / "jobs", deadline=deadline, profile=profile))
        f = out / f"{vid}.json"
        tmp = f.with_suffix(".part")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(f)
        print(line(vid, rec, labels.get(vid), codes), flush=True)
        recs.append(rec)
        if rec.get("killed"):  # сервис перезапустил бы процесс модели
            analyzer = make_analyzer()
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--factory", default="forte_vision.backend_adapter:create_analyzer", help="module:function")
    ap.add_argument("--rules-factory", default="forte_decision.backend_adapter:create_rules")
    ap.add_argument("--videos", nargs="+", required=True, help="X-33,X-37 | holdout | dev | quarantine | all")
    ap.add_argument("--out", required=True, help="каталог результатов (по JSON на ролик)")
    ap.add_argument("--transform", choices=sorted(TF), help="искажённые копии роликов вместо исходных")
    ap.add_argument("--deadline", type=float, default=300.0, help="секунд на ролик, как у сервиса")
    ap.add_argument("--profile", default="standard", help="context.profile (сервис по умолчанию: standard)")
    ap.add_argument("--force", action="store_true", help="пересчитать готовые")
    ap.add_argument("--video-dir", default=str(VIDEO_DIR))
    ap.add_argument("--cache", default=os.environ.get("FORTE_EVAL_CACHE")
                    or str(Path(tempfile.gettempdir()) / "forte_anchored_eval"),
                    help="искажённые копии и ссылки <job_id>.mov")
    ap.add_argument("--labels", default=str(sv.DEFAULT_LABELS), help="только для строки «эталон» и сводки")
    a = ap.parse_args(argv)

    absolutize_catalogs()
    import importlib

    from backend.worker import _factory

    code_paths = {}
    for name in ("backend", "forte_decision", "forte_vision"):  # не та копия кода — частая ошибка PYTHONPATH
        try:
            code_paths[name] = str(Path(importlib.import_module(name).__file__).resolve().parent)
        except ImportError:
            continue
        if ROOT not in Path(code_paths[name]).parents:
            print(f"ВНИМАНИЕ: {name} загружен из {code_paths[name]}, а не из {ROOT} — проверь PYTHONPATH",
                  file=sys.stderr)
    videos = list_videos(a.video_dir)
    ids = resolve_ids(a.videos, videos)
    labels = sv.load_labels(a.labels) if Path(a.labels).exists() else {}
    meta = {"factory": a.factory, "rules_factory": a.rules_factory, "code_paths": code_paths,
            "code_rev": _code_rev(), "env": _env()}
    out = Path(a.out)
    evaluate(ids, videos, lambda: _factory(a.factory), lambda: _factory(a.rules_factory), out,
             cache=Path(a.cache), transform=a.transform, deadline=a.deadline, profile=a.profile, force=a.force,
             meta=meta, labels=labels)
    if labels:  # сводка по роликам этого запуска (тем же независимым счётом)
        res = {k: v for k, v in sv.load_results(out, sv.load_catalog_codes()).items() if k in ids}
        rows = [r for r in sv.score(res, labels, include_dupes=True) if r["video"] in res]
        sv.print_summary(sv.by_split(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
