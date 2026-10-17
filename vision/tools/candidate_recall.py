"""Полнота механизма кандидатов: попадает ли размеченная секунда нарушения в отобранного кандидата.

Печатает только агрегаты (и id роликов-промахов): ни кадров, ни описаний. Подробности по роликам
пишутся в локальный JSONL (--out), который в git не идёт.
Разметка: JSON-массив {"file", "label", "timestamp_sec"} — формат data_tools/inventory.

    uv run python tools/candidate_recall.py --annotations ../local_data/annotations.json \
        --videos /path/to/videos --profile fast --detector nano --vlm none --out runs/recall.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

from forte_vision.observer import PROFILES, analyze_video
from forte_vision.vlm import StubBackend, make_backend


def _covered_frac(cands, duration):
    iv = sorted((c["start_sec"], c["end_sec"]) for c in cands)
    total, cur_s, cur_e = 0.0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total / duration if duration > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="fast")
    ap.add_argument("--detector", choices=["none", "nano", "small", "medium"], default="nano")
    ap.add_argument("--vlm", choices=["none", "mlx", "transformers", "openai"], default="none")
    ap.add_argument("--model")
    ap.add_argument("--tol", type=float, default=1.0, help="допуск к размеченной секунде, с")
    ap.add_argument("--only", nargs="*", help="ограничить списком файлов")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    rows = json.load(open(a.annotations, encoding="utf-8"))
    if a.only:
        rows = [r for r in rows if r["file"] in set(a.only)]
    from dataclasses import replace

    prof = PROFILES[a.profile]
    if a.vlm == "none":
        backend = StubBackend(lambda times, q: "[]" if "Categories:" in q else "{}", model_id="none")
        prof = replace(prof, screen=False)
    else:
        backend = make_backend(a.vlm, a.model)
    detector = None
    if a.detector != "none":
        from forte_vision.scene import Detector

        detector = Detector(a.detector)

    results = []
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fo:
        for r in rows:
            path = os.path.join(a.videos, r["file"])
            t0 = time.monotonic()
            obs, diag = analyze_video(path, backend, video_id=r["file"], profile=prof, detector=detector,
                                      deadline_sec=1e9)
            cands, t = obs["candidates"], r.get("timestamp_sec")
            res = {
                "file": r["file"],
                "label": r["label"],
                "n_cands": len(cands),
                "covered_frac": round(_covered_frac(cands, obs["duration_sec"]), 3),
                "status": obs["analysis_status"],
                "sec": round(time.monotonic() - t0, 1),
            }
            if r["label"] == 1 and t is not None:
                hits = [c for c in cands if c["start_sec"] - a.tol <= t <= c["end_sec"] + a.tol]
                res["hit"] = bool(hits)
                res["hit_sources"] = sorted({s for c in hits for s in c["source"].split("+")})
                res["main_err"] = round(min(abs(c["main_sec"] - t) for c in hits), 2) if hits else None
            results.append(res)
            fo.write(json.dumps({**res, "diag_scene": diag.get("scene"), "timings": diag.get("stage_timings")},
                                ensure_ascii=False) + "\n")

    pos = [x for x in results if x["label"] == 1]
    neg = [x for x in results if x["label"] == 0]
    hit = [x for x in pos if x.get("hit")]
    print(f"profile={a.profile} detector={a.detector} vlm={a.vlm} tol=±{a.tol}s videos={len(results)}")
    print(f"recall: {len(hit)}/{len(pos)}")
    srcs = sorted({s for x in hit for s in x["hit_sources"]})
    for s in srcs:
        print(f"  через {s}: {sum(s in x['hit_sources'] for x in hit)}/{len(pos)}")
    errs = [x["main_err"] for x in hit if x["main_err"] is not None]
    if errs:
        print(f"|main - метка| у попаданий: медиана {statistics.median(errs):.1f} с, макс {max(errs):.1f} с")
    for name, grp in (("с нарушением", pos), ("без нарушения", neg)):
        if grp:
            print(f"{name}: кандидатов в среднем {statistics.mean(x['n_cands'] for x in grp):.1f}, "
                  f"покрыто видео {statistics.median(x['covered_frac'] for x in grp):.0%} (медиана)")
    secs = [x["sec"] for x in results]
    print(f"время на ролик: медиана {statistics.median(secs):.1f} с, макс {max(secs):.1f} с")
    print("промахи:", " ".join(x["file"] for x in pos if not x.get("hit")) or "нет")


if __name__ == "__main__":
    main()
