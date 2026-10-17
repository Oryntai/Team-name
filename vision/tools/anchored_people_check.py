"""Проверка people.py на открытых роликах: кассир, уходы/приходы, визиты клиентов, посторонние.

  python vision/tools/anchored_people_check.py scan [--fps 5] [--only X-14,X-30]   # плотный проход в кеш
  python vision/tools/anchored_people_check.py eval [--only ...] [-v]              # сверка с разметкой

Разметка ниже — из описаний роликов (data/work/taxonomy, REPORT_cues «Присутствие кассира»), не из предсказаний.
X-47/X-48 — побайтные копии X-4/X-5, пропускаются.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

VIDEOS = Path("/Users/oryntajpazylbekov/projects/datathon/data/raw/open/videos")
CACHE = Path("/private/tmp/claude-501/-Users-oryntajpazylbekov-projects-datathon/"
             "ab6194b6-74ce-40d8-bc40-8ba5689bbe14/scratchpad/anchored_cache/scans")
SKIP = {"X-47", "X-48"}

# кассир: "leave" — (момент ухода с рабочего места, выход в дверь (a, b)); "enter" — вход в дверь (a, b)
LEAVE = {"X-11": (18.0, (17, 19)), "X-12": (None, (36, 40.5)), "X-13": (58.5, (60, 62)), "X-15": (15.5, (17, 18.5)),
         "X-22": (15.0, (15, 17.5)), "X-23": (None, (15, 16.5)), "X-24": (9.0, (9, 11)), "X-25": (None, (26, 28.5)),
         "X-26": (13.0, (13, 15)), "X-27": (None, (22.5, 25.5)), "X-28": (10.0, (12, 14)), "X-29": (None, (12, 13))}
ENTER = {"X-30": (2, 4), "X-31": (3, 7.5), "X-32": (4, 4.5), "X-33": (4.5, 5), "X-40": (6, 8)}
# визиты клиентов (вход в дверь .. выход), None = до конца ролика
VISITS = {"X-1": [(2.5, None)], "X-2": [(2.5, None)], "X-3": [(3, None)], "X-4": [(7, 54), (55, None)],
          "X-5": [(3, 60), (63, None)], "X-16": [(3, None)], "X-17": [(3.5, 12.5)], "X-18": [(2.5, None)],
          "X-19": [(3, 47), (49, None)], "X-34": [(2, 15.5)], "X-35": [(1.5, 54)], "X-37": [(3, None)],
          "X-38": [(2, 28), (31, 54)], "X-39": [(3, 40), (45, 75)]}
# посторонние / третий человек: (начало, конец, сторона по описанию)
THIRD = {"X-18": (16, 22.6, "across"), "X-30": (12.5, 18.8, "level"), "X-33": (11.5, 14.3, "behind")}


def vid_id(p: Path) -> str:
    return "X-" + p.stem.split("-")[1]  # кириллическая Х в именах файлов -> латинская X


def videos(only: set[str] | None = None) -> list[tuple[str, Path]]:
    out = [(vid_id(p), p) for p in VIDEOS.glob("*.mov")]
    out = [(v, p) for v, p in out if v not in SKIP and (not only or v in only)]
    return sorted(out, key=lambda x: int(x[0].split("-")[1]))


def cmd_scan(a):
    from forte_vision.anchored.detect import dense_scan, make_detector, save_scan

    det = make_detector(threshold=a.threshold)
    tot_v = tot_w = 0.0
    for v, p in videos(a.only):
        t0 = time.monotonic()
        scan = dense_scan(str(p), a.fps, detector=det)
        el = time.monotonic() - t0
        save_scan(scan, CACHE / v)
        tot_v += scan.duration
        tot_w += el
        print(f"{v:5s} {scan.duration:6.1f}s видео, {len(scan.frames):4d} кадров, {el:5.1f}s", flush=True)
    print(f"итого {tot_v:.0f}s видео за {tot_w:.0f}s ({tot_w / max(tot_v, 1):.3f}x)")


def transform(scan, every: int = 1, mirror: bool = False, crop: tuple[float, float] | None = None):
    """Проверка устойчивости без перепрогона детектора: реже кадры (every), зеркало по x, другой кадр по y
    (crop=(y0, y1): видна только эта полоса, рамки обрезаются и перенормируются)."""
    import numpy as np

    from forte_vision.anchored.types import Det, DenseScan, Frame

    def tb(b):
        x1, y1, x2, y2 = b
        if mirror:
            x1, x2 = 1 - x2, 1 - x1
        if crop:
            y0, yc = crop
            y1, y2 = (min(max(y1, y0), yc) - y0) / (yc - y0), (min(max(y2, y0), yc) - y0) / (yc - y0)
        return (x1, y1, x2, y2)

    def keep(b):
        return b[3] - b[1] > 0.02

    idx = list(range(0, len(scan.frames), every))
    frames = []
    for k in idx:
        f = scan.frames[k]
        frames.append(Frame(f.t, [Det(tb(d.box), d.conf, d.name) for d in f.persons if keep(tb(d.box))],
                            [Det(tb(d.box), d.conf, d.name) for d in f.objects if keep(tb(d.box))]))
    g = None if scan.gray is None else np.asarray(scan.gray)[idx]
    if g is not None and mirror:
        g = g[:, :, ::-1]
    if g is not None and crop:
        h = g.shape[1]
        g = g[:, int(crop[0] * h):int(np.ceil(crop[1] * h))]
    return DenseScan(scan.duration, scan.width, scan.height, scan.fps / every, frames,
                     [scan.gray_t[k] for k in idx], None if g is None else np.ascontiguousarray(g))


def _cov(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def cmd_eval(a):
    from forte_vision.anchored.detect import load_scan
    from forte_vision.anchored.people import analyze_people

    rows, s = [], {"n": 0, "cashier_ok": 0, "false_leave": 0, "leave_err": [], "exit_err": [], "enter_err": [],
                   "sit_err": [], "missed_leave": 0, "missed_enter": 0, "false_enter": 0, "vis_ok": 0,
                   "vis_n": 0, "false_vis": 0, "third_ok": 0, "false_third": 0, "change_ok": 0, "change_n": 0,
                   "arr_err": [], "gone_err": [], "cl_leave_spurious": 0, "cl_arr_spurious": 0}
    for v, _ in videos(a.only):
        stem = CACHE / v
        if not stem.with_suffix(".json").exists():
            print(f"{v}: нет кеша, сначала scan")
            continue
        scan = transform(load_scan(stem), a.every, a.mirror, a.crop)
        pp = analyze_people(scan)
        dur = scan.duration
        s["n"] += 1
        ev = {k: [e for e in pp.events if e.kind == k] for k in
              ("cashier_enters", "cashier_sits", "cashier_stands", "cashier_leaves", "client_arrives",
               "client_leaves", "client_change", "third_person")}
        notes = []
        # личность кассира: цепочка покрывает время, когда кассир по разметке в комнате
        g0 = ENTER[v][1] if v in ENTER else 0.0
        g1 = LEAVE[v][1][0] if v in LEAVE else dur
        pres = pp.cashier_present
        cov = sum(_cov(iv, (g0 + 0.5, g1 - 0.5)) for iv in pres) / max(g1 - g0 - 1, 1e-3)
        extra = sum(iv[1] - iv[0] for iv in pres) - sum(_cov(iv, (g0 - 1.5, g1 + 3.0)) for iv in pres)
        ok = pp.cashier is not None and cov >= 0.9 and extra <= 2.0
        s["cashier_ok"] += ok
        if not ok:
            notes.append(f"КАССИР? cov={cov:.2f} лишнее={extra:.1f}с")
        # уходы
        lv = ev["cashier_leaves"]
        if v in LEAVE:
            t_leave, door = LEAVE[v]
            if not lv:
                s["missed_leave"] += 1
                notes.append("уход ПРОПУЩЕН")
            else:
                e = lv[-1]
                te = e.meta.get("t_exit", e.t)
                s["exit_err"].append(te - door[1])
                if t_leave is not None:
                    s["leave_err"].append(e.t - t_leave)
                notes.append(f"уход {e.t:.1f} (эт {t_leave}) выход {te:.1f} (эт {door[0]}-{door[1]})")
                if len(lv) > 1:
                    s["false_leave"] += len(lv) - 1
                    notes.append(f"лишних уходов {len(lv) - 1}: {[round(x.t, 1) for x in lv[:-1]]}")
        elif lv:
            s["false_leave"] += len(lv)
            notes.append(f"ЛОЖНЫЙ уход {[round(x.t, 1) for x in lv]}")
        # приходы
        en = ev["cashier_enters"]
        if v in ENTER:
            if not en:
                s["missed_enter"] += 1
                notes.append("вход ПРОПУЩЕН")
            else:
                e = en[0]
                s["enter_err"].append(e.t - ENTER[v][1])
                sit = ev["cashier_sits"]
                notes.append(f"вход {e.t:.1f} (эт {ENTER[v][0]}-{ENTER[v][1]})"
                             + (f" сел {sit[0].t:.1f}" if sit else ""))
        elif en:
            s["false_enter"] += len(en)
            notes.append(f"ЛОЖНЫЙ вход {[round(x.t, 1) for x in en]}")
        # визиты
        gt = [(x, dur if y is None else y) for x, y in VISITS.get(v, [])]
        got = pp.visits
        for g in gt:
            s["vis_n"] += 1
            hit = any(_cov(g, x) >= 0.5 * (g[1] - g[0]) for x in got)
            s["vis_ok"] += hit
        for g0, g1 in gt:
            arr = [e for e in ev["client_arrives"] if abs(e.t - g0) <= 8]
            if g0 > 1.0 and arr:
                s["arr_err"].append(min((e.t - g0 for e in arr), key=abs))
            lv = [e for e in ev["client_leaves"] if abs(e.t_end - g1) <= 8]
            if g1 < dur - 1.0 and lv:
                s["gone_err"].append(min((e.t_end - g1 for e in lv), key=abs))
            if g1 >= dur - 1.0:
                s["cl_leave_spurious"] += sum(1 for e in ev["client_leaves"] if e.t_end > g0 + 1 and e.t_end >= g1 - 3)
        s["cl_arr_spurious"] += sum(1 for e in ev["client_arrives"] if not any(abs(e.t - g0) <= 8 for g0, _ in gt))
        fv = [x for x in got if not any(_cov(g, x) >= 0.3 * (x[1] - x[0]) for g in gt)]
        s["false_vis"] += len(fv)
        if gt or got:
            notes.append("визиты " + " ".join(f"{x:.1f}-{y:.1f}" for x, y in got)
                         + " (эт " + " ".join(f"{x}-{y:.0f}" for x, y in gt) + ")"
                         + (f" ЛИШНИХ {len(fv)}" if fv else ""))
        if len(gt) == 2:
            s["change_n"] += 1
            ch = ev["client_change"]
            gap = (gt[0][1], gt[1][0])
            hit = any(gap[0] - 6 <= e.t <= gap[1] + 6 for e in ch)
            s["change_ok"] += hit
            notes.append("смена " + (" ".join(f"{e.t:.1f}(gap {e.meta.get('gap', 0):.1f})" for e in ch) or "НЕТ"))
        elif ev["client_change"]:
            notes.append(f"ЛИШНЯЯ смена {[round(e.t, 1) for e in ev['client_change']]}")
        # посторонние
        tp = ev["third_person"]
        if v in THIRD:
            a0, a1, side = THIRD[v]
            hit = [e for e in tp if a0 - 3 <= e.t <= a1]
            s["third_ok"] += bool(hit)
            notes.append("третий " + (" ".join(f"{e.t:.1f}-{e.t_end or 0:.1f} {e.meta.get('side')}" for e in tp)
                                      or "НЕТ") + f" (эт {a0}-{a1} {side})")
            s["false_third"] += len(tp) - len(hit)
        elif tp:
            s["false_third"] += len(tp)
            notes.append("ЛОЖНЫЙ третий " + " ".join(f"{e.t:.1f}-{e.t_end or 0:.1f} {e.meta.get('side')}" for e in tp))
        stands = ev["cashier_stands"]
        rows.append((v, dur, pp.desk_dir, pp.roles_uncertain, notes, stands, pp))
        flag = "  " if all(not any(k in n for k in ("ЛОЖН", "ПРОПУЩ", "КАССИР?", "ЛИШН", "НЕТ")) for n in notes) else "!!"
        print(f"{flag} {v:5s} {dur:5.1f}s dir={pp.desk_dir:+d}{' U' if pp.roles_uncertain else ''} | "
              + "; ".join(notes) + (f"; встаёт {[round(e.t, 1) for e in stands]}" if a.verbose else ""))
        if a.verbose:
            print("      diag:", {k: val for k, val in pp.diag.items() if k != "tracks"})
            for tr in pp.visitors:
                print(f"      посетитель {tr.start:.1f}-{tr.end:.1f}")

    def stat(xs):
        if not xs:
            return "—"
        import numpy as np
        x = np.abs(np.array(xs))
        return f"n={len(xs)} медиана |ош| {np.median(x):.1f}с, макс {x.max():.1f}с, ср. знак {np.mean(xs):+.1f}с"

    print("\n=== итог ===")
    print(f"кассир верно: {s['cashier_ok']}/{s['n']}")
    print(f"уходы: пропущено {s['missed_leave']}/{len(LEAVE)}; момент ухода с места {stat(s['leave_err'])}; "
          f"выход (против конца выхода в дверь) {stat(s['exit_err'])}; ложных уходов {s['false_leave']}")
    print(f"входы: пропущено {s['missed_enter']}/{len(ENTER)}; ошибка (против конца входа) {stat(s['enter_err'])}; "
          f"ложных входов {s['false_enter']}")
    print(f"визиты клиентов найдено {s['vis_ok']}/{s['vis_n']}, лишних {s['false_vis']}; "
          f"смена клиента {s['change_ok']}/{s['change_n']}")
    print(f"клиенты: приход (против входа в дверь) {stat(s['arr_err'])}; ушёл (против выхода) {stat(s['gone_err'])}; "
          f"лишних приходов {s['cl_arr_spurious']}, лишних уходов {s['cl_leave_spurious']}")
    print(f"третий человек найден {s['third_ok']}/{len(THIRD)}, ложных {s['false_third']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "eval"])
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--only", type=lambda x: set(x.split(",")), default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--every", type=int, default=1, help="брать каждый N-й кадр скана (реже кадры)")
    ap.add_argument("--mirror", action="store_true", help="зеркало по горизонтали")
    ap.add_argument("--crop", type=lambda x: tuple(map(float, x.split(","))), default=None,
                    help="видна только полоса y0,y1 кадра (другой кадр камеры)")
    a = ap.parse_args(argv)
    {"scan": cmd_scan, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
