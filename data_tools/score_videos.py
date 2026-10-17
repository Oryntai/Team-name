"""Независимый счёт прогонов анализатора по роликам открытого набора.

Код решения не импортируется (ни forte_vision, ни backend, ни forte_decision): эталон — team_labels.csv,
русское название типа -> код проверки берётся из каталогов rules/*.json как из простого JSON
(type -> required[0] -> код). Так счёт не «подстраивается» под то, как решение само себя понимает.

  python -m data_tools.score_videos RESULTS                      # таблица по роликам + сводка по частям
  python -m data_tools.score_videos RESULTS --compare RESULTS2   # что поменялось между двумя прогонами
  python -m data_tools.score_videos angle/results.jsonl --tf mirror

RESULTS — каталог *.json (запись vision/tools/anchored_eval.py или голый ответ API; id из имени файла)
либо .jsonl от angle/run_angle.py (поле tf выбирается --tf, по умолчанию orig).
Части: holdout (отложенная), quarantine (Х-31: на ней подбирали правила дверей), dev — остальное.
Х-47 = Х-4 и Х-48 = Х-5 побайтно: по умолчанию в счёт идут только Х-4/Х-5 (--include-dupes — оба).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT.parent / "data/work/team_labels.csv"
TOL = 3.0  # допуск таймкода, с

HOLDOUT = frozenset(f"X-{n}" for n in (33, 34, 35, 37, 38, 39, 40, 43, 44, 45, 46))
QUARANTINE = frozenset({"X-31"})
DUPES = {"X-47": "X-4", "X-48": "X-5"}  # копия -> оригинал (один и тот же файл)
TWIN = {**DUPES, **{v: k for k, v in DUPES.items()}}

# Тип из разметки команды -> какие коды проверок засчитываются как верный тип.
ACCEPTED = {
    "T01": {"DOC"}, "T02": {"MIX"}, "T03": {"LOCK"}, "T04": {"REG", "BAG"}, "T05": {"QUE"},
    "T06": {"SAFE"}, "T07": {"KEYS"}, "T08": {"LEAVE", "FOOD"}, "T09": {"DOOR", "OUTSIDER"},
    "T10": {"DOOR"}, "T11": {"NOCLIENT"}, "T12": {"ITEMS"}, "T13": {"AUTH"},
}
# Предикат каталога -> код проверки. Новые предикаты без записи здесь узнаются по ключевым словам.
PRED2CODE = {
    "cash_accepted_without_id_document": "DOC", "client_cash_mixed_or_left_for_next_client": "MIX",
    "left_with_computer_unlocked": "LOCK", "left_valuables_on_desk": "LEAVE", "safe_left_open": "SAFE",
    "cash_room_door_left_open": "DOOR", "counted_after_client_left": "NOCLIENT",
    "personal_items_on_desk": "ITEMS", "food_or_drink_at_desk": "FOOD",
    "banknotes_not_checked_on_detector": "AUTH", "bag_cash_to_safe_without_registration": "REG",
    "collection_bag_left_uncounted": "BAG", "served_out_of_queue": "QUE", "keys_left_in_safe_lock": "KEYS",
    "client_absent_during_counting": "NOCLIENT",  # rules/cash_receipt.json
}
KEYWORDS = (("key", "KEYS"), ("outsider", "OUTSIDER"), ("stranger", "OUTSIDER"), ("door", "DOOR"),
            ("queue", "QUE"), ("safe", "SAFE"))


def norm_id(s: str) -> str:
    """'Х-33.mov' (кириллица), 'x33', '33', 'X-33.json' -> 'X-33'. Нераспознанное — как есть."""
    s = unicodedata.normalize("NFC", str(s)).strip()
    s = re.sub(r"\.(mov|json)$", "", s, flags=re.I)
    m = re.fullmatch(r"(?:[XxХх][\s_-]*)?(\d+)", s)
    return f"X-{int(m.group(1))}" if m else s


def _num(vid: str) -> int:
    m = re.search(r"\d+", vid)
    return int(m.group()) if m else 10**6


def split_of(vid: str) -> str:
    return "holdout" if vid in HOLDOUT else "quarantine" if vid in QUARANTINE else "dev"


@dataclass
class Gold:
    vid: str
    viol: bool
    sec: float | None
    type_code: str  # T01..T13 или ""
    accepted: set[str] = field(default_factory=set)
    note: str = ""


@dataclass
class Res:
    status: str  # done | failed | missing | иное (client_timeout ...)
    verdict: str | None = None
    viols: list[tuple[str, float | None, str]] = field(default_factory=list)  # (код, таймкод, название)
    elapsed: float | None = None
    error: str | None = None
    reused: str | None = None  # взят результат ролика-двойника


def _split_codes(s: str) -> set[str]:
    return {c.strip().upper() for c in re.split(r"[|,;\s]+", s or "") if c.strip()}


def load_accepted(path: str | Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """CSV-переопределение: столбцы type_code+accepted (на тип) и/или video|id+accepted|accepted_checks (на ролик).
    Формат eval_truth.csv аудита (video, accepted_checks) подходит как есть."""
    by_type, by_video = {}, {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            acc = _split_codes(r.get("accepted") or r.get("accepted_checks") or "")
            vid = r.get("video") or r.get("id")
            if vid:
                by_video[norm_id(vid)] = acc
            elif r.get("type_code"):
                by_type[r["type_code"].strip()[:3].upper()] = acc
    return by_type, by_video


def load_labels(path: str | Path = DEFAULT_LABELS, accepted_csv: str | Path | None = None) -> dict[str, Gold]:
    by_type, by_video = load_accepted(accepted_csv) if accepted_csv else ({}, {})
    amap = {**ACCEPTED, **by_type}
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            vid = norm_id(r["id"])
            viol = r["official_verdict"].strip() == "violation"
            tcode = (r.get("type_code") or "").strip()[:3].upper() if viol else ""
            sec = float(r["official_sec"]) if (r.get("official_sec") or "").strip() else None
            acc = set(by_video.get(vid, amap.get(tcode, set()))) if viol else set()
            out[vid] = Gold(vid, viol, sec, tcode, acc, r.get("team_note", ""))
    return out


def pred_code(pred: str) -> str:
    if pred in PRED2CODE:
        return PRED2CODE[pred]
    return next((c for k, c in KEYWORDS if k in pred), "?" + pred)


def load_catalog_codes(paths=None) -> tuple[dict[str, str], dict[str, str]]:
    """Каталоги правил как простой JSON -> (название типа -> код, rule_id -> код). По умолчанию rules/*.json
    репозитория, cash_checks.json первым (при совпадении названий выигрывает первый каталог)."""
    if paths is None:
        rules_dir = ROOT / "rules"
        paths = sorted(rules_dir.glob("*.json"), key=lambda p: (p.name != "cash_checks.json", p.name))
    title2code, rid2code = {}, {}
    for p in paths:
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rules = data if isinstance(data, list) else data.get("rules", []) if isinstance(data, dict) else []
        for r in rules:
            if not isinstance(r, dict) or not r.get("type") or not r.get("required"):
                continue
            codes = [pred_code(x) for x in r["required"]]
            code = next((c for c in codes if not c.startswith("?")), codes[0])
            title2code.setdefault(r["type"], code)
            if r.get("rule_id"):
                rid2code.setdefault(r["rule_id"], code)
    return title2code, rid2code


def violation_code(title: str, description: str, codes) -> str:
    title2code, rid2code = codes
    if title in title2code:
        return title2code[title]
    m = re.search(r"\(ID: ([^)]+)\)", description or "")
    if m and m.group(1).strip() in rid2code:
        return rid2code[m.group(1).strip()]
    return "?" + (title or "")[:30]


def _res_from_api(api: dict, codes, elapsed=None, error=None) -> Res:
    viols = []
    for v in api.get("violations") or []:
        t = v.get("timecode_sec", v.get("t"))
        viols.append((violation_code(v.get("type") or "", v.get("description") or "", codes),
                      None if t is None else float(t), v.get("type") or ""))
    return Res(str(api.get("status") or "missing"), api.get("verdict"), viols, elapsed, error)


def load_results(path: str | Path, codes, tf: str | None = None) -> dict[str, Res]:
    path = Path(path)
    out: dict[str, Res] = {}
    if path.is_dir():
        for p in sorted(path.glob("*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                print(f"пропуск {p.name}: битый JSON", file=sys.stderr)
                continue
            if isinstance(d.get("result"), dict):  # запись anchored_eval
                out[norm_id(d.get("video") or p.stem)] = _res_from_api(
                    d["result"], codes, d.get("elapsed_sec"), d.get("error_code"))
            else:  # голый ответ API
                out[norm_id(p.stem)] = _res_from_api(d, codes, d.get("elapsed_sec"))
        return out
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tfs = list(dict.fromkeys(r.get("tf") for r in recs))
    if tf is None and tfs != [None]:
        tf = "orig" if "orig" in tfs else tfs[0]
        print(f"jsonl: беру tf={tf} (есть: {', '.join(map(str, tfs))})", file=sys.stderr)
    for r in recs:
        if r.get("tf") == tf or tf is None:
            out[norm_id(r["video"])] = _res_from_api(r, codes, r.get("elapsed"), r.get("error"))
    return out


def judge(g: Gold, r: Res) -> dict:
    done = r.status == "done"
    pv = done and r.verdict == "violation"
    good = [(c, t) for c, t, _ in r.viols if c in g.accepted]
    near = lambda t: g.sec is not None and t is not None and abs(t - g.sec) <= TOL
    dts = [abs(t - g.sec) for _, t in good if t is not None and g.sec is not None]
    return {
        "video": g.vid, "split": split_of(g.vid), "gold": g.viol, "type_code": g.type_code, "sec": g.sec,
        "status": r.status, "pred": pv, "done": done,
        "verdict_ok": done and pv == g.viol,
        "type_ok": g.viol and pv and bool(good),
        "tc_ok": g.viol and any(near(t) for _, t in good),
        "tc_any": g.viol and any(near(t) for _, t, _ in r.viols),
        "dt": min(dts) if dts else None,
        "extra": sum(c not in g.accepted for c, _, _ in r.viols),  # лишние нарушения (не того типа / на чистом)
        "codes": "+".join(f"{c}@{t:.1f}" if t is not None else c for c, t, _ in r.viols),
        "elapsed": r.elapsed, "error": r.error, "reused": r.reused,
    }


def score(results: dict[str, Res], labels: dict[str, Gold], include_dupes: bool = False) -> list[dict]:
    """Строки по роликам, у которых есть результат. Двойник без своего результата берёт результат пары."""
    rows = []
    for vid in sorted(labels, key=_num):
        if vid in DUPES and not include_dupes:
            continue
        r = results.get(vid)
        if r is None and vid in TWIN and TWIN[vid] in results:
            src = results[TWIN[vid]]
            r = Res(src.status, src.verdict, src.viols, src.elapsed, src.error, reused=TWIN[vid])
        rows.append(judge(labels[vid], r) if r else {"video": vid, "split": split_of(vid), "status": "missing"})
    return rows


def summarize(rows: list[dict]) -> dict:
    """Метрики по строкам с результатом; missing не в знаменателе, failed — ошибка вердикта."""
    sc = [r for r in rows if r["status"] != "missing"]
    nv = sum(r["gold"] for r in sc)
    nc = len(sc) - nv
    el = sorted(r["elapsed"] for r in sc if r.get("elapsed") is not None)
    s = {
        "n": len(sc), "missing": len(rows) - len(sc), "n_viol": nv, "n_clean": nc,
        "done": sum(r["done"] for r in sc), "failed": sum(not r["done"] for r in sc),  # failed и любой не-done
        "verdict_ok": sum(r["verdict_ok"] for r in sc),
        "recall": sum(r["pred"] and r["gold"] for r in sc),
        "false_alarm": sum(r["pred"] and not r["gold"] for r in sc),
        "type_ok": sum(r["type_ok"] for r in sc), "tc_ok": sum(r["tc_ok"] for r in sc),
        "tc_any": sum(r["tc_any"] for r in sc), "extra": sum(r["extra"] for r in sc),
        "elapsed_median": round(statistics.median(el), 1) if el else None,
        "elapsed_max": round(el[-1], 1) if el else None,
    }
    # константные базы на тех же роликах
    s["base_violation"] = {"verdict_ok": nv, "recall": nv, "false_alarm": nc, "type_ok": 0, "tc_ok": 0}
    s["base_clean"] = {"verdict_ok": nc, "recall": 0, "false_alarm": 0, "type_ok": 0, "tc_ok": 0}
    return s


SPLITS = ("holdout", "dev", "quarantine", "all")


def by_split(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for sp in SPLITS:
        part = [r for r in rows if sp == "all" or r["split"] == sp]
        if any(r["status"] != "missing" for r in part):
            out[sp] = summarize(part)
    return out


def _f(k, n):
    return f"{k}/{n}" if n else "-"


def _mark(r, key):
    if r["status"] == "missing":
        return " "
    if key != "verdict_ok" and not r["gold"]:
        return "."
    return "+" if r[key] else "x"


def _gold_cell(g: Gold) -> str:
    return f"{g.type_code or '?'}@{g.sec:g}" if g.viol and g.sec is not None else ("violation" if g.viol else "чистый")


def _pred_cell(r: dict) -> str:
    if r["status"] == "missing":
        return "нет результата"
    if r["status"] != "done":
        return f"{r['status'].upper()}" + (f" {r['error']}" if r.get("error") else "")
    return ("V " + r["codes"]) if r["pred"] else "—"


def print_rows(rows, labels, out=None):
    print(f"{'ролик':7} {'часть':10} {'эталон':12} {'В':1} {'Т':1} {'±3':2} {'dt':>5} {'с':>6}  ответ", file=out)
    for r in rows:
        dt = f"{r['dt']:.1f}" if r.get("dt") is not None else ""
        el = f"{r['elapsed']:.0f}" if r.get("elapsed") is not None else ""
        tail = f" (результат {r['reused']})" if r.get("reused") else ""
        print(f"{r['video']:7} {r['split']:10} {_gold_cell(labels[r['video']]):12} {_mark(r, 'verdict_ok')} "
              f"{_mark(r, 'type_ok')} {_mark(r, 'tc_ok'):2} {dt:>5} {el:>6}  {_pred_cell(r)}{tail}", file=out)


def print_summary(summ: dict[str, dict], out=None):
    print("\nчасть       n  done fail | вердикт  recall  ложн.тревог | тип   ±3с(тип) ±3с(любой) лишних "
          "| медиана макс, с", file=out)
    for sp, s in summ.items():
        miss = f"  (нет результата: {s['missing']})" if s["missing"] else ""
        print(f"{sp:10} {s['n']:>3} {s['done']:>5} {s['failed']:>4} | {_f(s['verdict_ok'], s['n']):>7} "
              f"{_f(s['recall'], s['n_viol']):>7} {_f(s['false_alarm'], s['n_clean']):>12} | "
              f"{_f(s['type_ok'], s['n_viol']):>5} {_f(s['tc_ok'], s['n_viol']):>9} {_f(s['tc_any'], s['n_viol']):>10} "
              f"{s['extra']:>6} | {s['elapsed_median'] or '-':>7} {s['elapsed_max'] or '-':>5}{miss}", file=out)
        for name, b in (("всё=нарушение", s["base_violation"]), ("всё=чисто", s["base_clean"])):
            print(f"  {name:23} | {_f(b['verdict_ok'], s['n']):>7} {_f(b['recall'], s['n_viol']):>7} "
                  f"{_f(b['false_alarm'], s['n_clean']):>12} | {_f(0, s['n_viol']):>5} {_f(0, s['n_viol']):>9}", file=out)


def _key(r):  # что считается «ответ изменился»
    return (r["status"], r.get("pred"), r.get("codes"))


def _rank(r):  # чем больше, тем лучше; вердикт важнее типа, тип важнее таймкода
    return (r["verdict_ok"], r["type_ok"], r["tc_ok"], -r["extra"])


def compare(rows_a: list[dict], rows_b: list[dict], labels, out=None) -> list[dict]:
    """Ролики с результатом в обоих прогонах, где ответ B отличается от A; лучше/хуже — по
    (вердикт, тип, таймкод, лишние). Ролики только с одной стороны лишь считаются."""
    has = lambda rows: {r["video"]: r for r in rows if r["status"] != "missing"}
    a_by, b_by = has(rows_a), has(rows_b)
    diffs = []
    for vid in sorted(a_by.keys() & b_by.keys(), key=_num):
        a, b = a_by[vid], b_by[vid]
        if _key(a) == _key(b):
            continue
        ra, rb = _rank(a), _rank(b)
        how = "лучше" if rb > ra else "хуже" if rb < ra else "иначе"
        diffs.append({"video": vid, "how": how, "a": _pred_cell(a), "b": _pred_cell(b)})
        print(f"{vid:7} {split_of(vid):10} {_gold_cell(labels[vid]):12} {how:6} "
              f"A: {_pred_cell(a):34} B: {_pred_cell(b)}", file=out)
    cnt = {h: sum(d["how"] == h for d in diffs) for h in ("лучше", "хуже", "иначе")}
    print(f"общих роликов {len(a_by.keys() & b_by.keys())}, отличий {len(diffs)}"
          + "".join(f", {k} {v}" for k, v in cnt.items() if v)
          + f"; только в A: {len(a_by.keys() - b_by.keys())}, только в B: {len(b_by.keys() - a_by.keys())}", file=out)
    return diffs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="каталог *.json или .jsonl")
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--accepted", help="CSV: type_code,accepted или video,accepted_checks")
    ap.add_argument("--catalog", action="append", help="каталог правил (можно несколько); по умолчанию rules/*.json")
    ap.add_argument("--include-dupes", action="store_true", help="считать и Х-47/Х-48")
    ap.add_argument("--tf", help="для .jsonl run_angle: какое искажение брать")
    ap.add_argument("--compare", metavar="RESULTS2", help="показать отличия RESULTS2 от RESULTS")
    ap.add_argument("--tf2", help="--tf для RESULTS2")
    ap.add_argument("--split", choices=SPLITS, help="таблицу по роликам — только по этой части")
    ap.add_argument("--csv", help="записать строки по роликам в CSV")
    ap.add_argument("--json", help="записать сводку в JSON")
    a = ap.parse_args(argv)

    codes = load_catalog_codes(a.catalog)
    labels = load_labels(a.labels, a.accepted)
    res = load_results(a.results, codes, a.tf)
    unknown = sorted({c for r in res.values() for c, _, _ in r.viols if c.startswith("?")})
    if unknown:
        print("типы без кода (считаются неверными): " + "; ".join(unknown), file=sys.stderr)
    stray = sorted(set(res) - set(labels), key=_num)
    if stray:
        print("результаты без эталона (не в счёте): " + ", ".join(stray), file=sys.stderr)
    rows = score(res, labels, a.include_dupes)
    if a.compare:
        rows_b = score(load_results(a.compare, codes, a.tf2), labels, a.include_dupes)
        print(f"A = {a.results}\nB = {a.compare}")
        compare(rows, rows_b, labels)
        common = {r["video"] for r in rows if r["status"] != "missing"} & {
            r["video"] for r in rows_b if r["status"] != "missing"}
        for name, rr in (("A", rows), ("B", rows_b)):  # сводки на общих роликах, иначе несравнимы
            print(f"\n== {name} (общие ролики)", end="")
            print_summary(by_split([r for r in rr if r["video"] in common]))
        return 0
    shown = [r for r in rows if r["status"] != "missing" and (a.split in (None, "all") or r["split"] == a.split)]
    print_rows(shown, labels)
    summ = by_split(rows)
    print_summary(summ)
    if a.csv:
        cols = ["video", "split", "gold", "type_code", "sec", "status", "pred", "verdict_ok", "type_ok", "tc_ok",
                "tc_any", "dt", "extra", "codes", "elapsed", "error", "reused"]
        with open(a.csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    if a.json:
        Path(a.json).write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
