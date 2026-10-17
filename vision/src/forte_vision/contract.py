"""Контракт Observation между vision (наблюдения) и decision (правила, вердикт).

Зеркало формата, согласованного с decision/: строгий, лишние поля запрещены. Когда в репозитории
появится парсер forte_decision, этот валидатор остаётся только как самопроверка vision перед отдачей.
Числовые оценки, версии моделей и тайминги сюда не входят — они в diagnostics.
"""

from __future__ import annotations

import math
from typing import Any

TRI_STATE = ("observed", "not_observed", "unclear")
ANALYSIS_STATUS = ("complete", "partial", "failed")

_TOP = {"video_id", "duration_sec", "analysis_status", "coverage_complete", "candidates"}
_CAND = {"candidate_id", "start_sec", "end_sec", "main_sec", "source", "evidence", "predicates", "visibility"}
_EVID = {"frame_id", "pts_sec", "source"}
_EPS = 1e-6


class ContractError(ValueError):
    pass


def _num(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ContractError(f"{where}: ожидалось конечное число, получено {v!r}")
    return float(v)


def _keys(obj: Any, allowed: set[str], where: str) -> None:
    if not isinstance(obj, dict):
        raise ContractError(f"{where}: ожидался объект")
    extra, missing = set(obj) - allowed, allowed - set(obj)
    if extra:
        raise ContractError(f"{where}: лишние поля {sorted(extra)}")
    if missing:
        raise ContractError(f"{where}: нет полей {sorted(missing)}")


def _tri_map(m: Any, where: str) -> None:
    if not isinstance(m, dict):
        raise ContractError(f"{where}: ожидался объект")
    for k, v in m.items():
        if not isinstance(k, str) or not k:
            raise ContractError(f"{where}: пустое имя предиката")
        if v not in TRI_STATE:
            raise ContractError(f"{where}.{k}: {v!r} не из {TRI_STATE}")


def validate_observation(obs: Any) -> None:
    """Бросает ContractError при любом нарушении формата."""
    _keys(obs, _TOP, "observation")
    if not isinstance(obs["video_id"], str) or not obs["video_id"]:
        raise ContractError("video_id: непустая строка")
    dur = _num(obs["duration_sec"], "duration_sec")
    if dur <= 0:
        raise ContractError("duration_sec: должно быть > 0")
    if obs["analysis_status"] not in ANALYSIS_STATUS:
        raise ContractError(f"analysis_status: {obs['analysis_status']!r}")
    if not isinstance(obs["coverage_complete"], bool):
        raise ContractError("coverage_complete: bool")
    if obs["analysis_status"] == "complete" and not obs["coverage_complete"]:
        raise ContractError("complete без полного покрытия")
    if not isinstance(obs["candidates"], list):
        raise ContractError("candidates: список")

    seen_ids: set[str] = set()
    frame_pts: dict[str, tuple[float, str]] = {}
    for i, c in enumerate(obs["candidates"]):
        w = f"candidates[{i}]"
        _keys(c, _CAND, w)
        cid = c["candidate_id"]
        if not isinstance(cid, str) or not cid or cid in seen_ids:
            raise ContractError(f"{w}.candidate_id: пустой или повтор")
        seen_ids.add(cid)
        s, m, e = (_num(c[k], f"{w}.{k}") for k in ("start_sec", "main_sec", "end_sec"))
        if not (-_EPS <= s <= m + _EPS and m <= e + _EPS and e <= dur + _EPS):
            raise ContractError(f"{w}: нарушено 0 <= start <= main <= end <= duration ({s}, {m}, {e}, {dur})")
        if not isinstance(c["source"], str) or not c["source"]:
            raise ContractError(f"{w}.source: непустая строка")
        if not isinstance(c["evidence"], list) or not (1 <= len(c["evidence"]) <= 3):
            raise ContractError(f"{w}.evidence: от 1 до 3 кадров")
        for j, ev in enumerate(c["evidence"]):
            we = f"{w}.evidence[{j}]"
            _keys(ev, _EVID, we)
            pts = _num(ev["pts_sec"], f"{we}.pts_sec")
            if not (s - _EPS <= pts <= e + _EPS):
                raise ContractError(f"{we}: кадр {pts} вне интервала кандидата [{s}, {e}]")
            fid, src = ev["frame_id"], ev["source"]
            if not isinstance(fid, str) or not fid or not isinstance(src, str) or not src:
                raise ContractError(f"{we}: frame_id и source — непустые строки")
            prev = frame_pts.get(fid)
            if prev is not None and (abs(prev[0] - pts) > _EPS or prev[1] != src):
                raise ContractError(f"{we}: frame_id {fid} уже встречался с другим PTS/source")
            frame_pts[fid] = (pts, src)
        _tri_map(c["predicates"], f"{w}.predicates")
        _tri_map(c["visibility"], f"{w}.visibility")
