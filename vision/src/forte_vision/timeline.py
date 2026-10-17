"""Таймлайн шагов операции + чек-лист проверок.

VLM проходит всё видео перекрывающимися окнами и отмечает шаги из фиксированного словаря
(клиент пришёл, документ, деньги, пересчёт, компьютер, сейф, мешок, уход, дверь...). Код ниже
проверяет порядок шагов: нарушение — это шаг не там, не в том порядке или его нет, когда он
должен был быть. Таймкод — момент, когда шаг должен был случиться (как в разметке организаторов).

Проверки собраны по парам роликов открытой выборки «нарушение / правильно»
(work/scenario-catalog-draft.md). Юридический вывод делает decision/ по rules/cash_checks.json:
каждой проверке соответствует один предикат и одно правило.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

# Шаги, которые VLM отмечает на таймлайне. Описание — что именно должно быть видно.
EVENTS: dict[str, str] = {
    "client_arrives": "a client walks up to the desk or sits down in front of the cashier",
    "client_leaves": "the client stands up and walks away from the desk or out of the room",
    "id_document_shown": "the client hands or shows an ID card, passport or a phone with a digital ID to the cashier",
    "cash_given_by_client": "the client puts banknotes on the desk or hands banknotes to the cashier",
    "bag_received": "a cash collector or another person hands a sealed cash bag to the cashier",
    "bag_opened": "the cashier cuts or opens a cash bag and takes the bundles out",
    "cash_counted_by_hand": "the cashier counts banknotes by hand",
    "counting_machine": "banknotes are put into or run through the banknote counting machine",
    "uv_detector": "the cashier holds banknotes under the small banknote detector lamp to check them",
    "computer_work": "the cashier types or works on the laptop",
    "safe_open": "the safe door is standing open",
    "safe_closed": "the cashier closes the safe door",
    "put_in_safe": "the cashier puts cash, documents or a bag into the safe",
    "cashier_stands_up": "the cashier stands up from the chair",
    "cashier_leaves": "the cashier walks out of the room through the door",
    "cashier_enters": "the cashier walks into the room through the door",
    "screen_locked": "the laptop shows a lock screen / login picture, or its lid is fully closed",
    "screen_working": "the laptop screen shows a table, spreadsheet or another working application",
    "door_closed": "someone closes the room door",
    "door_open": "the room door is standing open",
    "outsider_enters": "a person who is neither the cashier nor the client being served enters the room or stands in the doorway",
    "items_on_desk": "a coat, jacket, handbag or other bag of the cashier is placed on the cash desk",
    "food_drink": "a cup, glass, bottle, snack or food is on the desk, or the cashier eats or drinks",
    "cash_on_desk": "banknotes, cash bundles, a cash bag or documents are lying on the desk",
    "stacks_combined": "banknotes of one client are put onto or together with another pile of banknotes",
    "queue_jump": "while one client is waiting, the cashier serves another person who came later",
}

# Проверки: код -> предикат (общий словарь с rules/cash_checks.json).
CHECKS: dict[str, str] = {
    "DOC": "cash_accepted_without_id_document",
    "MIX": "client_cash_mixed_or_left_for_next_client",
    "LOCK": "left_with_computer_unlocked",
    "LEAVE": "left_valuables_on_desk",
    "SAFE": "safe_left_open",
    "DOOR": "cash_room_door_left_open",
    "NOCLIENT": "counted_after_client_left",
    "ITEMS": "personal_items_on_desk",
    "FOOD": "food_or_drink_at_desk",
    "AUTH": "banknotes_not_checked_on_detector",
    "REG": "bag_cash_to_safe_without_registration",
    "BAG": "collection_bag_left_uncounted",
    "QUE": "served_out_of_queue",
}


def timeline_question() -> str:
    codes = "\n".join(f'- "{k}": {v}' for k, v in EVENTS.items())
    return (
        "List every step you can clearly see in these frames. Steps:\n"
        f"{codes}\n"
        "Report an action once, at the frame where it happens. Report a state (door open, safe open, cash on desk, "
        "screen locked/working, items, food) at most twice per answer: the first and the last frame where it is visible.\n"
        'Return ONLY a JSON list, no prose. Each item: {"event": <one of the keys>, "t": <seconds>}. '
        "Use the given timestamps. Return [] if nothing applies."
    )


def parse_timeline(text: str, t_min: float, t_max: float) -> tuple[list[tuple[float, str]], bool, int]:
    """Шаги окна, признак разобранного ответа (JSON-список) и число отброшенных элементов.

    Отдельный неизвестный код или битое время отбрасывается и считается в диагностике; окно
    невалидно только если ответ вообще не JSON-список (тогда покрытие видео неполное).
    """
    document = text.strip()
    if document.startswith("```"):
        fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", document, re.DOTALL | re.IGNORECASE)
        if fence is None:
            return [], False, 0
        document = fence.group(1)
    try:
        obj = json.loads(document)
    except (ValueError, TypeError):
        return [], False, 0
    if not isinstance(obj, list):
        return [], False, 0
    out, dropped = [], 0
    for item in obj:
        if not isinstance(item, dict) or item.get("event") not in EVENTS or isinstance(item.get("t"), bool):
            dropped += 1
            continue
        t = item.get("t")
        if isinstance(t, str):
            t = t.strip().rstrip("s").strip()
        try:
            t = float(t)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if not math.isfinite(t):
            dropped += 1
            continue
        out.append((min(max(t, t_min), t_max), item["event"]))
    return out, True, dropped


@dataclass
class Violation:
    check: str
    main_sec: float
    start_sec: float
    end_sec: float
    reason: str
    evidence_sec: list[float] = field(default_factory=list)


class _TL:
    def __init__(self, steps: list[tuple[float, str]]):
        self.by: dict[str, list[float]] = {k: [] for k in EVENTS}
        for t, e in sorted(steps):
            self.by[e].append(t)

    def any(self, e: str, lo: float = -math.inf, hi: float = math.inf) -> bool:
        return any(lo <= t <= hi for t in self.by[e])

    def first(self, e: str, lo: float = -math.inf, hi: float = math.inf) -> float | None:
        return next((t for t in self.by[e] if lo <= t <= hi), None)

    def last(self, e: str, lo: float = -math.inf, hi: float = math.inf) -> float | None:
        ts = [t for t in self.by[e] if lo <= t <= hi]
        return ts[-1] if ts else None


def _v(check, t, reason, duration, lo=2.0, hi=3.0, extra=()):
    s, e = max(0.0, t - lo), min(duration, t + hi)
    ev = sorted({round(min(max(x, s), e), 3) for x in (t, *extra)})
    return Violation(check, round(t, 3), round(s, 3), round(e, 3), reason, ev[:3])


def run_checklist(steps: list[tuple[float, str]], duration: float) -> list[Violation]:
    tl, out = _TL(steps), []
    count = lambda lo, hi: tl.any("cash_counted_by_hand", lo, hi) or tl.any("counting_machine", lo, hi)  # noqa: E731

    # DOC: деньги или мешок приняты без предъявленного документа
    for t in tl.by["cash_given_by_client"] + tl.by["bag_received"]:
        if not tl.any("id_document_shown", t - 30.0, t + 3.0):
            out.append(_v("DOC", t, "деньги/мешок приняты, документ клиента не предъявлен", duration))
            break

    # MIX: стопки разных клиентов сложены вместе, либо новый клиент пришёл, а деньги прошлого лежат на столе
    if tl.by["stacks_combined"]:
        out.append(_v("MIX", tl.by["stacks_combined"][0], "деньги разных клиентов сложены вместе", duration))
    else:
        for t1 in tl.by["client_leaves"]:
            t2 = tl.first("client_arrives", t1 + 1.0, t1 + 60.0)
            if t2 is not None and tl.any("cash_on_desk", t1, t2 + 3.0) and not tl.any("put_in_safe", t1, t2):
                out.append(_v("MIX", t2, "следующий клиент обслуживается при неубранных деньгах предыдущего",
                              duration, extra=(t1,)))
                break

    # LOCK / LEAVE / BAG: что осталось при уходе кассира
    bag_case = bool(tl.by["bag_received"] or tl.by["bag_opened"])
    for t_leave in tl.by["cashier_leaves"]:
        t_up = tl.last("cashier_stands_up", t_leave - 15.0, t_leave) or t_leave
        if tl.any("screen_working", t_leave - 10.0, t_leave + 10.0) and not tl.any("screen_locked", t_leave - 25.0, t_leave + 5.0):
            out.append(_v("LOCK", t_up, "кассир уходит, экран компьютера не заблокирован", duration, extra=(t_leave,)))
        if tl.any("cash_on_desk", t_up - 3.0, t_leave + 15.0) and not tl.any("put_in_safe", t_leave - 20.0, t_leave):
            if bag_case and not count(min(tl.by["bag_received"] + tl.by["bag_opened"]), t_leave):
                out.append(_v("BAG", t_up, "мешок не пересчитан и оставлен на столе при уходе", duration, extra=(t_leave,)))
            else:
                out.append(_v("LEAVE", t_up, "кассир уходит, ценности остаются на столе", duration, extra=(t_leave,)))
        break

    # SAFE: сейф открыт и дальше не закрывается, открытость держится не меньше 6 с
    t_open = tl.first("safe_open")
    if t_open is not None and not tl.any("safe_closed", t_open) and tl.any("safe_open", t_open + 6.0):
        t = tl.last("put_in_safe", hi=t_open + 6.0) or t_open
        out.append(_v("SAFE", t + 2.0 if t + 2.0 <= duration else t, "сейф оставлен открытым", duration))

    # DOOR: кассир вошла, а дверь после этого стоит открытой не меньше 6 с и не закрывается
    t_in = tl.first("cashier_enters")
    if t_in is not None:
        t_door = tl.first("door_open", t_in - 1.0)
        if t_door is not None and not tl.any("door_closed", t_in - 1.0) and tl.any("door_open", t_door + 6.0):
            extra = tuple(tl.by["outsider_enters"][:1])
            out.append(_v("DOOR", max(t_in, t_door), "кассир вошла и не закрыла дверь кассы", duration, extra=extra))

    # NOCLIENT: клиент ушёл, а его деньги пересчитываются после ухода
    for t0 in tl.by["cash_given_by_client"]:
        t1 = tl.first("client_leaves", t0)
        if t1 is None:
            continue
        t2 = min([t for e in ("cash_counted_by_hand", "counting_machine") for t in tl.by[e] if t1 <= t <= t1 + 30.0],
                 default=None)
        if t2 is not None and not tl.any("client_arrives", t1, t2):
            out.append(_v("NOCLIENT", t2, "пересчёт после ухода клиента", duration, extra=(t1,)))
            break

    # ITEMS / FOOD
    # ITEMS / FOOD: состояние должно быть замечено минимум дважды (разовые упоминания — частая галлюцинация)
    if len(set(tl.by["items_on_desk"])) >= 2:
        out.append(_v("ITEMS", tl.by["items_on_desk"][0], "личные вещи на кассовом столе", duration))
    if len(set(tl.by["food_drink"])) >= 2:
        out.append(_v("FOOD", tl.by["food_drink"][0], "еда или напитки на рабочем месте", duration))

    # AUTH: ручной пересчёт без машины и без детектора за всё видео, без клиента
    if (tl.by["cash_counted_by_hand"] and not tl.by["counting_machine"] and not tl.by["uv_detector"]
            and not tl.by["client_arrives"] and not bag_case):
        out.append(_v("AUTH", tl.by["cash_counted_by_hand"][0], "купюры не проверены на детекторе", duration))

    # REG: деньги из мешка в сейф без пересчёта и без работы с компьютером между ними
    t_bag = tl.first("bag_opened") or tl.first("bag_received")
    if t_bag is not None:
        t_put = tl.first("put_in_safe", t_bag)
        t_safe = None if t_put is None else min(t for t in (t_put, tl.first("safe_open", t_bag, t_put)) if t is not None)
        if t_safe is not None and not count(t_bag, t_safe) and not tl.any("computer_work", t_bag, t_safe):
            out.append(_v("REG", t_safe, "деньги из мешка в сейф без пересчёта и учёта в системе", duration,
                          extra=(t_bag,)))

    # QUE
    if tl.by["queue_jump"]:
        out.append(_v("QUE", tl.by["queue_jump"][0], "обслуживание вне очереди", duration))
    return out


# Верификатор: точный вопрос по плотным кадрам вокруг найденного нарушения.
# (вопрос, нарушение при ответе "no" вместо "yes", окно кадров относительно main_sec)
VERIFY: dict[str, tuple[str, bool, tuple[float, float]]] = {
    "DOC": ("Before the cashier takes the money or bag, does the client hand over or show an ID card, passport "
            "or a phone with a digital ID, and does the cashier look at it?", True, (-25.0, 2.0)),
    "MIX": ("Are banknotes of a previous client still lying on the desk while the next client is served, or are "
            "two different piles of banknotes put together?", False, (-6.0, 6.0)),
    "LOCK": ("When the cashier walks away, does the laptop screen still show a working application (a table or "
             "form) rather than a lock screen or a closed lid?", False, (-4.0, 8.0)),
    "LEAVE": ("When the cashier walks away, are cash or documents still lying on the desk?", False, (-2.0, 8.0)),
    "SAFE": ("Is the safe door left standing open while the cashier is not putting anything in or taking "
             "anything out of the safe?", False, (0.0, 8.0)),
    "DOOR": ("After the cashier comes into the room, is the room door left standing open?", False, (-1.0, 8.0)),
    "NOCLIENT": ("Is the client already gone while the cashier counts the client's money?", False, (-6.0, 4.0)),
    "ITEMS": ("Is a coat, jacket, handbag or other bag lying on the cash desk?", False, (-1.0, 6.0)),
    "FOOD": ("Is there a cup, glass, bottle or food on the cash desk, or does the cashier eat or drink?", False,
             (-1.0, 6.0)),
    "AUTH": ("Does the cashier check the banknotes with a detector lamp or a counting machine?", True, (-2.0, 10.0)),
    "REG": ("Between taking the money out of the cash bag and putting it into the safe, does the cashier count "
            "the money or work on the laptop?", True, (-15.0, 3.0)),
    "BAG": ("Is a cash bag left lying on the desk without being counted when the cashier walks away?", False,
            (-2.0, 8.0)),
    "QUE": ("While one client is waiting, does the cashier serve another person who came later?", False, (-8.0, 8.0)),
}


def verify_question(check: str) -> str:
    question = VERIFY[check][0]
    return question + ' Answer with ONLY one word: "yes", "no" or "unclear".'


def parse_verdict(text: str) -> str:
    word = re.sub(r"[^a-z]", " ", text.lower()).split()
    for w in word[:5]:
        if w in ("yes", "no", "unclear"):
            return w
    return "unclear"


def confirmed(check: str, answer: str) -> bool:
    """Нарушение остаётся, если верификатор его подтверждает или не может решить (пропуск хуже ложной тревоги)."""
    if answer == "unclear":
        return True
    inverted = VERIFY[check][1]
    return answer == ("no" if inverted else "yes")
