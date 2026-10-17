"""Вопросы к VLM: скрининг окна (что и когда происходит) и проверка предикатов на кандидате.

Имена предикатов и видимости — общий словарь с decision/rules (seed-набор из rules/README.md).
Здесь только вопросы и разбор ответов; юридического вывода vision не делает.
Ответ модели всегда сводится к observed / not_observed / unclear; всё, что не разобрали, — unclear.
"""

from __future__ import annotations

import json
import math
import re

# Что именно должно быть видно, чтобы ответить "yes". Формулировки нейтральные: только наблюдаемое.
LEGACY_PREDICATES: dict[str, str] = {
    "bank_cashier_role_established": "A person is working at the cashier workstation (seated or standing behind the cash desk, operating the desk/counter/laptop).",
    "cash_receipt_operation_established": "A client hands cash to the cashier (banknotes move from the client side to the cashier).",
    "cash_counting_observed": "The cashier counts banknotes, by hand or with a banknote counting machine.",
    "client_absent_during_counting": "While the cashier counts banknotes, no client is present at the client side of the desk.",
    "cashier_left_workstation": "The cashier walks away from the workstation and leaves the visible cash area.",
    "departure_before_client_service_completed": "The cashier leaves the workstation while a client is still being served (client still present, operation not finished).",
    # Рабочее место и помещение (rules/cash_workplace.json). Формулировки — по разметке открытой выборки:
    # в чистых роликах кассир тоже уходит, но сначала убирает ценности и блокирует ноутбук.
    "cashier_absent_from_room": "The cashier walks out of the room through the door, or the cashier is no longer in the room in the later frames.",
    "workstation_left_unsecured": "While the cashier is away from the desk, banknotes, a cash bundle, a cash bag or documents are still lying on the desk, or the laptop screen still shows a working application (table/form) instead of a lock screen or closed lid.",
    "cash_room_door_left_open": "The door of the cash room stays open (door leaf turned inward/outward, corridor visible through the doorway) for several seconds while the cashier works, not only while someone is walking through it.",
    "outsider_in_cash_room": "Besides the cashier and the one client sitting at the desk, a THIRD person enters the cash room or stands in the doorway or next to the desk. A single client walking in and sitting down to be served is normal and is NOT this.",
    "safe_left_open": "The safe door is open while the cashier is seated at the desk or doing other work, i.e. not at the moment of putting items into or taking items out of the safe.",
    "previous_client_cash_on_desk": "Banknotes that belong to an earlier client remain on the open desk while the next client is served, or banknotes of two different clients are put together into one stack.",
    "client_left_before_count_done": "The client stands up or walks away while the cashier is still counting the client's banknotes or has not yet counted them.",
    "collection_bag_accepted_without_checks": "A cash collector hands a cash bag to the cashier and leaves right away, with no document shown or signed and without the cashier inspecting the bag and its seal.",
    "personal_items_on_desk": "A handbag, backpack, jacket or other clothing is placed on the cashier's desk.",
    "food_or_drink_at_desk": "A cup, glass, bottle, can or food is on the cashier's desk, or the cashier eats or drinks at the desk.",
}

# Предметные признаки, которые принимаются только если детектор (COCO) видит такой объект на тех же кадрах.
DETECTOR_BACKED: dict[str, set[str]] = {
    "food_or_drink_at_desk": {"cup", "bottle"},
    "personal_items_on_desk": {"handbag", "backpack", "suitcase"},
}

# Признаки, которые принимаются только при минимальном числе людей на тех же кадрах (детектор).
# Обычный визит: кассир + клиент = 2 человека; посторонний означает третьего.
MIN_PERSONS: dict[str, int] = {
    "outsider_in_cash_room": 3,
}

LEGACY_VISIBILITY: dict[str, str] = {
    "counting_interval_fully_visible": "The whole counting action is visible in these frames (hands and banknotes are not hidden).",
    "all_possible_client_positions_visible": "Every place where a client could stand or sit at this desk is inside the frame.",
    "workstation_and_exit_visible": "Both the cashier workstation and the way out of the cash area are visible.",
    "service_sequence_around_departure_visible": "The frames show what happens with the client right before and after the cashier leaves.",
}

# Предикаты проверок чек-листа (timeline.CHECKS): один предикат = одно правило rules/cash_checks.json.
PREDICATES: dict[str, str] = {
    "cash_accepted_without_id_document": "Cash or a cash bag is accepted although no ID document was shown.",
    "client_cash_mixed_or_left_for_next_client": "A previous client's cash is left out for the next client or mixed with it.",
    "left_with_computer_unlocked": "The cashier leaves while the computer screen is not locked.",
    "left_valuables_on_desk": "The cashier leaves while cash or documents stay on the desk.",
    "safe_left_open": "The safe is left open.",
    "cash_room_door_left_open": "The cash room door is left open.",
    "counted_after_client_left": "Cash is counted after the client has left.",
    "personal_items_on_desk": "Personal items are placed on the cash desk.",
    "food_or_drink_at_desk": "Food or drinks at the cash desk.",
    "banknotes_not_checked_on_detector": "Banknotes are counted by hand without a detector or counting machine.",
    "bag_cash_to_safe_without_registration": "Cash from a collection bag goes into the safe without counting or computer entry.",
    "collection_bag_left_uncounted": "A collection bag is left on the desk uncounted.",
    "served_out_of_queue": "A client is served out of queue order.",
}
VISIBILITY: dict[str, str] = {}

# Скрининг: открытый список категорий + "other" для типов, которых нет в обучении.
SCREEN_CATEGORIES: dict[str, str] = {
    "cash_counting": "someone counts banknotes (by hand or machine)",
    "cash_handover": "banknotes or a cash bag pass between people",
    "cashier_leaves": "the cashier walks away from the workstation",
    "person_enters_cash_area": "a person enters the area behind or at the cash desk",
    "safe_activity": "a safe is opened, closed or items are put in/taken out",
    "door_open": "the door of the cash room is open or opened",
    "food_or_drink": "someone eats or drinks at the workstation",
    "personal_items": "a bag, phone or clothing is placed on the workstation",
    "valuables_left_on_desk": "cash, a cash bag or documents stay on the desk while the cashier is away",
    "client_leaves": "a client stands up or walks away from the desk",
    "other_suspicious": "any other action with cash, documents or valuables that looks irregular",
}

_TRI = {"yes": "observed", "no": "not_observed", "unclear": "unclear"}


def screen_question() -> str:
    cats = "\n".join(f'- "{k}": {v}' for k, v in SCREEN_CATEGORIES.items())
    return (
        "Report only what is clearly visible. Categories:\n"
        f"{cats}\n"
        "Return ONLY a JSON list, no prose. Each item: "
        '{"category": <one of the keys>, "start": <seconds>, "end": <seconds>}. '
        "Use the given timestamps. Return [] if nothing applies."
    )


def predicate_question(names: list[str], table: dict[str, str]) -> str:
    lines = "\n".join(f'- "{n}": {table[n]}' for n in names)
    return (
        'For each statement answer "yes" only if it is clearly visible in these frames, '
        '"no" if it is clearly not happening, "unclear" if it cannot be judged from what is visible.\n'
        f"{lines}\n"
        'Return ONLY a JSON object mapping each key to "yes", "no" or "unclear".'
    )


def _reject_constant(value):
    raise ValueError("non-finite JSON constant")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _json_blob(text: str, opener: str, closer: str):
    """Первый сбалансированный JSON-фрагмент: модели любят оборачивать ответ в ``` и прозу."""
    start = text.find(opener)
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(
                            text[start : i + 1],
                            parse_constant=_reject_constant,
                            object_pairs_hook=_unique_object,
                        )
                    except ValueError:
                        return None
    return None


def parse_tri_state(text: str, names: list[str]) -> dict[str, str]:
    obj = _json_blob(text, "{", "}")
    out = {n: "unclear" for n in names}
    if isinstance(obj, dict):
        for n in names:
            v = str(obj.get(n, "")).strip().lower()
            out[n] = _TRI.get(v, "unclear")
    return out


def parse_screen_checked(
    text: str, t_min: float, t_max: float
) -> tuple[list[tuple[str, float, float]], bool]:
    """Return candidates plus parse validity; valid [] differs from failed parsing.

    Preserve usable entries for diagnostics, but any malformed entry makes the
    whole window unresolved. Clipping never repairs non-finite or reversed time.
    """
    # A list buried in an error object/prose is not a valid negative response.
    # Accept a complete JSON document, optionally wrapped in one JSON code fence.
    document = text.strip()
    if document.startswith("```"):
        fence = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```", document, re.DOTALL | re.IGNORECASE
        )
        if fence is None:
            return [], False
        document = fence.group(1)
    try:
        obj = json.loads(
            document, parse_constant=_reject_constant, object_pairs_hook=_unique_object
        )
    except (ValueError, TypeError):
        return [], False
    out: list[tuple[str, float, float]] = []
    if not isinstance(obj, list):
        return out, False
    valid = True
    for item in obj:
        if not isinstance(item, dict):
            valid = False
            continue
        category = item.get("category")
        if not isinstance(category, str) or category.strip() not in SCREEN_CATEGORIES:
            valid = False
            continue
        try:
            start = float(_num_str(item.get("start")))
            end = float(_num_str(item.get("end")))
            if not math.isfinite(start) or not math.isfinite(end) or end < start:
                raise ValueError("invalid screen interval")
        except (TypeError, ValueError, OverflowError):
            valid = False
            continue
        start, end = max(t_min, start), min(t_max, end)
        if end < start:
            valid = False
            continue
        out.append((category.strip(), start, end))
    return out, valid


def parse_screen(
    text: str, t_min: float, t_max: float
) -> list[tuple[str, float, float]]:
    """Compatibility parser; coverage decisions must use parse_screen_checked."""
    return parse_screen_checked(text, t_min, t_max)[0]


def _num_str(value) -> str:
    if isinstance(value, bool):
        raise ValueError("boolean time")
    if isinstance(value, (int, float)):
        return str(value)
    if not isinstance(value, str):
        raise ValueError("invalid time")
    match = re.fullmatch(
        r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(?:s|sec|seconds)?\s*",
        value,
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError("invalid time string")
    return match.group(1)
