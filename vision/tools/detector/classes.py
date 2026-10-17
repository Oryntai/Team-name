"""Классы детектора предметов кассы.

Подобраны по матрице 42 роликов (схема «таймлайн + чек-лист»): 11 из 14 проверок не сработали ни разу,
и все упираются в мелкие предметы, которые VLM не видит на целом кадре. Людей, ноутбук, чашку и сумку
RF-DETR знает из COCO, но здесь они нужны в состояниях (экран заблокирован, сейф открыт), поэтому заведены
отдельно. Порядок задаёт id категорий в COCO-разметке: менять только вместе с переразметкой.

name — метка в разметке и в весах; prompt — фраза для grounding-запроса к Qwen3-VL; checks — проверки
чек-листа (rules/cash_checks.json), которым класс нужен.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetClass:
    name: str
    prompt: str
    checks: tuple[str, ...]


CLASSES: tuple[DetClass, ...] = (
    DetClass("cash", "banknotes or a stack of paper money", ("MIX", "LEAVE", "BAG", "AUTH", "NOCLIENT")),
    DetClass("document", "identity card, passport or paper document", ("DOC", "LEAVE")),
    DetClass("key", "key, including a key inserted in a lock", ("KEYS",)),
    DetClass("safe_open", "safe or metal cabinet with its door open", ("SAFE", "KEYS", "MIX")),
    DetClass("safe_closed", "safe or metal cabinet with its door closed", ("SAFE", "KEYS")),
    DetClass("door_open", "open glass door of the room", ("DOOR",)),
    DetClass("door_closed", "closed glass door of the room", ("DOOR",)),
    DetClass("screen_unlocked", "laptop screen showing open windows, tables or documents", ("LOCK",)),
    DetClass("screen_locked", "laptop screen showing a lock screen, a login screen or turned off", ("LOCK",)),
    DetClass("money_counter", "banknote counting machine", ("AUTH",)),
    DetClass("cash_bag", "cash collection bag or sealed money bag", ("BAG",)),
    DetClass("food_drink", "food, cup, mug, glass or bottle", ("FOOD",)),
    DetClass("personal_item", "handbag, backpack, jacket or other personal belongings", ("ITEMS",)),
)

NAMES = tuple(c.name for c in CLASSES)
BY_PROMPT = {c.prompt: c.name for c in CLASSES}


def select(names: str | None) -> tuple[DetClass, ...]:
    """Подмножество классов через запятую; None — все. Неизвестное имя — ошибка, а не тихий пропуск."""
    if not names:
        return CLASSES
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    unknown = sorted(set(wanted) - set(NAMES))
    if unknown:
        raise SystemExit(f"неизвестные классы: {', '.join(unknown)}; есть: {', '.join(NAMES)}")
    return tuple(c for c in CLASSES if c.name in wanted)
