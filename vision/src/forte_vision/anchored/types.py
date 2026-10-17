"""Общие типы анализатора «по якорям»: люди с детектора -> моменты -> вопросы VLM по вырезкам зон.

Всё в долях кадра (0..1), а не в пикселях: камера жюри может быть другой. Время — секунды PTS.
Модули: detect.py (плотный проход детектора), people.py (треки, роли, события), zones.py (зоны сцены),
ask.py (пакетные вопросы VLM по вырезкам с кешем), decide.py (события + ответы -> предикаты каталога),
analyzer.py (контракт сервиса). Типы ниже — единственный договор между модулями.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 в долях кадра


@dataclass(frozen=True)
class Det:
    box: Box
    conf: float
    name: str  # COCO-имя: person, laptop, tv, chair, dining table, handbag, cup, bottle, ...


@dataclass
class Frame:
    t: float  # PTS, с
    persons: list[Det] = field(default_factory=list)
    objects: list[Det] = field(default_factory=list)


@dataclass
class DenseScan:
    """Результат плотного прохода: детекции + маленькие серые кадры для датчика изменений."""

    duration: float
    width: int
    height: int
    fps: float
    frames: list[Frame]
    gray_t: list[float] = field(default_factory=list)  # времена серых кадров
    gray: object = None  # np.ndarray (N, h, w) uint8 или None
    hi_t: list[float] = field(default_factory=list)  # цветные кадры покрупнее (~половина кадра, ~2 к/с) — для стола
    hi: object = None  # np.ndarray (M, H, W, 3) uint8 RGB или None


@dataclass
class Track:
    tid: int
    t: list[float]
    boxes: list[Box]
    conf: list[float]

    @property
    def start(self) -> float:
        return self.t[0]

    @property
    def end(self) -> float:
        return self.t[-1]


# Виды событий людей (kind) и их смысл:
#   cashier_enters   — кассир вошла в комнату / появилась на рабочем месте после отсутствия
#   cashier_sits     — кассир заняла рабочее место (после входа)
#   cashier_stands   — кассир встала с рабочего места (без ухода из комнаты)
#   cashier_leaves   — кассир покинула рабочее место и ушла (t = момент ухода с места, t_exit в meta)
#   client_arrives   — начало визита клиента (сторона «через стол»)
#   client_leaves    — конец визита клиента
#   client_change    — один клиент ушёл, следующий пришёл (meta: gap)
#   third_person     — в комнате посторонний / второй посетитель во время визита (meta: side)
#   at_safe          — кассир у сейфа (если зона сейфа известна)
@dataclass
class Event:
    kind: str
    t: float
    t_end: float | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class People:
    cashier: Track | None  # склеенная цепочка кассира
    cashier_present: list[tuple[float, float]]  # интервалы присутствия на рабочем месте
    visits: list[tuple[float, float]]  # визиты клиентов
    visitors: list[Track]  # прочие треки в комнате (после фильтра «за стеклом»)
    desk_dir: int  # +1: стол/клиенты ниже кассира по кадру (y растёт), -1: выше, 0: неизвестно
    events: list[Event]
    roles_uncertain: bool = False
    diag: dict = field(default_factory=dict)


# Зоны сцены: safe, screen (ноутбук/монитор), desk (рабочая поверхность кассира), door (дверь в кассу),
# counter (счётная машинка / детектор), cashier_seat, client_seat. Отсутствующая зона — None.
ZONES = ("safe", "screen", "desk", "door", "counter")


@dataclass
class Zones:
    boxes: dict[str, Box | None]
    source: dict[str, str] = field(default_factory=dict)  # zone -> "vlm" | "detector" | "people" | "fallback"
    diag: dict = field(default_factory=dict)


@dataclass
class DeskBlob:
    """Новый предмет на столе по пикселям: появился в t_on и лежал (без рук на нём) до t_off."""

    box: Box
    t_on: float
    t_off: float
    area: float  # доля кадра
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CropItem:
    """Одна вырезка в пакетном вопросе: зона в момент t. key — уникальный ярлык, его повторяет ответ."""

    key: str
    t: float
    zone: str
    box: Box
    question: str  # вопрос да/нет про ЭТУ вырезку, по-английски
    zoom: float = 1.0  # доп. увеличение (например, 6 для замка сейфа)
    choices: tuple[str, ...] = ()  # непусто: вопрос «что из этого видно», ответ — подмножество choices
    open: bool = False  # с choices: модель сначала называет предмет своими словами (note), потом группу


@dataclass
class CropAnswer:
    key: str
    answer: str  # "yes" | "no" | "unclear"; для вопроса с choices — "cash+food" | "none" | "unclear"
    raw: str = ""
    note: str = ""  # open-вопрос: что это, словами модели («bundle of banknotes in plastic»)

    def has(self, want) -> bool:
        """Ответ «да» для проверки: yes, либо (для choices) в ответе есть что-то из want."""
        if not want:
            return self.answer == "yes"
        return self.answer not in ("unclear", "none", "no") and bool(set(self.answer.split("+")) & set(want))
