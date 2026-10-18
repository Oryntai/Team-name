# Контракты сервиса

Версия контракта: 1. Backend реализует `backend/contracts.py`; JSON-схемы и примеры в этом каталоге сгенерированы из него (`python -m backend.export_contracts`). Схема ответа жюри зафиксирована ТЗ; внутренние наблюдения в ответ жюри не попадают.

В проде: анализатор `forte_vision.anchored.analyzer:create_analyzer`, правила `forte_decision.backend_adapter:create_rules` с каталогом `rules/cash_checks.json`. Цепочка одного задания — [docs/ml-integration.md](../docs/ml-integration.md), описание для жюри — [docs/SUBMISSION.md](../docs/SUBMISSION.md).

## Интерфейс и API

- Разработка: backend `http://localhost:8000`, интерфейс раздаёт сам backend из `frontend/dist` (сборщика нет). В проде интерфейс, `/ui-api` и `/api/v1` на одном origin.
- `POST /ui-api/analyze`: multipart-поле `file`, файл `.mov`. Ответ `{"job_id":"..."}`.
- `GET /ui-api/jobs/{job_id}`: интерфейс опрашивает раз в 1,5 с до `done` или `failed`.
- Запросы интерфейса к `/ui-api` идут без ключа. `FORTE_API_KEY` не класть в код браузера. Общий ключ жюри страница «API» получает отдельно (`GET /ui-api/key`, см. [jury-api-access.md](../docs/jury-api-access.md)).
- Маршруты жюри — `/api/v1/analyze` и `/api/v1/jobs/{job_id}`: те же тела ответов, обязателен `X-API-Key`.
- Задание в очереди и в работе — одинаково `processing`. У `failed` в ответе только `job_id` и статус. Чистый завершённый анализ — `verdict: no_violation` и `violations: []`.
- У каждого нарушения `type`, `timecode_sec`, `description` и 1–3 `evidence_frames`. Кадры — JPEG в base64 без префикса `data:`; префикс `data:image/jpeg;base64,` добавляется только при построении `<img>`.
- `description` — шаблон правила, строка «Что видно: …» и строка «Нормативное основание: документ — тип; пункт (ID). URL».
- Коды HTTP: 401 — нет или неверный ключ; 400 — пустой файл; 413 — больше 50 МБ; 415 — не `.mov`; 408 — загрузка не уложилась в срок; 422 — нет поля `file`; 503 + `Retry-After: 5` — очередь заполнена; 404 — нет задания. Сбой анализа — успешный ответ опроса со `status: failed`.

## Анализатор

Фабрика без аргументов `package.module:create_analyzer` возвращает объект с методом:

```python
def analyze(self, context: AnalysisContext, video: VideoIO) -> ObservationBatch:
    ...
```

Задаётся `FORTE_ANALYZER_FACTORY`. Фабрика и модели создаются только в процессе модели worker, никогда в HTTP-сервере. Все веса и зависимости скачиваются до офлайн-работы. `complete=True` — только когда нужный анализ действительно выполнен; при сбое — исключение или `complete=False`, worker тогда выдаёт `failed`. Пустой список событий не даёт права пропустить видео.

`context` содержит `job_id`, абсолютный `video_path`, абсолютный Unix-срок `deadline_at` и `profile` (по умолчанию `standard`). Срок проверяется во время inference; worker дополнительно следит за ним снаружи и на сроке убивает процесс модели.

Все времена — секунды от начала видео по настоящим PTS. `frame_pts` событий должны быть из списка кадров переданного `video` (VideoIO): worker сверяет каждый PTS со списком ffprobe. Как анализатор получает пиксели — его дело. Анализатор «по якорям» читает кадры для детектора и вырезок сам через PyAV, поворачивая видео с тегом поворота так же, как ffmpeg (`anchored/detect.py`, `upright_k`), а таймкоды и кадры событий прижимает к PTS из VideoIO (`VideoIOSource.snap`). Сигнатуры VideoIO — [video_io.md](video_io.md).

Наблюдаемое событие содержит `t_start`, `t_end`, `t_main`, `frame_pts`, состояния предикатов (`observed`, `not_observed`, `unclear`), пообъектную видимость (`clear`, `partial`, `unusable`), источники (`vlm`, `motion`, `detector`) и `note` — что видно на этом видео (до 600 символов). Анализатор «по якорям» заполняет только один предикат в состоянии `observed` и `note`, без пообъектной видимости. Лица не распознаются, звук не анализируется.

`examples.json` содержит синтетический JPEG для иллюстрации контракта, а не находку анализа.

## Правила

Фабрика без аргументов `package.module:create_rules`:

```python
def evaluate(self, observations: ObservationBatch, context: AnalysisContext) -> list[RuleEvent]:
    ...
```

Задаётся `FORTE_RULES_FACTORY`. Возвращаются только события с проверенной нормой. `RuleEvent` содержит интервал, тип, описание, `rule_id`, `rule_title`, `rule_section`, `rule_url` и 1–3 `frame_pts`. Backend проверяет их, извлекает настоящие JPEG и ставит в ответ настоящие времена кадров. Backend сам не придумывает ни правил, ни находок. Граница backend ↔ decision — [decision-integration.md](decision-integration.md).

## Заглушка для разработки

По умолчанию (`.env.example`) анализатор и правила — `backend.adapters.unavailable`: каждое задание завершается `failed`. Так интерфейс можно проверять без модели, не выдавая «не обработано» за «чисто». Синтетические адаптеры с заданным исходом есть только в тестах. С `FORTE_ENVIRONMENT=production` сервис не стартует без настоящих фабрик анализатора и правил и без `FORTE_API_KEY` длиной от 32 символов.
