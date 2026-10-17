# Подключение ML наблюдений и решений

`vision/` принадлежит участнику, который реализует локальные модели. `decision/`, `rules/` и `data_tools/` содержат независимый слой решений и оценки. Backend отвечает за API, очередь, декодирование и реальные JPEG. Frontend получает официальный результат API. Не запускайте код из `decision/examples` как модель анализа: примеры полностью синтетические.

Установка из корня репозитория, Python 3.10 или новее:

```sh
python -m venv .venv
.venv/bin/python -m pip install -e './decision[dev]'
.venv/bin/python -m pytest -q decision/tests data_tools/test_inventory.py rules/test_catalog.py
.venv/bin/python -m forte_decision.evaluation evaluate decision/examples/predictions.json decision/examples/annotations.json
```

На Windows используйте соответствующий путь `.venv/Scripts/python`. Основной пакет не требует GPU и сторонних runtime-библиотек. Оборудование для локального observer выбирается по его фактическому профилированию.

## Задача локального observer

Реализуйте адаптер `analyze_video(video_path, job_id, deadline_at, profile)`, который формирует следующий объект. Это целевой интерфейс для интеграции, а не уже существующая функция этого модуля.

```json
{
  "video_id": "example-local-video",
  "duration_sec": 10.0,
  "analysis_status": "complete",
  "coverage_complete": true,
  "candidates": [
    {
      "candidate_id": "window-1-event-1",
      "start_sec": 2.0,
      "end_sec": 5.0,
      "main_sec": 3.0,
      "source": "local-observer-version",
      "evidence": [{"frame_id": "frame-3", "pts_sec": 3.0, "source": "video-io-version"}],
      "predicates": {},
      "visibility": {}
    }
  ]
}
```

Пример не описывает реальные данные и не подтверждает нарушение. Предикаты из [каталога](../rules/README.md) заполняются значениями `observed`, `not_observed`, `unclear`. Неизвестное или пропущенное условие не становится отрицательным фактом. `coverage_complete` подтверждается фактическим выполнением выбранного профиля по всему видео. Ошибка или досрочный выход дают `partial`/`failed`, а не готовый отрицательный ответ.

Observer объединяет кандидатов от обзора всего видео, движения и детектора; затем проверяет выбранные интервалы плотнее. Частоты, разрешение и модели сравниваются на development-данных. Детектор не должен быть обязательным фильтром для всех событий. Скорость и качество этого шага данным репозиторием пока не измерены.

Схема строгая: неизвестные поля отклоняются. `source` — строка, не массив. Диагностику производительности сохраняйте отдельно. Времена относятся к исходному видео; кадры внутри одного видео имеют стабильные ID и PTS. Нельзя подставлять `frame_index / 30` вместо фактического времени. Подробные ограничения — в [decision/README.md](../decision/README.md).

Согласовано с разработчиком `vision/`: несколько источников записываются в `source` отсортированными именами через `+`, например `motion+vlm`. Числовые оценки предикатов хранятся отдельно от Observation, в диагностике по `candidate_id` и имени предиката; они нужны для настройки порогов, но не заменяют три состояния в правилах. Там же сохраняются `model_id`, revision весов, `profile` и `stage_timings`. Адаптеры MLX и CUDA должны выдавать одинаковую схему Observation. Наличие этого соглашения не означает, что сами адаптеры уже реализованы.

## Подключение к backend

```python
import json
from pathlib import Path
from forte_decision import Observation, Rule, decide
from forte_decision.adapter import VerifiedJPEG, to_public_payload

rules = [Rule.from_json(json.dumps(item)) for item in
         json.loads(Path('rules/cash_receipt.json').read_text())]
observation = Observation.from_json(observation_json)
decision = decide(observation, rules, max_anchor_delta_sec=1.0)

if decision.status != 'complete':
    # Persist failed in the job store; never publish a clean negative.
    raise RuntimeError('analysis did not complete')

# The backend must extract, decode and verify each original frame first.
# verified_frames maps frame_id to VerifiedJPEG(video_id, frame_id, pts_sec, bytes).
payload = to_public_payload(decision, verified_frames, job_id=job_id)
```

`observation_json`, `verified_frames` и `job_id` в примере поступают из backend/observer. Serializer не извлекает изображение и не доказывает корректность его содержимого; `VerifiedJPEG` означает, что это уже сделал backend. Ошибки валидации также должны завершать задание с ошибкой.

Выход serializer точно содержит `job_id`, `status: done`, `verdict` и `violations`. Каждое нарушение содержит `type`, `timecode_sec`, `description`, `evidence_frames`, а каждый кадр — `timestamp_sec`, `image_base64`. Источник нормы включён в описание. Внутренние `events`, `diagnostics`, `rule_id` и статусы `complete`/`incomplete` не подменяют внешнюю схему.

## Проверка качества

Evaluator принимает **внутренние** записи с `video_id`, `status: complete`, `verdict`, `events`. Официальный API использует `status: done` и `violations`: независимый HTTP-клиент должен явно преобразовать их, сохраняя соответствие `job_id` исходному `video_id`. Не подавайте внешний JSON напрямую в evaluator. Статусы `failed` остаются ошибками, не отрицательными предсказаниями.

У эталона бинарный `verdict` и полнота событий — разные сведения. `exhaustive` касается списка событий; одна точка времени не делает разметку исчерпывающей. Отчёт отдельно показывает бинарные ошибки, полноту выполнения, точность событий на полной разметке и recall известных событий. Соблюдайте фиксированный split по съёмкам и дублям; `data_tools` оставляет его неназначенным до ручной группировки.

## Что ещё необходимо для работающего сервиса

Реальный observer, backend/video_io, локально проверенные аннотации, расширение начального каталога, реальные замеры качества/времени и UI реализуются отдельно. Эти компоненты нельзя считать готовыми по результатам синтетических unit-тестов слоя решений.
