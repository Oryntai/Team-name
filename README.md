# ForteBank Datathon 26 — проверка кассовой дисциплины по видео

**Сервис:** https://kassa-ai.oqcrm.kz — веб-интерфейс, ключ для него не нужен.

**API для жюри** (раздел 10 ТЗ). Ключ передаётся отдельно, в заголовке `X-API-Key`; без верного ключа ответ — 401.

| Метод | Путь | Что делает |
|---|---|---|
| `POST` | `/api/v1/analyze` | загрузка видео `.mov`, поле `file` → `{"job_id": "…"}` |
| `GET` | `/api/v1/jobs/{job_id}` | `processing` · `done` (вердикт и нарушения) · `failed` |

```bash
curl -X POST https://kassa-ai.oqcrm.kz/api/v1/analyze -H "X-API-Key: <ключ>" -F "file=@video.mov"
curl https://kassa-ai.oqcrm.kz/api/v1/jobs/<job_id> -H "X-API-Key: <ключ>"
```

## Что делает сервис

Сервис принимает запись камеры кассы и ищет на ней видимые признаки нарушения кассовой дисциплины. Вердикт — `violation` или `no_violation` на всё видео. По каждой находке сервис даёт:

- тип;
- таймкод;
- 1–3 настоящих кадра файла (JPEG);
- описание того, что видно;
- пункт нормативного акта РК со ссылкой на adilet.zan.kz.

Ответ — строго по схеме ТЗ, не дольше 5 минут с момента загрузки.

**Как устроено.** Детектор людей (RF-DETR) находит моменты: кассир ушёл с места, вошёл, сменился клиент, клиент ушёл. В эти моменты локальная модель Qwen3-VL отвечает на короткие вопросы по увеличенной вырезке нужной зоны: «что на экране», «что лежит на столе», «открыт ли сейф». Нарушение засчитывается только при подтверждении на двух кадрах. Тип, пункт и ссылку даёт каталог правил, а не модель.

Всё работает локально, без облачных AI-сервисов; лица не распознаются, звук не используется. Результат — подсказка для проверяющего, а не юридическое заключение.

**Документы для сдачи:**
- [docs/SUBMISSION.md](docs/SUBMISSION.md) — по разделу 8 ТЗ: инструкция, архитектура, модели и лицензии, типы нарушений, ограничения и известные ошибки;
- [docs/sources.md](docs/sources.md) — нормативные источники с дословными цитатами и принцип сопоставления;
- [rules/README.md](rules/README.md) — каталог правил.

## Требования ТЗ → как сделано → чем проверить

| Требование ТЗ | Как сделано | Чем проверить |
|---|---|---|
| Веб-интерфейс: загрузка, статус, результат без ручного запуска кода (разд. 4–5) | статический интерфейс `frontend/dist` на том же адресе. Экран анализа с таймером; результат — эпизоды, кадры, плеер, карточка «Основание» с цитатой нормы | открыть https://kassa-ai.oqcrm.kz и загрузить `.mov`; `node frontend/tests/api.test.mjs` |
| API в два шага, ключ в `X-API-Key`, статусы `processing / done / failed` (разд. 10) | `backend/app.py`, `backend/service.py`; ключ сравнивается за постоянное время | `curl` выше; `backend/tests/test_api.py` |
| Результат строго по схеме: `verdict`, `violations[]` с `type`, `timecode_sec`, `description`, 1–3 JPEG base64 | pydantic-модели с `extra="forbid"` (`backend/contracts.py`); вердикт всегда согласован со списком | `contracts/job-response.schema.json`; `backend/tests/test_end_to_end.py` |
| Не дольше 5 минут на видео | дедлайн 300 с от загрузки, вместе с очередью. Ролики залпа делят окно поровну (`fair_share_end`); процесс модели снимается на дедлайне | `backend/tests/test_fair_share.py`; `data_tools/burst_api.py` (залп через публичный API) |
| Конкретный таймкод и кадр (FAQ 17) | таймкод — PTS реального кадра; каждый кадр сверяется со списком кадров ffprobe и извлекается именно он | `backend/tests/test_video_io.py`, `test_end_to_end.py` |
| Тип, описание, правило и источник (разд. 4, 6) | `rules/cash_checks.json`: у каждого правила один видимый предикат, документ, пункт, ссылка; в интерфейсе — дословная цитата (`frontend/dist/norms.js`) | `rules/test_checks_catalog.py`; [docs/sources.md](docs/sources.md) |
| Не утверждать то, что не видно на видео (разд. 7) | предикаты — только видимые действия; учёт, проводки и сверку сервис не утверждает; «по смыслу» помечено в ответе | [docs/sources.md](docs/sources.md), раздел 5 |
| Ложных срабатываний как можно меньше (FAQ 16) | два подтверждения на разных кадрах; вопросы с вариантами вместо «да/нет»; модель недоступна — `failed`, а не «чисто» | таблица чисел ниже; `vision/tests/test_anchored_*.py` |
| Локальность: облачные AI-сервисы запрещены (разд. 7) | Qwen3-VL-8B за `llama-server` на машине команды; адрес сервера модели — только localhost или частная сеть, прокси отключены | `vision/src/forte_vision/anchored/ask.py` (`_endpoint`) |
| Переносимость: после загрузки весов интернет не нужен | веса Qwen (GGUF) и `rf-detr-nano.pth` кладутся заранее; в контуре банка всё ставится на один GPU-сервер | [docs/SUBMISSION.md](docs/SUBMISSION.md), разделы 2.3 и 3.4 |
| Лица не распознаются, звук не используется | роли по рамкам и времени, без внешности; ffmpeg отбрасывает аудиопотоки (`-an`) | `vision/src/forte_vision/anchored/people.py`; `backend/video_io.py` |
| Тестовые видео не хранить и не использовать для обучения | видео удаляется, как только задание получает `done` или `failed`; модели предобученные, своих весов нет | `backend/worker.py` (`purge_uploads`); `python infra/list_jobs.py` |
| Open source с коммерческой лицензией (FAQ 24) | Qwen3-VL — Apache-2.0, RF-DETR nano — Apache-2.0, llama.cpp — MIT | [docs/SUBMISSION.md](docs/SUBMISSION.md), раздел 4 |
| README, архитектура, источники, типы, ограничения (разд. 8) | этот файл, [docs/SUBMISSION.md](docs/SUBMISSION.md), [docs/sources.md](docs/sources.md) | — |

## Ожидаемые числа

**Набор.** Открытый набор — 40 уникальных роликов: 21 с нарушением, 19 чистых; X-47 = X-4 и X-48 = X-5, учтены один раз. Отложенная часть — 11 роликов, на которых ничего не подбирали. Эталон вердикта и секунды — от организаторов, тип — разметка команды.

**Счёт.** Независимый скрипт `data_tools/score_videos.py`, который код решения не импортирует. Прогон — тем же кодом, что сервис.

| Метрика | Константа «всё — нарушение» | v3, лучшая измеренная | Прод, финальный замер |
|---|---|---|---|
| Верный вердикт, 40 роликов | 21/40 | **27/40** | `<ФИНАЛ>` |
| Верный вердикт, отложенная часть | 6/11 | 8/11 | `<ФИНАЛ>` |
| Ложные тревоги на чистых | 19/19 | **0/19** | `<ФИНАЛ>` |
| Поймано нарушений | 21/21 | 8/21 | `<ФИНАЛ>` |
| Верный тип | 0/21 | 8/21 | `<ФИНАЛ>` |
| Таймкод в пределах ±3 с | 0/21 | 8/21 | `<ФИНАЛ>` |
| Время на ролик, медиана / максимум | — | 50 / 112 с | `<ФИНАЛ>` |
| Залп из 5 роликов, последний ответ | — | 293,7 с (до деления окна) | `<ФИНАЛ>` |

- **Сильная сторона v3** — ни одной ложной тревоги, и каждая находка с верным типом и таймкодом.
- **Слабая** — пропущено 13 нарушений из 21. Ответ `no_violation` нарушения не исключает.
- **Чего схема не ловит:**
  - нет проверок для приёма денег без документа клиента, пропуска счётной машины, ключей в замке сейфа;
  - на другом ракурсе качество не измерено.

  Подробно — [docs/SUBMISSION.md](docs/SUBMISSION.md), раздел 7.

## Запуск

Нужны Python 3.12, `ffmpeg` и `ffprobe`. Модель отдаёт OpenAI-совместимый сервер `llama-server` (llama.cpp) с Qwen3-VL-8B-Instruct (GGUF + mmproj) на GPU-машине. Команда установки ниже повторяет состав окружения прода; установку с нуля именно этой командой отдельно не проверяли.

```bash
python -m pip install -c requirements.lock -e '.[dev]' -e ./decision -e './vision[detector]'
export FORTE_ENVIRONMENT=production FORTE_API_KEY="$(cat /path/to/.api_key)"
export FORTE_ANALYZER_FACTORY=forte_vision.anchored.analyzer:create_analyzer
export FORTE_RULES_FACTORY=forte_decision.backend_adapter:create_rules
export FORTE_VLM_URL=http://127.0.0.1:8080 FORTE_VLM_KEY_FILE=/path/to/.vlm_key FORTE_DETECTOR=nano
export PYTHONPATH="$PWD:$PWD/decision:$PWD/vision/src"
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000   # терминал 1
python -m backend.worker                                         # терминал 2
curl -i http://127.0.0.1:8000/health/ready                        # 200 — готов
```

- На Mac всё поднимает `infra/mac/serve.sh` (см. `infra/mac/README.md`, образец — `infra/mac/mac.env.example`). Он перезапускает упавшие процессы и ставит новые коммиты `main` только после зелёных тестов.
- Ключ жюри создаёт `python infra/generate_jury_key.py`; порядок выдачи — [docs/jury-api-access.md](docs/jury-api-access.md).
- Подробная инструкция — [docs/SUBMISSION.md](docs/SUBMISSION.md), раздел 2.
- `Dockerfile` и `compose.yaml` собирают только API и worker, без анализатора и сервера модели. Для анализа нужен нативный запуск.

Основные настройки backend (префикс `FORTE_`, файл `.env` или окружение процесса):

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `FORTE_ENVIRONMENT` | `development` | в `production` сервис не стартует с заглушками и коротким ключом |
| `FORTE_API_KEY` | — | ключ жюри для `/api/v1` |
| `FORTE_DATA_DIR` | `./data` | SQLite и загруженные видео (удаляются после обработки) |
| `FORTE_DEADLINE_SECONDS` | `300` | срок задания от загрузки, не больше 300 |
| `FORTE_MAX_UPLOAD_MB` | `50` | лимит файла |
| `FORTE_MAX_PENDING_JOBS` | `5` | заданий в работе и очереди; дальше — 503 с `Retry-After` |
| `FORTE_PUBLIC_API_KEYS` | `true` | общий ключ хакатона на странице «API» сайта; `false` — только `FORTE_API_KEY` |
| `FORTE_ANALYZER_FACTORY` / `FORTE_RULES_FACTORY` | заглушки | анализатор «по якорям» и движок правил (см. выше) |
| `FORTE_RULE_CATALOGS` | `["rules/cash_checks.json"]` | каталоги правил |
| `FORTE_VLM_URL` / `FORTE_VLM_KEY_FILE` / `FORTE_VLM_MODEL` / `FORTE_VLM_TIMEOUT` | `http://127.0.0.1:18080` / — / `local` / `45` | сервер модели |
| `FORTE_DETECTOR` | `nano` | размер RF-DETR (только Apache-2.0: nano, small, medium) |

## Проверки

```bash
env -i HOME="$HOME" PATH="$PATH" PYTHONPATH="$PWD:$PWD/decision:$PWD/vision/src" \
  .venv/bin/python -m pytest -q -p no:cacheprovider backend/tests decision/tests data_tools rules/test_catalog.py vision/tests
python -m pytest -q rules/test_checks_catalog.py
node frontend/tests/api.test.mjs
```

Первая команда — тот же набор, что запускает автодеплой. Тесты используют синтетические видео и тестовые адаптеры; качество модели они не измеряют. Качество считается прогоном `vision/tools/anchored_eval.py` и подсчётом `data_tools/score_videos.py`.

## Как устроен репозиторий

| Каталог | Что внутри |
|---|---|
| `backend/` | API, очередь SQLite, worker, работа с видео (ffprobe/ffmpeg), контракт ответа |
| `vision/src/forte_vision/anchored/` | анализатор «по якорям»: `detect.py`, `people.py`, `zones.py`, `desk.py`, `ask.py`, `decide.py`, `analyzer.py` |
| `decision/` | движок правил: предикат → правило каталога → нарушение |
| `rules/` | каталог `cash_checks.json` и его тесты |
| `frontend/dist/` | веб-интерфейс без сборщика; `norms.js` — цитаты норм |
| `data_tools/` | независимый подсчёт качества, залп через API, проверки формата |
| `infra/` | запуск на Mac, генерация ключа, просмотр заданий без видео |
| `contracts/` | JSON-схемы и OpenAPI |

Правила работы с репозиторием — [AGENTS.md](AGENTS.md).
