# Граница backend ↔ decision

Rules-фабрика: `FORTE_RULES_FACTORY=forte_decision.backend_adapter:create_rules`.
Она принимает канонические ObservationBatch и AnalysisContext и возвращает list[RuleEvent].
JSON-схемы backend и внешний API не изменены.

В проде она подключена вместе с анализатором «по якорям» (`forte_vision.anchored.analyzer:create_analyzer`).
Цепочка, предикаты, условия `failed` и настройки — [docs/ml-integration.md](../docs/ml-integration.md),
описание для жюри — [docs/SUBMISSION.md](../docs/SUBMISSION.md).

- Vision реализует analyze(context, video). Таймкоды и кадры событий берутся из PTS переданного VideoIO; кадры для детектора и вырезок анализатор читает сам через PyAV.
- Decision оценивает правила. Standalone Observation не является вторым wire-контрактом.
- Worker проверяет реальные PTS и границы видео, извлекает JPEG, сериализует результат.
- Норма передаётся отдельными полями RuleEvent; backend добавляет её в описание один раз.
- Анализатор отдаёт только подтверждённые события: один предикат в состоянии `observed`, без пообъектной видимости.
- Анализ, который нельзя считать полным, анализатор помечает `complete=False`; worker выдаёт `failed`, а не `no_violation`.

Docker-образ ставит только backend. Для анализа нужны ещё decision, каталог правил (`FORTE_RULE_CATALOGS`), vision с RF-DETR и сервер модели, поэтому прод запускается нативно (`infra/mac/serve.sh`).
