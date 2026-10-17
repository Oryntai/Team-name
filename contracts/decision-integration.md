# Граница backend ↔ decision

Адаптер реализован: `FORTE_RULES_FACTORY=forte_decision.backend_adapter:create_rules`.
Он принимает канонические ObservationBatch и AnalysisContext и возвращает list[RuleEvent].
JSON-схемы backend и внешний API не изменены.

Актуальные команды, политика неопределённости, преобразование времён/доказательств и ограничения: [docs/ml-integration.md](../docs/ml-integration.md).

- Vision реализует analyze(context, video) и использует переданный VideoIO.
- Decision оценивает правила. Standalone Observation не является вторым wire-контрактом.
- Worker проверяет реальные PTS и границы видео, извлекает JPEG, сериализует результат.
- Норма передаётся отдельными полями RuleEvent; backend добавляет её в описание один раз.
- Пообъектное clear не доказывает полную видимость всех позиций.
- Неясные ключевые признаки без положительного события приводят к IncompleteAnalysis, не к no_violation.

Rules-фабрика не заменяет локальную модель. Пока не настроен рабочий analyzer, development unavailable по-прежнему даёт ошибку. Docker runtime должен дополнительно включать decision и каталог правил через FORTE_RULE_CATALOGS, а для реального ML — vision и локальные веса.
