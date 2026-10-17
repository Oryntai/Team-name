# Хост на Mac

`serve.sh` поднимает всё, что нужно для публичного сервиса на Mac с Apple Silicon:
SSH-туннель до публичного сервера, API (`uvicorn backend.app:app`), worker и автодеплой `origin/main`.

```bash
git worktree add --detach ../deploy origin/main     # отдельная копия только для сервиса
cp infra/mac/mac.env.example ../local_data/mac.env  # заполнить пути, ключи — из файлов
nohup ../deploy/infra/mac/serve.sh ../local_data/mac.env >/dev/null 2>&1 &
```

- Автодеплой раз в минуту проверяет `origin/main`. Новый коммит ставится, только когда в очереди нет заданий,
  и только если на нём проходят тесты; иначе остаётся прошлая версия, причина — в `deploy.log`.
- `touch $FORTE_DATA_DIR/DEPLOY_FREEZE` — пауза деплоя (на время проверки жюри), `rm` — снять.
- Изменения в `pyproject.toml` деплой не устанавливает сам: пишет предупреждение, окружение обновить вручную.
- Логи: `api.log`, `worker.log`, `tunnel.log`, `deploy.log`, `deploy-tests.log` в `FORTE_LOG_DIR`.
- Остановить: `kill` процесса `serve.sh` (он гасит API, worker и туннель).
