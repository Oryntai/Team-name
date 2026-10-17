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

## GPU-машина с моделью

Ноутбук с RTX 4060 запускает `llama-server` только на `127.0.0.1:8080` и публикует его на сервер своим ключом
(пользователь туннеля разрешает ему ровно один проброс):

```powershell
while ($true) { ssh -N -i $env:USERPROFILE\.ssh\forte_tunnel_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -R 127.0.0.1:18080:127.0.0.1:8080 forte-tunnel@<сервер>; Start-Sleep 3 }
```

Mac получает его через `-L 18080` своего туннеля; если ноут недоступен, worker считает на локальной MLX-модели.
