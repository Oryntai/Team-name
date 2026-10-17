#!/bin/bash
# Хост на Mac: туннель до публичного сервера + API + worker + автодеплой origin/main.
#
#   infra/mac/serve.sh /path/to/mac.env
#
# Запускать из ОТДЕЛЬНОЙ копии репозитория (git worktree), в которой никто не правит код:
# автодеплой переключает её на свежий origin/main. Настройки и секреты — в mac.env (вне git),
# образец — infra/mac/mac.env.example.
#
# Автодеплой: раз в FORTE_DEPLOY_INTERVAL секунд git fetch; если origin/main ушёл вперёд —
# ждём, пока очередь заданий опустеет (задачу жюри посреди обработки не прерываем),
# переключаемся, гоняем тесты; зелёные — перезапуск API и worker, красные — откат на прошлый
# коммит, этот коммит больше не пробуем. Файл $FORTE_DATA_DIR/DEPLOY_FREEZE ставит деплой на паузу.

set -u
ENV_FILE="${1:?usage: serve.sh /path/to/mac.env}"
# shellcheck disable=SC1090
source "$ENV_FILE"

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${FORTE_PYTHON:?FORTE_PYTHON: путь к python окружения сервиса}"
LOG_DIR="${FORTE_LOG_DIR:?FORTE_LOG_DIR: каталог логов}"
DATA_DIR="${FORTE_DATA_DIR:?FORTE_DATA_DIR: каталог данных сервиса}"
PORT="${FORTE_PORT:-8000}"
INTERVAL="${FORTE_DEPLOY_INTERVAL:-60}"
IDLE_WAIT="${FORTE_DEPLOY_IDLE_WAIT:-900}"
mkdir -p "$LOG_DIR" "$DATA_DIR"
export PYTHONPATH="$REPO:$REPO/decision:$REPO/vision/src"
cd "$REPO" || exit 1

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG_DIR/deploy.log"; }

API_PID="" WORKER_PID="" TUNNEL_PID=""

start_api() {
  "$PY" -m uvicorn backend.app:app --host 127.0.0.1 --port "$PORT" >>"$LOG_DIR/api.log" 2>&1 &
  API_PID=$!
}

start_worker() {
  "$PY" -m backend.worker >>"$LOG_DIR/worker.log" 2>&1 &
  WORKER_PID=$!
}

tunnel_loop() {
  # Публикуем API на сервере (-R) и получаем сервер модели GPU-машины (-L), если он настроен.
  local fwd=(-R "127.0.0.1:${FORTE_TUNNEL_REMOTE_PORT}:127.0.0.1:${PORT}")
  [ -n "${FORTE_TUNNEL_VLM_PORT:-}" ] && fwd+=(-L "127.0.0.1:${FORTE_TUNNEL_VLM_PORT}:127.0.0.1:${FORTE_TUNNEL_VLM_PORT}")
  while true; do
    ssh -N -i "$FORTE_TUNNEL_KEY" -o IdentitiesOnly=yes -o BatchMode=yes \
        -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
        "${fwd[@]}" "${FORTE_TUNNEL_USER}@${FORTE_TUNNEL_HOST}" >>"$LOG_DIR/tunnel.log" 2>&1
    echo "$(date '+%F %T') tunnel down, reconnect in 3s" >>"$LOG_DIR/tunnel.log"
    sleep 3
  done
}

stop_services() {
  for pid in $WORKER_PID $API_PID; do kill "$pid" 2>/dev/null; done
  for pid in $WORKER_PID $API_PID; do
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null
  done
  WORKER_PID="" API_PID=""
}

cleanup() {
  log "serve: stopping"
  stop_services
  [ -n "$TUNNEL_PID" ] && { pkill -P "$TUNNEL_PID" 2>/dev/null; kill "$TUNNEL_PID" 2>/dev/null; }
  exit 0
}
trap cleanup INT TERM

pending_jobs() {
  sqlite3 "$DATA_DIR/jobs.sqlite3" "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')" 2>/dev/null || echo 0
}

run_tests() {
  "$PY" -m pytest -q -p no:cacheprovider backend/tests decision/tests data_tools rules/test_catalog.py vision/tests \
    >>"$LOG_DIR/deploy-tests.log" 2>&1
}

BAD_SHA=""
deploy_if_needed() {
  [ -e "$DATA_DIR/DEPLOY_FREEZE" ] && return
  git fetch -q origin main 2>>"$LOG_DIR/deploy.log" || { log "deploy: fetch failed"; return; }
  local head target
  head=$(git rev-parse HEAD)
  target=$(git rev-parse origin/main)
  [ "$head" = "$target" ] && return
  [ "$target" = "$BAD_SHA" ] && return
  if ! git merge-base --is-ancestor "$head" "$target"; then
    log "deploy: origin/main ${target:0:7} не является продолжением ${head:0:7} — пропуск, нужен человек"
    BAD_SHA="$target"; return
  fi
  local waited=0
  while [ "$(pending_jobs)" != "0" ] && [ "$waited" -lt "$IDLE_WAIT" ]; do sleep 5; waited=$((waited + 5)); done
  if [ "$(pending_jobs)" != "0" ]; then log "deploy: очередь не опустела за ${IDLE_WAIT}s, отложено"; return; fi
  local deps_before deps_after
  deps_before=$(git show HEAD:pyproject.toml HEAD:vision/pyproject.toml HEAD:decision/pyproject.toml 2>/dev/null | shasum)
  git checkout -q --detach "$target" || { log "deploy: checkout failed"; return; }
  deps_after=$(git show HEAD:pyproject.toml HEAD:vision/pyproject.toml HEAD:decision/pyproject.toml 2>/dev/null | shasum)
  [ "$deps_before" != "$deps_after" ] && log "deploy: изменились зависимости (pyproject) — проверь окружение вручную"
  echo "=== $(date '+%F %T') ${target:0:7}" >>"$LOG_DIR/deploy-tests.log"
  if run_tests; then
    log "deploy: ${head:0:7} -> ${target:0:7}, тесты зелёные, перезапуск"
    stop_services; start_api; start_worker
  else
    log "deploy: тесты красные на ${target:0:7}, откат на ${head:0:7}"
    git checkout -q --detach "$head"
    BAD_SHA="$target"
  fi
}

log "serve: start at $(git rev-parse --short HEAD) in $REPO"
if [ -n "${FORTE_TUNNEL_HOST:-}" ]; then tunnel_loop & TUNNEL_PID=$!; fi
start_api
start_worker
last_check=0
while true; do
  [ -n "$API_PID" ] && ! kill -0 "$API_PID" 2>/dev/null && { log "serve: API упал, перезапуск"; start_api; }
  [ -n "$WORKER_PID" ] && ! kill -0 "$WORKER_PID" 2>/dev/null && { log "serve: worker упал, перезапуск"; start_worker; }
  now=$(date +%s)
  if [ $((now - last_check)) -ge "$INTERVAL" ]; then last_check=$now; deploy_if_needed; fi
  sleep 5
done
