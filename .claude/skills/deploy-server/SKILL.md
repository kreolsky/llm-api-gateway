---
name: deploy-server
description: >
  Use when the user asks to deploy, update, ship, push to server,
  "обнови сервер", "задеплой", "выкати на прод", "deploy", "push to docker host".
  Синкает src/ на удалённый Docker-хост через rsync и перезапускает ОБА роутера
  (main и ext). Различает code-only update и full rebuild по тому, что изменилось.
  Do NOT use for: локальный docker compose up, изменения конфигов на сервере
  (config/ и .env на сервере — authoritative и не трогаются).
---

# deploy-server: обновление llm-api-gateway на удалённом Docker-хосте

## Контекст

- **Хост:** `ssh docker` (alias)
- **Без git** — файлы синкаются напрямую rsync'ом.
- **Volume mount** `./src:/app/src` — изменения в `src/` подхватываются после restart.

На хосте **два роутера из одного кода** — деплой обновляет ОБА, по порядку main → ext:

| Роутер | Путь на сервере | Контейнер | Порт (host → container) |
|--------|-----------------|-----------|-------------------------|
| main | `/home/serge/docker/server-ai-api` | `server-ai-api-api-1` | 8777 → 8000 |
| ext | `/home/serge/docker/server-ai-api-ext` | `server-ai-api-ext` | 8778 → 8000 |

ext стоит цепочкой за main (кэширует capabilities main'а), поэтому main обновляется первым.
Роутер, оставленный на старой версии, — незавершённый деплой: `/health` обоих должен
показать один `$VER`.

**Важно:** `config/` и `.env` на сервере — authoritative, у каждого роутера свои.
- Никогда не rsync-ать корень проекта.
- Никогда не пушить локальные `config/*.yaml` или `.env`.
- На сервере другие провайдеры и ключи, чем локально (`dummy` там не работает).

## Алгоритм

### Шаг 1. Pre-deploy: проверить, что изменилось

```bash
git status --short
git diff --stat HEAD requirements.txt Dockerfile
```

Развилка:
- Изменён только `src/` → **code-only update** (Шаг 3a).
- Изменены `requirements.txt` или `Dockerfile` → **full rebuild** (Шаг 3b).
- Изменены `config/` → **остановиться и спросить пользователя** (это authoritative на сервере).

Версия, которую сервер покажет в `/health`:

```bash
VER=$(git describe --tags --always --dirty)   # v1.0.0 | v1.0.0-3-gabc1234 | …-dirty
```

Неизменённый релизный тег даёт чистое `vX.Y.Z`; всё остальное — честно видно как не-релиз.

### Шаг 2. Проверить состояние сервера (оба роутера)

```bash
ssh docker "docker logs server-ai-api-api-1 --tail 5; docker logs server-ai-api-ext --tail 5"
```

Если контейнер мёртв или сыпет ошибками — показать пользователю и подтвердить деплой.

Рассинхрон зависимостей — проверить для каждого роутера:

```bash
for DIR in server-ai-api server-ai-api-ext; do
  ssh docker "cat /home/serge/docker/$DIR/requirements.txt" | diff - requirements.txt && echo "$DIR ok"
done
```

Если diff есть, а локально `requirements.txt` не менялся — на сервере что-то руками поменяли, **остановиться и спросить**.

Если релиз ужесточил валидацию конфига (release note → *Upgrade actions*) — проверить
прод-конфиги ОБОИХ роутеров до рестарта: невалидный конфиг = роутер не стартует.

### Шаг 3a. Code-only update (main, затем ext)

```bash
for DIR in server-ai-api server-ai-api-ext; do
  rsync -a --delete src/ docker:/home/serge/docker/$DIR/src/
  ssh docker "echo '$VER' > /home/serge/docker/$DIR/src/VERSION"
  ssh docker "cd /home/serge/docker/$DIR && docker compose restart"
done
```

### Шаг 3b. Full rebuild (только если менялись requirements/Dockerfile)

```bash
for DIR in server-ai-api server-ai-api-ext; do
  rsync -a --delete src/ docker:/home/serge/docker/$DIR/src/
  scp requirements.txt Dockerfile docker:/home/serge/docker/$DIR/
  ssh docker "echo '$VER' > /home/serge/docker/$DIR/src/VERSION"
  ssh docker "cd /home/serge/docker/$DIR && docker compose up --build -d"
done
```

Если main после рестарта не прошёл Verify (Шаг 4) — ext НЕ трогать, показать ошибки.

### Шаг 4. Verify (оба роутера)

```bash
ssh docker "docker logs server-ai-api-api-1 --tail 20; docker logs server-ai-api-ext --tail 20"
ssh docker "curl -s localhost:8777/health; echo; curl -s localhost:8778/health"   # оба {"status":"ok","version":"$VER"}
```

`VERSION` пишется ПОСЛЕ rsync: `--delete` удаляет его на сервере, потому что локально файла нет.

Должно быть (в логах каждого):
- `Configuration manager initialized` — конфиги загрузились.
- `Application startup complete` — воркер стартанул (один: `API_WORKERS=1`, см. *Process Model* в `CLAUDE.md`).
- Нет `Traceback`, `ImportError`, `ModuleNotFoundError`.

Если ошибки — показать пользователю tail-50 и не считать деплой успешным.

## Правила

- **Никогда** не rsync-ать корень проекта или `config/`/`.env` — затрёт прод-конфиги.
- **Никогда** не запускать `docker compose down` на сервере без явной просьбы — рестарт достаточно для code-only.
- **Не использовать** `docker compose up --build` для code-only изменений — лишние 1-2 минуты на пересборку, при этом ничего не меняется (volume mount уже даёт код).
- Если pre-deploy проверка показывает, что сервер мёртв или нездоров — спросить пользователя, прежде чем накатывать.
- **Не использовать `--no-verify`** или другие обходы при `git`-операциях, если они вдруг возникнут в процессе.

## Отчёт пользователю

Краткий отчёт, по строке на роутер:

```
main (server-ai-api, :8777): synced src/, restarted, worker up clean, /health <VER>
ext  (server-ai-api-ext, :8778): synced src/, restarted, worker up clean, /health <VER>
```

При full rebuild — упомянуть, что пересобрался образ.
При ошибках — показать релевантные строки логов.
