# SYSTEMS — subsystem catalog

GENERATED — do not edit; run `.claude/scripts/systems-index.py --write`.

Discovery: find your system (or an alias) here, then `grep -rn "SYSTEM: <name>"`
and read the entry file(s). Aliases are hand-maintained in
`.claude/systems-aliases.json`.

| System | Description | Entry file(s) | Aliases |
|--------|-------------|---------------|---------|
| anthropic-provider | Anthropic Messages wire (x-api-key + anthropic-version) | `src/providers/anthropic.py:2` | anthropic, claude, messages, /v1/messages, антропик |
| api-app | FastAPI app, lifespan, routes, eager provider validation | `src/api/main.py:2` | startup, lifespan, роуты, приложение |
| auth | bearer authentication (constant-time key comparison) and per-key model access control | `src/core/auth.py:2` | ключи, доступ, авторизация, user_keys, api key |
| config | YAML load, 5s hot reload, env-backed settings | `src/core/config_manager.py:2` | конфиг, yaml, hot reload, перезагрузка конфига |
| error-format | OpenRouter-compatible error envelope | `src/core/error_handling/error_handler.py:2` | ошибки, error |
| header-policy | denylist for client headers forwarded upstream | `src/core/header_policy.py:10` |  |
| logging | the kwargs-style Logger used everywhere | `src/core/logging/logger.py:7` | логи, логирование |
| model-capabilities | manual layer + auto-cache + render | `src/core/model_capabilities/__init__.py:25` | модели, capabilities, /v1/models, кэш моделей |
| provider | base HTTP, retry, streaming and header merging | `src/providers/base.py:2` | провайдер, upstream, бэкенд |
| provider-registry | provider instances cached by name, drained on reload | `src/providers/__init__.py:2` | кэш провайдеров, реестр |
| request-context | the typed RequestContext carried per request | `src/core/context.py:7` | контекст запроса, request_id |
| request-logging | pure-ASGI request id + Incoming/Outgoing bookends | `src/api/middleware.py:16` | middleware, мидлварь |
| service-layer | validate access, resolve provider, dispatch | `src/services/base.py:2` | сервисы, сервисный слой |
| sse-stream | passthrough streaming body | `src/services/chat_service/stream_processor.py:2` | стрим, streaming, sse, чанки |
| stat-dashboard | the /stat/ usage HTML page (JSON endpoints in stat_routes.py) | `src/api/stat_page.py:2` | статистика, дашборд, usage |
| usage-stats | SQLite per-request usage rows and cost freezing | `src/core/usage_db/__init__.py:17` | статистика, usage, токены, стоимость |
