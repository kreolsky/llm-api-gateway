# LLM API Gateway

A self-hosted, OpenAI-compatible API gateway: one endpoint and one key per client in front of every model you use — cloud providers (OpenAI, DeepSeek, OpenRouter, z.ai, Kimi, any OpenAI-compatible API) and local servers (llama.cpp, local embeddings, speech-to-text) alike. Upstreams that speak the native Anthropic Messages protocol are served too, on `/v1/messages`, for clients built on the Anthropic SDK. Access, pricing and usage are controlled from one place. It is a single Docker container with a SQLite file — small enough for a home server.

**Where to start**

- *Connecting an app or agent to a running gateway* → [Using the gateway from a client](#using-the-gateway-from-a-client).
- *Running your own gateway* → [Quick Start](#quick-start), then [Configuration](#configuration).
- *Changing the code* → [`SYSTEMS.md`](SYSTEMS.md) (subsystem catalog with entry files) and [`CLAUDE.md`](CLAUDE.md) (architecture and contracts). Those files and the `INVARIANT:`/`ARCH:` markers in `src/` are authoritative; this README links to them instead of restating them.

## Use cases

**One API instead of N provider integrations.** Every app, script and agent is configured once: base URL `http://<host>:8777/v1`, one key. Which backend serves a model is the gateway's business — the client asks for `deepseek/chat` or `openrouter/claude` and never holds a provider key. Switching a model to another provider is a one-line edit in `models.yaml`, with no client change.

**Local and cloud models side by side.** A llama.cpp server on your LAN, a local embedding model, a local Whisper-style transcriber and paid cloud models all sit behind the same `/v1/chat/completions`, `/v1/embeddings` and `/v1/audio/transcriptions`. A provider without `api_key_env` is called without a key, so a local server needs no fake secret.

**Self-describing models.** `GET /v1/models` returns each model's context length, output limit, vision support, supported parameters, reasoning support and pricing (merged from the upstream and your own [`model_info.yaml`](#model_infoyaml--manual-capability-catalog)), and `GET /v1/capabilities` the reasoning-effort levels. A client or agent can configure itself from the gateway instead of from hand-written settings.

**A key per app, per purpose.** Give each app or agent its own key with its own allowed models and endpoints — e.g. a transcription bot that can only reach `/v1/audio/transcriptions`, a coding agent limited to two models. Keys, models and providers are hot-reloaded: add a user without a restart.

**Sharing with friends.** Two options, depending on how much separation you need:

- *A key on your instance* — a new entry in `user_keys.yaml` with the models you want to share. Their usage shows up as its own user on `/stat/`. Your provider accounts pay for it.
- *A second instance* — a separate copy with its own `.env` (their provider keys), its own models and its own statistics. See [Running a second instance](#running-a-second-instance) — two settings must change or both instances share one usage database.

**A small company or several offices.** A key per office, team or person; `/stat/` shows tokens and cost per key and model over time, with cost frozen at request time from the model's pricing (see [`docs/PRICING.md`](docs/PRICING.md)). Per-provider `max_concurrent` protects a rate-limited upstream from one busy office. There are **no per-key quotas or budgets** — limits are per provider, not per user.

**Upstream quirks handled once.** Reasoning parameters (`thinking`, `reasoning_effort`) are translated per provider family, 429s are retried with backoff, SSE streams are passed through byte-for-byte, and errors come back in one OpenRouter-compatible shape whichever provider failed (Anthropic-shaped on `/v1/messages`).

**OpenAI and Anthropic clients on one gateway.** A harness built on the Anthropic SDK calls `POST /v1/messages` with the same gateway key (sent as `x-api-key`) and gets the same access rules and usage stats; the request and the SSE stream pass through unchanged. `GET /v1/models` tells each client which protocol a model is served over.

## Endpoints

- `GET /health` — healthcheck; `version` is the deployed release (`dev` for an unstamped tree)
- `GET /v1/models` — list models (filtered by API key permissions) with declared capabilities (context, vision, pricing)
- `GET /v1/models/{model_id}` — model details; `?refresh=true` for a debug best-effort upstream refresh
- `GET /v1/capabilities` — flat per-model reasoning map `{supported, effort_levels}` (same access filtering as `/v1/models`)
- `POST /v1/chat/completions` — chat completion (streaming + non-streaming), for models on `openai`-type providers
- `POST /v1/messages` — native Anthropic Messages pass-through (streaming + non-streaming), for models on `anthropic`-type providers
- `POST /v1/embeddings` — text embeddings
- `POST /v1/audio/transcriptions` — speech-to-text (model optional, fallback to `DEFAULT_STT_MODEL`)
- `GET /stat/` — token usage dashboard (HTML); backed by `/stat/api/users`, `/stat/api/models`, `/stat/api/usage`, `/stat/api/summary`, `/stat/api/requests` (guarded by `X-Stat-Key` when `STAT_API_KEY` is set)

## Quick Start

1. Provider keys — one line per `api_key_env` you reference in `config/providers.yaml`:

   ```bash
   cp .env.example .env   # then fill in e.g. DEEPSEEK_API_KEY=sk-...
   ```

2. Providers and models — edit `config/providers.yaml` and `config/models.yaml` (format in [Configuration](#configuration)). Every `api_key_env` a provider names must be set in `.env`: startup refuses to run otherwise and reports every failing provider at once.

3. A client key — print one and paste it into `config/user_keys.yaml`:

   ```bash
   python scripts/generate_key.py        # prints nnp-v1-<64 hex>; never edits config
   ```

   ```yaml
   user_keys:
     my-laptop:                          # this name is the "user" on /stat/
       api_key: nnp-v1-...
       allowed_models: []                # empty = all models
   ```

4. Start and call it:

   ```bash
   docker compose up -d                  # host port 8777 -> container port 8000
   curl http://localhost:8777/health
   curl http://localhost:8777/v1/chat/completions \
     -H "Authorization: Bearer nnp-v1-..." \
     -H "Content-Type: application/json" \
     -d '{"model": "deepseek/chat", "messages": [{"role": "user", "content": "Hi"}]}'
   ```

`config/` and `src/` are bind-mounted: config edits apply within `CONFIG_RELOAD_INTERVAL` seconds; code edits need `docker compose restart api`; only `requirements.txt` or `Dockerfile` changes need `docker compose up -d --build`.

### Running a second instance

Copy the repository to another directory, give it its own `.env` and `config/`, then change two things in its `docker-compose.yml`:

- the host port (`"8777:8000"` → e.g. `"8778:8000"`);
- `volumes.usage_data.name` (`nnp-ai-router_usage_data` → e.g. `friends-router_usage_data`). The name is pinned on purpose (see the `INVARIANT(data-loss)` comment in [`docker-compose.yml`](docker-compose.yml)), so a copy that keeps it writes into **the same** usage database as the first instance.

## Using the gateway from a client

Any OpenAI SDK or OpenAI-compatible harness works — point it at the gateway:

| Setting | Value |
|---|---|
| Base URL | `http://<host>:8777/v1` |
| API key | the `nnp-v1-…` key from `user_keys.yaml`, sent as `Authorization: Bearer <key>` or `x-api-key: <key>` (Bearer wins when both are sent) |
| Model id | a key of `config/models.yaml`, e.g. `deepseek/chat` — list them with `GET /v1/models` |

- **Discovery.** `GET /v1/models` lists only the models this key may use, with capabilities (`context_length`, `top_provider.max_completion_tokens`, `supports_vision`, `supported_parameters`, `reasoning`, `pricing`) and `api` — the protocol the model is served over: `openai-completions` (call `/v1/chat/completions`) or `anthropic-messages` (call `/v1/messages`). A model outside the key's list is a 403 `model_not_allowed` whether or not it exists, so a restricted key cannot probe which models are configured; an unknown model for an unrestricted key is a 404 `model_not_found`.
- **Streaming.** `"stream": true` returns the provider's SSE stream unchanged. An upstream 401/429 before the first chunk keeps its real HTTP status instead of arriving as a 200 with an error frame. OpenAI-style `reasoning` deltas are duplicated into llama.cpp-style `reasoning_content`.
- **Reasoning.** Send `reasoning_effort` (or `thinking: {"type": "enabled" | "disabled"}`) as you would to OpenAI or DeepSeek; the gateway re-shapes it for the provider that actually serves the model. `GET /v1/capabilities` returns `{model_id: {supported, effort_levels}}`; a value outside `effort_levels` is a 400 `invalid_parameter_value`, never silently clamped.
- **Errors.** Every error has one shape: `{"error": {"code", "message", "metadata"}}` (OpenRouter-compatible); provider errors add `metadata.provider_name` and `metadata.raw`. The exception is `/v1/messages`, below.
- **Anthropic SDK clients.** Point the SDK's base URL at `http://<host>:8777` (the SDK appends `/v1/messages`) and pass the gateway key as its API key. The gateway checks the key and model access, rewrites only `model` to the provider's name, and forwards the body and SSE stream byte-for-byte; `anthropic-version` / `anthropic-beta` are forwarded (default `anthropic-version: 2023-06-01`). Gateway-made errors are Anthropic-shaped — `{"type": "error", "error": {"type", "message"}}`, and a mid-stream failure is an `event: error` frame. Calling a model over the other protocol's endpoint is a 400 `wrong_api`. The model's `reasoning_effort` policy is advertised in `/v1/capabilities` but not enforced on this path.
- **Other endpoints.** `POST /v1/embeddings`, and `POST /v1/audio/transcriptions` (multipart; `model` is optional and falls back to `DEFAULT_STT_MODEL`).

## How It Works

1. **Request arrives** at a FastAPI endpoint. Middleware generates a `request_id` and logs the lifecycle.
2. **Auth** extracts the Bearer token (falling back to `x-api-key`), looks it up in `user_keys.yaml` (constant-time comparison), sets `project_name` on request state.
3. **Service layer** validates the model: checks `allowed_models` *before* checking existence (prevents information leakage about configured models), then that the model's provider type matches the endpoint (`openai` → `/v1/chat/completions`, `anthropic` → `/v1/messages`; otherwise 400 `wrong_api`). Resolves `provider_name` and `provider_model_name` from `models.yaml`.
4. **Provider layer** gets a cached provider instance (keyed by provider name) and sends the request via its own `httpx.AsyncClient` connection pool, optionally through a per-provider proxy, with that provider type's credentials (`Authorization: Bearer` for `openai`, `x-api-key` + `anthropic-version` for `anthropic`).
5. **Streaming**: `_stream_request` yields raw bytes → `process_stream` forwards them byte-for-byte (no re-framing or re-encoding).
6. **Errors**: Provider HTTP errors are extracted from the JSON response body, logged, and returned in OpenRouter-compatible format `{"error": {"code", "message", "metadata": {"provider_name", "raw"}}}`.
7. **Rate limits**: 429 responses trigger exponential backoff retry (`min(base * 2^attempt, max)`), configurable via env vars.
8. **Concurrency limits**: Providers can have a `max_concurrent` cap (e.g. `3`). Requests beyond the cap wait for a slot (up to `QUEUE_WAIT_TIMEOUT` seconds); on timeout they receive an immediate 503.

## Configuration

Three YAML files in `config/`, hot-reloaded without restart (polled every `CONFIG_RELOAD_INTERVAL` seconds):

### providers.yaml — provider connections

```yaml
providers:
  deepseek:
    type: openai                          # openai | anthropic
    base_url: https://api.deepseek.com/v1
    api_key_env: DEEPSEEK_API_KEY         # env var name for the API key
    max_concurrent: 3                     # optional: cap concurrent requests
    proxy: socks5://host:1080             # optional: route this provider's traffic through a proxy
    headers:                              # extra headers (optional)
      HTTP-Referer: "https://myapp.com"
    identity: passthrough                 # optional: forward client headers minus the denylist (see below)
    reasoning_dialect: deepseek           # optional: openai (default) | deepseek | openrouter
```

A local server with no auth simply omits `api_key_env`:

```yaml
  local-llama:
    type: openai
    base_url: http://192.168.1.50:8080/v1   # llama.cpp llama-server, local embeddings, a Whisper server…
```

`reasoning_dialect` re-shapes the client's `thinking` / `reasoning_effort` fields for the upstream family (`src/services/reasoning_dialect.py`); an unknown value fails startup validation.

`type` is the protocol the provider speaks, and therefore the endpoint its models are served on: `openai` — any OpenAI-compatible API, on `/v1/chat/completions`, `/v1/embeddings`, `/v1/audio/transcriptions`; `anthropic` — the native Anthropic Messages API (`POST {base_url}/v1/messages`), on `/v1/messages`. A provider speaks exactly one protocol, so a vendor offering both is two entries sharing one key:

```yaml
  deepseek-anthropic:
    type: anthropic
    base_url: https://api.deepseek.com/anthropic
    api_key_env: DEEPSEEK_API_KEY
```

The optional `proxy` key routes all of that provider's traffic through a SOCKS5/HTTP proxy (requires the `httpx[socks]` extra); unset = direct connection.

#### identity — forwarding the client's headers

`identity: passthrough` forwards every client header upstream verbatim (the client's own spelling), minus a denylist of client credentials, transport/hop-by-hop headers and reverse-proxy topology headers. Only the router's own key (from `api_key_env`) reaches the upstream. The denylist is **fail-open** — an unknown client header is forwarded — so it must be re-audited whenever a new harness or proxy is put in front of the gateway. The exact list and the reasons for each group: [header denylist and why it is fail-open — `src/core/header_policy.py`](src/core/header_policy.py). Unset `identity` = plain gateway headers, nothing forwarded; removing the key is applied by hot reload.

##### Header precedence and the static fallback

Per-request headers are merged over the provider's static `headers:`, replacing case-insensitive duplicates, and `Authorization` is never overwritten. Under `passthrough`, a real client header wins over `headers:` — which makes `headers:` a **fallback**. If the client sends no `User-Agent` (a plain `curl`, a script, an SDK that omits it), nothing overrides the static entry, so upstream sees whatever you configured; with no `headers:` entry at all, upstream sees the bare httpx default. Set a static `User-Agent` when the upstream should never see a `python-httpx/*`:

```yaml
glm:
  identity: passthrough
  headers:
    User-Agent: "my-harness/1.0"   # used only when the client sends none
```

Static `headers:` is validated at startup and on every reload: entries must be `name: value` strings, and `Authorization`, `x-api-key` or transport/hop-by-hop names (e.g. `Content-Length`, `Host`) are rejected — the key comes from `api_key_env` and the router owns the transport values. Other credential names (e.g. Azure's `api-key`) stay allowed for backends keyed on them.

### models.yaml — model registry

```yaml
models:
  deepseek/chat:
    provider: deepseek                    # references providers.yaml key
    provider_model_name: deepseek-chat    # name sent to provider API
    options:                              # deep-merged into request body
      temperature: 0.7
    reasoning_effort:                     # optional: gate the effort values sent upstream
      allowed: [low, high, max]           # a client value outside this list -> 400
      param: reasoning_effort             # or reasoning.effort
  embeddings/local:
    provider: embedding
    provider_model_name: text-embedding
    is_hidden: true                       # hidden from /v1/models listing
```

`options` are deep-merged into the request body, so you can set default parameters per model. `is_hidden` keeps the model usable but invisible in the model list. `reasoning_effort` is soft-validated on load: a malformed block is logged and dropped (`src/services/reasoning_effort.py`). For a model on an `anthropic`-type provider it is advertisement only (`/v1/models`, `/v1/capabilities`): nothing is gated or injected on `/v1/messages`, and `param` is ignored.

### model_info.yaml — manual capability catalog

Declares per-model capabilities (context, output limit, vision/modalities, supported parameters, pricing). This is the **manual override** layer; the **auto-cache** (`src/core/model_capabilities/`, persisted to `data/model_cache.json`) fills the rest from upstream `/models` responses. `model_info.yaml` always wins over the auto-cache. All fields optional.

```yaml
model_info:
  gemini/mini:
    description: "Google Gemini 2.0 Flash — fast, multimodal, tool calling"
    context_length: 1048576               # int -> top_provider.context_length too
    max_completion_tokens: 8192           # int -> top_provider.max_completion_tokens too
    architecture:
      input_modalities: [text, image]     # presence of "image" == vision (derives supports_vision)
      output_modalities: [text]
      tokenizer: Gemini                   # optional
    supported_parameters: [tools, tool_choice, max_tokens, temperature, top_p, stream]
    reasoning:                            # overrides what the auto-cache derived from upstream
      supported: false
      default_enabled: false
    pricing:                              # numbers per-token; serialized to strings (no exponent)
      prompt: 0.0000001
      completion: 0.0000004
      input_cache_read: 0.000000025
      input_cache_write: 0.000000125      # optional; cache writes fall back to the prompt rate
      image: 0.0000258                    # optional
    compat:                               # optional, manual-only: pi-ai compat flags, rendered as stored;
      forceAdaptiveThinking: true         # only for real Claude models on anthropic-type providers
```

The auto-cache does not fill capabilities for `anthropic`-type providers, so their models need their `context_length`, `max_completion_tokens`, `architecture.input_modalities` and `pricing` here — otherwise `/v1/models` shows them bare and their usage rows are uncosted. The checklist is in the header of [`config/model_info.yaml`](config/model_info.yaml).

Derived fields (`architecture.modality`, `supports_vision`, `top_provider`, string `pricing`, `per_request_limits`) are computed by `render_capabilities()` on serialization — they are **not** stored here.

### user_keys.yaml — access control

```yaml
user_keys:
  admin:
    api_key: nnp-v1-...
    allowed_models: []                    # empty = all models
    allowed_endpoints: []                 # empty = all endpoints
  restricted:
    api_key: nnp-v1-...
    allowed_models:
      - deepseek/chat
    allowed_endpoints:
      - /v1/chat/completions
```

Two levels of restriction: `allowed_endpoints` controls which API paths are accessible, `allowed_models` controls which models can be used. Empty list = unrestricted. A key with a non-empty `allowed_endpoints` needs `/v1/messages` listed to reach Anthropic-protocol models.

### .env — provider API keys and tuning

```
DEEPSEEK_API_KEY=sk-...
OPENROUTER_API_KEY=sk-...
OPENAI_API_KEY=sk-...
```

## Project Structure

```
src/
├── api/
│   ├── main.py            # FastAPI app, lifespan, routes
│   ├── middleware.py      # Request ID injection, request/response logging, usage row per request
│   ├── stat_page.py       # /stat/ dashboard HTML
│   └── stat_routes.py     # /stat/api/* JSON endpoints
├── core/
│   ├── auth.py            # Bearer token extraction, constant-time comparison, endpoint access
│   ├── config_manager.py  # YAML loading, hot-reload task, frozen env Settings
│   ├── config_schema.py   # Config parsing and validation (a ConfigError refuses startup / vetoes a reload)
│   ├── context.py         # Typed RequestContext (request_id, project_name)
│   ├── header_policy.py   # Passthrough-identity header denylist
│   ├── model_capabilities/  # Capabilities: normalize/merge/render + auto-cache + background refresh
│   ├── usage_db/          # SQLite usage writer and dashboard queries
│   ├── error_handling/    # ErrorType enum, error factory, provider-error logging, Anthropic-shape converter
│   └── logging/           # Logger with request/response/debug_data methods
├── providers/
│   ├── __init__.py        # Provider registry: {type: class} factory, two-phase cache rebuild, lookup by name
│   ├── base.py            # Provider (the `openai` type): retry, requests, streaming, header merging, error extraction
│   ├── anthropic.py       # AnthropicProvider (the `anthropic` type): x-api-key credentials, /v1/messages
│   └── pool.py            # Per-provider httpx client, concurrency gate, graceful drain
├── services/
│   ├── base.py            # Model validation (access → existence → provider → protocol), dispatch funnel
│   ├── chat_service/
│   │   ├── chat_service.py    # Orchestrator: validation → provider → StreamingResponse/JSONResponse
│   │   └── stream_processor.py # Byte-for-byte SSE pass-through (OpenAI and Anthropic wires), usage capture, reasoning remap
│   ├── messages_service.py    # /v1/messages: Anthropic Messages pass-through
│   ├── capabilities_refresh.py # Background refresh of the capabilities auto-cache
│   ├── reasoning_effort.py    # Per-model effort policy (gate + default)
│   ├── reasoning_dialect.py   # Per-provider thinking/effort translation
│   ├── embedding_service.py
│   ├── model_service.py   # Model listing/retrieval from merged capabilities (no network)
│   └── transcription_service.py  # Default model fallback
└── utils/
    ├── deep_merge.py      # Recursive dict merge (for model options)
    ├── unicode.py         # Decode \uXXXX in provider error messages
    ├── mask.py            # Header masking for debug logs
    ├── client_address.py  # Client address for logs and stats
    └── generate_key.py    # nnp-v1-<64 hex chars> key generation (CLI: scripts/generate_key.py)
```

## Key Features

- **Streaming**: SSE pass-through — provider chunks are forwarded to the client byte-for-byte, with usage peeked from the stream and OpenAI-style `reasoning` duplicated into llama.cpp-style `reasoning_content`. On `/v1/messages` the Anthropic stream is forwarded the same way, with usage read from `message_start` / `message_delta`.
- **Anthropic Messages**: `anthropic`-type providers are served natively on `/v1/messages` — same keys, access rules and usage stats as the OpenAI endpoints; a model is reachable only over its provider's protocol (`wrong_api` otherwise).
- **Rate limit retry**: Exponential backoff on 429 — `min(base_delay * 2^attempt, max_delay)`. Detects rate limits via `status_code` and `original_exception.response.status_code`.
- **Concurrency limiting**: Per-provider `max_concurrent` (in `providers.yaml`) gates outbound requests via an `asyncio.Semaphore`. Queued requests fail fast with 503 after `QUEUE_WAIT_TIMEOUT` (default 30s).
- **Hot-reload**: A background task polls config file mtimes and, on change, validates the new config and rebuilds the provider cache. An invalid config (unknown `reasoning_dialect`, bad static `headers:`, a missing file) vetoes the reload and the old config stays live — see [config loading and reload — `src/core/config_manager.py`](src/core/config_manager.py).
- **Access control**: Per-key model and endpoint restrictions. Access check runs *before* existence check to prevent leaking information about configured models.
- **Per-provider proxy**: Optional `proxy` key (e.g. `socks5://host:1080`) routes a provider's outbound traffic through a SOCKS5/HTTP proxy. Requires the `httpx[socks]` extra.
- **Token usage dashboard**: `/stat/` serves an HTML dashboard of tokens and cost per user (the `user_keys.yaml` entry name) and model over time, one row per request (errors included; cache reads and cache writes counted separately), stored in SQLite at `USAGE_DB_PATH` on a named Docker volume. Never open the database file from a macOS host — read it via `/stat/api/*` or `docker compose exec`; why: [usage volume data-loss invariant — `docker-compose.yml`](docker-compose.yml).
- **Provider caching**: Provider instances are cached by provider name (the `providers.yaml` key); each owns its own `httpx.AsyncClient` pool. A config reload rebuilds the cache and closes the old pools only after their in-flight requests drain, so a reload never cuts a live stream — see [two-phase provider rebuild and drain — `src/providers/__init__.py`](src/providers/__init__.py).
- **Model capabilities**: `/v1/models` declares the full capability set per model (context, output limit, vision/modalities, supported parameters, pricing). Two layers — manual `model_info.yaml` (always wins) and an upstream auto-cache (`data/model_cache.json`, refreshed by a background task). The hot path never touches the network; `?refresh=true` is a debug-only best-effort refresh.
- **Error format**: All errors returned as `{"error": {"code", "message", "metadata"}}` — OpenRouter-compatible. Provider errors include `metadata.provider_name` and `metadata.raw`. On `/v1/messages` gateway-made errors are Anthropic-shaped `{"type": "error", "error": {"type", "message"}}`.

## Tests

```bash
# One-time: a project venv matching requirements.txt. Do not rely on a shared
# interpreter — a stale httpx there fails tests over features Docker has.
python -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
```

```bash
.venv/bin/python -m pytest tests/unit/ -v   # full unit suite (fast, no service needed)
.venv/bin/python -m pytest tests/api/ -v    # integration tests (service on :8777)
```

See [tests/README.md](tests/README.md) for details on what each test file covers.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HTTPX_MAX_CONNECTIONS` | 100 | Connection pool size |
| `HTTPX_MAX_KEEPALIVE_CONNECTIONS` | 20 | Keep-alive connections |
| `HTTPX_CONNECT_TIMEOUT` | 60.0 | Connection timeout (s) |
| `HTTPX_READ_TIMEOUT` | 60.0 | Non-streaming read timeout (s) |
| `HTTPX_POOL_TIMEOUT` | 5.0 | Pool wait timeout (s) |
| `STREAM_READ_TIMEOUT` | 300 | Streaming read timeout (s) |
| `OPENAI_CONNECT_TIMEOUT` | 60.0 | OpenAI connection timeout (s) |
| `OPENAI_TRANSCRIPTION_TIMEOUT` | 3600.0 | Transcription request timeout (s) |
| `OPENAI_EMBEDDINGS_READ_TIMEOUT` | 30.0 | Embeddings read timeout (s) |
| `QUEUE_WAIT_TIMEOUT` | 30.0 | Concurrency slot wait timeout (s) |
| `PROVIDER_MAX_RETRIES` | 3 | 429 retry attempts |
| `PROVIDER_RETRY_BASE_DELAY` | 1.0 | Retry base delay (s) |
| `PROVIDER_RETRY_MAX_DELAY` | 30.0 | Retry max delay (s) |
| `CONFIG_RELOAD_INTERVAL` | 5 | Config poll interval (s) |
| `MODEL_CACHE_ENABLED` | true | Populate the model capabilities auto-cache |
| `MODEL_CACHE_REFRESH_INTERVAL` | 3600 | Capabilities cache refresh interval (s) |
| `MODEL_CACHE_PATH` | data/model_cache.json | Persisted capabilities cache file |
| `DEBUG` | false | Enable debug-level JSON logging |
| `API_WORKERS` | 1 | Uvicorn worker processes. Keep at 1: the capabilities cache and usage writer are process-local |
| `LOG_LEVEL` | INFO | Logging level. `DEBUG` writes full request/response bodies to `logs/debug.log` |
| `LOG_MAX_BYTES` | 52428800 | Size at which a log file rotates (50 MB) |
| `LOG_BACKUP_COUNT` | 3 | Rotated log files kept per log |
| `DEFAULT_STT_MODEL` | stt/dummy | Fallback transcription model |
| `STAT_API_KEY` | *(unset)* | When set, `/stat/api/*` requires a matching `X-Stat-Key` header |
| `USAGE_DB_PATH` | data/usage.db | SQLite path for the token usage dashboard — container-only, see [usage volume data-loss invariant](docker-compose.yml) |

## Releases

Versions are annotated git tags `vMAJOR.MINOR.PATCH` on `main`, each with a release note in [`docs/release/`](docs/release/). Major = a change a client of the router has to adapt to (response shape, error envelope, access rule, removed endpoint or config key); minor = a new feature; patch = fixes only. The deployed version is visible at `GET /health`.

## License

[MIT](LICENSE) © 2025-2026 Serge Zaigraeff
