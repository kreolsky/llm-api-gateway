# LLM API Gateway — Project Bible

OpenAI-compatible API gateway for multiple LLM providers. Routes requests to OpenAI, DeepSeek, OpenRouter, and any OpenAI-compatible API through a unified endpoint.

**Development rules, coding standards, and lessons: see `RULES.md`**

## Tech Stack

* **Language**: Python 3.12
* **Framework**: FastAPI 0.111.0, Uvicorn 0.29.0
* **HTTP Client**: httpx (async, connection pooling)
* **Config**: YAML (hot-reloaded every 5s)
* **Testing**: pytest, pytest-asyncio
* **Infrastructure**: Docker (python:3.12-slim), Docker Compose
* **Entry point**: `src.api.main:app`

## Core Systems

**Request Flow**: Middleware (request ID, logging) → Auth (Bearer, falling back to `x-api-key`) → Service (model access validation + protocol gate) → Provider (HTTP to backend) → Response (JSON or SSE stream).

**Provider Abstraction**: Two provider types (`openai`, `anthropic` in `PROVIDER_TYPES`; the factory is a `{type: class}` dict in `src/providers/__init__.py`) — a provider speaks exactly one protocol, so a model's protocol is its provider's type: `openai` -> `/v1/chat/completions`, `anthropic` -> `POST {base_url}/v1/messages` (native pass-through via `MessagesService`, `/v1/messages` — the funnel `_resolve_target(api=...)` refuses a model over the wrong protocol with 400 `wrong_api`). A dual-protocol vendor gets two provider entries. One `Provider` base class (`src/providers/base.py`) owns pool/retry/timeout/header merging; `AnthropicProvider(Provider)` (`src/providers/anthropic.py`) swaps credentials (`x-api-key` + `anthropic-version: 2023-06-01`, never `Authorization`) and the endpoint paths; client `anthropic-version`/`anthropic-beta` are forwarded on the messages path as protocol (not identity) headers. Each provider instance owns its own `httpx.AsyncClient` (per-backend connection pool). Instances are cached by **provider name** (the dict key in `providers.yaml`) and built ONLY by the two-phase rebuild of the `ProviderRegistry` object (built in `lifespan`, stored on `app.state.provider_registry`, injected into every service and the capabilities refresh): `prepare` pre-swap, `publish` post-swap — `get` is a pure lookup and a miss is a 404, never a lazy build (see the `INVARIANT:` in `src/providers/__init__.py`). The cache is rebuilt on config reload, and the previous instances' pools are closed via `aclose()` **after their in-flight requests drain** (bounded by `stream_read_timeout`), so a reload never aborts a live SSE stream. `Provider` handles retry with exponential backoff on 429s. Optional per-provider `proxy` key (e.g. `socks5://host:port`) routes all of that provider's traffic through a SOCKS5 proxy; `None` (unset) = direct connection. Requires the `httpx[socks]` extra (`socksio`).

**Upstream Identity** (`identity: passthrough` in `providers.yaml`): opt-in per provider (unset = plain gateway headers, no forwarding) — with `passthrough`, every client header is forwarded upstream verbatim (the client's own spelling; casing is part of a harness fingerprint) minus a denylist. The denylist lives in `src/core/header_policy.py` and drops three groups: client credentials (`authorization`, `proxy-authorization`, `cookie`, `x-api-key`, `api-key`, `x-goog-api-key` — only the router's own key from `api_key_env` goes up), transport/hop-by-hop (`host`, `content-length`, `content-type`, `content-encoding`, `connection`, `transfer-encoding`, `te`, `upgrade`, `keep-alive`, `expect`, `accept-encoding` — the router re-serializes the body, so client framing values are stale), and reverse-proxy topology (`x-forwarded-*` prefix, `x-real-ip`, `forwarded`, `true-client-ip`, `cf-connecting-ip`, `cdn-loop`). The denylist is fail-open by design: an unknown client header IS forwarded, so it must be re-audited whenever a new harness or proxy is onboarded. Static `headers:` is validated at config parse (`core/config_schema.py`, a reload veto) (string names/values only; no `Authorization`; no transport names) and acts as a fallback — a real client header replaces its case-insensitive static twin in `_merge_request_headers`. Stream and non-stream paths merge headers identically; `Authorization` is never overwritten.

**Process Model**: **one uvicorn worker** (`API_WORKERS`, default `1`). The gateway is I/O-bound, and two subsystems are process-local singletons that extra workers would silently fork into independent copies: the `CapabilitiesCache` (N copies, N refresh loops, N× upstream polling) and the SQLite usage writer. `PRAGMA busy_timeout=5000` is set regardless so a concurrent writer waits instead of losing the event.

**Startup Validation**: Eager fail-fast — on startup every configured provider is instantiated (validating `base_url` + env API key). Any failure (collected, all reported) refuses to start.

**Request Context**: A typed `RequestContext` (`request_id`, `project_name`) frozen dataclass is stored on `request.state.request_context` by middleware and rebuilt by auth (to attach `project_name`). `core.context.request_context(request)` is the single accessor (services call it directly), and it returns the dataclass — no raw `request.state.request_id`/`project_name` keys, and no dict-shaped copy of the context.

**Configuration (YAML)**: Three files in `config/` — `providers.yaml` (connections, incl. the optional per-provider `reasoning_dialect: openai | deepseek | openrouter`, default `openai`; a typo is a `ConfigError` from `parse_config` in `core/config_schema.py` — refuses startup, vetoes a reload with the old config kept), `models.yaml` (model registry, incl. the optional per-model `reasoning_effort` policy), `user_keys.yaml` (API key access control). Hot-reloaded via background task.

**Access Control**: Per-key model restrictions. Access check runs BEFORE model existence check to prevent information leakage. Keys use `nnp-v1-<hex>` format. The client key is read from the `Authorization: Bearer` header, falling back to `x-api-key` when no Bearer is present (the Anthropic SDK sends its key there); Bearer wins when both are sent.

**Streaming**: SSE byte-for-byte pass-through. `process_stream` (`src/services/chat_service/stream_processor.py`) forwards provider chunks unchanged, peeking (never re-framing) per `StreamWire`: the default `OPENAI_STREAM` duplicates OpenAI-style `reasoning` into llama.cpp-style `reasoning_content` (best-effort, on complete `data:` lines); `ANTHROPIC_STREAM` (the `/v1/messages` path) captures `message_start`/`message_delta` usage (merged, later wins) into `set_anthropic_usage` and frames mid-stream router failures as `event: error` (no `[DONE]`). `open_provider_stream` primes the first chunk BEFORE the response starts, so an upstream 401/429 keeps its real HTTP status instead of arriving as a 200 with an error frame.

**Error Format**: OpenRouter-compatible JSON with `error.code`, `error.message`, `error.metadata` — except on `/v1/messages`, where router-made errors render Anthropic-shaped `{"type":"error","error":{"type","message"}}` (converter: `src/core/error_handling/anthropic.py`) AFTER the stats enrichment reads the OpenRouter envelope.

**Model Capabilities**: `/v1/models` and `/v1/models/{id}` declare the full capability set per model (context, output limit, vision/modalities, supported parameters, pricing) plus `api` — the protocol the model is served over (`openai-completions` | `anthropic-messages`), derived from its provider's type so a harness (Lore's pi-ai) knows which endpoint to call. Two layers, both in the same normalized stored shape:

1. **Manual layer** — `config/model_info.yaml` (operator-curated). Soft-validated on load: unknown keys / orphan entries (no matching model in `models.yaml`) → `logger.warning`, non-fatal.
2. **Auto-cache** — `src/core/model_capabilities/` `CapabilitiesCache`, persisted to `data/model_cache.json`. A background task (`capabilities_refresh_loop`) calls one `list_models()` per provider, normalizes the raw upstream response (`normalize_provider_model`, shape-dispatched: OpenRouter / llama-server / generic-empty), and stores it. Stale-if-error: on upstream failure the old entry is retained.

Priority: `model_info.yaml` **always wins** over the auto-cache (deep-merge where lists are *replaced*, not concatenated — `merge_capabilities`, distinct from `utils.deep_merge`). The single serializer `render_capabilities()` produces the response shape (derives `supports_vision`, `architecture.modality`, `top_provider`, string `pricing` without exponent, full `PRICING_KEYS` set incl. `input_cache_write`). The **hot path never touches the network** — `?refresh=true` is a debug-only best-effort refresh. `reasoning.supported` is auto-derived by the normalizer when upstream `supported_parameters` carries `reasoning` or `reasoning_effort`; when the upstream is itself a router of this kind, its top-level `reasoning` object is kept (stored keys, type-checked) — so `effort_levels`/`default_effort`/`default_enabled` survive the chain; real OpenRouter ships the same object with its own vocabulary (`default_effort: minimal`). A cached `default_effort` outside the advertised `effort_levels` is dropped with a one-time warning (see `INVARIANT` in `src/services/model_service.py`); `model_info.yaml` still has the last word. `compat` (manual-only pi-ai flags) is rendered as stored on non-anthropic provider types and, on anthropic-type models, filtered to dsh's offered set — see the `INVARIANT` on `ANTHROPIC_COMPAT_KEYS` in `src/core/config_schema.py`; the anthropic-type checklist for `model_info.yaml` lives in that file's header. `scripts/dump_model_info.py` prints a `model_info.yaml` draft from the cache (stdout only, never writes config).

**Reasoning Effort Policy** (per model, `models.yaml`): an optional `reasoning_effort:` block — `allowed: [low, high, max]`, `param: reasoning_effort | reasoning.effort` (`default:` used only where two model ids split one upstream model by effort, e.g. `gemini/flash` / `gemini/pro`; otherwise the client-side default is the harness's business, not the router's) — gates what effort values may leave for the upstream and injects the default when the client sends none (client value read from both dialects, left where the client put it; a value outside `allowed` is a 400 `invalid_parameter_value`, never a clamp). Enforced in the one funnel (`BaseService._prepare_dispatch` → `src/services/reasoning_effort.py`); `options.reasoning_effort` still wins over the injected default. Soft-validated at load (`parse_effort_policy` in `core/config_schema.py` — warn + drop the block on a typo; hot-reload safe). `/v1/models` derives `reasoning.effort_levels`/`default_effort` from the same key (beats the auto-cache, loses to `model_info.yaml`). For anthropic-type models the block is **advertisement only** — `/v1/messages` never gates or injects (that funnel is OpenAI-wire surgery); `param` is ignored, `allowed` is what `/v1/capabilities` advertises.

**Reasoning Dialect Translation** (per provider, `providers.yaml` `reasoning_dialect`, default `openai`): the same funnel, right after the effort policy, re-shapes the harness's `thinking: {type}` / `reasoning_effort` pair per upstream family (`src/services/reasoning_dialect.py`) — `deepseek` keeps both fields verbatim (the native wire shape, also spoken by z.ai); `openai` drops `thinking` (the dialect has no such field) and passes `reasoning_effort` through; `openrouter` re-nests the effort to `reasoning.effort` (vocabulary mapped `max → high`), maps a disabled thinking to `reasoning: {enabled: false}` (title turns), and never gates — the 400-gate stays the effort policy's. Keyed on the provider entry, not the model id, so the same model translates differently when repo and deploy route it to different backends. `GET /v1/capabilities` exposes the flat per-model `{supported, effort_levels}` map from the same derivation `/v1/models` renders (auth parity: `is_hidden` + `allowed_models` + per-key endpoint grants).

## Configuration (env vars)

All env-backed settings are read via `ConfigManager` **once, at construction** into a frozen `Settings` dataclass (`config_manager.settings`, passed to providers as a required argument; each field's env var is its upper-cased name) — env vars cannot change without a restart, so per-access reads only cost the hot path a parse and moved malformed-value failures into requests. No direct `os.getenv` in providers/services except initial logger/debug setup.

**Ports**: the container listens on `8000` (uvicorn `--port 8000`); `docker-compose` maps host `8777` → container `8000`. Clients and the test suite talk to `8777`.

* **HTTPX pools**: `HTTPX_MAX_CONNECTIONS`, `HTTPX_MAX_KEEPALIVE_CONNECTIONS`, `HTTPX_CONNECT_TIMEOUT`, `HTTPX_READ_TIMEOUT`, `HTTPX_POOL_TIMEOUT` — applied **per provider pool** (each provider instance owns its own `httpx.AsyncClient`). Total connections ≈ providers × limit.
* **Streaming**: `STREAM_READ_TIMEOUT` (default 300) — read timeout for SSE streams.
* **Transcription**: `DEFAULT_STT_MODEL` (default `stt/dummy`) — fallback model when none is requested.
* **Stats**: `STAT_API_KEY` (default unset) — when set, every `/stat/api/*` JSON endpoint requires an `X-Stat-Key` **header** with the same value (never a query param: the logging middleware logs full URLs). `GET /stat/` and `/stat/static` stay open — the page is what prompts for the key. Usage stats: one row per request (errors included, cost frozen at write time from merged capabilities pricing, cache writes priced at `input_cache_write` falling back to the prompt rate) in the SQLite file `USAGE_DB_PATH` (default `data/usage.db`), opened by the container process only — see `INVARIANT(data-loss)` on the `usage_data` volume in `docker-compose.yml`.
* **Per-provider concurrency**: optional `max_concurrent` key per provider in `providers.yaml` gates outbound requests via an `asyncio.Semaphore`; queued requests fail fast with 503 after `QUEUE_WAIT_TIMEOUT` (default `30.0`). Takes effect only after a config reload rebuilds the cache (semaphore is per-instance).
* **Model capabilities cache**: `MODEL_CACHE_ENABLED` (default `true`), `MODEL_CACHE_REFRESH_INTERVAL` (default `3600`s), `MODEL_CACHE_PATH` (default `data/model_cache.json`). `data/` is mounted in `docker-compose.yml`, so the cache survives restarts.

## Releases

A release is an annotated tag `vX.Y.Z` + release note (`docs/release/`) + fast-forward `dev` → `main` — the `release` skill (pre-flight: `.claude/scripts/release-audit.py`). Releasing does NOT deploy: `deploy-server` stamps `git describe` into the server's `src/VERSION`, which `/health` reports.

## Architecture Discovery

Start at **`SYSTEMS.md`** — the generated subsystem catalog (name · description · entry file
· aliases, incl. Russian task nouns). Find the system, then read its entry file. Regenerate
with `python3 .claude/scripts/systems-index.py --write`; `pre-commit-gates.sh` blocks on drift.

Markers (`.claude/rules/documentation.md`): `# ARCH:` cross-cutting decisions, `# INVARIANT:`
constraints (each with a `Why:`), `# WHY:` local non-obvious choices, `# DEBT:` the deferred-work
ledger, `# SYSTEM:` subsystem entry points. Every file has a module-level docstring.

* `grep -rn "SYSTEM:" src/` — subsystem entry points
* `grep -rn "ARCH:" src/` — all architectural decisions
* `grep -rn "INVARIANT:" src/` — constraints that must be preserved
* `grep -rn "DEBT:" src/` — the tech-debt ledger

## Design Principles

* **Thin gateway**: No business logic beyond routing. Translate formats, forward requests, return responses.
* **Provider isolation**: Each provider type handles its own format translation. Base class provides retry, HTTP, streaming.
* **Config-driven**: All provider connections, model mappings, and access control defined in YAML. No hardcoded endpoints.
* **Fail fast**: Crash on missing configs. Validate access before existence. Return clear errors.
