# POST /v1/rerank — reranker pass-through to an OpenAI-compatible backend

## Decisions

- New route `POST /v1/rerank`, `name="rerank"`, guarded by `check_endpoint_access("/v1/rerank")`, placed next to `/v1/embeddings` (src/api/main.py:258); the usage row's `endpoint` is the route name (`ARCH:` at src/api/middleware.py:32), so it is `rerank`.
- New `RerankService(BaseService)` in `src/services/rerank_service.py`, a copy of the embeddings shape (src/services/embedding_service.py:14): `_prepare_dispatch(..., api="openai-completions")` so access-before-existence, the protocol gate and identity headers come for free; wired in `lifespan` beside `EmbeddingService` (src/api/main.py:98).
- New `Provider.rerank()` beside `Provider.embeddings()` (src/providers/base.py:566): `_apply_model_config` then `POST {base_url}/rerank` — llama-server serves `/v1/rerank` (Jina/Cohere shape: `query`, `documents`, `top_n` → `results[{index, relevance_score}]`, `usage`), so with `base_url: .../v1` the path is `/rerank`.
- Rerank reuses `OPENAI_EMBEDDINGS_READ_TIMEOUT` (src/core/config_manager.py:61) — same class of call (one forward pass over short inputs); no new Settings field.
- The body is passed through untouched apart from `model`; the router does not validate the rerank schema — the backend's 400 comes back via `create_error` as for embeddings.
- `usage` from the response is recorded with `stats.set_usage` exactly like embeddings.
- No config schema change: a rerank model is an ordinary `models.yaml` entry on an `openai`-type provider (repo dev config gains `rerank/dummy` on a `rerank` provider for the api tests).

## Risks

- The route-name table test (tests/unit/test_middleware.py:122) fails until `"/v1/rerank": "rerank"` is added — expected, the test exists to force it.
- A key with a non-empty `allowed_endpoints` cannot call `/v1/rerank` until it is granted — intended; prod keys are edited on the server, not here.
- An anthropic-type model sent to `/v1/rerank` gets 400 `wrong_api` from the protocol gate — correct by construction.

## Order

1. `feat(rerank): POST /v1/rerank pass-through` — `src/providers/base.py` (`rerank`), `src/services/rerank_service.py`, `src/api/main.py` (wiring + route), `config/providers.yaml` + `config/models.yaml` (dev `rerank` provider, `rerank/dummy`), tests: `tests/unit/test_base_provider.py` (path `/rerank`, embeddings read timeout, extra headers), `tests/unit/test_rerank_service.py` (400 on bad JSON, identity headers, usage recorded, refused key denied before existence), `tests/unit/test_middleware.py` (route name), README endpoint list.

## Not doing

- Rerank-specific request validation or response re-shaping (Cohere `return_documents`, Jina field aliases).
- `/v1/models` capability flags for rerank models.
- Prod `config/` on the docker host — the operator adds the provider, model and key grants there.

## Validation

- `curl -s http://localhost:8777/v1/rerank -H "Authorization: Bearer dummy" -H "Content-Type: application/json" -d '{"model":"rerank/dummy","query":"capital of France","documents":["Berlin","Paris is the capital of France"]}'` → 200, `results[0].index == 1`.
- Same request with the `limited` key → 403 (model not allowed); with the `transctiber` key → 403 `endpoint_not_allowed`.
- `/stat/api` usage row for the request carries `endpoint == "rerank"`.
