# Declare structured-output support on the local orange models

## Decisions

- Context: Lore's extractor (`lore-knowledge-creator` `backend/pipeline/extractor/utils.py`, `call_llm_structured`) always sends `response_format: {type: json_schema, json_schema: {strict: true, ...}}`. Lore is about to gate its extractor model list on `/v1/models` `supported_parameters` containing `structured_outputs` — the OpenRouter vocabulary the auto-cache already copies for openrouter models — and to show ONLY models that declare it. A model that does not declare it disappears from Lore's extractor model picker and is refused at pipeline launch.
- Current behaviour: the production `config/model_info.yaml` on `ssh docker` (`/home/serge/docker/server-ai-api`, container `server-ai-api-api-1`, port 8777 = `api.ai.gray`) has entries for `local/orange/chat` and `local/orange/reasoner` with no `supported_parameters`, and no entry at all for `local/orange/instant`; llama-server's `/models` carries no `supported_parameters` either, so `/v1/models` serves none for these three. openrouter models (`gemini/pro`, `openai/luna`, `xiaomi/flash`) already serve `response_format` + `structured_outputs` from the auto-cache.
- Measured 2026-10-14 from Lore's prod worker through `api.ai.gray` with Lore's key, `response_format` json_schema strict, schema `{a: string}`: `local/orange/chat` 200, `local/orange/instant` 200, `local/orange/reasoner` 200 (valid JSON matching the schema each). `deepseek/flash` 400 `"This response_format type is unavailable now"` (json_object and no response_format both 200).
- The router itself never touches `response_format` on the chat path (only the transcription path sets it, `src/providers/base.py:538`), so declaring the parameter is pure advertisement — no code change.
- Decision: in the PRODUCTION `config/model_info.yaml`, give `local/orange/chat`, `local/orange/reasoner` and a new `local/orange/instant` entry `supported_parameters: [tools, tool_choice, max_tokens, temperature, top_p, stream, response_format, structured_outputs]`. `merge_capabilities` (`src/core/model_capabilities/render.py:107`) replaces lists, and the cache has no list for these models, so the manual list is exactly what is served. The `instant` entry carries only that key — every other field keeps coming from the cache.
- Decision: deepseek models stay as they are — they do NOT get `structured_outputs` (json_schema is refused upstream). Adding `response_format` alone for json_object is out of scope.
- Production config is not in the deploy (`deploy-server` rsyncs `src/` only; `config/` on the server is authoritative), so the change ships as a hand edit on the server, hot-reloaded within 5s. No code commit.
- Repo side: one line in the `config/model_info.yaml` header (`config/model_info.yaml:23`, the `supported_parameters` schema line) gains the meaning of the two tokens: `response_format` = accepts json_object, `structured_outputs` = accepts json_schema strict; Lore's extractor shows only models declaring `structured_outputs`.
- Size: S — config + one header comment; no runtime code.

## Risks

- A future llama-server model swap that breaks grammar-constrained output keeps the declaration. Signal: Lore extractor error notes now carry the provider body (`HTTP 4xx from model local/orange/...`).
- Order with Lore: if Lore's gate ships first, the three orange models vanish from Lore's extractor picker and CIR pipelines on `local/orange/chat` are refused at launch. This edit must land BEFORE Lore deploys its gate.

## Order

1. Commit the header line in `config/model_info.yaml` (repo).
2. Hand edit of the production `config/model_info.yaml` on `ssh docker` per the decision above (operator confirms the edit first — server config is authoritative).

## Not doing

- Probing upstreams at refresh time to auto-detect structured-output support.
- `system/interface`, glm, kimi, deepseek entries — not on Lore's key or not supporting json_schema.

## Validation

- `ssh docker "curl -s localhost:8777/v1/models -H 'Authorization: Bearer <lore key from Lore .env AI_API_KEY>' | jq -c '.data[] | {id, sp: .supported_parameters}'"` → the three `local/orange/*` ids list `structured_outputs`; `deepseek/flash` does not.
