# Serve non-streaming chat completions for anthropic-type models

## Decisions

- Context: a Lore partner deployment has only Anthropic models. Lore's pipeline (extractor + digest) has ONE LLM path — `lore-knowledge-creator` `backend/pipeline/extractor/utils.py` `call_llm_structured` — which always POSTs `{AI_API_URL}/chat/completions` non-streaming with `temperature: 0` and `response_format: {type: json_schema, json_schema: {name, strict: true, schema}}`, and reads `choices[0].message.content`. Lore's image prompt refiner (`backend/image_generation/refine.py`) is a second plain chat-completions caller. Lore's chat agent is NOT affected: it reads `/v1/models` `api` and speaks `/v1/messages` itself.
- Current behaviour: the protocol gate refuses an anthropic-type model on `/v1/chat/completions` with 400 `wrong_api` (`src/services/base.py:249`), so the partner's pipeline cannot run at all. Separately, Lore's extractor model picker lists only models whose `/v1/models` `supported_parameters` carries `structured_outputs` (`config/model_info.yaml:24` defines the token), and no Claude model declares it, so the picker is empty.
- Operator ruling: the fix lives in the router, not in Lore — Lore stays OpenAI-wire for its direct calls, and every such caller is fixed at once.
- Decision: `/v1/chat/completions` accepts an anthropic-type model when the request is NON-streaming, translating the body to Messages and the response back. A streaming chat-completions request for an anthropic-type model keeps the 400 `wrong_api` — nothing in Lore needs it and stream re-framing is a different-sized job.
- Decision: the translation lives in the provider layer, per this repo's "provider isolation" principle — `AnthropicProvider` gains a `chat_completions` override (base: `src/providers/base.py:486`) that translates, calls its own `messages` (`src/providers/anthropic.py:46`), and translates the result back. Pure translation functions go in a new `src/providers/anthropic_translate.py` so they are unit-testable without HTTP.
- Decision: the protocol gate (`src/services/base.py:38` table, check at `:249`) learns one exception — `api="openai-completions"` with an anthropic-type provider passes when the body is non-streaming. The gate has no body today (it is body-agnostic for multipart), so the stream check sits in `ChatService.chat_completions` (`src/services/chat_service/chat_service.py:22`) before dispatch, passing a flag into the funnel; embeddings/transcription keep the strict gate.
- Decision: the OpenAI-wire funnel surgery still runs first (`src/services/base.py:279` effort policy, `:290` dialect). For anthropic-type models the effort block is advertisement-only per CLAUDE.md, so the translator DROPS every reasoning field (`reasoning_effort`, `reasoning`, `thinking`) rather than mapping it — no Lore caller sends one.
- Request mapping: `system`/`developer` messages → top-level `system` (joined); `user`/`assistant` string or text-part content → Messages content; `max_tokens` or `max_completion_tokens` → `max_tokens`, defaulting to the model's `max_completion_tokens` from the rendered capabilities (Anthropic refuses a request without it); `stop` → `stop_sequences`; `response_format.json_schema.schema` → `output_config: {format: {type: "json_schema", schema}}` (the GA field; the old `output_format` is deprecated); `response_format: json_object` → 400 `invalid_parameter_value` (no Anthropic equivalent). Anything else unmappable (tools, images, `n>1`, logprobs) → 400 naming the field, never a silent drop.
- Request mapping, sampling: `temperature`/`top_p` are DROPPED. Current Claude models (Opus 5.5, Sonnet 5.5, Haiku 5.5, Fable) answer 400 to any sampling parameter, and API version 2026-10-21 refuses them on every model. The cost is determinism — see Risks.
- Request mapping, thinking: the translator never sends `thinking`. Opus 5.5 refuses `{type: disabled}` at every effort and runs adaptive by default, so the response can carry thinking blocks.
- Schema mapping: a property `type: [T, "null"]` is rewritten to `anyOf: [{type: T}, {type: "null"}]` with the other keys kept. Lore emits that shape for every numeric extractor field (an INVARIANT in Lore's `build_json_schema`), and the documented supported list names `anyOf` and `null` but not type arrays. `minimum`/`maximum`/`minLength`/`maxLength`/`multipleOf` are stripped (unsupported); Lore emits none today.
- Response mapping: concatenate the `text` blocks only (skip `thinking`/`redacted_thinking`) into `choices[0].message.content`; `stop_reason` `end_turn`/`stop_sequence` → `finish_reason: stop`, `max_tokens` → `length`, `refusal` → `content_filter`; `id`, `model` (the client's requested id), `created`, `object: chat.completion`.
- Usage: the response carries OpenAI-shaped usage for the client, but stats are recorded from the RAW Anthropic usage via `set_anthropic_usage` (`src/core/usage_db/writer.py:106`), so cache read/write split and cost stay exactly as on `/v1/messages`.
- Errors: an upstream 4xx/5xx body passes through as on the native path (the router's OpenRouter envelope on this route); Lore already surfaces `resp.text` in its error note, so a schema Anthropic rejects names its own cause.
- `/v1/models`: the `api` field stays `anthropic-messages` (it names the NATIVE protocol for harnesses). Advertising `structured_outputs` on Claude models is a `model_info.yaml` declaration on the deployment that serves them, not code.
- Size: M — one commit, one repo, moves the protocol-gate contract.

## Risks

- Determinism: the extractor's `temperature: 0` is dropped on Claude, so identical input can extract differently. Accepted: the partner's schemas are small and Claude offers no greedy mode on current models. Signal: repeated runs on the same transcript disagree.
- Schema complexity: Anthropic compiles each schema; an over-complex one is refused with 400. No union-count limit is documented in the reference used to write this plan; the partner needs ~10 fields. Signal: a 400 from upstream quoting the schema, visible verbatim in Lore's extractor error note.
- DeepSeek's Anthropic endpoint (`deepseek-anthropic` in `config/providers.yaml`) may not support `output_config.format`. Signal: 400 from `api.deepseek.com/anthropic` on a structured request; it then simply does not get `structured_outputs` declared.
- Refusal: `stop_reason: refusal` yields content that may not match the schema; Lore's `json.loads` fails loudly. Accepted.
- A future gate change that re-derives the protocol rule without the stream flag would re-break the partner silently. Signal: `wrong_api` back on non-streaming chat to a Claude model — covered by a test.

## Order

1. `src/providers/anthropic_translate.py` (request/schema/response translators), `AnthropicProvider.chat_completions` override in `src/providers/anthropic.py`, the non-stream exception in the protocol gate (`src/services/base.py`) with the flag from `src/services/chat_service/chat_service.py`, usage recorded via `set_anthropic_usage` on that path, tests for the translators and for gate behaviour (non-stream passes, stream still `wrong_api`, embeddings still `wrong_api`), a `CLAUDE.md` Request-Flow line and the `config/model_info.yaml` header note that `structured_outputs` is valid on anthropic-type models.

## Not doing

- Streaming chat completions for anthropic-type models (SSE re-framing).
- Tools/tool_choice, images, multi-choice translation.
- Mapping OpenAI reasoning fields onto Anthropic thinking/effort.
- Any change in Lore beyond dropping its `wrong_api` warnings in the `CHAT_MODEL`/`COMFYUI_PROMPT_MODEL` setting help after this ships (Lore-side task).
- A forced-tool-use fallback for upstreams without `output_config.format` — current Claude models refuse forced `tool_choice`, so it would not even work there.

## Validation

- Unit tests over the translators: a Lore-shaped extractor body (system-free, one user message, `temperature: 0`, strict json_schema with a `["number","null"]` field) → expected Messages body with `output_config.format`, the `anyOf` rewrite, no `temperature`, `max_tokens` filled.
- Live, on the deployment serving the partner, after the operator adds `structured_outputs` to its Claude models' `supported_parameters` in that server's `config/model_info.yaml` (hand edit, hot-reloaded): `curl -s :8777/v1/chat/completions -H 'Authorization: Bearer <partner Lore key>' -d '{"model":"<claude id>","messages":[{"role":"user","content":"Extract: age 42, smoker unknown"}],"temperature":0,"response_format":{"type":"json_schema","json_schema":{"name":"extraction","strict":true,"schema":{"type":"object","properties":{"age":{"type":["integer","null"]},"smoker":{"type":"string"}},"required":["age","smoker"],"additionalProperties":false}}}}'` → 200, `choices[0].message.content` parses to `{"age":42,...}`, and the usage row in `/stat` shows the call with Anthropic token counts.
- Same request with `"stream":true` → 400 `wrong_api`. A `/v1/embeddings` call naming the Claude id → 400 `wrong_api`.
- `/v1/models` with the partner key → the Claude ids list `structured_outputs`; Lore's extractor model dropdown in the partner project then shows them, and one pipeline run on a short transcript completes.
