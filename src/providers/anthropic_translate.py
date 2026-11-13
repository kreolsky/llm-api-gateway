"""Pure OpenAI chat-completions <-> Anthropic Messages translators (non-streaming only)."""

import time
from typing import Any

from ..core.error_handling import ErrorType, create_error

# WHY dropped, not mapped: current Claude models answer 400 to any sampling
# parameter, and the reasoning fields are OpenAI-wire surgery (the effort
# policy is advertisement-only for anthropic-type models). `stream` is false
# by the time this runs — the protocol gate refuses a streaming request.
_DROPPED_KEYS = frozenset({
    "model", "stream", "temperature", "top_p",
    "reasoning_effort", "reasoning", "thinking",
})
_MAPPED_KEYS = frozenset({
    "messages", "max_tokens", "max_completion_tokens", "stop", "response_format",
})
_SYSTEM_ROLES = frozenset({"system", "developer"})
_TURN_ROLES = frozenset({"user", "assistant"})
_UNSUPPORTED_SCHEMA_KEYS = frozenset({
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minLength", "maxLength", "multipleOf",
})
_FINISH_REASON = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "refusal": "content_filter",
}


def _unsupported(field: str, value: Any, request_id: str) -> Exception:
    """400 naming the field a Messages body cannot carry — never a silent drop."""
    return create_error(
        ErrorType.INVALID_PARAMETER_VALUE, parameter_name=field, parameter_value=repr(value),
        allowed_values="not supported on /v1/chat/completions for anthropic-type models",
        request_id=request_id,
    )


def _text_of(content: Any, field: str, request_id: str) -> str:
    """A message's content as plain text: a string or a list of text parts."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise _unsupported(field, content, request_id)
    texts: list[str] = []
    for index, part in enumerate(content):
        if not isinstance(part, dict) or part.get("type") != "text":
            part_type = part.get("type") if isinstance(part, dict) else part
            raise _unsupported(f"{field}[{index}].type", part_type, request_id)
        texts.append(part.get("text") or "")
    return "".join(texts)


def _translate_messages(messages: Any, request_id: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Split OpenAI messages into Messages' top-level system parts and turns."""
    if not isinstance(messages, list):
        raise create_error(ErrorType.MISSING_REQUIRED_FIELD, field_name="messages",
                           request_id=request_id)
    system: list[str] = []
    turns: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        field = f"messages[{index}]"
        role = message.get("role") if isinstance(message, dict) else None
        if role not in _SYSTEM_ROLES and role not in _TURN_ROLES:
            raise _unsupported(f"{field}.role", role, request_id)
        extra = set(message) - {"role", "content", "name"}
        if extra:
            key = sorted(extra)[0]
            raise _unsupported(f"{field}.{key}", message[key], request_id)
        text = _text_of(message.get("content"), f"{field}.content", request_id)
        if role in _SYSTEM_ROLES:
            system.append(text)
        else:
            turns.append({"role": role, "content": text})
    return system, turns


def _translate_schema(node: Any) -> Any:
    """Rewrite a JSON schema into the subset Anthropic structured outputs compile.

    ``type: [T, "null"]`` -> ``anyOf: [{type: T}, {type: "null"}]`` (other keys
    kept on the node); numeric/string bound keywords are stripped.
    """
    if isinstance(node, list):
        return [_translate_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    out = {key: _translate_schema(value) for key, value in node.items()
           if key not in _UNSUPPORTED_SCHEMA_KEYS}
    node_type = out.get("type")
    if isinstance(node_type, list):
        del out["type"]
        out["anyOf"] = [{"type": member} for member in node_type]
    return out


def _translate_response_format(response_format: Any, request_id: str) -> dict[str, Any] | None:
    """OpenAI response_format -> Messages output_config (None for plain text)."""
    if response_format is None:
        return None
    kind = response_format.get("type") if isinstance(response_format, dict) else response_format
    if kind == "text":
        return None
    if kind != "json_schema":
        raise _unsupported("response_format.type", kind, request_id)
    schema = (response_format.get("json_schema") or {}).get("schema")
    if not isinstance(schema, dict):
        raise create_error(ErrorType.MISSING_REQUIRED_FIELD,
                           field_name="response_format.json_schema.schema", request_id=request_id)
    return {"format": {"type": "json_schema", "schema": _translate_schema(schema)}}


def chat_to_messages(body: dict[str, Any], request_id: str = "unknown") -> dict[str, Any]:
    """Translate a non-streaming chat-completions body into a Messages body.

    Raises 400 on any field with no Messages equivalent. ``max_tokens`` is
    omitted when the body carries neither limit (the models.yaml options merge
    may still supply it; otherwise the upstream refuses and its error passes
    through).
    """
    for key, value in body.items():
        if key in _DROPPED_KEYS or key in _MAPPED_KEYS:
            continue
        if key == "n" and value in (None, 1):
            continue
        raise _unsupported(key, value, request_id)

    system, turns = _translate_messages(body.get("messages"), request_id)
    out: dict[str, Any] = {"messages": turns}
    if system:
        out["system"] = "\n\n".join(system)
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
    if max_tokens:
        out["max_tokens"] = max_tokens
    stop = body.get("stop")
    if stop:
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    output_config = _translate_response_format(body.get("response_format"), request_id)
    if output_config:
        out["output_config"] = output_config
    return out


def openai_usage(usage: dict[str, Any]) -> dict[str, Any]:
    """Anthropic usage -> OpenAI usage, under the router's prompt_tokens convention.

    INVARIANT: RequestStats.set_usage(openai_usage(u)) records the same row as
    RequestStats.set_anthropic_usage(u).
    Why: the chat-completions funnel records stats from the OpenAI shape, and
    the cache read/write split (and so the frozen cost) must stay exactly as on
    /v1/messages — pinned by a test over both extractors.
    """
    input_tokens = int(usage.get("input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    prompt_tokens = input_tokens + cache_read + cache_write
    completion_tokens = int(usage.get("output_tokens") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": cache_read, "cache_write_tokens": cache_write},
    }


def messages_to_chat(response: dict[str, Any], requested_model: str) -> dict[str, Any]:
    """Translate a Messages response into a chat.completion body.

    Only ``text`` blocks reach the content (thinking blocks are skipped).
    """
    text = "".join(block.get("text") or "" for block in response.get("content") or []
                   if isinstance(block, dict) and block.get("type") == "text")
    stop_reason = response.get("stop_reason")
    return {
        "id": response.get("id") or "",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": _FINISH_REASON.get(stop_reason, "stop"),
        }],
        "usage": openai_usage(response.get("usage") or {}),
    }
