"""Unit tests for src/providers/anthropic_translate.py — chat-completions <-> Messages."""

import pytest
from fastapi import HTTPException

from src.core.usage_db import RequestStats
from src.providers.anthropic_translate import chat_to_messages, messages_to_chat, openai_usage

LORE_SCHEMA = {
    "type": "object",
    "properties": {
        "age": {"type": ["number", "null"], "description": "years", "minimum": 0},
        "smoker": {"type": "string", "enum": ["yes", "no", ""]},
    },
    "required": ["age", "smoker"],
    "additionalProperties": False,
}


def _lore_extractor_body(**extra):
    """The body Lore's call_llm_structured sends (system-free, temperature 0)."""
    return {
        "model": "claude/sonnet",
        "messages": [{"role": "user", "content": "Extract: age 42"}],
        "temperature": 0,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "extraction", "strict": True, "schema": LORE_SCHEMA}},
        **extra,
    }


def _error_code(exc_info):
    return exc_info.value.detail["error"]["metadata"]["error_code"]


class TestRequest:

    def test_lore_extractor_body(self):
        out = chat_to_messages(_lore_extractor_body(max_tokens=512))
        assert out == {
            "messages": [{"role": "user", "content": "Extract: age 42"}],
            "max_tokens": 512,
            "output_config": {"format": {"type": "json_schema", "schema": {
                "type": "object",
                "properties": {
                    "age": {"anyOf": [{"type": "number"}, {"type": "null"}],
                            "description": "years"},
                    "smoker": {"type": "string", "enum": ["yes", "no", ""]},
                },
                "required": ["age", "smoker"],
                "additionalProperties": False,
            }}},
        }

    def test_system_and_developer_join_into_top_level_system(self):
        out = chat_to_messages({"messages": [
            {"role": "system", "content": "A"},
            {"role": "developer", "content": [{"type": "text", "text": "B"}]},
            {"role": "user", "content": [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}]},
            {"role": "assistant", "content": "z"},
        ], "max_completion_tokens": 7, "stop": "END"})
        assert out == {
            "system": "A\n\nB",
            "messages": [{"role": "user", "content": "xy"},
                         {"role": "assistant", "content": "z"}],
            "max_tokens": 7,
            "stop_sequences": ["END"],
        }

    def test_reasoning_and_sampling_fields_are_dropped(self):
        out = chat_to_messages({"messages": [], "temperature": 0, "top_p": 1, "stream": False,
                                "reasoning_effort": "high", "thinking": {"type": "enabled"},
                                "reasoning": {"effort": "high"}, "n": 1})
        assert out == {"messages": []}

    @pytest.mark.parametrize("body", [
        {"messages": [], "tools": [{"type": "function"}]},
        {"messages": [], "n": 2},
        {"messages": [], "logprobs": True},
        {"messages": [], "response_format": {"type": "json_object"}},
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]},
        {"messages": [{"role": "tool", "content": "x"}]},
        {"messages": [{"role": "assistant", "content": None, "tool_calls": []}]},
    ])
    def test_unmappable_field_is_400_never_dropped(self, body):
        with pytest.raises(HTTPException) as exc_info:
            chat_to_messages(body)
        assert exc_info.value.status_code == 400
        assert _error_code(exc_info) == "invalid_parameter_value"


class TestResponse:

    def test_text_blocks_only_and_requested_model(self):
        out = messages_to_chat({
            "id": "msg_1", "model": "claude-sonnet-5-5", "stop_reason": "end_turn",
            "content": [{"type": "thinking", "thinking": "hmm"},
                        {"type": "text", "text": '{"age"'},
                        {"type": "text", "text": ": 42}"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }, "claude/sonnet")
        assert out["id"] == "msg_1"
        assert out["object"] == "chat.completion"
        assert out["model"] == "claude/sonnet"
        assert out["choices"] == [{"index": 0, "finish_reason": "stop",
                                   "message": {"role": "assistant", "content": '{"age": 42}'}}]

    @pytest.mark.parametrize(("stop_reason", "finish_reason"), [
        ("end_turn", "stop"), ("stop_sequence", "stop"),
        ("max_tokens", "length"), ("refusal", "content_filter"),
    ])
    def test_finish_reason(self, stop_reason, finish_reason):
        out = messages_to_chat({"content": [], "stop_reason": stop_reason}, "m")
        assert out["choices"][0]["finish_reason"] == finish_reason

    def test_usage_records_the_same_row_as_the_messages_path(self):
        """The INVARIANT on openai_usage: the chat funnel's set_usage over the
        translated usage equals set_anthropic_usage over the raw one."""
        raw = {"input_tokens": 11, "cache_read_input_tokens": 300,
               "cache_creation_input_tokens": 40, "output_tokens": 7}
        via_chat, via_messages = RequestStats(), RequestStats()
        via_chat.set_usage(openai_usage(raw))
        via_messages.set_anthropic_usage(raw)
        assert via_chat == via_messages
        assert via_chat.cached_tokens == 300 and via_chat.cache_write_tokens == 40
