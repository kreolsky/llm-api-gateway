"""Unit tests for the SSE pass-through (process_stream and its helpers)."""

import json
import logging

import pytest
from fastapi import HTTPException

from src.core.usage_db import RequestStats
from src.services.chat_service.stream_processor import (
    ANTHROPIC_STREAM,
    duplicate_reasoning_field,
    process_stream,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def async_gen(items):
    for item in items:
        yield item


async def collect(agen):
    return [x async for x in agen]


def sse(data_str, sep="\n\n"):
    return f"data: {data_str}{sep}".encode()


# ---------------------------------------------------------------------------
# 1. Transparent mode
# ---------------------------------------------------------------------------

class TestTransparentMode:

    @pytest.mark.asyncio
    async def test_chunks_pass_through(self):
        chunks = [b"data: hello\n\n", b"data: world\n\n"]
        result = await collect(process_stream(async_gen(chunks), "m", "r", "u", "p", RequestStats()))
        assert result == chunks

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        result = await collect(process_stream(async_gen([]), "m", "r", "u", "p", RequestStats()))
        assert result == []


# ---------------------------------------------------------------------------
# 6. Mid-stream error frame (via process_stream)
# ---------------------------------------------------------------------------

def _parse_error_frame(result: bytes):
    """Parse the first SSE data frame from a stream's error output."""
    text = result.decode("utf-8")
    first_frame = text.split("\n\n", 1)[0]
    return json.loads(first_frame[len("data: "):])


def raising_gen(exc):
    """A provider stream that fails on its first read."""
    async def gen():
        raise exc
        yield b""  # pragma: no cover
    return gen()


class TestErrorFrame:

    @pytest.mark.asyncio
    async def test_generic_exception(self):
        result = b"".join(await collect(
            process_stream(raising_gen(ValueError("oops")), "m", "r", "u", "p", RequestStats())))
        decoded = _parse_error_frame(result)
        assert decoded["error"]["code"] == 500
        assert "oops" in decoded["error"]["message"]

    @pytest.mark.asyncio
    async def test_http_exception_string_detail(self):
        exc = HTTPException(status_code=403, detail="forbidden")
        result = b"".join(await collect(
            process_stream(raising_gen(exc), "m", "r", "u", "p", RequestStats())))
        decoded = _parse_error_frame(result)
        assert decoded["error"]["code"] == 403
        assert "forbidden" in decoded["error"]["message"]

    @pytest.mark.asyncio
    async def test_http_exception_dict_detail(self):
        detail = {"error": {"code": 429, "message": "rate limited"}}
        exc = HTTPException(status_code=429, detail=detail)
        result = b"".join(await collect(
            process_stream(raising_gen(exc), "m", "r", "u", "p", RequestStats())))
        decoded = _parse_error_frame(result)
        assert decoded["error"]["code"] == 429
        assert decoded["error"]["message"] == "rate limited"

    @pytest.mark.asyncio
    async def test_returns_bytes_with_sse_framing(self):
        result = b"".join(await collect(
            process_stream(raising_gen(RuntimeError("x")), "m", "r", "u", "p", RequestStats())))
        assert isinstance(result, bytes)
        assert result.startswith(b"data: ")
        assert result.endswith(b"\n\n")

    @pytest.mark.asyncio
    async def test_ends_with_done_sentinel(self):
        result = b"".join(await collect(
            process_stream(raising_gen(RuntimeError("x")), "m", "r", "u", "p", RequestStats())))
        assert result.endswith(b"data: [DONE]\n\n")


# ---------------------------------------------------------------------------
# 7b. Concurrency: per-stream captured usage isolation
# ---------------------------------------------------------------------------

class TestConcurrentStreamUsageIsolation:

    @pytest.mark.asyncio
    async def test_two_interleaved_streams_record_own_usage(self):
        """Two streams each capture their own usage."""

        usage_a = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        usage_b = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        chunk_a = sse(json.dumps({"usage": usage_a}))
        chunk_b = sse(json.dumps({"usage": usage_b}))

        recorded = []

        async def gen_a():
            for c in [chunk_a, chunk_b]:
                yield c

        async def gen_b():
            for c in [chunk_b, chunk_a]:
                yield c

        async def run_and_capture(gen, tag):
            chunks = await collect(process_stream(gen, tag, tag, tag, tag, RequestStats()))
            recorded.append((tag, chunks))

        await run_and_capture(gen_a(), "A")
        await run_and_capture(gen_b(), "B")

        # Each stream saw both usages; verify usage was captured without
        # cross-stream interference (no instance-level shared overwrite crash).
        assert len(recorded) == 2


# ---------------------------------------------------------------------------
# 9. reasoning -> reasoning_content duplication
# ---------------------------------------------------------------------------

class TestReasoningFieldDuplication:

    def test_stream_delta_duplicated(self):
        data = {"choices": [{"delta": {"reasoning": "We"}}]}
        assert duplicate_reasoning_field(data) is True
        assert data["choices"][0]["delta"]["reasoning_content"] == "We"
        assert data["choices"][0]["delta"]["reasoning"] == "We"

    def test_non_stream_message_duplicated(self):
        data = {"choices": [{"message": {"role": "assistant", "reasoning": "think", "content": "42"}}]}
        assert duplicate_reasoning_field(data) is True
        assert data["choices"][0]["message"]["reasoning_content"] == "think"

    def test_existing_reasoning_content_not_overwritten(self):
        data = {"choices": [{"delta": {"reasoning": "new", "reasoning_content": "old"}}]}
        assert duplicate_reasoning_field(data) is False
        assert data["choices"][0]["delta"]["reasoning_content"] == "old"

    def test_no_reasoning_untouched(self):
        data = {"choices": [{"delta": {"content": "hi"}}]}
        assert duplicate_reasoning_field(data) is False
        assert data == {"choices": [{"delta": {"content": "hi"}}]}

    def test_malformed_shapes_ignored(self):
        assert duplicate_reasoning_field(None) is False
        assert duplicate_reasoning_field("string") is False
        assert duplicate_reasoning_field({"choices": "oops"}) is False
        assert duplicate_reasoning_field({"choices": ["oops"]}) is False

    @pytest.mark.asyncio
    async def test_transparent_stream_remapped(self):
        chunk = sse(json.dumps({"choices": [{"delta": {"reasoning": "We"}}]}))
        result = await collect(process_stream(async_gen([chunk]), "m", "r", "u", "p", RequestStats()))
        parsed = json.loads(result[0].decode("utf-8").split("data: ", 1)[1].split("\n\n")[0])
        assert parsed["choices"][0]["delta"]["reasoning_content"] == "We"

    @pytest.mark.asyncio
    async def test_transparent_stream_without_reasoning_unchanged(self):
        chunk = sse(json.dumps({"choices": [{"delta": {"content": "hi"}}]}))
        result = await collect(process_stream(async_gen([chunk]), "m", "r", "u", "p", RequestStats()))
        assert result == [chunk]

    @pytest.mark.asyncio
    async def test_transparent_split_json_not_corrupted(self):
        """A JSON line split across chunks must pass through without corruption."""
        full = 'data: {"choices": [{"delta": {"reasoning": "hello"}}]}\n\n'
        encoded = full.encode("utf-8")
        mid = encoded.index(b'"hello') + 3
        result = await collect(process_stream(
            async_gen([encoded[:mid], encoded[mid:]]), "m", "r", "u", "p", RequestStats()))
        combined = b"".join(result).decode("utf-8")
        parsed = json.loads(combined.split("data: ", 1)[1].split("\n\n")[0])
        assert parsed["choices"][0]["delta"]["reasoning"] == "hello"


# ---------------------------------------------------------------------------
# 11. Stats-holder enrichment (frame and row cannot drift)
# ---------------------------------------------------------------------------

class TestStatsEnrichment:

    @pytest.mark.asyncio
    async def test_mid_stream_error_writes_error_code_and_partial_usage(self):
        """A failure after the 200 must not read as a success."""
        stats = RequestStats(model_id="m", provider_name="p", stream=True)
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        chunks = [sse(json.dumps({"usage": usage}))]

        async def gen():
            for c in chunks:
                yield c
            raise HTTPException(
                status_code=500,
                detail={"error": {"code": 500, "message": "upstream died",
                                  "metadata": {"error_code": "internal_server_error"}}},
            )

        result = await collect(process_stream(gen(), "m", "r", "u", "p", stats=stats))
        assert any(b"[DONE]" in r for r in result)
        assert stats.error_code == "internal_server_error"
        assert "upstream died" in stats.error_message
        # Partial usage captured before the failure survives the error path
        assert stats.prompt_tokens == 10
        assert stats.completion_tokens == 5
        assert stats.has_usage is True

    @pytest.mark.asyncio
    async def test_generic_exception_classified_internal_server_error(self):
        stats = RequestStats()

        async def gen():
            yield b"data: {}\n\n"
            raise ValueError("kaboom")

        await collect(process_stream(gen(), "m", "r", "u", "p", stats=stats))
        assert stats.error_code == "internal_server_error"
        assert "kaboom" in stats.error_message

    @pytest.mark.asyncio
    async def test_http_exception_without_metadata_is_coarse(self):
        stats = RequestStats()

        async def gen():
            yield b"data: {}\n\n"
            raise HTTPException(status_code=403, detail="forbidden")

        await collect(process_stream(gen(), "m", "r", "u", "p", stats=stats))
        assert stats.error_code == "internal_server_error"
        assert stats.error_message == "forbidden"

    @pytest.mark.asyncio
    async def test_successful_stream_keeps_usage_only(self):
        stats = RequestStats(model_id="m")
        usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
        await collect(process_stream(
            async_gen([sse(json.dumps({"usage": usage}))]), "m", "r", "u", "p", stats=stats))
        assert stats.has_usage is True
        assert stats.total_tokens == 10
        assert stats.error_code is None
        assert stats.error_message is None

    @pytest.mark.asyncio
    async def test_usage_stays_empty_when_provider_sent_no_usage_chunk(self):
        stats = RequestStats()
        await collect(process_stream(
            async_gen([b"data: {}\n\n"]), "m", "r", "u", "p", stats=stats))
        assert stats.has_usage is False
        assert stats.total_tokens == 0

    @pytest.mark.asyncio
    async def test_upstream_error_frame_is_forwarded_verbatim_and_recorded(self):
        """A llama-server late failure must not read as a clean success."""
        stats = RequestStats(model_id="m", provider_name="p", stream=True)
        chunks = [_DELTA_CHUNK, _LLAMA_ERROR_FRAME]
        result = await collect(process_stream(
            async_gen(chunks), "m", "r", "u", "p", stats=stats))
        assert b"".join(result) == b"".join(chunks)
        assert stats.error_code == "provider_stream_error"
        assert "peg-native" in stats.error_message
        assert stats.has_usage is False

    @pytest.mark.asyncio
    async def test_upstream_error_frame_is_logged(self, caplog):
        with caplog.at_level(logging.ERROR, logger="nnp-llm-router"):
            await collect(process_stream(
                async_gen([_DELTA_CHUNK, _LLAMA_ERROR_FRAME]), "m", "r", "u", "p",
                stats=RequestStats()))
        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert any(m.startswith("Provider 'p' returned error 500:") for m in errors)
        line = _LLAMA_ERROR_FRAME.decode().split("\n", 1)[0]
        assert any(line[:200] in m for m in errors)

    @pytest.mark.asyncio
    async def test_delta_containing_error_substring_is_not_an_error(self, caplog):
        stats = RequestStats()
        delta = sse(json.dumps({"choices": [{"delta": {"content": "error"}}]}))
        with caplog.at_level(logging.ERROR, logger="nnp-llm-router"):
            result = await collect(process_stream(
                async_gen([delta]), "m", "r", "u", "p", stats=stats))
        assert b'"error"' in delta  # the byte gate fires; only the parse must reject it
        assert result == [delta]
        assert stats.error_code is None
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    @pytest.mark.asyncio
    async def test_error_frame_with_prior_usage_keeps_partial_usage(self):
        stats = RequestStats()
        usage = {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
        await collect(process_stream(
            async_gen([sse(json.dumps({"usage": usage})), _LLAMA_ERROR_FRAME]),
            "m", "r", "u", "p", stats=stats))
        assert stats.has_usage is True
        assert stats.total_tokens == 6
        assert stats.error_code == "provider_stream_error"


_DELTA_CHUNK = sse(json.dumps({"choices": [{"delta": {"content": "Hi"}}]}))
_LLAMA_ERROR_FRAME = sse(json.dumps({"error": {
    "code": 500,
    "message": "Failed to parse input at pos 12: the model output does not match "
               "the peg-native format",
    "type": "server_error",
}}))


# ---------------------------------------------------------------------------
# 12. Anthropic wire (ANTHROPIC_STREAM): byte-identical frames, usage capture
# ---------------------------------------------------------------------------

_MESSAGE_START = (
    'event: message_start\n'
    'data: {"type":"message_start","message":{"id":"msg_1","type":"message","usage":'
    '{"input_tokens":25,"output_tokens":1,"cache_creation_input_tokens":0,'
    '"cache_read_input_tokens":0}}}\n\n'
)
_THINKING_DELTA = (
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta",'
    '"thinking":"Hmm"}}\n\n'
)
_TEXT_DELTA = (
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta",'
    '"text":"Hi there"}}\n\n'
)
_MESSAGE_DELTA = (
    'event: message_delta\n'
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":'
    '{"input_tokens":25,"output_tokens":12,"cache_creation_input_tokens":128,'
    '"cache_read_input_tokens":64}}\n\n'
)
_MESSAGE_STOP = 'event: message_stop\ndata: {"type":"message_stop"}\n\n'

_DEEPSEEK_ANTHROPIC_FRAMES = [
    _MESSAGE_START.encode(), _THINKING_DELTA.encode(), _TEXT_DELTA.encode(),
    _MESSAGE_DELTA.encode(), _MESSAGE_STOP.encode(),
]


class TestAnthropicWire:
    """The DeepSeek anthropic-wire frames observed during planning, passed
    through byte-identically, with usage captured from message_start and
    message_delta and recorded via set_anthropic_usage."""

    def _run(self, chunks, stats=None):
        return collect(process_stream(
            async_gen(chunks), "m", "r", "u", "p",
            stats=stats or RequestStats(), wire=ANTHROPIC_STREAM))

    @pytest.mark.asyncio
    async def test_frames_pass_byte_identical_and_usage_recorded(self):
        stats = RequestStats(model_id="m", provider_name="p")
        result = await self._run(_DEEPSEEK_ANTHROPIC_FRAMES, stats)

        assert b"".join(result) == b"".join(_DEEPSEEK_ANTHROPIC_FRAMES)
        # merged (later wins): input 25, output 12 (delta over start's 1),
        # cache_read 64, cache_creation 128
        assert stats.has_usage is True
        assert stats.prompt_tokens == 25 + 64 + 128
        assert stats.completion_tokens == 12
        assert stats.cached_tokens == 64
        assert stats.cache_write_tokens == 128
        assert stats.total_tokens == 25 + 64 + 128 + 12
        assert stats.error_code is None

    @pytest.mark.asyncio
    async def test_usage_split_across_chunks_recorded_only_when_whole(self):
        """Best-effort capture (documented): a data line split across chunk
        boundaries is not parsed; the earlier whole frame's usage survives."""
        encoded = _MESSAGE_DELTA.encode()
        # cut at every offset INSIDE the `data: {...}` JSON line: neither
        # chunk then carries the line whole, so the delta usage must be
        # skipped and the message_start numbers (output 1) survive. A cut
        # before the data line (e.g. inside "event: ...") leaves it whole in
        # the second chunk and is correctly captured — not asserted here.
        data_at = encoded.index(b"data: ")
        json_end = encoded.rindex(b"}") + 1
        for cut in range(data_at + 1, json_end):
            stats = RequestStats()
            chunks = [_MESSAGE_START.encode(), encoded[:cut], encoded[cut:],
                      _MESSAGE_STOP.encode()]
            result = await self._run(chunks, stats)
            assert b"".join(result) == b"".join(chunks)
            assert stats.has_usage is True
            assert stats.completion_tokens == 1, f"cut={cut}"
            assert stats.cache_write_tokens == 0, f"cut={cut}"

    @pytest.mark.asyncio
    async def test_no_usage_frames_leaves_stats_empty(self):
        stats = RequestStats()
        await self._run([_THINKING_DELTA.encode(), _MESSAGE_STOP.encode()], stats)
        assert stats.has_usage is False

    @pytest.mark.asyncio
    async def test_mid_stream_failure_frames_anthropic_error_without_done(self):
        """A router-side mid-stream failure answers an Anthropic error event,
        with no [DONE] sentinel (that is an OpenAI-wire convention)."""
        stats = RequestStats(model_id="m", provider_name="p", stream=True)

        async def gen():
            yield _MESSAGE_START.encode()
            raise HTTPException(
                status_code=429,
                detail={"error": {"code": 429, "message": "rate limited",
                                  "metadata": {"error_code": "provider_http_error"}}},
            )

        result = b"".join(await collect(
            process_stream(gen(), "m", "r", "u", "p", stats=stats, wire=ANTHROPIC_STREAM)))
        assert result.startswith(_MESSAGE_START.encode())
        tail = result[len(_MESSAGE_START):]
        assert tail.startswith(b"event: error\n")
        assert b"[DONE]" not in tail
        payload = json.loads(tail.decode().split("data: ", 1)[1].split("\n\n")[0])
        assert payload == {"type": "error",
                           "error": {"type": "rate_limit_error", "message": "rate limited"}}
        assert stats.error_code == "provider_http_error"

    @pytest.mark.asyncio
    async def test_openai_default_wire_unchanged(self):
        """process_stream without wire= keeps the OpenAI capture and frame."""
        usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
        stats = RequestStats()
        result = await collect(process_stream(
            async_gen([sse(json.dumps({"usage": usage}))]), "m", "r", "u", "p", stats=stats))
        assert stats.has_usage is True and stats.total_tokens == 10
        assert result == [sse(json.dumps({"usage": usage}))]


# ---------------------------------------------------------------------------
# 10. Eager priming: upstream errors must keep their HTTP status
# ---------------------------------------------------------------------------

from src.services.chat_service.stream_processor import open_provider_stream  # noqa: E402


class TestOpenProviderStream:

    @pytest.mark.asyncio
    async def test_first_chunk_error_propagates(self):
        """An error on the first read raises instead of becoming a stream body."""
        async def failing():
            raise HTTPException(status_code=429, detail={"error": {"code": 429}})
            yield b""  # pragma: no cover

        with pytest.raises(HTTPException) as exc_info:
            await open_provider_stream(failing())
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_stream_content_is_unchanged(self):
        """Priming does not consume or reorder the stream."""
        stream = await open_provider_stream(async_gen([b"a", b"b", b"c"]))
        assert await collect(stream) == [b"a", b"b", b"c"]

    @pytest.mark.asyncio
    async def test_empty_stream_stays_empty(self):
        stream = await open_provider_stream(async_gen([]))
        assert await collect(stream) == []

    @pytest.mark.asyncio
    async def test_later_error_still_reaches_the_body(self):
        """An error after the first chunk cannot change the status; it is framed."""
        async def fails_late():
            yield b"first"
            raise HTTPException(status_code=500, detail="boom")

        primed = await open_provider_stream(fails_late())
        result = b"".join(await collect(process_stream(primed, "m", "r", "u", "p", RequestStats())))
        assert result.startswith(b"first")
        assert b"[DONE]" in result

    @pytest.mark.asyncio
    async def test_generator_cleanup_runs_on_first_read_failure(self):
        """The failing generator's finally block runs, releasing its slot."""
        released = []

        async def failing():
            try:
                raise HTTPException(status_code=401, detail="nope")
                yield b""  # pragma: no cover
            finally:
                released.append(True)

        with pytest.raises(HTTPException):
            await open_provider_stream(failing())
        assert released == [True]

    @pytest.mark.asyncio
    async def test_abandoning_primed_stream_closes_the_source(self):
        """A client disconnect must close the wrapped provider stream.

        Otherwise the provider's in-flight slot is never released and a config
        reload waits out the whole drain timeout before closing the pool.
        """
        closed = []

        async def source():
            try:
                yield b"a"
                yield b"b"
            finally:
                closed.append(True)

        primed = await open_provider_stream(source())
        assert await primed.__anext__() == b"a"
        await primed.aclose()
        assert closed == [True]
