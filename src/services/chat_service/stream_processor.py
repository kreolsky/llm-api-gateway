"""Stream processor for forwarding provider SSE streams."""
# SYSTEM: sse-stream — passthrough streaming body

import json
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from ...core.error_handling import enrich_stats_from_envelope, log_provider_error
from ...core.error_handling.anthropic import anthropic_error_type
from ...core.logging import logger
from ...core.usage_db import RequestStats


@dataclass(frozen=True)
class StreamWire:
    """Per-protocol behavior of the SSE pass-through.

    ARCH: chunks are forwarded byte-for-byte on every wire; what varies is
    (a) the byte gate + parser that captures usage frames, (b) the stats
    recorder for the captured usage shape (OpenAI tokens vs Anthropic
    input/output), (c) the mid-stream router-failure frame, and (d) whether
    the reasoning->reasoning_content duplication applies (it is OpenAI-wire
    shaped: choices/delta/message holders). Frozen on purpose: a wire is
    protocol knowledge, not per-stream state.
    """
    chunk_may_carry_usage: Callable[[bytes], bool]
    capture_usage: Callable[[bytes, dict[str, Any]], None]
    record_usage: Callable[[RequestStats, dict[str, Any]], None]
    frame_error: Callable[[dict[str, Any]], bytes]
    remap_reasoning: bool


def _frame_anthropic_error(error_payload: dict[str, Any]) -> bytes:
    """Frame a router failure as an Anthropic error event.

    WHY no [DONE] sentinel here: that is an OpenAI-wire convention; the
    Anthropic wire terminates a failed stream with the error event alone.
    """
    error = error_payload.get("error") or {}
    status_code = error.get("code") if isinstance(error.get("code"), int) else 500
    payload = {
        "type": "error",
        "error": {"type": anthropic_error_type(status_code),
                  "message": str(error.get("message") or "stream failed")},
    }
    return f"event: error\ndata: {json.dumps(payload)}\n\n".encode()


def _openai_chunk_may_carry_usage(chunk: bytes) -> bool:
    return b'"usage"' in chunk and b'"prompt_tokens"' in chunk


def _anthropic_chunk_may_carry_usage(chunk: bytes) -> bool:
    return b'"input_tokens"' in chunk or b'"output_tokens"' in chunk


def _capture_usage_from_chunk(chunk: bytes, captured_usage: dict[str, Any]) -> None:
    """Parse usage out of a transparent-mode chunk, writing into the holder."""
    try:
        text = chunk.decode('utf-8')
        for line in text.split('\n'):
            if line.startswith('data: ') and line != 'data: [DONE]':
                data = json.loads(line[6:])
                if 'usage' in data:
                    captured_usage["usage"] = data['usage']
    except Exception:
        pass


def _capture_anthropic_usage_from_chunk(chunk: bytes, captured_usage: dict[str, Any]) -> None:
    """Capture Anthropic-wire usage from complete ``data:`` lines.

    ``message_start`` carries usage under ``message.usage`` and
    ``message_delta`` at the top level (observed on DeepSeek: both carry the
    full set). Merged key by key into the holder — later frames win, so the
    final ``message_delta`` numbers are recorded. Best-effort like the OpenAI
    capture: a line split across chunks is not parsed.
    """
    try:
        text = chunk.decode('utf-8')
    except UnicodeDecodeError:
        return
    for line in text.split('\n'):
        line = line.rstrip('\r')
        if not line.startswith('data: '):
            continue
        try:
            payload = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        usage = None
        if payload.get("type") == "message_start":
            message = payload.get("message")
            usage = message.get("usage") if isinstance(message, dict) else None
        elif payload.get("type") == "message_delta":
            usage = payload.get("usage")
        if isinstance(usage, dict):
            merged = dict(captured_usage.get("usage") or {})
            merged.update(usage)
            captured_usage["usage"] = merged


def _frame_error(error_payload: dict[str, Any]) -> bytes:
    """Frame an error payload as SSE data.

    WHY: many OpenAI-compatible clients block until they see [DONE]; an
    error frame alone leaves them waiting until the read timeout fires.
    """
    return f"data: {json.dumps(error_payload)}\n\ndata: [DONE]\n\n".encode()


# The two wires, built here (after their helpers) so every reference is a
# direct name and OPENAI_STREAM can serve as a default argument below.
OPENAI_STREAM = StreamWire(
    chunk_may_carry_usage=_openai_chunk_may_carry_usage,
    capture_usage=_capture_usage_from_chunk,
    record_usage=RequestStats.set_usage,
    frame_error=_frame_error,
    remap_reasoning=True,
)

ANTHROPIC_STREAM = StreamWire(
    chunk_may_carry_usage=_anthropic_chunk_may_carry_usage,
    capture_usage=_capture_anthropic_usage_from_chunk,
    record_usage=RequestStats.set_anthropic_usage,
    frame_error=_frame_anthropic_error,
    remap_reasoning=False,
)


def duplicate_reasoning_field(data: Any) -> bool:
    """Duplicate OpenAI-style 'reasoning' into llama.cpp-style 'reasoning_content'.

    vLLM (and some other backends) stream thinking into delta/message.reasoning,
    while clients in this lab were built against llama.cpp, which emits
    reasoning_content. Duplicating keeps both client styles working; the
    original field is left intact. Returns True when the data was modified.
    """
    if not isinstance(data, dict):
        return False
    choices = data.get("choices")
    if not isinstance(choices, list):
        return False
    changed = False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        for holder in (choice.get("message"), choice.get("delta")):
            if (isinstance(holder, dict) and "reasoning" in holder
                    and "reasoning_content" not in holder):
                holder["reasoning_content"] = holder["reasoning"]
                changed = True
    return changed


async def open_provider_stream(
        provider_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[bytes, None]:
    """Pull the first chunk eagerly, then return an equivalent stream.

    # WHY: a provider generator does nothing until it is iterated, and iteration
    # starts inside StreamingResponse — after the 200 and the response headers
    # have already gone out. An upstream 401/429 then reached the client as a
    # 200 carrying an SSE error frame, losing the status code entirely.
    #
    # Priming moves the first read (which is where upstream errors surface) in
    # front of the response, so the HTTPException propagates through the normal
    # error handler and the client gets the real status. If the first read
    # raises, the generator is already finished, so its cleanup — the in-flight
    # count and the concurrency slot — has run.
    """
    try:
        first_chunk = await provider_stream.__anext__()
    except StopAsyncIteration:
        return _empty_stream()
    return _prepend(first_chunk, provider_stream)


async def _empty_stream() -> AsyncGenerator[bytes, None]:
    """An already-exhausted byte stream."""
    return
    yield  # pragma: no cover - makes this an async generator


async def _prepend(first_chunk: bytes,
                   rest: AsyncGenerator[bytes, None]) -> AsyncGenerator[bytes, None]:
    """Re-attach an already-consumed first chunk to the front of a stream.

    # INVARIANT: closing this wrapper must close the wrapped stream. Without the
    # finally, a client disconnect throws GeneratorExit in here and leaves `rest`
    # suspended, so the provider's in-flight count and concurrency slot are never
    # released — and a config reload would then wait out the full drain timeout.
    """
    try:
        yield first_chunk
        async for chunk in rest:
            yield chunk
    finally:
        await rest.aclose()


async def process_stream(provider_stream: AsyncGenerator[bytes, None],
                         model_id: str,
                         request_id: str,
                         user_id: str,
                         provider_name: str,
                         stats: RequestStats,
                         wire: StreamWire = OPENAI_STREAM) -> AsyncGenerator[bytes, None]:
    """Forward a provider SSE stream byte-for-byte (logging, usage capture,
    error framing) for one wire (OPENAI_STREAM by default, ANTHROPIC_STREAM
    for the Messages protocol).

    Holds no state across calls: captured usage lives in a per-stream holder,
    so concurrent streams never affect each other.

    The per-request RequestStats holder is enriched in place: mid-stream
    failures write error_code/error_message from the same payload as the
    wire's error frame (so the frame and the row cannot drift), and the
    captured usage is recorded in a ``finally`` so the error path and a
    client disconnect keep partial usage.
    """
    captured_usage: dict[str, Any] = {}
    chunk_stats = _StreamStats()
    ctx = _StreamContext(request_id, user_id, model_id, provider_name, stats)
    start_time = time.time()

    logger.info("Starting stream processing", extra={
        "request_id": request_id,
        "user_id": user_id,
        "model": model_id
    })

    try:
        async for chunk in _passthrough(provider_stream, captured_usage, chunk_stats, ctx, wire):
            yield chunk
    except Exception as e:
        yield _stream_failure(e, ctx, chunk_stats, start_time, wire)
        return
    finally:
        if captured_usage.get("usage"):
            wire.record_usage(stats, captured_usage["usage"])

    logger.info("Stream completed", extra={
            "request_id": request_id,
            "duration": round(time.time() - start_time, 3),
            "total_bytes": chunk_stats.bytes
        })


def _stream_failure(e: Exception, ctx: "_StreamContext", chunk_stats: "_StreamStats",
                    start_time: float, wire: StreamWire) -> bytes:
    """Log a mid-stream failure, record it on the stats row, return its SSE frame."""
    logger.error("Stream processing failed", extra={
        "request_id": ctx.request_id,
        "user_id": ctx.user_id,
        "model": ctx.model_id,
        "stream_processing": {
            "duration_seconds": time.time() - start_time,
            "chunks_processed": chunk_stats.chunks,
            "error": str(e),
            "error_type": type(e).__name__
        }
    }, exc_info=True)
    error_payload = _error_payload(e)
    # The ONE envelope extractor (core/error_handling/envelope.py),
    # shared with the HTTP exception handler: the row the dashboard
    # sees and the frame the client sees cannot drift. Enrichment reads
    # the OpenRouter envelope on BOTH wires; only the frame is wire-shaped.
    # overwrite=True: this payload is the terminal error for the
    # request, so it owns error_code/error_message outright.
    enrich_stats_from_envelope(
        ctx.req_stats, error_payload,
        default_error_code="internal_server_error", overwrite=True,
    )
    return wire.frame_error(error_payload)


async def _passthrough(provider_stream: AsyncGenerator[bytes, None],
                       captured_usage: dict[str, Any],
                       stats: "_StreamStats",
                       ctx: "_StreamContext",
                       wire: StreamWire = OPENAI_STREAM) -> AsyncGenerator[bytes, None]:
    """Forward chunks unchanged, only peeking for reasoning fields, usage
    and upstream error frames.

    Every peek is gated on a raw byte substring test, so a chunk that
    carries none of them costs one scan and no parsing.
    """
    is_debug = logger.is_debug_enabled()
    async for chunk in provider_stream:
        stats.chunks += 1
        stats.bytes += len(chunk)

        if is_debug:
            preview = chunk.decode('utf-8', errors='replace')[:200].replace('\n', '\\n')
            logger.debug(f"Chunk {stats.chunks} ({len(chunk)}B): {preview}", request_id=ctx.request_id)

        if wire.remap_reasoning and b'"reasoning"' in chunk:
            chunk = _remap_reasoning_in_chunk(chunk)

        # INVARIANT: the chunk is yielded before the usage and error peeks and
        # is never rewritten by them.
        # Why: the router is transparent — the client must see the upstream's
        # own error text and framing; these peeks only log and record.
        yield chunk

        if wire.chunk_may_carry_usage(chunk):
            wire.capture_usage(chunk, captured_usage)

        if b'"error"' in chunk:
            _note_upstream_error(chunk, ctx)


@dataclass
class _StreamStats:
    """Per-stream counters used only for the completion and failure log lines."""
    chunks: int = 0
    bytes: int = 0


@dataclass(frozen=True)
class _StreamContext:
    """Per-stream identity the error peek logs and records against."""
    request_id: str
    user_id: str
    model_id: str
    provider_name: str
    req_stats: RequestStats


def _note_upstream_error(chunk: bytes, ctx: _StreamContext) -> None:
    """Log and record every complete ``data:`` line carrying a top-level error.

    An upstream (llama-server) can fail after the 200 by sending a
    ``{"error": {...}}`` frame and closing without ``[DONE]``; without this
    the request logged as completed and its usage row read as a success.
    Best-effort like the other peeks: a line split across chunks, or one that
    is not JSON, passes unlogged.
    """
    try:
        text = chunk.decode('utf-8')
    except UnicodeDecodeError:
        return
    for line in text.split('\n'):
        line = line.rstrip('\r')
        if not line.startswith('data: '):
            continue
        try:
            payload = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("error"), dict):
            continue
        code = payload["error"].get("code")
        log_provider_error(
            ctx.provider_name, line[:200],
            code if isinstance(code, int) and not isinstance(code, bool) else 0,
            request_id=ctx.request_id, user_id=ctx.user_id, model_id=ctx.model_id,
        )
        # overwrite=True: the upstream frame is the terminal error of the request.
        enrich_stats_from_envelope(
            ctx.req_stats, payload,
            default_error_code="provider_stream_error", overwrite=True,
        )


def _remap_reasoning_in_chunk(chunk: bytes) -> bytes:
    """Best-effort reasoning->reasoning_content duplication for a raw transparent-mode chunk.

    Only rewrites 'data: ' lines that parse as complete JSON objects; a line
    split across chunk boundaries fails to parse here and passes through
    unchanged (it will be handled on its completing chunk or, worst case,
    reach the client with the original field name only).
    """
    try:
        text = chunk.decode('utf-8')
    except UnicodeDecodeError:
        return chunk
    out_lines = []
    changed = False
    for line in text.split('\n'):
        stripped = line.rstrip('\r')
        if stripped.startswith('data: ') and '"reasoning"' in stripped:
            try:
                data = json.loads(stripped[6:])
            except json.JSONDecodeError:
                out_lines.append(line)
                continue
            if duplicate_reasoning_field(data):
                changed = True
                line = 'data: ' + json.dumps(data, ensure_ascii=False)
        out_lines.append(line)
    if not changed:
        return chunk
    return '\n'.join(out_lines).encode('utf-8')


def _error_payload(error: Exception) -> dict[str, Any]:
    """Build the OpenRouter-shaped error payload for a mid-stream failure.

    Single construction point shared by the SSE error frame and the stats
    enrichment, so the frame the client sees and the row the dashboard sees
    cannot drift.
    """
    if isinstance(error, HTTPException):
        error_detail = error.detail
        if isinstance(error_detail, dict) and "error" in error_detail:
            return error_detail
        return {
            "error": {
                "code": error.status_code,
                "message": str(error_detail) if error_detail else str(error)
            }
        }
    return {
        "error": {
            "code": 500,
            "message": f"An unexpected error occurred during streaming: {error}"
        }
    }
