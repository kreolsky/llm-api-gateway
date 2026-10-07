"""Unit tests for src/services/messages_service.py — the /v1/messages funnel.

The Messages path deliberately does NOT ride _prepare_dispatch: the effort
policy and dialect translation are OpenAI-wire body surgery and must never
touch a Messages body. What it shares with the other endpoints is exactly
_parse_json_request + _resolve_target(api="anthropic-messages").
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from src.core.config_schema import parse_config, parse_provider
from src.core.context import AuthContext, RequestContext
from src.core.usage_db import RequestStats
from src.services.messages_service import MessagesService


def _make_auth_context():
    return AuthContext(allowed_models=[], allowed_endpoints=[])


class _StubAnthropicProvider:
    """AnthropicProvider double: records dispatches, returns canned bodies."""

    identity = None
    entry = parse_provider({"type": "anthropic", "base_url": "https://upstream.invalid"})

    def __init__(self, message_body=None, stream_frames=None):
        self.messages_calls = []
        self.stream_calls = []
        self.message_body = message_body or {
            "id": "msg_1", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 4,
                      "cache_read_input_tokens": 5,
                      "cache_creation_input_tokens": 2},
        }
        self.stream_frames = stream_frames or []

    async def messages(self, request_body, provider_model_name, model_config,
                       request_id="unknown", extra_headers=None):
        self.messages_calls.append((dict(request_body), provider_model_name, extra_headers))
        return self.message_body

    def messages_stream(self, request_body, provider_model_name, model_config,
                        request_id="unknown", extra_headers=None):
        self.stream_calls.append((dict(request_body), provider_model_name, extra_headers))

        async def gen():
            for frame in self.stream_frames:
                yield frame

        return gen()


def _service(provider, models=None):
    registry = MagicMock()
    registry.get.return_value = provider
    cm = MagicMock()
    cm.get_config.return_value = parse_config({
        "models": models if models is not None else {
            "claude/flash": {"provider": "prov-a", "provider_model_name": "upstream-a"}},
        "providers": {"prov-a": {"type": "anthropic",
                                 "base_url": "https://upstream.invalid"}},
    })
    return MessagesService(cm, registry)


def _request(body, headers=None):
    request = MagicMock()
    request.state = SimpleNamespace(
        request_context=RequestContext(request_id="req-msg", project_name="proj"),
        request_stats=RequestStats(endpoint="messages"),
    )
    request.json = AsyncMock(return_value=body)
    request.headers = dict(headers or {})
    return request


_ANNOTATE_BODY = {
    "model": "claude/flash", "max_tokens": 300,
    "messages": [{"role": "user", "content": "Say hi"}],
    "thinking": {"type": "enabled", "budget_tokens": 1024},
}


class TestNonStream:

    @pytest.mark.asyncio
    async def test_body_untouched_under_an_effort_policy(self):
        """A model carrying a reasoning_effort block (advertisement only for
        anthropic-type models) must not have anything injected: the effort
        policy lives in the OpenAI-wire funnel this path does not ride."""
        models = {"claude/flash": {
            "provider": "prov-a", "provider_model_name": "upstream-a",
            "reasoning_effort": {"allowed": ["low", "high", "max"],
                                 "param": "reasoning_effort"}}}
        provider = _StubAnthropicProvider()
        request = _request(dict(_ANNOTATE_BODY))

        await _service(provider, models).messages(request, _make_auth_context())

        sent = provider.messages_calls[0][0]
        assert sent["thinking"] == {"type": "enabled", "budget_tokens": 1024}
        assert "reasoning_effort" not in sent
        assert "reasoning" not in sent
        # only `model` is rewritten (by the provider's _apply_model_config)
        assert sent["model"] == "claude/flash"

    @pytest.mark.asyncio
    async def test_response_body_passed_through_as_is(self):
        provider = _StubAnthropicProvider()
        request = _request(dict(_ANNOTATE_BODY))

        response = await _service(provider).messages(request, _make_auth_context())

        assert isinstance(response, JSONResponse)
        assert json.loads(response.body) == provider.message_body

    @pytest.mark.asyncio
    async def test_usage_recorded_with_cache_read_and_write(self):
        provider = _StubAnthropicProvider()
        request = _request(dict(_ANNOTATE_BODY))

        await _service(provider).messages(request, _make_auth_context())

        stats = request.state.request_stats
        assert stats.has_usage is True
        assert stats.prompt_tokens == 10 + 5 + 2
        assert stats.cached_tokens == 5
        assert stats.cache_write_tokens == 2
        assert stats.completion_tokens == 4
        assert stats.stream is False
        assert stats.model_id == "claude/flash"
        assert stats.provider_name == "prov-a"

    @pytest.mark.asyncio
    async def test_no_usage_block_leaves_stats_empty(self):
        provider = _StubAnthropicProvider(message_body={"type": "message", "content": []})
        request = _request(dict(_ANNOTATE_BODY))

        await _service(provider).messages(request, _make_auth_context())

        assert request.state.request_stats.has_usage is False


class TestProtocolHeaders:

    @pytest.mark.asyncio
    async def test_anthropic_headers_forwarded_without_passthrough(self):
        """anthropic-version / anthropic-beta are protocol, not identity:
        they ride upstream on this path even when the provider has no
        identity profile."""
        provider = _StubAnthropicProvider()
        request = _request(dict(_ANNOTATE_BODY), headers={
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "interleaved-thinking-2025-05-14",
            "user-agent": "pi-ai/1.0",
        })

        await _service(provider).messages(request, _make_auth_context())

        extra = provider.messages_calls[0][2]
        assert extra["anthropic-version"] == "2023-06-01"
        assert extra["anthropic-beta"] == "interleaved-thinking-2025-05-14"
        # non-protocol client headers stay behind without identity passthrough
        assert "user-agent" not in extra

    @pytest.mark.asyncio
    async def test_protocol_headers_merge_into_identity_headers(self):
        """With identity passthrough the protocol headers join the SAME
        per-request dict (one merge object feeds both branches)."""
        provider = _StubAnthropicProvider()
        provider.identity = "passthrough"
        request = _request(dict(_ANNOTATE_BODY), headers={
            "anthropic-beta": "context-1m-2025-08-07",
            "user-agent": "pi-ai/1.0",
            "x-custom": "yes",
        })

        await _service(provider).messages(request, _make_auth_context())

        extra = provider.messages_calls[0][2]
        assert extra == {"user-agent": "pi-ai/1.0", "x-custom": "yes",
                         "anthropic-beta": "context-1m-2025-08-07"}

    @pytest.mark.asyncio
    async def test_client_credential_headers_never_forwarded(self):
        """x-api-key from the client is a credential: dropped here (it is on
        the passthrough denylist anyway), the provider's own key goes up."""
        provider = _StubAnthropicProvider()
        provider.identity = "passthrough"
        request = _request(dict(_ANNOTATE_BODY), headers={
            "x-api-key": "client-key", "authorization": "Bearer nnp-v1-x"})

        await _service(provider).messages(request, _make_auth_context())

        extra = provider.messages_calls[0][2]
        assert extra is None or "x-api-key" not in extra


class TestStream:

    FRAMES = [
        b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":3,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n',
        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":3,"output_tokens":2,"cache_creation_input_tokens":7,"cache_read_input_tokens":1}}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]

    @pytest.mark.asyncio
    async def test_stream_passthrough_records_usage(self):
        provider = _StubAnthropicProvider(stream_frames=self.FRAMES)
        request = _request({**_ANNOTATE_BODY, "stream": True})

        response = await _service(provider).messages(request, _make_auth_context())

        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"
        chunks = [chunk async for chunk in response.body_iterator]
        assert b"".join(chunks) == b"".join(self.FRAMES)

        stats = request.state.request_stats
        assert stats.stream is True
        assert stats.has_usage is True
        assert stats.prompt_tokens == 3 + 1 + 7
        assert stats.cache_write_tokens == 7
        assert stats.completion_tokens == 2

    @pytest.mark.asyncio
    async def test_stream_headers_match_chat_service(self):
        provider = _StubAnthropicProvider(stream_frames=self.FRAMES)
        request = _request({**_ANNOTATE_BODY, "stream": True})

        response = await _service(provider).messages(request, _make_auth_context())

        assert response.headers["x-accel-buffering"] == "no"
        assert response.headers["cache-control"] == "no-cache"


class TestRefusals:

    @pytest.mark.asyncio
    async def test_wrong_protocol_model_is_wrong_api(self):
        """An openai-type model requested over /v1/messages -> 400 wrong_api
        (the funnel's gate, shared by construction)."""
        provider = SimpleNamespace(entry=parse_provider({"type": "openai"}))
        registry = MagicMock()
        registry.get.return_value = provider
        cm = MagicMock()
        cm.get_config.return_value = parse_config({
            "models": {"chat/m": {"provider": "prov-openai"}},
            "providers": {"prov-openai": {"type": "openai"}},
        })
        service = MessagesService(cm, registry)
        request = _request({"model": "chat/m", "max_tokens": 1, "messages": []})

        with pytest.raises(HTTPException) as exc_info:
            await service.messages(request, _make_auth_context())
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"]["metadata"]["error_code"] == "wrong_api"
        assert "/v1/chat/completions" in exc_info.value.detail["error"]["message"]


# ---------------------------------------------------------------------------
# Error shape at the HTTP boundary: Anthropic on /v1/messages, OpenRouter
# everywhere else — driven through the REAL handlers (like test_auth).
# ---------------------------------------------------------------------------

import httpx  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from src.api.main import custom_http_exception_handler  # noqa: E402
from src.core.error_handling import (
    ErrorType,  # noqa: E402
    create_error,  # noqa: E402
)


def _error_app(path: str) -> FastAPI:
    """App raising a router-made error at <path> through the real handler."""
    application = FastAPI()
    application.add_exception_handler(HTTPException, custom_http_exception_handler)

    @application.post(path)
    async def boom():
        raise create_error(ErrorType.MODEL_NOT_ALLOWED, model_id="claude/flash")

    return application


class TestErrorShapeAtTheBoundary:

    async def _post(self, app, path):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(path, json={})

    @pytest.mark.asyncio
    async def test_messages_route_renders_anthropic_shape(self):
        resp = await self._post(_error_app("/v1/messages"), "/v1/messages")
        assert resp.status_code == 403
        assert resp.json() == {"type": "error",
                               "error": {"type": "permission_error",
                                         "message": "Model 'claude/flash' is not available for your account"}}

    @pytest.mark.asyncio
    async def test_chat_route_keeps_openrouter_shape(self):
        resp = await self._post(_error_app("/v1/chat/completions"), "/v1/chat/completions")
        assert resp.status_code == 403
        envelope = resp.json()
        assert envelope["error"]["code"] == 403
        assert envelope["error"]["metadata"]["error_code"] == "model_not_allowed"

    @pytest.mark.asyncio
    async def test_unhandled_500_on_messages_is_anthropic(self):
        """The unhandled-exception handler also renders Anthropic shape on
        /v1/messages. Driven through raw ASGI: ServerErrorMiddleware
        re-raises after responding and httpx would surface that re-raise
        instead of the sent response."""
        from src.api.main import unhandled_exception_handler

        app = FastAPI()
        app.add_exception_handler(Exception, unhandled_exception_handler)

        @app.post("/v1/messages")
        async def boom():
            raise ValueError("kaboom")

        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1", "method": "POST", "scheme": "http",
            "path": "/v1/messages", "raw_path": b"/v1/messages",
            "query_string": b"", "root_path": "",
            "headers": [(b"host", b"testserver")],
            "client": ("127.0.0.1", 123), "server": ("testserver", 80), "state": {},
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        sent: list[dict] = []

        async def send(message):
            sent.append(message)

        raised = None
        try:
            await app(scope, receive, send)
        except Exception as e:  # noqa: BLE001 — ServerErrorMiddleware re-raises
            raised = e

        assert isinstance(raised, ValueError)  # Starlette re-raised after responding
        status = next(m["status"] for m in sent if m["type"] == "http.response.start")
        body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
        assert status == 500
        assert json.loads(body) == {"type": "error",
                                    "error": {"type": "api_error",
                                              "message": "Internal server error"}}
