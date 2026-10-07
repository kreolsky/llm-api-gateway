"""Unit tests for src/providers/anthropic.py — AnthropicProvider.

The Anthropic-protocol sibling of Provider: same pool, retry loop and header
merge, but POSTs {base_url}/v1/messages with x-api-key + anthropic-version and
no Authorization.
"""

import functools
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from src.core.config_manager import Settings
from src.core.config_schema import ModelEntry, parse_provider
from src.providers.anthropic import AnthropicProvider

AnthropicStub = functools.partial(AnthropicProvider, provider_name="astub")

NO_OPTIONS = ModelEntry(provider=None)


def _anthropic_entry(**extra):
    return parse_provider({"type": "anthropic", "base_url": "https://api.anthropic.com",
                           "api_key_env": "TEST_API_KEY", **extra})


def _build_provider(env_vars=None, settings=None, **extra):
    env = {"TEST_API_KEY": "sk-ant-test"}
    if env_vars is not None:
        env.update(env_vars)
    with patch.dict("os.environ", env, clear=False):
        return AnthropicStub(_anthropic_entry(**extra), settings=settings or Settings())


class TestStaticHeaders:

    def test_x_api_key_and_version_set_no_authorization(self):
        provider = _build_provider()
        assert provider.headers["x-api-key"] == "sk-ant-test"
        assert provider.headers["anthropic-version"] == "2023-06-01"
        assert provider.headers["Content-Type"] == "application/json"
        assert "Authorization" not in provider.headers

    def test_missing_api_key_env_var_raises(self):
        with patch.dict("os.environ", {}, clear=True), \
             pytest.raises(HTTPException) as exc_info:
            AnthropicStub(_anthropic_entry(), Settings())
        assert exc_info.value.status_code == 500

    def test_no_api_key_env_no_credential_header(self):
        entry = parse_provider({"type": "anthropic", "base_url": "https://api.anthropic.com"})
        provider = AnthropicStub(entry, Settings())
        assert "x-api-key" not in provider.headers
        # the protocol version header is not a credential — always present
        assert provider.headers["anthropic-version"] == "2023-06-01"

    @pytest.mark.parametrize("name", ["anthropic-version", "Anthropic-Version"])
    def test_static_version_replaces_default_in_any_casing(self, name):
        """One version header goes upstream: the operator's, never both."""
        provider = _build_provider(headers={name: "2024-10-22"})
        versions = {k: v for k, v in provider.headers.items()
                    if k.lower() == "anthropic-version"}
        assert versions == {name: "2024-10-22"}

    def test_operator_headers_kept(self):
        provider = _build_provider(headers={"X-Title": "lore"})
        assert provider.headers["X-Title"] == "lore"


class TestMessages:

    @pytest.mark.asyncio
    async def test_messages_posts_to_v1_messages(self):
        provider = _build_provider()
        provider._request = AsyncMock(return_value={"type": "message"})
        body = {"model": "claude-x", "max_tokens": 10, "messages": []}

        result = await provider.messages(body, "claude-real", NO_OPTIONS, request_id="r1")

        assert result == {"type": "message"}
        call = provider._request.call_args
        assert call.kwargs["method"] == "POST"
        assert call.kwargs["path"] == "/v1/messages"
        assert call.kwargs["request_body"]["model"] == "claude-real"
        # the body is forwarded with only `model` rewritten
        assert call.kwargs["request_body"]["max_tokens"] == 10
        assert call.kwargs["extra_headers"] is None

    @pytest.mark.asyncio
    async def test_message_options_deep_merged(self):
        """models.yaml options (Anthropic-wire: thinking, cache_control) merge
        into the outgoing body, with the stream guard of _apply_model_config."""
        provider = _build_provider()
        provider._request = AsyncMock(return_value={"type": "message"})
        model_config = ModelEntry(provider=None, options={
            "thinking": {"type": "enabled", "budget_tokens": 2048}})

        await provider.messages({"max_tokens": 10}, "m", model_config, request_id="r1")

        sent = provider._request.call_args.kwargs["request_body"]
        assert sent["thinking"] == {"type": "enabled", "budget_tokens": 2048}

    @pytest.mark.asyncio
    async def test_message_options_cannot_override_stream(self):
        provider = _build_provider()
        provider._request = AsyncMock(return_value={"type": "message"})
        model_config = ModelEntry(provider=None, options={"stream": True})

        await provider.messages({"max_tokens": 10}, "m", model_config, request_id="r1")

        assert "stream" not in provider._request.call_args.kwargs["request_body"]

    @pytest.mark.asyncio
    async def test_messages_stream_uses_stream_request(self):
        provider = _build_provider()

        async def fake_stream(path, body, request_id="unknown", extra_headers=None):
            assert path == "/v1/messages"
            assert body["model"] == "claude-real"
            yield b"event: message_start\n\n"

        provider._stream_request = fake_stream
        gen = provider.messages_stream({"model": "claude-x"}, "claude-real", NO_OPTIONS,
                                       request_id="r1")
        chunks = [c async for c in gen]
        assert chunks == [b"event: message_start\n\n"]

    @pytest.mark.asyncio
    async def test_429_retried(self):
        """messages() rides _request, so the 429 backoff loop covers it."""
        provider = _build_provider(settings=Settings(provider_max_retries=1,
                                                     provider_retry_base_delay=0.001,
                                                     provider_retry_max_delay=0.01))
        calls = []

        async def post(*a, **k):
            calls.append(1)
            raise HTTPException(status_code=429, detail="rate limited")

        provider.pool.client.post = post
        with patch("src.providers.base.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(HTTPException) as exc_info:
                await provider.messages({"max_tokens": 1}, "m", NO_OPTIONS, request_id="r1")
        assert exc_info.value.status_code == 429
        assert len(calls) == 2


class TestHeaderMerge:

    def test_extra_headers_cannot_replace_x_api_key(self):
        provider = _build_provider()
        merged = provider._merge_request_headers({"x-api-key": "attacker",
                                                  "X-Api-Key": "attacker2"})
        assert merged["x-api-key"] == "sk-ant-test"

    def test_client_anthropic_version_replaces_the_default(self):
        provider = _build_provider()
        merged = provider._merge_request_headers({"anthropic-version": "2024-01-01"})
        assert merged["anthropic-version"] == "2024-01-01"

    def test_other_extra_headers_ride_along(self):
        provider = _build_provider()
        merged = provider._merge_request_headers({"anthropic-beta": "prompt-caching-2024-07-31"})
        assert merged["anthropic-beta"] == "prompt-caching-2024-07-31"
        assert merged["x-api-key"] == "sk-ant-test"


class TestListModels:

    @pytest.mark.asyncio
    async def test_list_models_gets_v1_models(self):
        """Anthropic serves GET /v1/models; a 404 upstream (DeepSeek) lands in
        the stale-if-error path of the capabilities refresh, not here."""
        provider = _build_provider()
        provider._request = AsyncMock(return_value={"data": []})
        result = await provider.list_models(request_id="r1")
        provider._request.assert_called_once_with(
            method="GET", path="/v1/models", request_id="r1")
        assert result == {"data": []}
