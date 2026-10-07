"""Unit tests for src/providers/base.py — Provider and the inline retry loop."""

import asyncio
import functools
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from src.core.config_manager import Settings
from src.core.config_schema import (
    DEFAULT_REASONING_DIALECT,
    ConfigError,
    ModelEntry,
    ProviderEntry,
    parse_provider,
)
from src.providers.base import Provider

# ---------------------------------------------------------------------------
# The provider under its test name ("stub" shows up in error metadata)
# ---------------------------------------------------------------------------

ProviderStub = functools.partial(Provider, provider_name="stub")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NO_OPTIONS = ModelEntry(provider=None)


def _entry(config):
    """Parse a providers.yaml-shaped dict (type defaulted) into a ProviderEntry."""
    return parse_provider({"type": "openai", **config})


def _make_config(base_url="https://api.example.com", api_key_env="TEST_API_KEY", **extra):
    cfg = {"base_url": base_url, "api_key_env": api_key_env, **extra}
    return cfg


def _build_provider(base_url="https://api.example.com", api_key_env="TEST_API_KEY",
                     env_vars=None, settings=None, headers=None, proxy=None):
    """Build a ProviderStub with mocked env.

    The provider owns its own httpx.AsyncClient (built in __init__). Settings
    default to a bare Settings() — the single source of no-config defaults.
    """
    config = {"base_url": base_url}
    if api_key_env is not None:
        config["api_key_env"] = api_key_env
    if headers is not None:
        config["headers"] = headers
    if proxy is not None:
        config["proxy"] = proxy

    env = {"TEST_API_KEY": "sk-test-123"}
    if env_vars is not None:
        env.update(env_vars)

    with patch.dict("os.environ", env, clear=False):
        provider = ProviderStub(_entry(config), settings=settings or Settings())
    return provider


def _make_settings(**overrides) -> Settings:
    """Build a Settings with overrides applied over the single defaults source."""
    return Settings(**overrides)


def _build_limited_provider(max_concurrent, settings=None):
    """Build a ProviderStub with max_concurrent set.

    The entry is built directly, not via parse_provider: the schema now
    refuses a non-positive/non-int max_concurrent (ConfigError), while the
    pool keeps its defensive check for direct construction — pinned by
    test_max_concurrent_non_positive_disables_limit below.
    """
    entry = ProviderEntry(
        type="openai", base_url="https://api.example.com",
        api_key_env="TEST_API_KEY", max_concurrent=max_concurrent)
    with patch.dict("os.environ", {"TEST_API_KEY": "sk-test-123"}, clear=False):
        return ProviderStub(entry, settings=settings or Settings())


def _mock_response(json_body=None):
    """Build a mock httpx Response usable by _request."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock(return_value=None)
    resp.json.return_value = json_body if json_body is not None else {"ok": True}
    resp.text = ""
    return resp


# ===================================================================
# Inline 429 retry loop (_request)
# ===================================================================

class TestRetryLoop:
    """429 backoff loop inlined in _request, bounds from settings."""

    def _provider(self, **settings_overrides):
        provider = _build_provider(settings=_make_settings(**settings_overrides))
        return provider

    @staticmethod
    def _rate_limited_post(status_code=429):
        def post(*a, **k):
            raise HTTPException(status_code=status_code, detail="rate limited")
        return post

    @pytest.mark.asyncio
    async def test_successful_first_try_no_retries(self):
        """Successful call on first try — no sleeps, one HTTP call."""
        provider = self._provider()
        provider.pool.client.post = AsyncMock(return_value=_mock_response())
        with patch("src.providers.base.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await provider._request("POST", "/x", request_id="r1")
        assert result == {"ok": True}
        provider.pool.client.post.assert_awaited_once()
        mock_sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_429_retried_up_to_max_then_raised(self):
        """429 error retried up to max_retries, then re-raised."""
        provider = self._provider(provider_max_retries=2,
                                  provider_retry_base_delay=0.001,
                                  provider_retry_max_delay=0.01)
        calls = []

        async def post(*a, **k):
            calls.append(1)
            raise HTTPException(status_code=429, detail="rate limited")

        provider.pool.client.post = post
        with patch("src.providers.base.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(HTTPException) as exc_info:
                await provider._request("POST", "/x", request_id="r1")
        assert exc_info.value.status_code == 429
        # initial attempt + 2 retries = 3 calls
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_non_429_error_raised_immediately(self):
        """Non-429 error raised immediately (no retry)."""
        provider = self._provider(provider_max_retries=3)
        calls = []

        async def post(*a, **k):
            calls.append(1)
            raise HTTPException(status_code=500, detail="server error")

        provider.pool.client.post = post
        with patch("src.providers.base.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(HTTPException) as exc_info:
                await provider._request("POST", "/x", request_id="r1")
        assert exc_info.value.status_code == 500
        assert len(calls) == 1
        mock_sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exponential_backoff_formula(self):
        """Backoff delay = min(base * 2^attempt, max)."""
        provider = self._provider(provider_max_retries=4, provider_retry_base_delay=1.0,
                                  provider_retry_max_delay=10.0)
        recorded_delays = []

        async def mock_sleep(delay):
            recorded_delays.append(delay)

        provider.pool.client.post = self._rate_limited_post()
        with patch("src.providers.base.asyncio.sleep", side_effect=mock_sleep), \
             pytest.raises(HTTPException):
            await provider._request("POST", "/x", request_id="r1")

        # attempts 0..3 → delays: min(1*2^0,10)=1, min(1*2^1,10)=2, min(1*2^2,10)=4, min(1*2^3,10)=8
        assert recorded_delays == [1.0, 2.0, 4.0, 8.0]

    @pytest.mark.asyncio
    async def test_bounds_come_from_settings(self):
        """provider_max_retries=1 in settings → exactly one retry."""
        provider = self._provider(provider_max_retries=1,
                                  provider_retry_base_delay=0.001,
                                  provider_retry_max_delay=0.01)
        calls = []

        async def post(*a, **k):
            calls.append(1)
            raise HTTPException(status_code=429, detail="rate limited")

        provider.pool.client.post = post
        with patch("src.providers.base.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(HTTPException):
                await provider._request("POST", "/x", request_id="r1")
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_defaults_from_bare_settings(self):
        """A bare Settings() gives the documented 3 retries."""
        provider = self._provider()
        calls = []

        async def post(*a, **k):
            calls.append(1)
            raise HTTPException(status_code=429, detail="rate limited")

        provider.pool.client.post = post
        with patch("src.providers.base.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(HTTPException):
                await provider._request("POST", "/x", request_id="r1")
        # default max_retries=3 → 1 initial + 3 retries = 4
        assert len(calls) == 4

    def test_retry_decorator_is_gone(self):
        """retry_on_rate_limit no longer exists — the loop is inlined."""
        import src.providers.base as base_module
        assert not hasattr(base_module, "retry_on_rate_limit")


# ===================================================================
# Provider.__init__
# ===================================================================

class TestProviderInit:

    def test_missing_base_url_raises(self):
        """Missing base_url raises HTTPException."""
        config = {"api_key_env": "TEST_API_KEY"}
        with patch.dict("os.environ", {"TEST_API_KEY": "sk-123"}, clear=False):
            with pytest.raises(HTTPException) as exc_info:
                ProviderStub(_entry(config), Settings())
        assert exc_info.value.status_code == 500

    def test_missing_api_key_env_var_raises(self):
        """Missing API key env var raises HTTPException."""
        config = {"base_url": "https://api.example.com", "api_key_env": "MISSING_KEY"}
        with patch.dict("os.environ", {}, clear=True), pytest.raises(HTTPException) as exc_info:
            ProviderStub(_entry(config), Settings())
        assert exc_info.value.status_code == 500

    def test_no_api_key_env_no_error(self):
        """No api_key_env in config means no Authorization header, no error."""
        config = {"base_url": "https://api.example.com"}
        provider = ProviderStub(_entry(config), Settings())
        assert "Authorization" not in provider.headers
        assert provider.api_key is None

    def test_successful_init(self):
        """Successful init sets headers, provider_name, base_url."""
        provider = _build_provider()
        assert provider.base_url == "https://api.example.com"
        assert provider.headers["Authorization"] == "Bearer sk-test-123"
        assert provider.headers["Content-Type"] == "application/json"
        assert provider.provider_name == "stub"

    def test_provider_name_is_required(self):
        """No class-name fallback: omitting provider_name is a TypeError."""
        with pytest.raises(TypeError):
            Provider(_entry(_make_config()), Settings())


# ===================================================================
# Typed settings argument
# ===================================================================

class TestTypedSettingsArgument:
    """The provider takes a required, typed Settings — not a duck-typed manager."""

    def test_settings_is_required(self):
        """Constructing without settings is a TypeError, not a silent fallback."""
        config = _make_config()
        with patch.dict("os.environ", {"TEST_API_KEY": "sk-test-123"}, clear=False):
            with pytest.raises(TypeError):
                ProviderStub(config)

    def test_settings_stored_verbatim(self):
        """The exact Settings object passed in is what the provider reads."""
        settings = Settings(stream_read_timeout=123.0)
        provider = _build_provider(settings=settings)
        assert provider.settings is settings

    def test_client_limits_come_from_settings(self):
        """_build_client reads pool limits/timeouts off Settings."""
        settings = Settings(httpx_max_connections=42, httpx_max_keepalive_connections=7,
                            httpx_connect_timeout=12.0, httpx_read_timeout=33.0,
                            httpx_pool_timeout=4.0)
        provider = _build_provider(settings=settings)
        assert provider.pool.client.timeout.connect == 12.0
        assert provider.pool.client.timeout.read == 33.0
        assert provider.pool.client.timeout.pool == 4.0

    def test_no_config_manager_attribute(self):
        """The duck-typed config_manager attribute is gone."""
        provider = _build_provider()
        assert not hasattr(provider, "config_manager")
        assert hasattr(provider, "settings")


# ===================================================================
# _apply_model_config
# ===================================================================

class TestApplyModelConfig:

    def test_sets_model_name(self):
        """Sets model name in request body."""
        provider = _build_provider()
        body = {"messages": []}
        model_config = NO_OPTIONS
        result = provider._apply_model_config(body, "gpt-4", model_config)
        assert result["model"] == "gpt-4"

    def test_merges_options(self):
        """Merges options via deep_merge when present."""
        provider = _build_provider()
        body = {"messages": [], "temperature": 0.5}
        model_config = ModelEntry(provider=None, options={"temperature": 0.9, "top_p": 0.8})
        result = provider._apply_model_config(body, "gpt-4", model_config)
        assert result["model"] == "gpt-4"
        # deep_merge: options override existing keys
        assert result["temperature"] == 0.9
        assert result["top_p"] == 0.8

    def test_options_cannot_override_stream(self):
        """`stream` in options is dropped: the service already branched on the client's value."""
        provider = _build_provider()
        body = {"messages": [], "stream": False}
        model_config = ModelEntry(provider=None, options={"stream": True, "temperature": 0.9})
        result = provider._apply_model_config(body, "gpt-4", model_config)
        assert result["stream"] is False
        assert result["temperature"] == 0.9

    def test_options_stream_absent_from_body_stays_absent(self):
        """Options may not INTRODUCE stream either — an absent key means non-stream."""
        provider = _build_provider()
        body = {"messages": []}
        model_config = ModelEntry(provider=None, options={"stream": True})
        result = provider._apply_model_config(body, "gpt-4", model_config)
        assert "stream" not in result

    def test_options_are_not_mutated_by_the_stream_guard(self):
        """The guard copies: dropping stream must not edit the live config dict."""
        provider = _build_provider()
        options = {"stream": True, "top_p": 0.8}
        model_config = ModelEntry(provider=None, options=options)
        provider._apply_model_config({"messages": []}, "gpt-4", model_config)
        assert options == {"stream": True, "top_p": 0.8}

    def test_no_options_no_merge(self):
        """No options in model_config means no merge, body unchanged except model."""
        provider = _build_provider()
        body = {"messages": [], "temperature": 0.5}
        model_config = NO_OPTIONS
        result = provider._apply_model_config(body, "gpt-4", model_config)
        assert result == {"messages": [], "temperature": 0.5, "model": "gpt-4"}


# ===================================================================
# _raise_provider_http_error
# ===================================================================

class TestRaiseProviderHttpError:
    # INVARIANT: patch log_provider_error where it is CALLED (error_handler), not where
    # base.py once imported it.
    # Why: base.py only calls create_provider_http_error, which logs internally — the old
    # `src.providers.base.log_provider_error` target was inert, and these tests passed
    # without ever suppressing the log until ruff deleted the unused import.

    def _make_http_status_error(self, status_code, json_body=None, text_body="error text"):
        """Helper to build a mock httpx.HTTPStatusError."""
        response = MagicMock(spec=httpx.Response)
        response.status_code = status_code
        response.text = text_body
        if json_body is not None:
            response.json.return_value = json_body
        else:
            response.json.side_effect = json.JSONDecodeError("", "", 0)
        request = MagicMock(spec=httpx.Request)
        error = httpx.HTTPStatusError("error", request=request, response=response)
        return error

    @patch("src.core.error_handling.error_handler.log_provider_error")
    def test_extracts_nested_error_message(self, mock_log):
        """Extracts error message from JSON {"error": {"message": ...}}."""
        provider = _build_provider()
        err = self._make_http_status_error(
            400, json_body={"error": {"message": "bad request details"}}
        )
        with pytest.raises(HTTPException) as exc_info:
            provider._raise_provider_http_error(err, "req-1")
        assert exc_info.value.detail["error"]["message"] == "bad request details"
        assert exc_info.value.status_code == 400

    @patch("src.core.error_handling.error_handler.log_provider_error")
    def test_extracts_flat_message(self, mock_log):
        """Extracts error message from {"message": ...}."""
        provider = _build_provider()
        err = self._make_http_status_error(
            422, json_body={"message": "validation failed"}
        )
        with pytest.raises(HTTPException) as exc_info:
            provider._raise_provider_http_error(err, "req-1")
        assert exc_info.value.detail["error"]["message"] == "validation failed"

    @patch("src.core.error_handling.error_handler.log_provider_error")
    def test_falls_back_to_response_text(self, mock_log):
        """Falls back to response text when JSON parse fails."""
        provider = _build_provider()
        err = self._make_http_status_error(502, json_body=None, text_body="Bad Gateway")
        with pytest.raises(HTTPException) as exc_info:
            provider._raise_provider_http_error(err, "req-1")
        assert exc_info.value.detail["error"]["message"] == "Bad Gateway"

    @patch("src.core.error_handling.error_handler.log_provider_error")
    def test_raises_with_correct_detail_structure(self, mock_log):
        """HTTPException detail has correct structure with code, message, metadata."""
        provider = _build_provider()
        err = self._make_http_status_error(
            503, json_body={"error": {"message": "overloaded"}}
        )
        with pytest.raises(HTTPException) as exc_info:
            provider._raise_provider_http_error(err, "req-1")
        detail = exc_info.value.detail
        assert detail["error"]["code"] == 503
        assert detail["error"]["metadata"]["provider_name"] == "stub"
        assert "raw" in detail["error"]["metadata"]


# ===================================================================
# _create_timeout
# ===================================================================

class TestCreateTimeout:

    def test_returns_timeout_with_overrides(self):
        """Returns httpx.Timeout with specified overrides."""
        provider = _build_provider()
        timeout = provider._create_timeout(connect=1.0, read=2.0, write=3.0, pool=4.0)
        assert timeout.connect == 1.0
        assert timeout.read == 2.0
        assert timeout.write == 3.0
        assert timeout.pool == 4.0

    def test_inherits_connect_pool_from_client(self):
        """Inherits connect/pool from the owned client when not specified."""
        provider = _build_provider()
        client_timeout = provider.pool.client.timeout
        timeout = provider._create_timeout(read=99.0)
        assert timeout.connect == client_timeout.connect
        assert timeout.pool == client_timeout.pool
        assert timeout.read == 99.0
        assert timeout.write is None   # default when not specified


# ===================================================================
# Per-provider proxy (SOCKS5)
# ===================================================================

class TestProxySupport:
    """Tests for the per-provider `proxy` config key (SOCKS5)."""

    def test_provider_without_proxy_defaults_none(self):
        """A provider without the `proxy` key has self.proxy is None."""
        provider = _build_provider()
        assert provider.proxy is None
        assert isinstance(provider.pool.client, httpx.AsyncClient)
        assert not provider.pool.client.is_closed

    def test_provider_with_proxy_builds_client(self):
        """Provider with a socks5 proxy URL builds a client without error."""
        provider = _build_provider(proxy="socks5://proxy.red:1331")
        assert provider.proxy == "socks5://proxy.red:1331"
        assert isinstance(provider.pool.client, httpx.AsyncClient)
        assert not provider.pool.client.is_closed

    @pytest.mark.asyncio
    async def test_proxy_client_closes_cleanly(self):
        """A proxy-backed client can be closed via aclose()."""
        provider = _build_provider(proxy="socks5://proxy.red:1331")
        await provider.aclose()
        assert provider.pool.client.is_closed

    def test_proxy_with_settings(self):
        """Proxy and Settings limits coexist on the same client."""
        settings = Settings(httpx_max_connections=10, httpx_max_keepalive_connections=2,
                            httpx_connect_timeout=12.0, httpx_read_timeout=33.0,
                            httpx_pool_timeout=4.0)
        provider = _build_provider(proxy="socks5://proxy.red:1331", settings=settings)
        assert provider.proxy == "socks5://proxy.red:1331"
        assert provider.pool.client.timeout.connect == 12.0
        assert provider.pool.client.timeout.read == 33.0


# ===================================================================
# list_models
# ===================================================================

def _build_openai_provider(settings=None):
    config = {"base_url": "https://api.example.com", "api_key_env": "TEST_API_KEY"}
    with patch.dict("os.environ", {"TEST_API_KEY": "sk-test-123"}, clear=False):
        return ProviderStub(_entry(config), settings=settings or Settings())


class TestListModels:

    @pytest.mark.asyncio
    async def test_list_models_happy_path(self):
        """list_models calls GET /models and returns the JSON body."""
        provider = _build_openai_provider()
        provider._request = AsyncMock(return_value={"data": [{"id": "gpt-4"}]})
        result = await provider.list_models(request_id="req-1")
        provider._request.assert_called_once_with(
            method="GET", path="/models", request_id="req-1"
        )
        assert result == {"data": [{"id": "gpt-4"}]}

    @pytest.mark.asyncio
    async def test_list_maps_provider_error(self):
        """list_models propagates provider errors via _request."""
        provider = _build_openai_provider()
        provider._request = AsyncMock(side_effect=HTTPException(status_code=502, detail="bad"))
        with pytest.raises(HTTPException) as exc_info:
            await provider.list_models(request_id="req-1")
        assert exc_info.value.status_code == 502


# ===================================================================
# Per-provider concurrency limit (acquire_slot / semaphore)
# ===================================================================

class TestConcurrencyLimit:
    """Tests for the per-provider max_concurrent semaphore gate."""

    def test_no_semaphore_when_max_concurrent_unset(self):
        """Provider without max_concurrent has _semaphore is None."""
        provider = _build_provider()
        assert provider.pool._semaphore is None
        assert provider.pool._max_concurrent is None

    def test_semaphore_created_when_max_concurrent_set(self):
        """Provider with max_concurrent creates an asyncio.Semaphore."""
        provider = _build_limited_provider(2)
        assert provider.pool._max_concurrent == 2
        assert provider.pool._semaphore is not None

    def test_max_concurrent_non_positive_disables_limit(self):
        """Non-positive / non-int max_concurrent disables the limit."""
        provider = _build_limited_provider(0)
        assert provider.pool._semaphore is None
        assert provider.pool._max_concurrent is None

    @pytest.mark.asyncio
    async def test_no_limit_requests_run_concurrently(self):
        """Without max_concurrent, two requests run concurrently."""
        provider = _build_provider()
        started = [asyncio.Event(), asyncio.Event()]
        release = asyncio.Event()
        idx = [0]

        async def post(*a, **k):
            i = idx[0]
            idx[0] += 1
            started[i].set()
            await release.wait()
            return _mock_response()

        provider.pool.client.post = post
        t1 = asyncio.create_task(provider._request("POST", "/x", request_id="r1"))
        t2 = asyncio.create_task(provider._request("POST", "/x", request_id="r2"))
        await asyncio.wait_for(asyncio.gather(started[0].wait(), started[1].wait()), timeout=2)
        release.set()
        r1, r2 = await asyncio.gather(t1, t2)
        assert r1 == {"ok": True}
        assert r2 == {"ok": True}

    @pytest.mark.asyncio
    async def test_limit_queues_second_request(self):
        """max_concurrent=1: second request waits until the first releases its slot."""
        provider = _build_limited_provider(1, settings=_make_settings())
        started = [asyncio.Event(), asyncio.Event()]
        release = [asyncio.Event(), asyncio.Event()]
        idx = [0]

        async def post(*a, **k):
            i = idx[0]
            idx[0] += 1
            started[i].set()
            await release[i].wait()
            return _mock_response()

        provider.pool.client.post = post
        t1 = asyncio.create_task(provider._request("POST", "/x", request_id="r1"))
        await asyncio.wait_for(started[0].wait(), timeout=2)

        t2 = asyncio.create_task(provider._request("POST", "/x", request_id="r2"))
        await asyncio.sleep(0.05)
        assert not started[1].is_set()  # queued, not running

        release[0].set()  # first finishes → slot freed
        await asyncio.wait_for(started[1].wait(), timeout=2)
        release[1].set()
        await asyncio.gather(t1, t2)

    @pytest.mark.asyncio
    async def test_queue_timeout_returns_503(self):
        """Queued request exceeding queue_wait_timeout fails fast with 503."""
        provider = _build_limited_provider(1, settings=_make_settings(queue_wait_timeout=0.05))
        first_release = asyncio.Event()
        started = [asyncio.Event(), asyncio.Event()]
        idx = [0]

        async def post(*a, **k):
            i = idx[0]
            idx[0] += 1
            started[i].set()
            if i == 0:
                await first_release.wait()
            return _mock_response()

        provider.pool.client.post = post
        t1 = asyncio.create_task(provider._request("POST", "/x", request_id="r1"))
        await asyncio.wait_for(started[0].wait(), timeout=2)

        t2 = asyncio.create_task(provider._request("POST", "/x", request_id="r2"))
        with pytest.raises(HTTPException) as exc_info:
            await t2
        assert exc_info.value.status_code == 503

        first_release.set()
        await t1

    @pytest.mark.asyncio
    async def test_queue_timeout_does_not_leak_slot(self):
        """A 503 from queue timeout must not acquire/release a slot (no double-release)."""
        provider = _build_limited_provider(1, settings=_make_settings(queue_wait_timeout=0.02))
        blocker = asyncio.Event()
        running = asyncio.Event()

        async def post(*a, **k):
            running.set()
            await blocker.wait()
            return _mock_response()

        provider.pool.client.post = post
        t1 = asyncio.create_task(provider._request("POST", "/x", request_id="r1"))
        await asyncio.wait_for(running.wait(), timeout=2)

        with pytest.raises(HTTPException):
            await provider._request("POST", "/x", request_id="r2")

        # Semaphore value must still be 0 (first call holds it; the timed-out one never acquired).
        assert provider.pool._semaphore._value == 0
        blocker.set()
        await t1
        assert provider.pool._semaphore._value == 1

    @pytest.mark.asyncio
    async def test_stream_slot_released_on_completion(self):
        """After a stream finishes, the slot is released; a second stream starts at once."""
        provider = _build_limited_provider(1, settings=_make_settings())

        async def inner(url, body, rid, extra_headers=None):
            yield b"a"
            yield b"b"

        provider._stream_request_inner = inner
        chunks = []
        async for c in provider._stream_request("/x", {}, "r1"):
            chunks.append(c)
        assert chunks == [b"a", b"b"]
        assert provider.pool._semaphore._value == 1

        # second stream must start immediately (slot was released)
        second_started = asyncio.Event()

        async def inner2(url, body, rid, extra_headers=None):
            second_started.set()
            yield b"c"

        provider._stream_request_inner = inner2
        out = []
        async for c in provider._stream_request("/x", {}, "r2"):
            out.append(c)
        assert second_started.is_set()
        assert out == [b"c"]

    @pytest.mark.asyncio
    async def test_stream_slot_released_on_early_close(self):
        """aclose() mid-stream releases the slot."""
        provider = _build_limited_provider(1, settings=_make_settings())
        gate = asyncio.Event()

        async def inner(url, body, rid, extra_headers=None):
            yield b"a"
            await gate.wait()
            yield b"b"

        provider._stream_request_inner = inner
        gen = provider._stream_request("/x", {}, "r1")
        first = await gen.__anext__()
        assert first == b"a"
        await gen.aclose()
        assert provider.pool._semaphore._value == 1  # released on close

    @pytest.mark.asyncio
    async def test_stream_slot_released_on_exception(self):
        """An exception inside the stream releases the slot."""
        provider = _build_limited_provider(1, settings=_make_settings())

        async def inner(url, body, rid, extra_headers=None):
            yield b"a"
            raise RuntimeError("boom")

        provider._stream_request_inner = inner
        gen = provider._stream_request("/x", {}, "r1")
        first = await gen.__anext__()
        assert first == b"a"
        with pytest.raises(RuntimeError):
            await gen.__anext__()
        assert provider.pool._semaphore._value == 1

    @pytest.mark.asyncio
    async def test_retry_holds_slot_across_attempts(self):
        """The slot is held across a 429 retry; released once after success."""
        provider = _build_limited_provider(1)  # bare Settings() retry defaults
        seen_values = []
        calls = [0]

        async def post(*a, **k):
            calls[0] += 1
            seen_values.append(provider.pool._semaphore._value)
            if calls[0] == 1:
                resp = MagicMock()
                resp.status_code = 429
                resp.text = "rate limited"
                resp.json.side_effect = json.JSONDecodeError("", "", 0)
                raise httpx.HTTPStatusError("429", request=MagicMock(spec=httpx.Request), response=resp)
            return _mock_response()

        provider.pool.client.post = post
        with patch("src.providers.base.asyncio.sleep", new_callable=AsyncMock):
            result = await provider._request("POST", "/x", request_id="r1")

        assert result == {"ok": True}
        assert calls[0] == 2
        # Slot value was 0 during BOTH attempts (held across retry).
        assert seen_values == [0, 0]
        # Released exactly once after success.
        assert provider.pool._semaphore._value == 1


# ===================================================================
# Header merge: stream / non-stream parity + extra_headers
# ===================================================================

class _FakeStreamResponse:
    """Minimal httpx stream response stand-in."""

    status_code = 200
    headers = {}

    def raise_for_status(self):
        return None

    async def aiter_bytes(self):
        yield b"data: chunk\n\n"


class _FakeStreamCtx:
    """Async context manager returned by a mocked client.stream()."""

    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *exc):
        return False


class TestHeaderMergeParity:
    """Stream and non-stream paths must send an identical header set."""

    EXTRA = {"User-Agent": "opencode/1.18.23", "X-Session-Id": "ses_abc",
             "x-session-affinity": "ses_abc"}

    @pytest.mark.asyncio
    async def test_stream_and_non_stream_send_identical_headers(self):
        """Same extra_headers → byte-identical header dicts on both paths."""
        provider = _build_provider()
        provider.pool.client.post = AsyncMock(return_value=_mock_response())
        stream_response = _FakeStreamResponse()
        provider.pool.client.stream = MagicMock(return_value=_FakeStreamCtx(stream_response))

        await provider._request("POST", "/chat", request_body={}, request_id="r1",
                                     extra_headers=self.EXTRA)
        chunks = [c async for c in provider._stream_request("/chat", {},
                                                            "r2", extra_headers=self.EXTRA)]
        assert chunks == [b"data: chunk\n\n"]

        post_headers = provider.pool.client.post.call_args.kwargs["headers"]
        stream_headers = provider.pool.client.stream.call_args.kwargs["headers"]
        assert post_headers == stream_headers

    @pytest.mark.asyncio
    async def test_stream_without_extra_headers_sends_static_headers(self):
        """Stream path with no extra_headers sends exactly self.headers."""
        provider = _build_provider()
        provider.pool.client.stream = MagicMock(return_value=_FakeStreamCtx(_FakeStreamResponse()))

        async for _ in provider._stream_request("/chat", {}, "r1"):
            pass

        assert provider.pool.client.stream.call_args.kwargs["headers"] == provider.headers

    def test_merge_replaces_case_insensitive_duplicates(self):
        """An extra header replaces its case-insensitive base twin, not duplicates it."""
        provider = _build_provider(headers={"User-Agent": "configured/1.0"})
        merged = provider._merge_request_headers({"user-agent": "client/2.0"})
        assert merged["user-agent"] == "client/2.0"
        assert "User-Agent" not in merged

    def test_merge_never_overwrites_authorization(self):
        """extra_headers cannot replace Authorization (INVARIANT)."""
        provider = _build_provider()
        merged = provider._merge_request_headers({"Authorization": "Bearer attacker",
                                                  "authorization": "Bearer attacker2"})
        assert merged["Authorization"] == "Bearer sk-test-123"
        assert "authorization" not in merged

    def test_merge_without_extra_returns_copy_of_static(self):
        """No extra_headers → a plain copy of self.headers (no aliasing)."""
        provider = _build_provider()
        merged = provider._merge_request_headers(None)
        assert merged == provider.headers
        assert merged is not provider.headers


class TestIdentityProfileInit:
    """identity config key — validated by parse_provider before construction."""

    def test_identity_passthrough_sets_no_user_agent(self):
        config = {"base_url": "https://api.example.com", "api_key_env": "TEST_API_KEY",
                  "identity": "passthrough"}
        with patch.dict("os.environ", {"TEST_API_KEY": "sk-test-123"}, clear=False):
            provider = ProviderStub(_entry(config), Settings())
        assert provider.identity == "passthrough"
        assert "User-Agent" not in provider.headers

    def test_unknown_identity_fails_fast(self):
        config = {"base_url": "https://api.example.com", "api_key_env": "TEST_API_KEY",
                  "identity": "opencode"}
        with patch.dict("os.environ", {"TEST_API_KEY": "sk-test-123"}, clear=False):
            with pytest.raises(ConfigError) as exc_info:
                ProviderStub(_entry(config), Settings())
        assert "expected 'passthrough'" in str(exc_info.value)

    def test_no_identity_keeps_current_behavior(self):
        provider = _build_provider()
        assert provider.identity is None
        assert "User-Agent" not in provider.headers


class TestReasoningDialectInit:
    """reasoning_dialect config key — parse_provider fails before construction
    so a typo never reaches the funnel as a silent `openai`."""

    @pytest.mark.parametrize("dialect", ["openai", "deepseek", "openrouter"])
    def test_known_dialects_construct(self, dialect):
        config = {"base_url": "https://api.example.com", "api_key_env": "TEST_API_KEY",
                  "reasoning_dialect": dialect}
        with patch.dict("os.environ", {"TEST_API_KEY": "sk-test-123"}, clear=False):
            provider = ProviderStub(_entry(config), Settings())
        assert provider.base_url == config["base_url"]

    def test_unknown_dialect_fails_fast(self):
        config = {"base_url": "https://api.example.com", "api_key_env": "TEST_API_KEY",
                  "reasoning_dialect": "openrouter-compatible"}
        with patch.dict("os.environ", {"TEST_API_KEY": "sk-test-123"}, clear=False):
            with pytest.raises(ConfigError) as exc_info:
                ProviderStub(_entry(config), Settings())
        assert "reasoning_dialect" in str(exc_info.value)

    def test_absent_key_keeps_current_behavior(self):
        """No key: no validation fires, and the entry carries the default dialect."""
        config = {"base_url": "https://api.example.com", "api_key_env": "TEST_API_KEY"}
        with patch.dict("os.environ", {"TEST_API_KEY": "sk-test-123"}, clear=False):
            provider = ProviderStub(_entry(config), Settings())
        assert provider.base_url == config["base_url"]
        assert provider.entry.reasoning_dialect == DEFAULT_REASONING_DIALECT


class TestStaticHeadersValidation:
    """Static `headers:` from providers.yaml fails fast in parse_provider."""

    def test_non_string_value_fails_fast(self):
        """X-Title: 12345 (YAML int) is rejected at startup, not on first request."""
        config = {"base_url": "https://api.example.com", "api_key_env": "TEST_API_KEY",
                  "headers": {"X-Title": 12345}}
        with patch.dict("os.environ", {"TEST_API_KEY": "sk-test-123"}, clear=False):
            with pytest.raises(ConfigError) as exc_info:
                ProviderStub(_entry(config), Settings())
        assert "headers" in str(exc_info.value)

    def test_non_string_key_fails_fast(self):
        config = {"base_url": "https://api.example.com", "api_key_env": "TEST_API_KEY",
                  "headers": {123: "value"}}
        with patch.dict("os.environ", {"TEST_API_KEY": "sk-test-123"}, clear=False):
            with pytest.raises(ConfigError) as exc_info:
                ProviderStub(_entry(config), Settings())
        assert "headers" in str(exc_info.value)

    def test_authorization_in_headers_fails_fast(self):
        config = {"base_url": "https://api.example.com", "api_key_env": "TEST_API_KEY",
                  "headers": {"Authorization": "Bearer literal-key"}}
        with patch.dict("os.environ", {"TEST_API_KEY": "sk-test-123"}, clear=False):
            with pytest.raises(ConfigError) as exc_info:
                ProviderStub(_entry(config), Settings())
        assert "headers" in str(exc_info.value)

    @pytest.mark.parametrize("name", ["x-api-key", "X-Api-Key"])
    def test_router_owned_credential_header_fails_fast(self, name):
        """x-api-key is built from api_key_env by AnthropicProvider, so a
        static twin would collide with it — refused at parse."""
        config = {"base_url": "https://api.example.com", "api_key_env": "TEST_API_KEY",
                  "headers": {name: "literal-key"}}
        with patch.dict("os.environ", {"TEST_API_KEY": "sk-test-123"}, clear=False):
            with pytest.raises(ConfigError) as exc_info:
                ProviderStub(_entry(config), Settings())
        assert "headers" in str(exc_info.value)

    @pytest.mark.parametrize("name", ["api-key", "x-goog-api-key", "Cookie"])
    def test_credential_header_api_key_env_cannot_emit_is_accepted(self, name):
        """api_key_env emits only Authorization / x-api-key, so a static
        `api-key` (Azure OpenAI) is the only way to reach such a backend."""
        provider = _build_provider(headers={name: "literal-key"})
        assert provider.headers[name] == "literal-key"

    @pytest.mark.parametrize("name", ["Content-Length", "host", "Transfer-Encoding",
                                      "Connection", "Accept-Encoding"])
    def test_hop_by_hop_in_headers_fails_fast(self, name):
        config = {"base_url": "https://api.example.com", "api_key_env": "TEST_API_KEY",
                  "headers": {name: "x"}}
        with patch.dict("os.environ", {"TEST_API_KEY": "sk-test-123"}, clear=False):
            with pytest.raises(ConfigError) as exc_info:
                ProviderStub(_entry(config), Settings())
        assert "headers" in str(exc_info.value)

    def test_valid_static_headers_accepted(self):
        """Attribution headers pass validation; Content-Type default still applied."""
        provider = _build_provider(headers={"HTTP-Referer": "https://nnp.space",
                                            "X-Title": "nnp.space"})
        assert provider.headers["HTTP-Referer"] == "https://nnp.space"
        assert provider.headers["Content-Type"] == "application/json"


class TestChatExtraHeadersForwarding:
    """Provider endpoint methods forward extra_headers."""

    @pytest.mark.asyncio
    async def test_chat_completions_forwards_extra_headers(self):
        provider = _build_openai_provider()
        provider._request = AsyncMock(return_value={"ok": True})
        extra = {"X-Session-Id": "ses_x", "x-session-affinity": "ses_x"}
        await provider.chat_completions({"messages": []}, "gpt-4", NO_OPTIONS, request_id="r1",
                                        extra_headers=extra)
        assert provider._request.call_args.kwargs["extra_headers"] == extra

    @pytest.mark.asyncio
    async def test_chat_completions_stream_forwards_extra_headers(self):
        provider = _build_openai_provider()

        async def fake_stream(path, body, request_id="unknown", extra_headers=None):
            assert extra_headers == {"X-Session-Id": "ses_x"}
            yield b""

        provider._stream_request = fake_stream
        gen = provider.chat_completions_stream({"messages": []}, "gpt-4", NO_OPTIONS, request_id="r1",
                                               extra_headers={"X-Session-Id": "ses_x"})
        async for _ in gen:
            pass

    @pytest.mark.asyncio
    async def test_embeddings_forwards_extra_headers(self):
        provider = _build_openai_provider()
        provider._request = AsyncMock(return_value={"data": []})
        extra = {"user-agent": "Kilo-Code/7.5.5"}
        await provider.embeddings({"input": "hi"}, "emb", NO_OPTIONS, request_id="r1",
                                  extra_headers=extra)
        assert provider._request.call_args.kwargs["extra_headers"] == extra

    @pytest.mark.asyncio
    async def test_transcriptions_forwards_extra_headers(self):
        provider = _build_openai_provider()
        provider._request = AsyncMock(return_value={"text": "ok"})
        extra = {"user-agent": "Kilo-Code/7.5.5"}
        body = {"audio": {"filename": "a.wav", "content_type": "audio/wav", "data": b"x"},
                "params": {}}
        await provider.transcriptions(body, "stt", NO_OPTIONS, request_id="r1", extra_headers=extra)
        assert provider._request.call_args.kwargs["extra_headers"] == extra


# ===================================================================
# _create_timeout fallback semantics and call-site timeouts
# ===================================================================

class TestCreateTimeoutFallback:
    """Unspecified values inherit from the client's timeout — read and write included."""

    def test_unspecified_read_write_inherit_from_client(self):
        settings = _make_settings(httpx_connect_timeout=11.0, httpx_read_timeout=77.0, httpx_pool_timeout=4.0)
        provider = _build_provider(settings=settings)

        t = provider._create_timeout()

        assert t.connect == 11.0
        assert t.read == 77.0
        assert t.write == provider.pool.client.timeout.write
        assert t.pool == 4.0

    def test_explicit_values_win_over_client(self):
        provider = _build_provider(settings=_make_settings())

        t = provider._create_timeout(connect=1.0, read=2.0, write=3.0, pool=4.0)

        assert (t.connect, t.read, t.write, t.pool) == (1.0, 2.0, 3.0, 4.0)


class TestCallSiteTimeouts:
    """Every provider call site must produce a Timeout with a real read cap."""

    def _openai_provider(self, settings):
        config = {"base_url": "https://api.example.com", "api_key_env": "TEST_API_KEY"}
        with patch.dict("os.environ", {"TEST_API_KEY": "sk-test-123"}, clear=False):
            provider = ProviderStub(_entry(config), settings=settings)
        captured = {}

        async def fake_request(**kwargs):
            captured.update(kwargs)
            return {"ok": True}

        provider._request = fake_request
        return provider, captured

    @pytest.mark.asyncio
    async def test_chat_non_stream_read_is_stream_read_timeout(self):
        """Non-stream chat is capped by stream_read_timeout — a silent upstream must
        not hold its concurrency slot and in-flight count forever."""
        settings = _make_settings(httpx_connect_timeout=60.0, httpx_read_timeout=60.0,
                      httpx_pool_timeout=5.0, stream_read_timeout=300.0)
        provider, captured = self._openai_provider(settings)

        await provider.chat_completions({"messages": []}, "m", NO_OPTIONS)

        t = captured["timeout"]
        assert t.read == 300.0
        assert t.connect == 60.0
        assert t.pool == 5.0
        assert t.write == provider.pool.client.timeout.write

    @pytest.mark.asyncio
    async def test_transcription_read_is_transcription_timeout(self):
        settings = _make_settings(httpx_connect_timeout=60.0, httpx_pool_timeout=5.0,
                      openai_transcription_timeout=3600.0)
        provider, captured = self._openai_provider(settings)

        await provider.transcriptions(
            {"audio": {"filename": "a.ogg", "content_type": "audio/ogg", "data": b"x"},
             "params": {}},
            "m", NO_OPTIONS)

        t = captured["timeout"]
        assert t.read == 3600.0
        assert t.connect == 60.0
        assert t.pool == 5.0

    @pytest.mark.asyncio
    async def test_embeddings_hardcoded_timeouts_dropped(self):
        """The old hardcoded connect=10/write=10/pool=10 are gone; client defaults cover them."""
        settings = _make_settings(httpx_connect_timeout=60.0, httpx_read_timeout=60.0,
                      httpx_pool_timeout=5.0, openai_embeddings_read_timeout=30.0)
        provider, captured = self._openai_provider(settings)

        await provider.embeddings({"input": "hi"}, "m", NO_OPTIONS)

        t = captured["timeout"]
        assert t.read == 30.0
        assert t.connect == 60.0
        assert t.pool == 5.0
        assert t.write == provider.pool.client.timeout.write

    @pytest.mark.asyncio
    async def test_stream_read_is_stream_read_timeout(self):
        settings = _make_settings(httpx_connect_timeout=60.0, httpx_read_timeout=60.0,
                      httpx_pool_timeout=5.0, stream_read_timeout=321.0)
        provider = self._openai_provider(settings)[0]

        captured = {}

        class _FakeStreamResponse:
            status_code = 200
            headers: dict = {}

            def raise_for_status(self):
                return None

            async def aiter_bytes(self):
                if False:
                    yield b""

        class _FakeStreamCM:
            def __init__(self, resp):
                self._resp = resp

            async def __aenter__(self):
                return self._resp

            async def __aexit__(self, *exc):
                return False

        fake_client = MagicMock()
        fake_client.timeout = httpx.Timeout(connect=60.0, read=60.0, write=None, pool=5.0)

        def stream(*args, **kwargs):
            captured.update(kwargs)
            return _FakeStreamCM(_FakeStreamResponse())

        fake_client.stream = stream
        provider.pool.client = fake_client

        chunks = [c async for c in provider._stream_request("/chat/completions", {})]
        assert chunks == []

        t = captured["timeout"]
        assert t.read == 321.0
        assert t.connect == 60.0
        assert t.pool == 5.0



class TestRetryUploadSafety:
    """A retried multipart upload must resend the file, not an empty body.

    Regression: the files tuple used to carry an io.BytesIO; httpx consumed it
    on the first attempt, so a 429 retry posted an empty file part. Driven
    through httpx.MockTransport so the REAL multipart encoder runs on every
    attempt — mocking client.post would hide exactly this defect.
    """

    def _provider_with_mock_transport(self, handler, cm=None):
        config = {"base_url": "https://api.example.com", "api_key_env": "TEST_API_KEY"}
        with patch.dict("os.environ", {"TEST_API_KEY": "sk-test-123"}, clear=False):
            provider = ProviderStub(_entry(config), settings=cm or _make_settings(
                provider_retry_base_delay=0.001, provider_retry_max_delay=0.01))
        provider.pool.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return provider

    @pytest.mark.asyncio
    async def test_retried_429_upload_resends_the_audio(self):
        """429-then-200: BOTH encoded multipart bodies contain the audio bytes."""
        audio = b"OggS-fake-audio-payload" * 64
        seen_bodies: list[bytes] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_bodies.append(request.read())
            if len(seen_bodies) == 1:
                return httpx.Response(429, json={"error": {"message": "rate limited"}})
            return httpx.Response(200, json={"text": "transcribed"})

        provider = self._provider_with_mock_transport(handler)
        body = {"audio": {"filename": "a.ogg", "content_type": "audio/ogg", "data": audio},
                "params": {}}
        result = await provider.transcriptions(body, "stt/dummy", NO_OPTIONS, request_id="r1")

        assert result == {"text": "transcribed"}
        assert len(seen_bodies) == 2, "the 429 must have been retried exactly once"
        for attempt, raw in enumerate(seen_bodies, start=1):
            assert audio in raw, f"attempt {attempt} lost the audio payload"

    @pytest.mark.asyncio
    async def test_first_attempt_429_is_surfaced_only_after_retries(self):
        """Sanity for the driver: the mocked 429 really flows through the retry path."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})

        provider = self._provider_with_mock_transport(handler)
        body = {"audio": {"filename": "a.ogg", "content_type": "audio/ogg", "data": b"x"},
                "params": {}}
        with pytest.raises(HTTPException) as exc_info:
            await provider.transcriptions(body, "stt/dummy", NO_OPTIONS, request_id="r1")
        assert exc_info.value.status_code == 429


# ===================================================================
# Error parity between the stream and non-stream paths
# ===================================================================

class TestErrorPathParity:
    """PoolTimeout answers 503 service_unavailable on BOTH paths — the
    non-stream path used to let it fall into RequestError's 500
    provider_network_error — and an upstream 4xx/5xx in a stream is logged
    once, by the provider-error channel, never again as "Stream request
    failed" with a traceback."""

    def _provider_with_mock_transport(self, handler):
        config = {"base_url": "https://api.example.com", "api_key_env": "TEST_API_KEY"}
        with patch.dict("os.environ", {"TEST_API_KEY": "sk-test-123"}, clear=False):
            provider = ProviderStub(_entry(config), settings=_make_settings(
                provider_retry_base_delay=0.001, provider_retry_max_delay=0.01))
        provider.pool.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return provider

    @pytest.mark.asyncio
    async def test_non_stream_pool_timeout_is_503_service_unavailable(self):
        """A pool timeout on a non-stream call answers 503
        service_unavailable — the same status the stream path gives."""
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.PoolTimeout("pool exhausted")

        provider = self._provider_with_mock_transport(handler)
        with pytest.raises(HTTPException) as exc_info:
            await provider.chat_completions(
                {"messages": [{"role": "user", "content": "hi"}]},
                "gpt-x", NO_OPTIONS, request_id="r1")
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail["error"]["metadata"]["error_code"] == "service_unavailable"

    @pytest.mark.asyncio
    async def test_stream_429_raises_without_the_generic_error_log(self):
        """An upstream 429 in a stream raises its HTTPException (already
        logged by log_provider_error via _raise_provider_http_error) and
        never logs "Stream request failed" — the old except Exception arm
        doubled the same error at ERROR with a full traceback."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})

        provider = self._provider_with_mock_transport(handler)
        with patch("src.providers.base.logger") as mock_logger:
            with pytest.raises(HTTPException) as exc_info:
                async for _ in provider.chat_completions_stream(
                        {"messages": [{"role": "user", "content": "hi"}]},
                        "gpt-x", NO_OPTIONS, request_id="r1"):
                    pass
        assert exc_info.value.status_code == 429
        assert not [c for c in mock_logger.error.call_args_list
                    if "Stream request failed" in str(c)]

    async def _stream_error(self, handler) -> HTTPException:
        provider = self._provider_with_mock_transport(handler)
        with pytest.raises(HTTPException) as exc_info:
            async for _ in provider.chat_completions_stream(
                    {"messages": [{"role": "user", "content": "hi"}]},
                    "gpt-x", NO_OPTIONS, request_id="r1"):
                pass
        return exc_info.value

    @pytest.mark.asyncio
    async def test_stream_error_json_body_reaches_the_client(self):
        """A streaming upstream error carries the upstream's own message, not
        the "Unable to read error response" placeholder — the stream context
        never buffers the body unless it is read explicitly.

        The body is an unread `stream=`: `httpx.Response(json=...)` pre-buffers
        its content, so it would pass without the fix."""
        body = b'{"error": {"message": "rate limited"}}'
        exc = await self._stream_error(
            lambda request: httpx.Response(429, stream=httpx.ByteStream(body)))
        assert exc.status_code == 429
        assert exc.detail["error"]["message"] == "rate limited"
        assert "rate limited" in exc.detail["error"]["metadata"]["raw"]

    @pytest.mark.asyncio
    async def test_stream_error_text_body_is_the_raw(self):
        exc = await self._stream_error(
            lambda request: httpx.Response(500, stream=httpx.ByteStream(b"upstream exploded")))
        assert exc.status_code == 500
        assert exc.detail["error"]["metadata"]["raw"] == "upstream exploded"


# ===================================================================
# Retry 429-detection robustness
# ===================================================================

class TestRetryRateLimitDetection:
    """429 detection must survive a wrapped exception whose .response is None.

    Driven through _request with client.post raising the wrapped error —
    the same shape create_error produces when it re-raises an httpx error.
    """

    def _provider(self):
        return _build_provider(settings=_make_settings(provider_max_retries=2,
                                                       provider_retry_base_delay=0.001,
                                                       provider_retry_max_delay=0.01))

    @pytest.mark.asyncio
    async def test_wrapped_exception_with_none_response_is_not_rate_limit(self):
        provider = self._provider()
        inner = SimpleNamespace(response=None)
        wrapped = HTTPException(status_code=500, detail="wrapped network failure")
        wrapped.original_exception = inner
        calls = []

        async def post(*a, **k):
            calls.append(1)
            raise wrapped

        provider.pool.client.post = post
        with patch("src.providers.base.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(HTTPException):
                await provider._request("POST", "/x", request_id="r1")
        assert len(calls) == 1
        mock_sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_wrapped_429_response_still_retries(self):
        provider = self._provider()
        inner = SimpleNamespace(response=SimpleNamespace(status_code=429))
        wrapped = HTTPException(status_code=502, detail="upstream 429")
        wrapped.original_exception = inner
        calls = []

        async def post(*a, **k):
            calls.append(1)
            if len(calls) < 3:
                raise wrapped
            return _mock_response()

        provider.pool.client.post = post
        with patch("src.providers.base.asyncio.sleep", new_callable=AsyncMock):
            result = await provider._request("POST", "/x", request_id="r1")
        assert result == {"ok": True}
        assert len(calls) == 3
