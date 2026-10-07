"""The provider: HTTP, streaming, retry and error handling for an OpenAI-compatible backend."""
# SYSTEM: provider — base HTTP, retry, streaming and header merging
import asyncio
import contextlib
import json
import os
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import HTTPException

from ..core.config_manager import Settings
from ..core.config_schema import ModelEntry, ProviderEntry
from ..core.error_handling import ErrorType, create_error, create_provider_http_error
from ..core.logging import logger
from ..utils.deep_merge import deep_merge
from ..utils.mask import mask_headers
from .pool import ProviderPool


def _is_rate_limit_error(e: BaseException) -> bool:
    """429 detection for the retry loop in _request.

    Either the exception itself carries status 429, or it wraps one
    (original_exception.response.status_code == 429, wrapped httpx errors) —
    the wrapped response may be None, which must mean "not a rate limit",
    not a crash.
    """
    if hasattr(e, 'status_code') and e.status_code == 429:
        return True
    original = getattr(e, 'original_exception', None)
    response = getattr(original, 'response', None)
    return response is not None and getattr(response, 'status_code', None) == 429




# INVARIANT: the provider's credential headers are set once in __init__ from
# os.environ[api_key_env] — Authorization for the OpenAI wire (this class),
# x-api-key for the Anthropic wire (AnthropicProvider). They are never
# replaced per-request. Client API keys stay in auth.py and are not
# propagated to providers.
class Provider:
    def __init__(self, entry: ProviderEntry, settings: Settings, provider_name: str):
        """Initialize provider from its parsed providers.yaml entry.

        Static checks (type, identity, reasoning_dialect, headers) already ran
        in core/config_schema.py parse_provider; only the env-dependent ones
        (base_url, the api_key_env variable) run here.
        Composes a ProviderPool: its own httpx.AsyncClient (per-provider
        connection pool), concurrency gate and drain accounting. Limits
        come from settings (global env applied per pool).
        An optional `proxy` URL (e.g. socks5://host:port) routes all of that
        provider's traffic through that proxy.
        provider_name is the providers.yaml dict key (used in logs and
        startup-validation errors).

        Raises:
            HTTPException: If base_url is missing or the env var for the API key is unset.
        """
        self.base_url = entry.base_url
        self.api_key_env = entry.api_key_env
        self.api_key = os.environ.get(self.api_key_env) if self.api_key_env else None
        self.settings = settings
        self.proxy = entry.proxy
        self.provider_name = provider_name
        # WHY: the providers.yaml entry this instance was built from. The
        # registry's reuse check compares it by value with the freshly parsed
        # entry, so equality means "the operator did not touch this backend".
        self.entry = entry
        # See the identity ARCH on ProviderEntry (core/config_schema.py).
        self.identity = entry.identity

        if not self.base_url:
            raise create_error(ErrorType.PROVIDER_CONFIG_ERROR,
                             error_details="Provider base_url is not configured.",
                             provider_name=self.provider_name)
        headers = dict(entry.headers or {})
        headers.setdefault("Content-Type", "application/json")
        self.headers = {**headers, **self._auth_headers()}

        # ARCH: the httpx pool is a composed component (providers/pool.py), not a
        # base-class role — client construction, the concurrency gate and the
        # graceful-drain invariants live there, directly testable.
        self.pool = ProviderPool(settings=settings, provider_name=self.provider_name,
                                 proxy=self.proxy, max_concurrent=entry.max_concurrent)

    def _auth_headers(self) -> dict[str, str]:
        """This protocol's credential headers, built once in __init__.

        The OpenAI-compatible wire carries its key as a Bearer Authorization;
        subclasses speaking another protocol override this (AnthropicProvider
        sends x-api-key). Static checks on entry.headers already ran in
        parse_provider; only the env-dependent one (the api_key_env variable
        is set) runs here.

        Raises:
            HTTPException: If api_key_env names an unset environment variable.
        """
        if not self.api_key_env:
            return {}
        if not self.api_key:
            raise create_error(ErrorType.PROVIDER_CONFIG_ERROR,
                             error_details=f"API key for {self.api_key_env} is not set in environment variables.",
                             provider_name=self.provider_name)
        return {"Authorization": f"Bearer {self.api_key}"}

    async def aclose(self, drain_timeout: float | None = None) -> None:
        """Close the owned pool once in-flight requests have drained.

        Thin delegate to the composed ProviderPool (see there for the drain
        contract); kept on the provider because the registry closes providers.
        """
        await self.pool.aclose(drain_timeout)

    def _log_provider_data(self, title: str, data: dict[str, Any], request_id: str, data_flow: str, component: str = None) -> None:
        """Log request/response data with standardized provider context."""
        if component is None:
            component = f"{self.provider_name}_provider"

        logger.debug_data(
            title=title,
            data=data,
            request_id=request_id,
            component=component,
            data_flow=data_flow
        )

    def _create_timeout(self, connect: float = None, read: float = None,
                        write: float = None, pool: float = None) -> httpx.Timeout:
        """
        Create an httpx.Timeout with the client's defaults for anything unspecified.

        WHY: every unspecified field — read and write included, exactly like
        connect/pool — inherits from the client's timeout. Passing None would
        mean "no timeout", letting a silent upstream hold its concurrency slot
        and in-flight count until the aclose() drain timeout.
        """
        client_timeout = self.pool.client.timeout
        return httpx.Timeout(
            connect=connect if connect is not None else client_timeout.connect,
            read=read if read is not None else client_timeout.read,
            write=write if write is not None else client_timeout.write,
            pool=pool if pool is not None else client_timeout.pool
        )

    def _apply_model_config(self, request_body: dict[str, Any], provider_model_name: str,
                            model_config: ModelEntry) -> dict[str, Any]:
        """
        Set provider model name and merge model-level options into request body.

        `stream` is stripped from the options before the merge — see the
        INVARIANT below.
        """
        request_body["model"] = provider_model_name
        if options := model_config.options:
            # INVARIANT: models.yaml `options:` may not set `stream`.
            # Why: the service picks the streaming or the JSON branch from the
            # CLIENT's stream value and only then reaches this merge, so an
            # options-supplied `stream: true` sent an SSE body to the
            # non-streaming path, where .json() fails and the client gets a
            # 502 provider_invalid_response instead of its answer. The
            # transport is the router's, not the model config's.
            if "stream" in options:
                logger.warning(
                    f"Model options for '{provider_model_name}' set 'stream'; ignoring it "
                    f"(the client's request decides the transport)",
                    extra={"component": "base_provider",
                           "provider_name": self.provider_name,
                           "provider_model_name": provider_model_name},
                )
                options = {k: v for k, v in options.items() if k != "stream"}
            request_body = deep_merge(request_body, options)
        return request_body

    def _raise_provider_http_error(self, e: httpx.HTTPStatusError, request_id: str = "unknown") -> None:
        """Extract error message from provider response and raise HTTPException.

        Handles ResponseNotRead (a streaming error body whose read failed).
        Logs via log_provider_error before raising.
        """
        response_text = ""
        try:
            response_text = e.response.text
        except httpx.ResponseNotRead:
            response_text = "Unable to read error response from provider"

        error_message = f"Provider API error: {e.response.status_code}"
        try:
            error_json = e.response.json()
            if "error" in error_json and isinstance(error_json["error"], dict):
                error_message = error_json["error"].get("message", error_message)
            elif "message" in error_json:
                error_message = error_json["message"]
        except (json.JSONDecodeError, ValueError, httpx.ResponseNotRead):
            error_message = response_text or error_message

        raise create_provider_http_error(
            status_code=e.response.status_code,
            message=error_message,
            provider_name=self.provider_name,
            raw=response_text,
            request_id=request_id,
            original_exception=e,
        ) from e

    def _merge_request_headers(self, extra_headers: dict[str, str] | None) -> dict[str, str]:
        """Merge per-request extra_headers over self.headers.

        ARCH: shared by the stream and non-stream paths so both send an
        identical header set — a diverging set is itself a fingerprint.
        The credential headers (Authorization / x-api-key — see the INVARIANT
        above the class) are never overwritten; case-insensitive duplicates of
        extra keys replace their base counterparts instead of being sent twice.
        """
        merged = dict(self.headers)
        if not extra_headers:
            return merged
        # WHY: credentials are not replaceable, so they must also survive the
        # case-insensitive duplicate elimination below.
        protected = {"authorization", "x-api-key"}
        extra_lower = {name.lower() for name in extra_headers} - protected
        merged = {k: v for k, v in merged.items() if k.lower() not in extra_lower}
        for name, value in extra_headers.items():
            if name.lower() in protected:
                continue
            merged[name] = value
        return merged

    # WHY noqa ASYNC109 (_request, _send_once, _send): `timeout` is the provider
    # API parameter passed straight to httpx (per-request httpx.Timeout), not
    # a wait bound this function owns — wrapping the body in asyncio.timeout()
    # would double-cap streaming-adjacent calls for no benefit.
    async def _request(
        self,
        method: str,
        path: str,
        request_body: dict[str, Any] = None,
        extra_headers: dict[str, str] = None,
        timeout: httpx.Timeout = None,  # noqa: ASYNC109
        files: dict[str, Any] = None,
        data: dict[str, Any] = None,
        request_id: str = "unknown"
    ) -> dict[str, Any]:
        """Unified non-streaming HTTP request to provider APIs.

        Holds a per-provider concurrency slot across the whole call. The retry
        loop runs inside the held slot, so retries reuse the same slot and it
        is released exactly once.
        Backoff formula: min(base_delay * 2^attempt, max_delay), bounds from
        settings. Rate-limit detection lives in _is_rate_limit_error. Only
        429-shaped errors are retried; everything else surfaces on the first
        attempt. extra_headers may add non-credential headers (e.g. Accept) but
        cannot overwrite Authorization — see INVARIANT above the class.
        """
        max_retries = self.settings.provider_max_retries
        async with self.pool.acquire_slot(request_id):
            for attempt in range(max_retries + 1):
                try:
                    return await self._send_once(
                        method, path, request_body=request_body, extra_headers=extra_headers,
                        timeout=timeout, files=files, data=data, request_id=request_id,
                    )
                except Exception as e:
                    if _is_rate_limit_error(e) and attempt < max_retries:
                        delay = min(self.settings.provider_retry_base_delay * (2 ** attempt),
                                    self.settings.provider_retry_max_delay)
                        logger.warning(
                            f"Rate limit exceeded, retrying in {delay}s (attempt {attempt + 1}/{max_retries})",
                            extra={
                                "delay_seconds": delay,
                                "attempt": attempt + 1,
                                "max_retries": max_retries,
                                "component": "base_provider"
                            })
                        await asyncio.sleep(delay)
                        continue
                    raise
        raise RuntimeError("retry loop exhausted without an exception")

    async def _send_once(
        self,
        method: str,
        path: str,
        request_body: dict[str, Any] = None,
        extra_headers: dict[str, str] = None,
        timeout: httpx.Timeout = None,  # noqa: ASYNC109
        files: dict[str, Any] = None,
        data: dict[str, Any] = None,
        request_id: str = "unknown"
    ) -> dict[str, Any]:
        """One HTTP attempt: send, raise_for_status, parse the JSON response.

        HTTPStatusError: extracts error message from provider JSON response.
        PoolTimeout: maps to 503 (connection pool exhausted), like the stream path.
        RequestError: maps to a network error.
        """
        url = f"{self.base_url}{path}"
        merged_headers = self._merge_request_headers(extra_headers)
        self._log_request_start(url, merged_headers, request_body, files, data, request_id)

        try:
            response = await self._send(method, url, merged_headers, request_body, files, data, timeout)
            response.raise_for_status()
            response_json = response.json()
        except json.JSONDecodeError as e:
            raise create_error(ErrorType.PROVIDER_INVALID_RESPONSE, original_exception=e,
                             error_details=f"Non-JSON response (status {response.status_code})",
                             request_id=request_id, provider_name=self.provider_name) from e
        except httpx.HTTPStatusError as e:
            self._raise_provider_http_error(e, request_id)
        except httpx.PoolTimeout as e:
            # Same condition as the stream path: the pool is exhausted, not
            # the network broken — 503, not a 500 network error.
            raise create_error(ErrorType.SERVICE_UNAVAILABLE, original_exception=e,
                             error_details="Connection pool exhausted. Please retry later.",
                             request_id=request_id, provider_name=self.provider_name) from e
        except httpx.RequestError as e:
            raise create_error(ErrorType.PROVIDER_NETWORK_ERROR, original_exception=e,
                             error_details=str(e), request_id=request_id, provider_name=self.provider_name) from e

        self._log_provider_data(title="Provider Response", data=response_json,
                                request_id=request_id, data_flow="from_provider")
        return response_json

    def _log_request_start(self, url: str, headers: dict[str, str], request_body: dict[str, Any],
                           files: dict[str, Any], data: dict[str, Any], request_id: str) -> None:
        """Log the outgoing request (masked headers) before it is sent."""
        self._log_provider_data(
            title="Provider Request",
            data={
                "url": url,
                "headers": mask_headers(headers),
                "request_body": request_body,
                "has_files": files is not None,
                "has_data": data is not None
            },
            request_id=request_id,
            data_flow="to_provider"
        )

    async def _send(self, method: str, url: str, headers: dict[str, str],
                    request_body: dict[str, Any] | None, files: dict[str, Any] | None,
                    data: dict[str, Any] | None,
                    timeout: httpx.Timeout | None) -> httpx.Response:  # noqa: ASYNC109
        """Issue the POST or GET on the owned client; no status check."""
        if method.upper() == "POST":
            # WHY: multipart uploads (files) need httpx to set Content-Type with boundary;
            # explicit Content-Type: application/json would break the multipart encoding
            if files:
                headers.pop("Content-Type", None)

            return await self.pool.client.post(
                url,
                headers=headers,
                json=request_body if not files else None,
                files=files,
                data=data,
                timeout=timeout
            )
        if method.upper() == "GET":
            return await self.pool.client.get(url, headers=headers, params=request_body, timeout=timeout)
        raise ValueError(f"Unsupported HTTP method: {method}")

    async def _stream_request(self, url_path: str, request_body: dict[str, Any],
                              request_id: str = "unknown",
                              extra_headers: dict[str, str] = None) -> AsyncGenerator[bytes, None]:
        """Async generator streaming raw bytes from a provider API.

        Holds a per-provider concurrency slot across the ENTIRE iteration by the
        downstream consumer. `async with` releases on normal completion, on
        exception, and on generator close (AClose) — so a client disconnect also
        frees the slot. extra_headers are merged exactly like in _request.
        """
        async with self.pool.acquire_slot(request_id):
            async for chunk in self._stream_request_inner(url_path, request_body, request_id,
                                                          extra_headers):
                yield chunk

    # WHY: no retry loop here, unlike _request — streaming is driven through
    # open_provider_stream, which primes the first chunk BEFORE the response
    # starts, so an upstream 429 already surfaces to the client as its real
    # HTTP status instead of a 200 with an error frame; a retry loop at this
    # layer would be unreachable for connection-time failures and would
    # double-send a generation the client may already have partially received.
    async def _stream_request_inner(self, url_path: str, request_body: dict[str, Any],
                                    request_id: str = "unknown",
                                    extra_headers: dict[str, str] = None) -> AsyncGenerator[bytes, None]:
        """Actual streaming implementation. See _stream_request for the slot wrapper.

        Uses client.stream() context manager for memory-efficient chunk iteration.
        Headers go through _merge_request_headers so the stream and non-stream
        paths send an identical set.
        Error hierarchy inside stream context:
        - HTTPStatusError: the error body is read first, ResponseNotRead fallback if that fails
        - PoolTimeout → 503 (connection pool exhausted)
        - RequestError → generic network error
        """
        url = f"{self.base_url}{url_path}"
        merged_headers = self._merge_request_headers(extra_headers)
        stream_timeout = self._create_timeout(read=self.settings.stream_read_timeout)
        self._log_stream_start(url, merged_headers, request_body, stream_timeout, request_id)

        start_time = time.time()
        try:
            async with self.pool.client.stream("POST", url, headers=merged_headers,
                                               json=request_body,
                                               timeout=stream_timeout) as response:
                await self._check_stream_response(response, start_time, request_id)
                async for chunk in response.aiter_bytes():
                    yield chunk
                logger.debug(f"Provider stream finished for {request_id}")
        # WHY: PoolTimeout means all connections in use, not a network failure — maps to 503
        except httpx.PoolTimeout as e:
            raise create_error(ErrorType.SERVICE_UNAVAILABLE,
                             original_exception=e,
                             error_details="Connection pool exhausted. Please retry later.",
                             request_id=request_id,
                             provider_name=self.provider_name) from e
        except httpx.RequestError as e:
            raise create_error(ErrorType.PROVIDER_NETWORK_ERROR,
                             original_exception=e,
                             error_details=str(e),
                             request_id=request_id,
                             provider_name=self.provider_name) from e
        except HTTPException:
            # already logged via log_provider_error in _raise_provider_http_error
            raise
        except Exception as e:
            logger.error(f"Stream request failed after {time.time() - start_time:.2f}s: {str(e)}", extra={
                "error_type": type(e).__name__,
                "request_id": request_id
            }, exc_info=True)
            raise

    def _log_stream_start(self, url: str, headers: dict[str, str], request_body: dict[str, Any],
                          stream_timeout: httpx.Timeout, request_id: str) -> None:
        """Log the outgoing stream request (masked headers) before it is sent."""
        self._log_provider_data(
            title="Provider Stream Request",
            data={
                "url": url,
                "headers": mask_headers(headers),
                "request_body": request_body
            },
            request_id=request_id,
            data_flow="to_provider"
        )
        logger.debug(f"Starting stream request to {url}", extra={
            "url": url,
            "timeout": str(stream_timeout),
            "request_id": request_id
        })

    async def _check_stream_response(self, response: httpx.Response, start_time: float,
                                     request_id: str) -> None:
        """Log the stream's response headers and raise on a non-2xx status."""
        logger.debug(f"Stream response headers received after {time.time() - start_time:.2f}s", extra={
            "status_code": response.status_code,
            "request_id": request_id
        })
        self._log_provider_data(
            title="Provider Response Headers",
            data={
                "status_code": response.status_code,
                "headers": dict(response.headers)
            },
            request_id=request_id,
            data_flow="from_provider"
        )
        if response.status_code >= 400:
            # WHY: the stream context never buffers the body, so without this
            # read the client gets a placeholder instead of the upstream's error
            # text. An error body is small and bounded by stream_read_timeout; a
            # failed read still surfaces the upstream status via the
            # ResponseNotRead fallback in _raise_provider_http_error.
            with contextlib.suppress(httpx.HTTPError):
                await response.aread()
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            self._raise_provider_http_error(e, request_id)

    async def chat_completions(self, request_body: dict[str, Any], provider_model_name: str,
                               model_config: ModelEntry, request_id: str = "unknown",
                               extra_headers: dict[str, str] = None) -> dict[str, Any]:
        """Forward a non-streaming chat completion. Returns the parsed provider JSON response."""
        request_body = self._apply_model_config(request_body, provider_model_name, model_config)

        connect_timeout = self.settings.openai_connect_timeout
        # WHY: read is capped by stream_read_timeout — the same env knob aclose()
        # drains on as "the longest a legitimate request may run" — so a silent
        # upstream cannot hold the concurrency slot and _inflight forever.
        read_timeout = self.settings.stream_read_timeout
        non_stream_timeout = self._create_timeout(connect=connect_timeout, read=read_timeout)

        return await self._request(
            method="POST",
            path="/chat/completions",
            request_body=request_body,
            extra_headers=extra_headers,
            timeout=non_stream_timeout,
            request_id=request_id
        )

    def chat_completions_stream(self, request_body: dict[str, Any], provider_model_name: str,
                                model_config: ModelEntry, request_id: str = "unknown",
                                extra_headers: dict[str, str] = None) -> AsyncGenerator[bytes, None]:
        """Forward a streaming chat completion. Yields raw SSE bytes from the provider."""
        request_body = self._apply_model_config(request_body, provider_model_name, model_config)
        return self._stream_request("/chat/completions", request_body,
                                    request_id=request_id, extra_headers=extra_headers)

    async def transcriptions(self, request_body: dict[str, Any], provider_model_name: str,
                             model_config: ModelEntry, request_id: str = "unknown",
                             extra_headers: dict[str, str] = None) -> dict[str, Any]:
        """Send audio to an OpenAI-compatible /audio/transcriptions endpoint.

        request_body shape:

            {"audio": {"filename": str, "content_type": str, "data": bytes},
             "params": {"language"?, "temperature"?, "response_format"?,
                         "return_timestamps"?, "prompt"?}}

        Uses provider's own credentials from self.headers (set in __init__).
        extra_headers (client identity) ride along; the multipart Content-Type
        is still set by httpx — _send pops it for multipart bodies.
        """
        audio = request_body["audio"]
        params = dict(request_body.get("params") or {})

        # WHY: return_timestamps is a non-standard convenience flag; OpenAI Whisper
        # exposes the same data via response_format=verbose_json
        return_timestamps = params.pop("return_timestamps", None)
        if return_timestamps:
            params["response_format"] = "verbose_json"
        params.setdefault("response_format", "json")

        # Drop None values so we don't send empty form fields
        form = {k: v for k, v in params.items() if v is not None}
        form = self._apply_model_config(form, provider_model_name, model_config)

        # WHY raw bytes, not io.BytesIO: the 429 retry loop lives below this
        # construction site (in _request), so the
        # files tuple is encoded once per attempt. httpx's multipart encoder
        # happens to seek(0) seekable file objects today, but bytes make each
        # attempt self-contained by construction instead of by accommodation —
        # the provider layer must not depend on upload rewinding.
        files = {"file": (audio["filename"], audio["data"], audio["content_type"])}

        transcription_read_timeout = self.settings.openai_transcription_timeout
        transcription_timeout = self._create_timeout(read=transcription_read_timeout)

        return await self._request(
            method="POST",
            path="/audio/transcriptions",
            files=files,
            data=form,
            extra_headers=extra_headers,
            timeout=transcription_timeout,
            request_id=request_id
        )

    async def embeddings(self, request_body: dict[str, Any], provider_model_name: str,
                         model_config: ModelEntry, request_id: str = "unknown",
                         extra_headers: dict[str, str] = None) -> Any:
        """Forward an embedding request to an OpenAI-compatible API."""
        request_body = self._apply_model_config(request_body, provider_model_name, model_config)

        read_timeout = self.settings.openai_embeddings_read_timeout
        # WHY: no hardcoded connect/write/pool — the client's own defaults
        # (HTTPX_CONNECT_TIMEOUT etc.) cover them via _create_timeout fallback.
        embeddings_timeout = self._create_timeout(read=read_timeout)

        return await self._request(
            method="POST",
            path="/embeddings",
            request_body=request_body,
            extra_headers=extra_headers,
            timeout=embeddings_timeout,
            request_id=request_id
        )

    async def list_models(self, request_id: str = "unknown") -> dict[str, Any]:
        """Return the provider's /models list (raw response)."""
        return await self._request(
            method="GET",
            path="/models",
            request_id=request_id
        )
