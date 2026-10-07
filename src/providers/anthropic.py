"""The Anthropic-protocol provider: POST {base_url}/v1/messages pass-through."""
# SYSTEM: anthropic-provider — Anthropic Messages wire (x-api-key + anthropic-version)

from collections.abc import AsyncGenerator
from typing import Any

from ..core.config_schema import ModelEntry
from ..core.error_handling import ErrorType, create_error
from .base import Provider

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(Provider):
    """Speaks exactly one protocol: the Anthropic Messages API.

    ARCH: a provider speaks exactly one protocol, so this class only swaps the
    credential/version headers and the endpoint paths — pool, retry loop,
    timeouts, header merging and the options merge (with its `stream` guard)
    are inherited unchanged from Provider. A dual-protocol vendor gets two
    provider entries in providers.yaml, one per protocol.
    """

    def _auth_headers(self) -> dict[str, str]:
        """Anthropic wire credentials: x-api-key (never Authorization), plus
        the default protocol version."""
        headers: dict[str, str] = {}
        # WHY: anthropic-version is a DEFAULT, not a credential — a static
        # `headers:` entry in any casing replaces it (a case-sensitive dict
        # merge would send both spellings upstream), and a client value still
        # replaces it per request (it is not protected in _merge_request_headers).
        static_names = {name.lower() for name in self.entry.headers or {}}
        if "anthropic-version" not in static_names:
            headers["anthropic-version"] = ANTHROPIC_VERSION
        if self.api_key_env:
            if not self.api_key:
                raise create_error(
                    ErrorType.PROVIDER_CONFIG_ERROR,
                    error_details=(f"API key for {self.api_key_env} is not set "
                                   f"in environment variables."),
                    provider_name=self.provider_name,
                )
            headers["x-api-key"] = self.api_key
        return headers

    async def messages(self, request_body: dict[str, Any], provider_model_name: str,
                       model_config: ModelEntry, request_id: str = "unknown",
                       extra_headers: dict[str, str] = None) -> dict[str, Any]:
        """Forward a non-streaming Messages request. Returns the parsed JSON body."""
        request_body = self._apply_model_config(request_body, provider_model_name, model_config)
        # Same budget as chat_completions: read is capped by stream_read_timeout
        # so a silent upstream cannot hold its slot and the drain window open.
        timeout = self._create_timeout(connect=self.settings.openai_connect_timeout,
                                       read=self.settings.stream_read_timeout)
        return await self._request(
            method="POST",
            path="/v1/messages",
            request_body=request_body,
            extra_headers=extra_headers,
            timeout=timeout,
            request_id=request_id,
        )

    def messages_stream(self, request_body: dict[str, Any], provider_model_name: str,
                        model_config: ModelEntry, request_id: str = "unknown",
                        extra_headers: dict[str, str] = None) -> AsyncGenerator[bytes, None]:
        """Forward a streaming Messages request. Yields raw SSE bytes."""
        request_body = self._apply_model_config(request_body, provider_model_name, model_config)
        return self._stream_request("/v1/messages", request_body,
                                    request_id=request_id, extra_headers=extra_headers)

    async def list_models(self, request_id: str = "unknown") -> dict[str, Any]:
        """GET {base_url}/v1/models (Anthropic serves it; a 404 from another
        anthropic-wire backend lands in the capabilities refresh's
        stale-if-error path)."""
        return await self._request(
            method="GET",
            path="/v1/models",
            request_id=request_id,
        )
