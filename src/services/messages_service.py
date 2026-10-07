"""Messages service: the /v1/messages (Anthropic Messages) funnel.

ARCH: deliberately does NOT ride BaseService._prepare_dispatch — the effort
policy and the dialect translation are OpenAI-wire body surgery, and a
Messages body must reach its provider untouched except for the `model`
rewrite and the models.yaml `options:` merge (both live in the provider
layer). What it shares with every other endpoint is exactly
``_parse_json_request`` + ``_resolve_target(api="anthropic-messages")``, so
access control, the protocol gate, stats enrichment and identity headers
reach this path by construction.
"""

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..core.context import AuthContext, request_context
from ..core.logging import logger
from ..services.base import BaseService, ResolvedTarget
from .chat_service.stream_processor import ANTHROPIC_STREAM, open_provider_stream, process_stream

# Protocol (not identity) headers forwarded on this path even without
# identity: passthrough — the Messages wire requires the version header and
# beta flags to mean the same thing upstream.
_PROTOCOL_HEADERS = ("anthropic-version", "anthropic-beta")


class MessagesService(BaseService):
    """Forwards native Anthropic Messages requests to anthropic-type providers."""

    async def messages(self, request: Request, auth_context: AuthContext) -> Any:
        """Process a Messages request, returning StreamingResponse or JSONResponse."""
        request_body = await self._parse_json_request(request)

        logger.debug_data(title="Messages Request JSON", data=request_body,
                          request_id=request_context(request).request_id,
                          component="messages_service", data_flow="incoming")

        target = await self._resolve_target(
            request, auth_context, request_body.get("model"), api="anthropic-messages")

        async with self._guard_service_errors(target.error_ctx):
            # INVARIANT: the protocol headers merge into the SAME per-request
            # header dict the resolver built (one object, both branches).
            # Why: the provider layer requires the stream and non-stream paths
            # to send an identical set (providers/base.py
            # _merge_request_headers ARCH) — see the INVARIANT on
            # _resolve_target for the same rule.
            extra_headers = self._messages_headers(target, request)

            if request_body.get("stream", False):
                return await self._stream_response(target, request_body, extra_headers)

            response_data = await target.provider.messages(
                request_body, target.provider_model_name, target.model_config,
                request_id=target.request_id, extra_headers=extra_headers,
            )

            logger.debug_data(
                title="Messages Response JSON", data=response_data,
                request_id=target.request_id, component="messages_service",
                data_flow="from_provider",
            )

            usage = response_data.get("usage", {})
            if usage:
                target.stats.set_anthropic_usage(usage)

            return JSONResponse(content=response_data)

    def _messages_headers(self, target: ResolvedTarget, request: Request) -> dict[str, str] | None:
        """Identity headers (if any) plus the protocol headers — or None when
        the client sent neither an identity profile nor protocol headers."""
        extra = dict(target.identity_headers or {})
        for name in _PROTOCOL_HEADERS:
            if name in request.headers:
                extra[name] = request.headers[name]
        return extra or None

    async def _stream_response(self, target: ResolvedTarget, request_body: dict[str, Any],
                               extra_headers: dict[str, str] | None) -> StreamingResponse:
        """Stream branch: mark stats, open the provider stream before the
        response starts, wrap it in the Anthropic-wire SSE processor."""
        target.stats.stream = True
        provider_stream = target.provider.messages_stream(
            request_body, target.provider_model_name, target.model_config,
            request_id=target.request_id, extra_headers=extra_headers,
        )
        # Surface an upstream failure as a real HTTP status instead of a
        # 200 carrying an SSE error frame (see open_provider_stream).
        provider_stream = await open_provider_stream(provider_stream)
        logger.debug_data(
            title="Streaming Response Started",
            data={"streaming": True, "model": request_body.get("model"),
                  "request_id": target.request_id},
            request_id=target.request_id, component="messages_service",
            data_flow="from_provider",
        )

        return StreamingResponse(
            process_stream(
                provider_stream, request_body.get("model"), target.request_id,
                target.user_id, target.provider_name, stats=target.stats,
                wire=ANTHROPIC_STREAM,
            ),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )
