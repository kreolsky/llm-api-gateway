"""Base service: model access resolution, provider lookup and the shared dispatch funnel."""
# SYSTEM: service-layer — validate access, resolve provider, dispatch

import contextlib
from dataclasses import dataclass, fields
from typing import Any

from fastapi import HTTPException, Request

from ..core.config_manager import ConfigManager
from ..core.config_schema import ModelEntry
from ..core.context import AuthContext, request_context
from ..core.error_handling import ErrorType, create_error
from ..core.header_policy import (
    FORWARDED_HEADER_DENY_PREFIXES,
    FORWARDED_HEADER_DENYLIST,
)
from ..core.logging import logger
from ..core.usage_db import RequestStats, request_stats
from ..providers import ProviderRegistry
from ..providers.base import Provider
from .reasoning_dialect import translate_reasoning_fields
from .reasoning_effort import apply_reasoning_effort


@dataclass(frozen=True)
class Protocol:
    """The wire protocol a provider type speaks: the api name services pass to
    the dispatch funnel and the client-facing path that protocol is served
    over (named in WRONG_API so a client learns where to resend)."""
    api: str
    path: str


# ARCH: one protocol per provider type — a model's protocol is its provider's
# type, so the /v1/messages <-> /v1/chat/completions routing question has one
# answer per model and every endpoint enforces it through the same funnel.
PROTOCOL_BY_PROVIDER_TYPE: dict[str, Protocol] = {
    "openai": Protocol(api="openai-completions", path="/v1/chat/completions"),
    "anthropic": Protocol(api="anthropic-messages", path="/v1/messages"),
}


@dataclass(frozen=True)
class ResolvedTarget:
    """Result of the body-agnostic dispatch resolver (BaseService._resolve_target).

    Everything a dispatch needs once the model id is known: validated config,
    the registry's provider instance, and the per-request identity headers.
    Carries no request body — JSON (chat/embeddings) and multipart
    (transcription) dispatches both ride it, so a cross-cutting policy added
    here reaches every endpoint by construction.
    """
    request_id: str
    user_id: str
    stats: RequestStats
    error_ctx: dict[str, Any]
    model_config: ModelEntry
    provider_name: str
    provider_model_name: str
    provider: Provider
    identity_headers: dict[str, str] | None


@dataclass(frozen=True)
class PreparedDispatch(ResolvedTarget):
    """Result of the shared service preamble (BaseService._prepare_dispatch).

    The JSON wrapper around ResolvedTarget: adds the parsed body and the
    requested model, applies the reasoning-effort policy, then the per-provider
    dialect translation (both body-shaped concerns that must NOT ride the
    shared resolver — multipart endpoints carry neither).
    """
    request_body: dict[str, Any]
    requested_model: str | None


class BaseService:
    """Common base for ChatService, EmbeddingService, ModelService, and TranscriptionService."""

    def __init__(self, config_manager: ConfigManager, registry: ProviderRegistry):
        self.config_manager = config_manager
        self.registry = registry

    @contextlib.asynccontextmanager
    async def _guard_service_errors(self, error_ctx: dict[str, Any]):
        """Wrap a service block: re-raise HTTPException as-is, wrap other errors.

        De-duplicates the identical try/except error-wrapping pattern across
        chat/embedding/transcription services.
        """
        try:
            yield
        except HTTPException:
            raise
        except Exception as e:
            raise create_error(
                ErrorType.INTERNAL_SERVER_ERROR,
                original_exception=e,
                error_details=str(e),
                **error_ctx,
            ) from e

    def _extract_passthrough_headers(self, request: Request | None) -> dict[str, str]:
        """Collect every client header minus the denylist (core/header_policy.py).

        WHY: passthrough forwards headers verbatim (the client's own spelling —
        casing is part of a harness fingerprint), so there is no whitelist to
        apply; the denylist is what keeps client credentials, stale transport
        values, and lab topology from going upstream.
        """
        if request is None:
            return {}
        forwarded: dict[str, str] = {}
        for name, value in request.headers.items():
            low = name.lower()
            if low in FORWARDED_HEADER_DENYLIST:
                continue
            if any(low.startswith(prefix) for prefix in FORWARDED_HEADER_DENY_PREFIXES):
                continue
            forwarded[name] = value
        return forwarded

    def _build_identity_headers(self, provider_instance: Any,
                                 request: Request | None) -> dict[str, str] | None:
        """Per-request upstream headers for providers with an identity profile.

        passthrough: forward the client's headers verbatim minus the denylist.
        Unset identity: returns None — no client headers go upstream.
        """
        identity = getattr(provider_instance, "identity", None)
        if not identity:
            return None
        return self._extract_passthrough_headers(request) or None

    def resolve_model(
        self,
        model_id: str,
        auth_context: AuthContext,
        /,
        **error_ctx: Any,
    ) -> ModelEntry:
        """The one access resolver: allowed -> exists -> its provider exists.

        Shared by the dispatch funnel (_resolve_target) and the model-detail
        endpoint (ModelService.retrieve_model). Returns the model's entry; the
        caller decides how to name the upstream model. model_id is
        positional-only because error_ctx carries a model_id of its own.
        """
        # INVARIANT: check allowed_models BEFORE checking existence to prevent
        # information leakage about configured models
        allowed_models = auth_context.allowed_models
        if allowed_models and model_id not in allowed_models:
            raise create_error(ErrorType.MODEL_NOT_ALLOWED, **error_ctx)

        current_config = self.config_manager.get_config()
        model_config = current_config.models.get(model_id)
        if not model_config:
            raise create_error(ErrorType.MODEL_NOT_FOUND, **error_ctx)

        if model_config.provider not in current_config.providers:
            raise create_error(ErrorType.PROVIDER_NOT_FOUND, provider_name=model_config.provider, **error_ctx)
        return model_config

    async def _parse_json_request(self, request: Request) -> dict[str, Any]:
        """Parse the JSON request body, answering 400 on malformed input."""
        try:
            return await request.json()
        # WHY: ValueError, not json.JSONDecodeError — invalid UTF-8 bodies raise
        # UnicodeDecodeError (also a ValueError) and must answer 400, not 500
        except ValueError:
            ctx = request_context(request)
            # from None: the client's own malformed body is the whole story
            raise create_error(ErrorType.MISSING_REQUIRED_FIELD, field_name="valid JSON body",
                             request_id=ctx.request_id, user_id=ctx.user_id) from None

    async def _resolve_target(
        self,
        request: Request,
        auth_context: AuthContext,
        model_id: Any,
        *,
        api: str,
    ) -> ResolvedTarget:
        """Body-agnostic dispatch funnel: enrich stats, validate access,
        resolve the provider, enforce the protocol gate, and build the
        identity headers.

        A non-string model id (client sent JSON garbage) reaches the usage row
        as "" but keeps its raw value in error_ctx — the error messages quote
        what the client actually sent.

        INVARIANT: identity_headers is computed exactly ONCE per request here,
        and the SAME object feeds the stream and non-stream branches.
        Why: the provider layer requires both paths to send an identical set
        (providers/base.py _merge_request_headers ARCH), and a per-branch
        recompute would hold that by luck, not by construction.
        """
        ctx = request_context(request)
        request_id = ctx.request_id
        user_id = ctx.user_id

        stats = request_stats(request)
        stats.model_id = model_id if isinstance(model_id, str) else ""

        error_ctx = {"request_id": request_id, "user_id": user_id, "model_id": model_id}

        if not model_id:
            raise create_error(ErrorType.MODEL_NOT_SPECIFIED, **error_ctx)
        model_config, provider_instance = self._resolve_provider(
            model_id, auth_context, api, error_ctx)
        stats.provider_name = model_config.provider

        identity_headers = self._build_identity_headers(provider_instance, request)

        return ResolvedTarget(
            request_id=request_id, user_id=user_id, stats=stats, error_ctx=error_ctx,
            model_config=model_config, provider_name=model_config.provider,
            provider_model_name=model_config.provider_model_name or model_id,
            provider=provider_instance, identity_headers=identity_headers,
        )

    def _resolve_provider(
        self,
        model_id: str,
        auth_context: AuthContext,
        api: str,
        error_ctx: dict[str, Any],
    ) -> tuple[ModelEntry, Provider]:
        """resolve_model + registry lookup + the protocol gate.

        ``api`` is the calling endpoint's protocol ("openai-completions" from
        chat/embeddings/transcription, "anthropic-messages" from /v1/messages);
        AFTER resolve_model (access before existence stays intact) the
        provider instance's type is checked against it, so a model is only
        ever dispatched over the protocol its provider speaks — by
        construction, for every endpoint.
        """
        model_config = self.resolve_model(model_id, auth_context, **error_ctx)
        provider_instance = self.registry.get(model_config.provider)
        # INVARIANT: the protocol is read from the provider INSTANCE's own
        # entry, never from the config dict.
        # Why: the config and the registry are swapped separately on a reload,
        # so a config-side entry can belong to a different generation than the
        # backend actually called; the instance's entry is the one it was built
        # from (same rule as the reasoning dialect in _prepare_dispatch).
        protocol = PROTOCOL_BY_PROVIDER_TYPE[provider_instance.entry.type]
        if protocol.api != api:
            raise create_error(ErrorType.WRONG_API, expected_path=protocol.path, **error_ctx)
        return model_config, provider_instance

    async def _prepare_dispatch(
        self,
        request: Request,
        auth_context: AuthContext,
        *,
        component: str,
        log_title: str,
        api: str,
    ) -> PreparedDispatch:
        """Thin JSON wrapper over _resolve_target: parse the body, log it,
        delegate, then apply the reasoning-effort policy to the parsed body.

        The effort policy is deliberately HERE and not in the resolver: it is
        body-shaped (reads/writes the JSON body), and multipart endpoints
        riding the resolver must never get it. ``api`` is passed through to
        the resolver's protocol gate unchanged.
        """
        request_body = await self._parse_json_request(request)
        requested_model = request_body.get("model")

        logger.debug_data(title=log_title, data=request_body,
                          request_id=request_context(request).request_id,
                          component=component, data_flow="incoming")

        target = await self._resolve_target(request, auth_context, requested_model, api=api)

        # ARCH: the effort policy rides the one dispatch funnel (services/reasoning_effort.py).
        request_body = apply_reasoning_effort(request_body, target.model_config.effort_policy,
                                              **target.error_ctx)

        # ARCH: the dialect translation rides the same funnel
        # (services/reasoning_dialect.py), AFTER the policy — the value the
        # gate ruled legal is what gets re-nested for the upstream's dialect.
        # INVARIANT: the dialect is read from the provider INSTANCE's own entry,
        # never from the config dict.
        # Why: the config and the registry are swapped separately on a reload,
        # so a config-side entry can belong to a different generation than the
        # backend actually called; the instance's entry is the one it was built from.
        request_body = translate_reasoning_fields(request_body, target.provider.entry.reasoning_dialect)

        # Fields are DERIVED from ResolvedTarget, not re-listed: a field added
        # to the resolver reaches the JSON wrapper without a second edit.
        return PreparedDispatch(
            request_body=request_body, requested_model=requested_model,
            **{f.name: getattr(target, f.name) for f in fields(ResolvedTarget)},
        )
