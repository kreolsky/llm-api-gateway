"""Model listing and detail retrieval with merged capability data.

ARCH: the hot path (/v1/models, /v1/models/{id}) reads capabilities from an
in-memory merged store and NEVER touches the network. Capability data is
sourced from two layers (see src/core/model_capabilities/):

  1. the auto-cache (CapabilitiesCache, refreshed by a background task);
  2. config/model_info.yaml (manual override).

INVARIANT: model_info.yaml ALWAYS wins over the auto-cache.
"""
import time
from typing import Any

from ..core.context import AuthContext
from ..core.logging import logger
from ..core.model_capabilities import (
    CapabilitiesCache,
    merge_capabilities,
    render_capabilities,
)
from ..providers import ProviderRegistry
from .base import PROTOCOL_BY_PROVIDER_TYPE, BaseService
from .capabilities_refresh import refresh_provider_capabilities


class ModelService(BaseService):
    """Lists models and retrieves single-model details from merged capabilities."""

    def __init__(self, config_manager, registry: ProviderRegistry,
                 capabilities_cache: CapabilitiesCache):
        super().__init__(config_manager, registry)
        self.capabilities_cache = capabilities_cache
        self._warned_effort_defaults: set[tuple[str, Any, tuple[Any, ...]]] = set()

    def _build_model_response(self, model_id: str, **extra_fields) -> dict[str, Any]:
        """Build a standardized model response object."""
        return {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nnp-llm-router",
            "parent": None,
            "permission": [{
                "id": f"model-perm-{model_id}",
                "object": "model_permission",
                "created": int(time.time()),
                "allow_create_engine": False,
                "allow_sampling": True,
                "allow_logprobs": False,
                "allow_search_indices": False,
                "allow_view": True,
                "allow_fine_tuning": False,
                "organization": "*",
                "group": None,
                "is_blocking": False
            }],
            "root": model_id,
            **extra_fields
        }

    def _resolve_stored_capabilities(self, model_id: str) -> dict[str, Any]:
        """Merge the auto-cache, the derived effort policy, and manual model_info
        into the STORED form.

        INVARIANT: model_info.yaml always wins over the auto-cache (deep merge
        where lists are replaced, not concatenated).

        Layering: merge_capabilities(merge_capabilities(cache, derived), model_info)
        — the reasoning_effort key derived from models.yaml beats the auto-cache,
        model_info.yaml still beats everything, so the invariant holds unchanged.
        The derived block (reasoning.supported/effort_levels/default_effort) comes
        from the SAME key the dispatch funnel enforces, never re-typed, so
        list_models and retrieve_model cannot diverge from the wire behaviour.
        """
        cache_data = self.capabilities_cache.get(model_id) or {}
        config = self.config_manager.get_config()
        model_info = config.model_info.get(model_id) or {}
        derived: dict[str, Any] = {}
        model_entry = config.models.get(model_id)
        policy = model_entry.effort_policy if model_entry is not None else None
        if policy is not None:
            reasoning: dict[str, Any] = {
                "supported": True,
                "effort_levels": list(policy.allowed),
            }
            if policy.default is not None:
                reasoning["default_effort"] = policy.default
            derived = {"reasoning": reasoning}
        layered = merge_capabilities(cache_data, derived)
        self._drop_foreign_default_effort(model_id, layered)
        return merge_capabilities(layered, model_info)

    def _drop_foreign_default_effort(self, model_id: str, stored: dict[str, Any]) -> None:
        """Drop a cache-sourced ``default_effort`` that is outside ``effort_levels``.

        INVARIANT: /v1/models never advertises a default effort outside the
        advertised effort_levels — the dispatch gate would 400 that value.
        Why: upstream entries carry their own vocabulary (OpenRouter ships
        "minimal"; a chained router ships its own wider policy), and a merge
        replaces the levels list but keeps the foreign default.
        Runs before model_info, which stays the operator's final word. Warned
        once per (model, value, levels): this path runs on every request.
        """
        reasoning = stored.get("reasoning")
        if not isinstance(reasoning, dict):
            return
        levels = reasoning.get("effort_levels")
        default = reasoning.get("default_effort")
        if not isinstance(levels, list) or default is None or default in levels:
            return
        del reasoning["default_effort"]
        key = (model_id, default, tuple(levels))
        if key in self._warned_effort_defaults:
            return
        self._warned_effort_defaults.add(key)
        logger.warning(
            f"Model '{model_id}': upstream default_effort {default!r} is not in "
            f"effort_levels {levels} — dropped from /v1/models. Fix: add it to "
            f"reasoning_effort.allowed in models.yaml, or set reasoning.default_effort "
            f"in model_info.yaml",
            error_type="capabilities_config_conflict",
        )

    def get_pricing(self, model_id: str) -> dict[str, Any] | None:
        """Stored per-token pricing for a model, or None when unknown.

        Public wrapper over _resolve_stored_capabilities for the usage-stats
        flush. In-memory, never touches the network. Returned values follow
        the stored form: USD per token, only the keys the sources provided.
        """
        if not model_id:
            return None
        pricing = self._resolve_stored_capabilities(model_id).get("pricing")
        return pricing if isinstance(pricing, dict) and pricing else None

    def _model_api(self, model_id: str) -> str | None:
        """The protocol this model is served over — its provider's type.

        One derivation for list_models and retrieve_model, so /v1/models and
        /v1/models/{id} cannot disagree about where to send a client. None
        when the provider entry is missing or of an unknown type (a dangling
        reference already warned at load; retrieval 404s before rendering).
        """
        config = self.config_manager.get_config()
        model_entry = config.models.get(model_id)
        if model_entry is None or model_entry.provider not in config.providers:
            return None
        provider_type = config.providers[model_entry.provider].type
        protocol = PROTOCOL_BY_PROVIDER_TYPE.get(provider_type)
        return protocol.api if protocol else None

    def _capability_meta(self, model_id: str) -> dict[str, Any]:
        """Provenance fields (source, fetched_at) for diagnostics, when available."""
        meta = self.capabilities_cache.get_meta(model_id)
        if meta is None:
            return {}
        return {
            "capabilities_source": meta.get("source"),
            "capabilities_fetched_at": meta.get("fetched_at"),
        }

    def _iter_visible_models(self, allowed_models: Any):
        """Yield ``(model_id, model_data)`` for every model the key may see.

        ONE visibility filter for /v1/models and /v1/capabilities:
        ``is_hidden`` models are dropped and ``allowed_models`` applies
        (empty/None = unrestricted) — the two listings cannot drift because
        neither carries its own copy of the rule.
        """
        models_config = self.config_manager.get_config().models
        for model_id, model_data in models_config.items():
            if model_data.is_hidden:
                continue
            if allowed_models and model_id not in allowed_models:
                continue
            yield model_id, model_data

    async def list_models(self, auth_context: AuthContext) -> dict[str, Any]:
        """Return OpenAI-compatible model list filtered by allowed_models and is_hidden.

        Capability fields are rendered identically to retrieve_model, so the
        list and the detail endpoint never diverge.
        """
        models_list = []
        for model_id, _model_data in self._iter_visible_models(auth_context.allowed_models):
            stored = self._resolve_stored_capabilities(model_id)
            rendered = render_capabilities(stored)
            api = self._model_api(model_id)
            if api is not None:
                rendered["api"] = api
            models_list.append(self._build_model_response(model_id, **rendered))
        return {"object": "list", "data": models_list}

    async def capabilities(self, auth_context: AuthContext) -> dict[str, dict[str, Any]]:
        """Flat per-model reasoning map for the GET /v1/capabilities endpoint.

        Reuses _resolve_stored_capabilities — the same derivation /v1/models
        renders — so the two listings cannot drift (one derivation, no
        re-typing of the policy). Shape per model:
        ``{"supported": bool, "effort_levels": [...]}`` with an empty list
        when nothing is advertised. Filtered by the same visibility iterator
        as list_models (_iter_visible_models).
        """
        out: dict[str, dict[str, Any]] = {}
        for model_id, _model_data in self._iter_visible_models(auth_context.allowed_models):
            reasoning = self._resolve_stored_capabilities(model_id).get("reasoning") or {}
            out[model_id] = {
                "supported": bool(reasoning.get("supported")),
                "effort_levels": list(reasoning.get("effort_levels") or []),
            }
        return out

    async def retrieve_model(
        self,
        model_id: str,
        auth_context: AuthContext,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Return model details from merged capabilities (no live upstream call).

        The hot path reads only from the in-memory store. ``refresh=True``
        (debug) triggers a best-effort background refresh of the model's
        provider before reading; failures are non-fatal (stale-if-error).
        """
        model_data = self.resolve_model(model_id, auth_context, model_id=model_id)
        if refresh:
            await self._debug_refresh(model_id, model_data.provider)

        stored = self._resolve_stored_capabilities(model_id)
        rendered = render_capabilities(stored)
        meta = self._capability_meta(model_id)
        api = self._model_api(model_id)
        if api is not None:
            rendered["api"] = api

        return self._build_model_response(
            model_id,
            provider=model_data.provider,
            provider_model_name=model_data.provider_model_name,
            params=model_data.params,
            options=model_data.options,
            **rendered,
            **meta,
        )

    async def _debug_refresh(self, model_id: str, provider_name: str) -> None:
        """The ``?refresh=true`` path: refresh one provider's capabilities in place."""
        # ARCH: optional debug refresh — NOT the default path. Best effort,
        # errors swallowed (stale-if-error). The result is still read from cache.
        if not self.config_manager.settings.model_cache_enabled:
            return
        try:
            await refresh_provider_capabilities(self.config_manager, self.registry,
                                                self.capabilities_cache, provider_name)
        except Exception as e:
            logger.warning(
                f"Debug capabilities refresh failed for {model_id}: {e}",
                error_type="capabilities_refresh_error",
            )
