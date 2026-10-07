"""Unit tests for src/services/model_service.py — ModelService class."""

import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from fastapi import HTTPException

from src.core.config_schema import parse_config
from src.core.context import AuthContext
from src.core.model_capabilities import CapabilitiesCache, normalize_provider_model
from src.providers import ProviderRegistry
from src.services.model_service import ModelService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_auth_context(allowed_models=None, allowed_endpoints=None):
    """Return an AuthContext matching what auth.get_api_key builds."""
    return AuthContext(allowed_models or [], allowed_endpoints or [])


def _make_config(models=None, providers=None, model_info=None):
    """Return a RouterConfig suitable for ConfigManager.get_config()."""
    result = {
        "models": models or {},
        "providers": providers or {},
    }
    if model_info:
        result["model_info"] = model_info
    return parse_config(result)


def _build_service(models=None, providers=None, model_info=None, cache=None, registry=None):
    """Build a ModelService with a mocked ConfigManager and registry; the
    cache defaults to an empty, never-loaded one (the service requires a cache)."""
    cm = MagicMock()
    cm.get_config.return_value = _make_config(models, providers, model_info)
    return ModelService(cm, registry or MagicMock(spec=ProviderRegistry),
                        cache if cache is not None else CapabilitiesCache(os.devnull))


def _make_cache(entries=None):
    """Build an in-memory CapabilitiesCache pre-populated with entries."""
    cache = CapabilitiesCache("data/unused.json")
    for model_id, data in (entries or {}).items():
        cache.upsert(model_id, data, source="test-provider")
    return cache


# Sample model configs used across tests
SAMPLE_MODELS = {
    "model-a": {"provider": "prov-a", "provider_model_name": "a-real"},
    "model-b": {"provider": "prov-b", "provider_model_name": "b-real"},
    "model-c": {"provider": "prov-c", "provider_model_name": "c-real"},
    "hidden-model": {"provider": "prov-a", "provider_model_name": "h-real", "is_hidden": True},
}

SAMPLE_PROVIDERS = {
    "prov-a": {"type": "openai", "base_url": "https://a.example.com"},
    "prov-b": {"type": "openai", "base_url": "https://b.example.com"},
    "prov-c": {"type": "openai", "base_url": "https://c.example.com"},
}


# ===================================================================
# list_models
# ===================================================================

class TestListModels:

    @pytest.mark.asyncio
    async def test_unrestricted_user_sees_all_visible(self):
        """Empty allowed_models → all non-hidden models returned."""
        svc = _build_service(models=SAMPLE_MODELS)
        auth_ctx = _make_auth_context(allowed_models=[])

        result = await svc.list_models(auth_ctx)

        ids = [m["id"] for m in result["data"]]
        assert sorted(ids) == ["model-a", "model-b", "model-c"]
        assert "hidden-model" not in ids

    @pytest.mark.asyncio
    async def test_restricted_user_sees_only_allowed(self):
        """Non-empty allowed_models → only those models returned."""
        svc = _build_service(models=SAMPLE_MODELS)
        auth_ctx = _make_auth_context(allowed_models=["model-a"])

        result = await svc.list_models(auth_ctx)

        ids = [m["id"] for m in result["data"]]
        assert ids == ["model-a"]

    @pytest.mark.asyncio
    async def test_restricted_user_multiple_allowed(self):
        """Partial overlap: only allowed models that exist in config are returned."""
        svc = _build_service(models=SAMPLE_MODELS)
        auth_ctx = _make_auth_context(allowed_models=["model-a", "model-c"])

        result = await svc.list_models(auth_ctx)

        ids = sorted(m["id"] for m in result["data"])
        assert ids == ["model-a", "model-c"]

    @pytest.mark.asyncio
    async def test_hidden_model_excluded_even_if_allowed(self):
        """Hidden model is excluded from list even when in allowed_models."""
        svc = _build_service(models=SAMPLE_MODELS)
        auth_ctx = _make_auth_context(allowed_models=["model-a", "hidden-model"])

        result = await svc.list_models(auth_ctx)

        ids = [m["id"] for m in result["data"]]
        assert "hidden-model" not in ids
        assert ids == ["model-a"]

    @pytest.mark.asyncio
    async def test_allowed_model_not_in_config(self):
        """allowed_models references a model not in config — no crash, empty result."""
        svc = _build_service(models=SAMPLE_MODELS)
        auth_ctx = _make_auth_context(allowed_models=["nonexistent-model"])

        result = await svc.list_models(auth_ctx)

        assert result["data"] == []

    @pytest.mark.asyncio
    async def test_all_models_hidden(self):
        """When every model is hidden, list is empty for unrestricted user."""
        models = {
            "h1": {"provider": "p", "is_hidden": True},
            "h2": {"provider": "p", "is_hidden": True},
        }
        svc = _build_service(models=models)
        auth_ctx = _make_auth_context(allowed_models=[])

        result = await svc.list_models(auth_ctx)

        assert result["data"] == []

    @pytest.mark.asyncio
    async def test_response_structure(self):
        """Response has OpenAI-compatible structure."""
        svc = _build_service(models=SAMPLE_MODELS)
        auth_ctx = _make_auth_context(allowed_models=[])

        result = await svc.list_models(auth_ctx)

        assert result["object"] == "list"
        assert isinstance(result["data"], list)
        for model in result["data"]:
            assert model["object"] == "model"
            assert "id" in model
            assert "created" in model
            assert "owned_by" in model

    @pytest.mark.asyncio
    async def test_empty_config(self):
        """No models in config → empty list."""
        svc = _build_service(models={})
        auth_ctx = _make_auth_context(allowed_models=[])

        result = await svc.list_models(auth_ctx)

        assert result["data"] == []

    @pytest.mark.asyncio
    async def test_model_info_rendered_in_list(self):
        """model_info fields from catalog are rendered into the list response."""
        model_info = {
            "model-a": {
                "description": "desc-a",
                "context_length": 8192,
                "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
                "pricing": {"completion": 0.001},
            },
        }
        svc = _build_service(models=SAMPLE_MODELS, model_info=model_info)
        auth_ctx = _make_auth_context(allowed_models=[])

        result = await svc.list_models(auth_ctx)

        model_a = next(m for m in result["data"] if m["id"] == "model-a")
        assert model_a["description"] == "desc-a"
        assert model_a["context_length"] == 8192
        # vision derived from input_modalities
        assert model_a["supports_vision"] is True
        assert model_a["architecture"]["modality"] == "text+image->text"
        # top_provider mirrors flat fields
        assert model_a["top_provider"]["context_length"] == 8192
        # pricing rendered to strings, missing keys filled with "0"
        assert model_a["pricing"]["completion"] == "0.001"
        assert model_a["pricing"]["prompt"] == "0"
        # model-b has no info entry → no capability fields
        model_b = next(m for m in result["data"] if m["id"] == "model-b")
        assert "description" not in model_b
        assert "supports_vision" not in model_b

    @pytest.mark.asyncio
    async def test_model_info_reasoning_passthrough_in_list(self):
        """reasoning block passes through unchanged (manual-only, not derived)."""
        model_info = {
            "model-b": {
                "reasoning": {"supported": True, "default_enabled": False},
            },
        }
        svc = _build_service(models=SAMPLE_MODELS, model_info=model_info)
        auth_ctx = _make_auth_context(allowed_models=[])

        result = await svc.list_models(auth_ctx)

        model_b = next(m for m in result["data"] if m["id"] == "model-b")
        assert model_b["reasoning"] == {"supported": True, "default_enabled": False}

    @pytest.mark.asyncio
    async def test_model_info_empty_catalog_no_crash(self):
        """No model_info in config → no extra fields, no crash."""
        svc = _build_service(models=SAMPLE_MODELS)
        auth_ctx = _make_auth_context(allowed_models=[])

        result = await svc.list_models(auth_ctx)

        for model in result["data"]:
            assert "description" not in model
            assert "context_length" not in model

    @pytest.mark.asyncio
    async def test_cache_feeds_list_when_no_model_info(self):
        """Auto-cache provides capabilities when no manual model_info entry exists."""
        cache = _make_cache({"model-a": {"context_length": 32768, "max_completion_tokens": 4096}})
        svc = _build_service(models=SAMPLE_MODELS, cache=cache)
        auth_ctx = _make_auth_context(allowed_models=[])

        result = await svc.list_models(auth_ctx)

        model_a = next(m for m in result["data"] if m["id"] == "model-a")
        assert model_a["context_length"] == 32768
        assert model_a["top_provider"]["max_completion_tokens"] == 4096


# ===================================================================
# retrieve_model
# ===================================================================

class TestRetrieveModel:

    @pytest.mark.asyncio
    async def test_unrestricted_user_retrieves_model(self):
        """Unrestricted user can retrieve any existing model."""
        svc = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS)
        auth_ctx = _make_auth_context(allowed_models=[])

        result = await svc.retrieve_model("model-a", auth_ctx)

        assert result["id"] == "model-a"
        assert result["provider"] == "prov-a"

    @pytest.mark.asyncio
    async def test_restricted_user_retrieves_allowed(self):
        """Restricted user can retrieve a model in their allowed_models."""
        svc = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS)
        auth_ctx = _make_auth_context(allowed_models=["model-a"])

        result = await svc.retrieve_model("model-a", auth_ctx)

        assert result["id"] == "model-a"

    @pytest.mark.asyncio
    async def test_restricted_user_denied_disallowed(self):
        """Restricted user gets 403 for a model not in allowed_models."""
        svc = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS)
        auth_ctx = _make_auth_context(allowed_models=["model-a"])

        with pytest.raises(HTTPException) as exc_info:
            await svc.retrieve_model("model-b", auth_ctx)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_access_check_before_existence(self):
        """INVARIANT: disallowed + nonexistent model → 403, not 404."""
        svc = _build_service(models={})
        auth_ctx = _make_auth_context(allowed_models=["model-a"])

        with pytest.raises(HTTPException) as exc_info:
            await svc.retrieve_model("secret-model", auth_ctx)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_unrestricted_user_nonexistent_model(self):
        """Unrestricted user gets 404 for nonexistent model."""
        svc = _build_service(models={})
        auth_ctx = _make_auth_context(allowed_models=[])

        with pytest.raises(HTTPException) as exc_info:
            await svc.retrieve_model("no-such-model", auth_ctx)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_missing_provider_raises_404(self):
        """Model exists but its provider is not in config → 404."""
        models = {"orphan": {"provider": "missing-prov", "provider_model_name": "x"}}
        svc = _build_service(models=models, providers={})
        auth_ctx = _make_auth_context(allowed_models=[])

        with pytest.raises(HTTPException) as exc_info:
            await svc.retrieve_model("orphan", auth_ctx)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_retrieve_does_not_touch_network(self):
        """ARCH: retrieve_model never instantiates a provider / makes no HTTP call."""
        registry = MagicMock(spec=ProviderRegistry)
        svc = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS, registry=registry)
        auth_ctx = _make_auth_context(allowed_models=[])

        result = await svc.retrieve_model("model-a", auth_ctx)

        registry.get.assert_not_called()
        assert result["id"] == "model-a"

    @pytest.mark.asyncio
    async def test_model_info_overrides_cache(self):
        """INVARIANT: manual model_info wins over auto-cache."""
        cache = _make_cache({"model-a": {"context_length": 4096, "description": "cache-desc"}})
        model_info = {"model-a": {"context_length": 16384, "description": "catalog-desc"}}
        svc = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS, model_info=model_info, cache=cache)
        auth_ctx = _make_auth_context(allowed_models=[])

        result = await svc.retrieve_model("model-a", auth_ctx)

        assert result["description"] == "catalog-desc"
        assert result["context_length"] == 16384

    @pytest.mark.asyncio
    async def test_cache_and_model_info_deep_merge(self):
        """Cache fills fields absent from manual model_info (deep merge)."""
        cache = _make_cache({"model-a": {
            "context_length": 32768,
            "max_completion_tokens": 4096,
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
            "pricing": {"prompt": 0.001},
        }})
        # manual only overrides description; rest comes from cache
        model_info = {"model-a": {"description": "manual"}}
        svc = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS, model_info=model_info, cache=cache)
        auth_ctx = _make_auth_context(allowed_models=[])

        result = await svc.retrieve_model("model-a", auth_ctx)

        assert result["description"] == "manual"
        assert result["context_length"] == 32768
        assert result["supports_vision"] is True  # from cache architecture

    @pytest.mark.asyncio
    async def test_supports_vision_in_list_and_detail_match(self):
        """supports_vision is present and identical in both list and detail."""
        model_info = {
            "model-a": {"architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]}},
        }
        svc = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS, model_info=model_info)
        auth_ctx = _make_auth_context(allowed_models=[])

        listed = await svc.list_models(auth_ctx)
        detail = await svc.retrieve_model("model-a", auth_ctx)

        listed_a = next(m for m in listed["data"] if m["id"] == "model-a")
        assert listed_a["supports_vision"] is True
        assert detail["supports_vision"] is True

    @pytest.mark.asyncio
    async def test_capability_meta_fields(self):
        """Detail includes capabilities_source / capabilities_fetched_at from cache."""
        cache = _make_cache({"model-a": {"context_length": 8192}})
        svc = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS, cache=cache)
        auth_ctx = _make_auth_context(allowed_models=[])

        result = await svc.retrieve_model("model-a", auth_ctx)

        assert result["capabilities_source"] == "test-provider"
        assert result["capabilities_fetched_at"] is not None

    @pytest.mark.asyncio
    async def test_no_meta_fields_without_cache(self):
        """Without a cache, no provenance fields are added."""
        svc = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS)
        auth_ctx = _make_auth_context(allowed_models=[])

        result = await svc.retrieve_model("model-a", auth_ctx)

        assert "capabilities_source" not in result
        assert "capabilities_fetched_at" not in result

    @pytest.mark.asyncio
    async def test_refresh_true_triggers_provider_refresh(self):
        """refresh=True triggers a best-effort refresh of the provider (non-blocking)."""
        svc = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS,
                             cache=_make_cache({"model-a": {"context_length": 1}}))
        svc.config_manager.settings.model_cache_enabled = True
        auth_ctx = _make_auth_context(allowed_models=[])

        with patch("src.services.model_service.refresh_provider_capabilities", new_callable=AsyncMock) as mock_refresh:
            await svc.retrieve_model("model-a", auth_ctx, refresh=True)

        mock_refresh.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_refresh_false_does_not_trigger_refresh(self):
        """Default refresh=False keeps the hot path network-free."""
        svc = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS, cache=_make_cache())
        auth_ctx = _make_auth_context(allowed_models=[])

        with patch("src.services.model_service.refresh_provider_capabilities", new_callable=AsyncMock) as mock_refresh:
            await svc.retrieve_model("model-a", auth_ctx, refresh=False)

        mock_refresh.assert_not_called()


# ===================================================================
# reasoning effort — derived layer from models.yaml
# ===================================================================

class TestReasoningEffortDerived:
    """reasoning.effort_levels / default_effort derive from the models.yaml
    reasoning_effort key — the same key the dispatch funnel enforces."""

    @pytest.mark.asyncio
    async def test_list_and_detail_carry_effort_fields_from_models_yaml(self):
        """Asserted against the value read from the loaded models.yaml — the
        contract mirrors what the operator configured, not a literal."""
        with open("config/models.yaml") as f:
            models = yaml.safe_load(f)["models"]
        with open("config/providers.yaml") as f:
            providers = yaml.safe_load(f)["providers"]
        cfg = models["local/reasoner"]["reasoning_effort"]
        svc = _build_service(models=models, providers=providers)
        auth_ctx = _make_auth_context(allowed_models=[])

        listed = await svc.list_models(auth_ctx)
        listed_r = next(m for m in listed["data"] if m["id"] == "local/reasoner")
        assert listed_r["reasoning"]["effort_levels"] == cfg["allowed"]
        assert listed_r["reasoning"]["supported"] is True
        # no `default` configured -> nothing injected and nothing advertised:
        # the client-side default is the harness's business, not the router's
        assert "default" not in cfg
        assert "default_effort" not in listed_r["reasoning"]

        # list and detail render from the same stored form — they cannot diverge
        detail = await svc.retrieve_model("local/reasoner", auth_ctx)
        assert detail["reasoning"] == listed_r["reasoning"]

    @pytest.mark.asyncio
    async def test_model_without_policy_has_no_reasoning_block(self):
        svc = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS)
        auth_ctx = _make_auth_context(allowed_models=[])

        listed = await svc.list_models(auth_ctx)
        model_a = next(m for m in listed["data"] if m["id"] == "model-a")
        assert "reasoning" not in model_a

    @pytest.mark.asyncio
    async def test_derived_beats_auto_cache(self):
        """The derived layer wins over the auto-cache for the reasoning block."""
        cache = _make_cache({"model-a": {"reasoning": {"supported": False,
                                                       "effort_levels": ["cache"]}}})
        models = {"model-a": {"provider": "prov-a", "provider_model_name": "a-real",
                              "reasoning_effort": {"allowed": ["low", "high"],
                                                   "default": "high"}}}
        svc = _build_service(models=models, providers=SAMPLE_PROVIDERS, cache=cache)
        auth_ctx = _make_auth_context(allowed_models=[])

        detail = await svc.retrieve_model("model-a", auth_ctx)
        assert detail["reasoning"]["effort_levels"] == ["low", "high"]
        assert detail["reasoning"]["supported"] is True

    @pytest.mark.asyncio
    async def test_cached_supported_surfaces_without_any_policy(self):
        """A model with no reasoning_effort policy still advertises the
        auto-derived reasoning.supported coming from the upstream normalizer."""
        cache = _make_cache({"model-a": normalize_provider_model({
            "id": "model-a",
            "context_length": 8192,
            "supported_parameters": ["reasoning_effort", "tools"],
        })})
        svc = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS, cache=cache)
        auth_ctx = _make_auth_context(allowed_models=[])

        detail = await svc.retrieve_model("model-a", auth_ctx)
        assert detail["reasoning"] == {"supported": True}

    @pytest.mark.asyncio
    async def test_policy_effort_levels_layer_over_cached_supported(self):
        """The models.yaml policy supplies effort_levels on top of the cached
        supported flag — the two layers merge key-wise, neither replaces the other."""
        cache = _make_cache({"model-a": normalize_provider_model({
            "id": "model-a",
            "context_length": 8192,
            "supported_parameters": ["reasoning", "tools"],
        })})
        models = {"model-a": {"provider": "prov-a", "provider_model_name": "a-real",
                              "reasoning_effort": {"allowed": ["low", "high"],
                                                   "default": "high"}}}
        svc = _build_service(models=models, providers=SAMPLE_PROVIDERS, cache=cache)
        auth_ctx = _make_auth_context(allowed_models=[])

        detail = await svc.retrieve_model("model-a", auth_ctx)
        assert detail["reasoning"] == {
            "supported": True, "effort_levels": ["low", "high"], "default_effort": "high",
        }

    @pytest.mark.asyncio
    async def test_model_info_reasoning_still_overrides_derived(self):
        """INVARIANT: model_info.yaml beats everything — per key, deep merge."""
        models = {"model-a": {"provider": "prov-a", "provider_model_name": "a-real",
                              "reasoning_effort": {"allowed": ["low", "high"],
                                                   "default": "low"}}}
        model_info = {"model-a": {"reasoning": {"supported": False,
                                                "effort_levels": ["minimal", "full"]}}}
        svc = _build_service(models=models, providers=SAMPLE_PROVIDERS, model_info=model_info)
        auth_ctx = _make_auth_context(allowed_models=[])

        detail = await svc.retrieve_model("model-a", auth_ctx)
        assert detail["reasoning"]["supported"] is False  # model_info wins
        assert detail["reasoning"]["effort_levels"] == ["minimal", "full"]  # list replaced
        # deep merge: a derived key model_info does not mention survives
        assert detail["reasoning"]["default_effort"] == "low"

    @staticmethod
    def _effort_warnings(caplog):
        return [r.getMessage() for r in caplog.records
                if r.levelno == logging.WARNING and "default_effort" in r.getMessage()]

    @pytest.mark.asyncio
    async def test_foreign_cached_default_dropped_and_warned_once(self, caplog):
        """A real OpenRouter entry ships its own default_effort ("minimal");
        under a local policy it is outside effort_levels, so it is dropped —
        the router's own gate would 400 it — and the operator is warned once."""
        cache = _make_cache({"model-a": normalize_provider_model({
            "id": "model-a",
            "context_length": 8192,
            "supported_parameters": ["reasoning", "reasoning_effort"],
            "reasoning": {"supported": True, "default_effort": "minimal",
                          "default_enabled": True},
        })})
        models = {"model-a": {"provider": "prov-a", "provider_model_name": "a-real",
                              "reasoning_effort": {"allowed": ["low", "high"]}}}
        svc = _build_service(models=models, providers=SAMPLE_PROVIDERS, cache=cache)
        auth_ctx = _make_auth_context(allowed_models=[])

        with caplog.at_level(logging.WARNING, logger="nnp-llm-router"):
            detail = await svc.retrieve_model("model-a", auth_ctx)
            await svc.list_models(auth_ctx)
        assert detail["reasoning"] == {
            "supported": True, "effort_levels": ["low", "high"], "default_enabled": True,
        }
        warnings = self._effort_warnings(caplog)
        assert len(warnings) == 1
        assert "model-a" in warnings[0] and "'minimal'" in warnings[0]

    @pytest.mark.asyncio
    async def test_chained_default_outside_local_policy_dropped(self, caplog):
        """Chained routers: the upstream router's default ("max") is outside
        this router's narrower policy — dropped, not advertised."""
        cache = _make_cache({"model-a": {"reasoning": {
            "supported": True, "effort_levels": ["low", "high", "max"],
            "default_effort": "max"}}})
        models = {"model-a": {"provider": "prov-a", "provider_model_name": "a-real",
                              "reasoning_effort": {"allowed": ["low", "high"]}}}
        svc = _build_service(models=models, providers=SAMPLE_PROVIDERS, cache=cache)

        with caplog.at_level(logging.WARNING, logger="nnp-llm-router"):
            detail = await svc.retrieve_model("model-a", _make_auth_context())
        assert detail["reasoning"] == {"supported": True, "effort_levels": ["low", "high"]}
        assert len(self._effort_warnings(caplog)) == 1

    @pytest.mark.asyncio
    async def test_cached_default_inside_levels_kept_silently(self, caplog):
        cache = _make_cache({"model-a": {"reasoning": {
            "supported": True, "effort_levels": ["low", "high", "max"],
            "default_effort": "high"}}})
        svc = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS, cache=cache)

        with caplog.at_level(logging.WARNING, logger="nnp-llm-router"):
            detail = await svc.retrieve_model("model-a", _make_auth_context())
        assert detail["reasoning"]["default_effort"] == "high"
        assert self._effort_warnings(caplog) == []


# ===================================================================
# capabilities — the flat /v1/capabilities map
# ===================================================================

class TestCapabilities:
    """The flat per-model reasoning map: one derivation with /v1/models."""

    @pytest.mark.asyncio
    async def test_flat_map_shape(self):
        """Policy models advertise levels; unknown models are supported=false
        with an empty list — every visible model is present exactly once."""
        models = {
            "m": {"provider": "prov-a", "reasoning_effort": {"allowed": ["low", "high"],
                                                             "param": "reasoning_effort"}},
            "plain": {"provider": "prov-a"},
            "hidden": {"provider": "prov-a", "is_hidden": True},
        }
        svc = _build_service(models=models, providers=SAMPLE_PROVIDERS)

        result = await svc.capabilities(_make_auth_context(allowed_models=[]))

        assert result == {
            "m": {"supported": True, "effort_levels": ["low", "high"]},
            "plain": {"supported": False, "effort_levels": []},
        }

    @pytest.mark.asyncio
    async def test_restricted_user_sees_only_allowed(self):
        models = {
            "m": {"provider": "prov-a", "reasoning_effort": {"allowed": ["low"],
                                                             "param": "reasoning_effort"}},
            "other": {"provider": "prov-a"},
        }
        svc = _build_service(models=models, providers=SAMPLE_PROVIDERS)

        result = await svc.capabilities(_make_auth_context(allowed_models=["other"]))

        assert list(result) == ["other"]
        assert result["other"] == {"supported": False, "effort_levels": []}

    @pytest.mark.asyncio
    async def test_matches_the_models_listing(self):
        """One derivation: the map's supported/effort_levels equal what
        /v1/models renders per model — the two cannot drift."""
        with open("config/models.yaml") as f:
            models = yaml.safe_load(f)["models"]
        with open("config/providers.yaml") as f:
            providers = yaml.safe_load(f)["providers"]
        svc = _build_service(models=models, providers=providers)
        auth_ctx = _make_auth_context(allowed_models=[])

        caps = await svc.capabilities(auth_ctx)
        listed = await svc.list_models(auth_ctx)

        for entry in listed["data"]:
            reasoning = entry.get("reasoning", {})
            assert caps[entry["id"]] == {
                "supported": bool(reasoning.get("supported")),
                "effort_levels": list(reasoning.get("effort_levels") or []),
            }
        assert set(caps) == {m["id"] for m in listed["data"]}

    @pytest.mark.asyncio
    async def test_manual_reasoning_block_keeps_derived_levels(self):
        """model_info.yaml's manual `reasoning` blocks (deepseek/flash+pro
        carry {supported, default_enabled}) deep-merge OVER the derived block
        without erasing the derived effort_levels."""
        models = {
            "m": {"provider": "prov-a", "reasoning_effort": {"allowed": ["low", "high", "max"],
                                                             "param": "reasoning_effort"}},
        }
        model_info = {"m": {"reasoning": {"supported": True, "default_enabled": True}}}
        svc = _build_service(models=models, providers=SAMPLE_PROVIDERS,
                             model_info=model_info)

        caps = await svc.capabilities(_make_auth_context(allowed_models=[]))
        detail = await svc.retrieve_model("m", _make_auth_context(allowed_models=[]))

        assert caps["m"] == {"supported": True, "effort_levels": ["low", "high", "max"]}
        # the merge survival itself, asserted on the stored form the map reads:
        # manual keys and derived keys coexist key-wise
        assert detail["reasoning"] == {
            "supported": True, "default_enabled": True,
            "effort_levels": ["low", "high", "max"],
        }


# ===================================================================
# api field + compat — the Lore-facing protocol contract
# ===================================================================

_ANTHROPIC_PROVIDERS = {
    "prov-a": {"type": "openai", "base_url": "https://a.example.com"},
    "prov-an": {"type": "anthropic", "base_url": "https://an.example.com"},
}


class TestApiField:
    """Every listed model carries `api` derived from its provider's type —
    that is how /v1/models tells Lore (pi-ai) which protocol to use."""

    @pytest.mark.asyncio
    async def test_api_derived_from_provider_type_in_list(self):
        models = {
            "open/model": {"provider": "prov-a"},
            "anthropic/model": {"provider": "prov-an"},
        }
        svc = _build_service(models=models, providers=_ANTHROPIC_PROVIDERS)

        listed = await svc.list_models(_make_auth_context(allowed_models=[]))
        by_id = {m["id"]: m for m in listed["data"]}
        assert by_id["open/model"]["api"] == "openai-completions"
        assert by_id["anthropic/model"]["api"] == "anthropic-messages"

    @pytest.mark.asyncio
    async def test_api_in_detail_matches_list(self):
        models = {"anthropic/model": {"provider": "prov-an"}}
        svc = _build_service(models=models, providers=_ANTHROPIC_PROVIDERS)

        listed = await svc.list_models(_make_auth_context(allowed_models=[]))
        detail = await svc.retrieve_model("anthropic/model", _make_auth_context(allowed_models=[]))

        assert detail["api"] == "anthropic-messages"
        assert detail["api"] == next(m for m in listed["data"]
                                     if m["id"] == "anthropic/model")["api"]

    @pytest.mark.asyncio
    async def test_compat_rendered_as_stored(self):
        """model_info compat (manual-only) renders verbatim on /v1/models."""
        compat = {"forceAdaptiveThinking": True, "supportsMidConvoEffort": True,
                  "title": "deep thinker"}
        models = {"anthropic/model": {"provider": "prov-an"}}
        model_info = {"anthropic/model": {"compat": compat}}
        svc = _build_service(models=models, providers=_ANTHROPIC_PROVIDERS,
                             model_info=model_info)

        detail = await svc.retrieve_model("anthropic/model", _make_auth_context(allowed_models=[]))
        assert detail["compat"] == compat

    @pytest.mark.asyncio
    async def test_anthropic_model_with_effort_policy_in_capabilities(self):
        """Lore reads effort levels from /v1/capabilities: an anthropic-type
        model with a reasoning_effort block (advertisement only on that wire)
        appears by construction, and a key without access does not see it."""
        models = {
            "anthropic/model": {"provider": "prov-an",
                                "reasoning_effort": {"allowed": ["low", "high", "max"],
                                                     "param": "reasoning_effort"}},
            "plain": {"provider": "prov-a"},
        }
        svc = _build_service(models=models, providers=_ANTHROPIC_PROVIDERS)

        caps = await svc.capabilities(_make_auth_context(allowed_models=[]))
        assert caps["anthropic/model"] == {"supported": True,
                                           "effort_levels": ["low", "high", "max"]}

        restricted = await svc.capabilities(
            _make_auth_context(allowed_models=["plain"]))
        assert "anthropic/model" not in restricted


# ===================================================================
# get_pricing
# ===================================================================

class TestGetPricing:

    def test_returns_stored_pricing_from_cache(self):
        cache = _make_cache({"model-a": {"pricing": {"prompt": 1e-6, "completion": 2e-6}}})
        svc = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS, cache=cache)
        pricing = svc.get_pricing("model-a")
        assert pricing == {"prompt": 1e-6, "completion": 2e-6}

    def test_model_info_overrides_cache(self):
        cache = _make_cache({"model-a": {"pricing": {"prompt": 9e-6}}})
        model_info = {"model-a": {"pricing": {"prompt": 1e-6, "completion": 2e-6}}}
        svc = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS,
                             model_info=model_info, cache=cache)
        assert svc.get_pricing("model-a") == {"prompt": 1e-6, "completion": 2e-6}

    def test_unknown_model_returns_none(self):
        svc = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS)
        assert svc.get_pricing("no-such-model") is None

    def test_empty_model_id_returns_none(self):
        svc = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS)
        assert svc.get_pricing("") is None
        assert svc.get_pricing(None) is None

    def test_model_without_pricing_returns_none(self):
        cache = _make_cache({"model-a": {"context_length": 8192}})
        svc = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS, cache=cache)
        assert svc.get_pricing("model-a") is None

    def test_empty_pricing_dict_returns_none(self):
        model_info = {"model-a": {"pricing": {}}}
        svc = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS,
                             model_info=model_info)
        assert svc.get_pricing("model-a") is None


# ===================================================================
# model_info pricing values — floats in the stored form
# ===================================================================

class TestModelInfoPricingValues:
    """_parse_pricing (config_schema) normalizes model_info pricing to floats.

    PyYAML loads `1e-7` (exponent, no dot) as the STRING "1e-7"; the usage
    writer multiplies prices with token counts, so a str price would record
    string repetition (or a TypeError) instead of a cost."""

    @pytest.mark.asyncio
    async def test_numeric_string_renders_like_the_float_form(self):
        """`prompt: "1e-7"` renders the same pricing strings as `prompt: 1e-7`,
        and get_pricing returns floats."""
        model_info = {"model-a": {"pricing": {"prompt": "1e-7", "completion": 3e-07}}}
        float_info = {"model-a": {"pricing": {"prompt": 1e-7, "completion": 3e-07}}}
        svc = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS,
                             model_info=model_info)
        svc_float = _build_service(models=SAMPLE_MODELS, providers=SAMPLE_PROVIDERS,
                                   model_info=float_info)
        auth_ctx = _make_auth_context(allowed_models=[])

        detail = await svc.retrieve_model("model-a", auth_ctx)
        detail_float = await svc_float.retrieve_model("model-a", auth_ctx)

        assert detail["pricing"] == detail_float["pricing"]
        assert detail["pricing"]["prompt"] == "0.0000001"
        pricing = svc.get_pricing("model-a")
        assert pricing == {"prompt": 1e-7, "completion": 3e-07}
        assert all(isinstance(v, float) for v in pricing.values())

    def test_unparsable_value_drops_only_that_key_with_warning(self):
        model_info = {"model-a": {"pricing": {"prompt": "abc", "completion": 0.001}}}
        with patch("src.core.config_schema.logger") as mock_logger:
            svc = _build_service(models=SAMPLE_MODELS, model_info=model_info)

        assert svc.get_pricing("model-a") == {"completion": 0.001}
        mock_logger.warning.assert_called_once()
        message = mock_logger.warning.call_args[0][0]
        assert "model-a" in message and "prompt" in message

    def test_bool_value_dropped(self):
        model_info = {"model-a": {"pricing": {"prompt": True, "completion": 0.001}}}
        with patch("src.core.config_schema.logger"):
            svc = _build_service(models=SAMPLE_MODELS, model_info=model_info)
        assert svc.get_pricing("model-a") == {"completion": 0.001}

    @pytest.mark.parametrize("bad", ["nan", "inf", float("inf"), -1e-7, "-1e-7"])
    def test_non_finite_or_negative_value_dropped_with_warning(self, bad):
        model_info = {"model-a": {"pricing": {"prompt": bad, "completion": 0.001}}}
        with patch("src.core.config_schema.logger") as mock_logger:
            svc = _build_service(models=SAMPLE_MODELS, model_info=model_info)
        assert svc.get_pricing("model-a") == {"completion": 0.001}
        mock_logger.warning.assert_called_once()

    def test_zero_price_kept(self):
        model_info = {"model-a": {"pricing": {"prompt": 0, "completion": "0"}}}
        with patch("src.core.config_schema.logger") as mock_logger:
            svc = _build_service(models=SAMPLE_MODELS, model_info=model_info)
        assert svc.get_pricing("model-a") == {"prompt": 0.0, "completion": 0.0}
        mock_logger.warning.assert_not_called()

    def test_non_mapping_pricing_dropped_with_warning(self):
        model_info = {"model-a": {"pricing": "free"}}
        with patch("src.core.config_schema.logger") as mock_logger:
            svc = _build_service(models=SAMPLE_MODELS, model_info=model_info)
        assert svc.get_pricing("model-a") is None
        mock_logger.warning.assert_called_once()
