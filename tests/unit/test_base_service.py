"""Unit tests for src/services/base.py — BaseService class."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.core.config_schema import ModelEntry, parse_config, parse_model, parse_provider
from src.core.context import AuthContext, RequestContext, request_context
from src.core.usage_db import RequestStats
from src.providers import ProviderRegistry
from src.services.base import BaseService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_auth_context(allowed_models=None, allowed_endpoints=None):
    """Return an AuthContext matching what auth.get_api_key builds."""
    return AuthContext(allowed_models or [], allowed_endpoints or [])


def _make_config_manager(models=None, providers=None):
    """Return a mock ConfigManager with get_config wired up."""
    cm = MagicMock()
    config = {
        "models": models or {},
        "providers": providers or {},
    }
    cm.get_config.return_value = parse_config(config)
    return cm


def _make_request(request_id="req-abc", project_name=None):
    """Return a mock FastAPI Request with a typed RequestContext."""
    request = MagicMock()
    request.state = SimpleNamespace(
        request_context=RequestContext(request_id=request_id, project_name=project_name)
    )
    return request


def _provider(identity=None, dialect=None):
    """A stand-in provider instance carrying the entry it was built from."""
    raw = {"type": "openai", "base_url": "https://x.example.com"}
    if dialect is not None:
        raw["reasoning_dialect"] = dialect
    return SimpleNamespace(identity=identity, entry=parse_provider(raw))


def _registry():
    """A stand-in registry: services only ever call get(name)."""
    return MagicMock(spec=ProviderRegistry)


@pytest.fixture
def registry():
    return _registry()


@pytest.fixture
def mock_get(registry):
    return registry.get


def _build_service(models=None, providers=None, registry=None):
    cm = _make_config_manager(models, providers)
    return BaseService(cm, registry or _registry())


# ===================================================================
# request_context — the accessor services read the context through
# ===================================================================

class TestRequestContextAccessor:

    def test_with_request_id(self):
        """Request with request_id returns it in context."""
        request = _make_request("req-42", project_name="my-project")
        ctx = request_context(request)
        assert ctx.request_id == "req-42"
        assert ctx.user_id == "my-project"

    def test_without_request_returns_unknown(self):
        """Without request, request_id is 'unknown'."""
        ctx = request_context(None)
        assert ctx.request_id == "unknown"
        assert ctx.user_id == "unknown"

    def test_extracts_user_id_from_project_name(self):
        """user_id matches the project_name from the typed context."""
        request = _make_request("req-1", project_name="acme-corp")
        ctx = request_context(request)
        assert ctx.user_id == "acme-corp"

    def test_user_id_unknown_when_project_name_none(self):
        """user_id is 'unknown' when project_name is not yet set."""
        request = _make_request("req-1", project_name=None)
        ctx = request_context(request)
        assert ctx.user_id == "unknown"

    def test_returns_typed_context(self):
        """The accessor returns the dataclass, not a stringly-typed dict."""
        ctx = request_context(_make_request("req-9", project_name="p"))
        assert isinstance(ctx, RequestContext)

    def test_request_without_middleware_context(self):
        """A request that never passed through the middleware degrades to 'unknown'."""
        request = MagicMock()
        request.state = SimpleNamespace()
        ctx = request_context(request)
        assert ctx.request_id == "unknown"
        assert ctx.user_id == "unknown"


# ===================================================================
# resolve_model — the one access resolver
# ===================================================================

class TestResolveModel:

    @pytest.mark.asyncio
    async def test_empty_model_raises_400(self):
        """Empty model string raises model_not_specified (400) in the dispatch funnel."""
        svc = _build_service()
        auth_ctx = _make_auth_context()
        with pytest.raises(HTTPException) as exc_info:
            await svc._resolve_target(_make_identity_request(), auth_ctx, "", api="openai-completions")
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_none_model_raises_400(self):
        """None model raises model_not_specified (400) in the dispatch funnel."""
        svc = _build_service()
        auth_ctx = _make_auth_context()
        with pytest.raises(HTTPException) as exc_info:
            await svc._resolve_target(_make_identity_request(), auth_ctx, None, api="openai-completions")
        assert exc_info.value.status_code == 400

    def test_model_not_in_allowed_raises_403(self):
        """Model not in allowed_models raises handle_model_not_allowed (403)."""
        svc = _build_service(
            models={"gpt-4": {"provider": "openai"}},
            providers={"openai": {"type": "openai", "base_url": "https://api.openai.com"}}
        )
        auth_ctx = _make_auth_context(allowed_models=["gpt-3.5-turbo"])
        with pytest.raises(HTTPException) as exc_info:
            svc.resolve_model("gpt-4", auth_ctx, model_id="gpt-4")
        assert exc_info.value.status_code == 403

    def test_model_not_in_config_raises_404(self):
        """Model not in config raises handle_model_not_found (404)."""
        svc = _build_service(models={})
        auth_ctx = _make_auth_context()  # empty allowed_models = unrestricted
        with pytest.raises(HTTPException) as exc_info:
            svc.resolve_model("nonexistent-model", auth_ctx, model_id="nonexistent-model")
        assert exc_info.value.status_code == 404

    def test_provider_not_in_config_raises_404(self):
        """Provider not in config raises handle_provider_not_found (404)."""
        svc = _build_service(
            models={"gpt-4": {"provider": "missing-provider"}},
            providers={}
        )
        auth_ctx = _make_auth_context()
        with pytest.raises(HTTPException) as exc_info:
            svc.resolve_model("gpt-4", auth_ctx, model_id="gpt-4")
        assert exc_info.value.status_code == 404

    def test_happy_path_returns_model_entry(self):
        """Happy path returns the model's entry (its provider and upstream name ride on it)."""
        models = {
            "my-model": {
                "provider": "openai",
                "provider_model_name": "gpt-4-turbo",
                "options": {"temperature": 0.7}
            }
        }
        providers = {
            "openai": {"type": "openai", "base_url": "https://api.openai.com"}
        }
        svc = _build_service(models=models, providers=providers)
        auth_ctx = _make_auth_context()

        model_config = svc.resolve_model("my-model", auth_ctx, model_id="my-model")

        assert model_config == parse_model("my-model", models["my-model"])
        assert model_config.provider == "openai"
        assert model_config.provider_model_name == "gpt-4-turbo"

    @pytest.mark.asyncio
    async def test_dispatch_defaults_provider_model_name(self, registry, mock_get):
        """When provider_model_name is absent, the dispatch funnel sends the requested model name."""
        mock_get.return_value = _provider()
        models = {"gpt-4": {"provider": "openai"}}
        providers = {"openai": {"type": "openai"}}
        svc = _build_service(models=models, providers=providers, registry=registry)
        auth_ctx = _make_auth_context()

        target = await svc._resolve_target(_make_identity_request(), auth_ctx, "gpt-4", api="openai-completions")
        assert target.provider_model_name == "gpt-4"

    def test_empty_allowed_models_unrestricted(self):
        """Empty allowed_models list means unrestricted access -- no 403."""
        models = {"gpt-4": {"provider": "openai"}}
        providers = {"openai": {"type": "openai"}}
        svc = _build_service(models=models, providers=providers)
        auth_ctx = _make_auth_context(allowed_models=[])

        model_config = svc.resolve_model("gpt-4", auth_ctx, model_id="gpt-4")
        assert model_config is not None

    def test_model_in_allowed_models_succeeds(self):
        """When allowed_models is non-empty and requested model IS in the list, validation passes."""
        models = {"gpt-4": {"provider": "openai", "provider_model_name": "gpt-4-turbo"}}
        providers = {"openai": {"type": "openai", "base_url": "https://api.openai.com"}}
        svc = _build_service(models=models, providers=providers)
        auth_ctx = _make_auth_context(allowed_models=["gpt-4"])

        model_config = svc.resolve_model("gpt-4", auth_ctx, model_id="gpt-4")

        assert model_config == parse_model("gpt-4", models["gpt-4"])
        assert model_config.provider == "openai"
        assert model_config.provider_model_name == "gpt-4-turbo"

    def test_invariant_access_check_before_existence(self):
        """INVARIANT: access check runs before existence check.

        A model that is NOT in allowed_models AND NOT in config should
        produce 403 (not 404), preventing information leakage.
        """
        svc = _build_service(models={})  # model does not exist in config
        auth_ctx = _make_auth_context(allowed_models=["only-this-model"])

        with pytest.raises(HTTPException) as exc_info:
            svc.resolve_model("secret-model", auth_ctx, model_id="secret-model")
        # Must be 403 (access denied), not 404 (not found)
        assert exc_info.value.status_code == 403


# ===================================================================
# _prepare_dispatch — provider resolution
# ===================================================================

class TestPrepareDispatchProviders:
    """_prepare_dispatch resolves the provider via the registry directly.

    The old pass-through _get_provider wrapper is gone (inlined); these tests
    pin the resolution contract at its new call site.
    """

    @pytest.mark.asyncio
    async def test_resolves_provider_from_registry(self, registry, mock_get):
        """A valid config yields the registry's instance on PreparedDispatch."""
        mock_provider = _provider()
        mock_get.return_value = mock_provider

        svc = _build_service(
            registry=registry,
            models={"gpt-4": {"provider": "openai"}},
            providers={"openai": {"type": "openai", "base_url": "https://api.example.com"}}
        )
        auth_ctx = _make_auth_context()
        request = _make_request("req-1")
        request.json = AsyncMock(return_value={"model": "gpt-4"})

        prepared = await svc._prepare_dispatch(
            request, auth_ctx, component="chat_service", log_title="Request JSON", api="openai-completions"
        )

        assert prepared.provider is mock_provider
        mock_get.assert_called_once_with("openai")

    @pytest.mark.asyncio
    async def test_registry_error_propagates(self, registry, mock_get):
        """A registry lookup error propagates unchanged (an unknown provider
        type never gets this far — parse_config refuses it)."""
        mock_get.side_effect = HTTPException(status_code=404, detail="not found")

        svc = _build_service(
            registry=registry,
            models={"gpt-4": {"provider": "bad"}},
            providers={"bad": {"type": "openai"}}
        )
        auth_ctx = _make_auth_context()
        request = _make_request("req-1")
        request.json = AsyncMock(return_value={"model": "gpt-4"})

        with pytest.raises(HTTPException) as exc_info:
            await svc._prepare_dispatch(
                request, auth_ctx, component="chat_service", log_title="Request JSON", api="openai-completions"
            )
        assert exc_info.value.status_code == 404


# ===================================================================
# _prepare_dispatch — preamble parity with the old duplicated lines
# ===================================================================

class TestPrepareDispatchPreamble:

    def _svc(self, registry=None):
        return _build_service(
            models={"m": {"provider": "openai"}},
            providers={"openai": {"type": "openai", "base_url": "https://x.example.com"}},
            registry=registry,
        )

    def _request(self, body):
        request = _make_request("req-1", project_name="proj")
        request.json = AsyncMock(return_value=body)
        return request

    @pytest.mark.asyncio
    async def test_malformed_json_answers_400(self):
        svc = self._svc()
        request = self._request({})
        request.json = AsyncMock(side_effect=ValueError("bad utf-8"))
        with pytest.raises(HTTPException) as exc_info:
            await svc._prepare_dispatch(
                request, _make_auth_context(),
                component="c", log_title="t", api="openai-completions",
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_stats_enriched_and_identity_headers_built(self, registry, mock_get):
        mock_get.return_value = _provider(identity="passthrough")
        svc = self._svc(registry)
        request = self._request({"model": "m"})
        request.headers = {"user-agent": "oc/1.0"}

        prepared = await svc._prepare_dispatch(
            request, _make_auth_context(), component="c", log_title="t", api="openai-completions"
        )

        assert prepared.requested_model == "m"
        assert prepared.stats.model_id == "m"
        assert prepared.stats.provider_name == "openai"
        assert prepared.identity_headers == {"user-agent": "oc/1.0"}
        assert prepared.error_ctx == {"request_id": "req-1", "user_id": "proj", "model_id": "m"}

    @pytest.mark.asyncio
    async def test_non_string_model_blanks_stats_model_id(self, registry, mock_get):
        """A non-string "model" reaches the usage row as "", never as the raw value.

        The stats holder is enriched BEFORE validation rejects the request, so
        the blanking branch is observable on the real RequestStats even though
        the call raises. A truthy non-string (123) is used deliberately: None
        would short-circuit at MODEL_NOT_SPECIFIED without exercising it.
        """
        mock_get.return_value = _provider()
        svc = self._svc(registry)
        request = self._request({"model": 123})
        request.headers = {}
        stats = RequestStats()
        request.state.request_stats = stats

        with pytest.raises(HTTPException) as exc_info:
            await svc._prepare_dispatch(
                request, _make_auth_context(), component="c", log_title="t", api="openai-completions"
            )
        assert exc_info.value.status_code == 404
        assert stats.model_id == ""


# ===================================================================
# _prepare_dispatch — reasoning-field dialect translation
# ===================================================================

class TestPrepareDispatchDialectTranslation:
    """_prepare_dispatch applies the per-provider dialect translation AFTER
    the effort policy — the gated-legal value is what gets re-nested."""

    def _svc(self, registry, providers):
        return _build_service(models={"m": {"provider": "p"}}, providers=providers, registry=registry)

    def _request(self, body):
        request = _make_request("req-1", project_name="proj")
        request.json = AsyncMock(return_value=body)
        request.headers = {}
        return request

    @pytest.mark.asyncio
    async def test_openrouter_body_translated_on_the_funnel(self, registry, mock_get):
        mock_get.return_value = _provider(dialect="openrouter")
        svc = self._svc(registry, {"p": {"type": "openai", "base_url": "https://x",
                               "reasoning_dialect": "openrouter"}})
        request = self._request({"model": "m", "messages": [],
                                 "thinking": {"type": "enabled"},
                                 "reasoning_effort": "high"})

        prepared = await svc._prepare_dispatch(
            request, _make_auth_context(), component="c", log_title="t", api="openai-completions")

        assert prepared.request_body["reasoning"] == {"effort": "high"}
        assert "thinking" not in prepared.request_body
        assert "reasoning_effort" not in prepared.request_body

    @pytest.mark.asyncio
    async def test_default_dialect_drops_thinking_keeps_effort(self, registry, mock_get):
        mock_get.return_value = _provider()
        svc = self._svc(registry, {"p": {"type": "openai", "base_url": "https://x"}})
        request = self._request({"model": "m", "messages": [],
                                 "thinking": {"type": "enabled"},
                                 "reasoning_effort": "low"})

        prepared = await svc._prepare_dispatch(
            request, _make_auth_context(), component="c", log_title="t", api="openai-completions")

        assert prepared.request_body["reasoning_effort"] == "low"
        assert "thinking" not in prepared.request_body

    @pytest.mark.asyncio
    async def test_deepseek_body_untouched_on_the_funnel(self, registry, mock_get):
        """The native dialect: both fields reach the provider verbatim,
        title-shaped requests included."""
        mock_get.return_value = _provider(dialect="deepseek")
        svc = self._svc(registry, {"p": {"type": "openai", "base_url": "https://x",
                               "reasoning_dialect": "deepseek"}})
        body = {"model": "m", "messages": [], "thinking": {"type": "disabled"}}
        request = self._request(dict(body))

        prepared = await svc._prepare_dispatch(
            request, _make_auth_context(), component="c", log_title="t", api="openai-completions")

        assert prepared.request_body == body

    @pytest.mark.asyncio
    async def test_dialect_follows_the_called_instance_not_the_config(self, registry, mock_get):
        """Across a reload the config dict and the registry can hold two
        generations: the body is shaped for the backend actually called."""
        mock_get.return_value = _provider(dialect="openrouter")
        svc = self._svc(registry, {"p": {"type": "openai", "base_url": "https://x"}})
        request = self._request({"model": "m", "messages": [], "reasoning_effort": "high"})

        prepared = await svc._prepare_dispatch(
            request, _make_auth_context(), component="c", log_title="t", api="openai-completions")

        assert prepared.request_body["reasoning"] == {"effort": "high"}
        assert "reasoning_effort" not in prepared.request_body

    @pytest.mark.asyncio
    async def test_policy_gate_fires_before_the_translation(self, registry, mock_get):
        """A value outside the model's allowed list is a 400 from the policy —
        the openrouter re-nest never runs for refused principals."""
        mock_get.return_value = _provider(dialect="openrouter")
        svc = _build_service(
            registry=registry,
            models={"m": {"provider": "p",
                          "reasoning_effort": {"allowed": ["low", "high"],
                                               "param": "reasoning_effort"}}},
            providers={"p": {"type": "openai", "base_url": "https://x",
                             "reasoning_dialect": "openrouter"}})
        request = self._request({"model": "m", "messages": [],
                                 "thinking": {"type": "enabled"},
                                 "reasoning_effort": "max"})

        with pytest.raises(HTTPException) as exc_info:
            await svc._prepare_dispatch(
                request, _make_auth_context(), component="c", log_title="t", api="openai-completions")
        assert exc_info.value.status_code == 400


# ===================================================================
# _resolve_target — the body-agnostic dispatch funnel
# ===================================================================

class TestResolveTarget:
    """The shared resolver: enrich stats, validate access, resolve the
    provider, build identity headers — without touching any request body.

    JSON (chat/embeddings) and multipart (transcription) both ride it, which
    is what makes the dispatch funnel true by construction.
    """

    def _svc(self, registry=None, models=None, providers=None):
        return _build_service(
            models=models or {"m": {"provider": "openai"}},
            providers=providers or {"openai": {"type": "openai", "base_url": "https://x.example.com"}},
            registry=registry,
        )

    @pytest.mark.asyncio
    async def test_resolves_target_without_parsing_a_body(self, registry, mock_get):
        """A body-agnostic call resolves everything the dispatch needs."""
        mock_get.return_value = _provider(identity="passthrough")
        svc = self._svc(registry)
        request = _make_identity_request({"user-agent": "oc/1.0"})

        target = await svc._resolve_target(request, _make_auth_context(), "m", api="openai-completions")

        assert target.provider is mock_get.return_value
        assert target.provider_name == "openai"
        assert target.provider_model_name == "m"
        assert target.model_config == ModelEntry(provider="openai")
        assert target.identity_headers == {"user-agent": "oc/1.0"}
        assert target.request_id == "req-1"
        assert target.user_id == "proj"
        assert target.error_ctx == {"request_id": "req-1", "user_id": "proj", "model_id": "m"}
        # stats enrichment rides the resolver (the JSON wrapper no longer owns it)
        assert target.stats.model_id == "m"
        assert target.stats.provider_name == "openai"

    @pytest.mark.asyncio
    async def test_access_denied_raises_before_provider_resolution(self, registry, mock_get):
        """The resolver keeps the INVARIANT: access check before existence."""
        svc = self._svc(registry)
        request = _make_identity_request({})

        with pytest.raises(HTTPException) as exc_info:
            await svc._resolve_target(
                request, _make_auth_context(allowed_models=["other"]), "m", api="openai-completions"
            )
        assert exc_info.value.status_code == 403
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_prepare_dispatch_delegates_to_the_resolver(self, registry, mock_get):
        """The JSON wrapper is thin: parse, delegate, apply the effort policy."""
        mock_get.return_value = _provider()
        svc = self._svc(registry)
        request = _make_request("req-1")
        request.json = AsyncMock(return_value={"model": "m"})
        request.headers = {}

        with patch.object(BaseService, "_resolve_target",
                          new_callable=AsyncMock) as mock_resolve:
            mock_resolve.return_value = SimpleNamespace(
                request_id="req-1", user_id="unknown", stats=RequestStats(),
                error_ctx={}, model_config=ModelEntry(provider="openai"), provider_name="openai",
                provider_model_name="m",
                provider=mock_get.return_value, identity_headers=None,
            )
            prepared = await svc._prepare_dispatch(
                request, _make_auth_context(), component="c", log_title="t", api="openai-completions"
            )

        mock_resolve.assert_awaited_once()
        assert mock_resolve.call_args.args[2] == "m"
        assert prepared.provider is mock_get.return_value
        assert prepared.request_body == {"model": "m"}
        assert prepared.requested_model == "m"

    @pytest.mark.asyncio
    async def test_reasoning_effort_policy_stays_in_the_json_wrapper(self, registry, mock_get):
        """The effort policy is body-shaped, so it is applied by the JSON
        wrapper — a multipart endpoint riding the resolver must never get it."""
        mock_get.return_value = _provider()
        models = {"m": {
            "provider": "openai",
            "reasoning_effort": {"allowed": ["low", "medium", "high"], "default": "high"},
        }}
        svc = self._svc(registry, models=models)
        request = _make_request("req-1")
        request.json = AsyncMock(return_value={"model": "m"})
        request.headers = {}

        prepared = await svc._prepare_dispatch(
            request, _make_auth_context(), component="c", log_title="t", api="openai-completions"
        )

        assert prepared.request_body["reasoning_effort"] == "high"


# ===================================================================
# Protocol gate — one api per provider type
# ===================================================================

class TestProtocolGate:
    """_resolve_target enforces the calling endpoint's api against the
    provider instance's type, AFTER resolve_model — access before existence
    before protocol."""

    ENDPOINT_APIS = ("openai-completions", "anthropic-messages")

    def _service(self, registry, provider_type):
        return _build_service(
            models={"m": {"provider": "p"}},
            providers={"p": {"type": provider_type, "base_url": "https://x.example.com"}},
            registry=registry,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider_type", ["openai", "anthropic"])
    @pytest.mark.parametrize("api", ENDPOINT_APIS)
    async def test_every_api_provider_pair(self, registry, mock_get, provider_type, api):
        """Match passes, mismatch is a 400 wrong_api naming the served path."""
        mock_get.return_value = _provider_of_type(provider_type)
        svc = self._service(registry, provider_type)

        expected = {"openai": "/v1/chat/completions", "anthropic": "/v1/messages"}[provider_type]
        if provider_type == "openai" and api == "openai-completions" or \
           provider_type == "anthropic" and api == "anthropic-messages":
            target = await svc._resolve_target(
                _make_identity_request(), _make_auth_context(), "m", api=api)
            assert target.provider is mock_get.return_value
        else:
            with pytest.raises(HTTPException) as exc_info:
                await svc._resolve_target(
                    _make_identity_request(), _make_auth_context(), "m", api=api)
            assert exc_info.value.status_code == 400
            assert exc_info.value.detail["error"]["metadata"]["error_code"] == "wrong_api"
            assert expected in exc_info.value.detail["error"]["message"]
            assert "'m'" in exc_info.value.detail["error"]["message"]

    @pytest.mark.asyncio
    async def test_access_denied_beats_protocol_mismatch(self, registry, mock_get):
        """A key without access to the model gets 403 even on a mismatch —
        the INVARIANT (access before existence) extends to the protocol gate."""
        mock_get.return_value = _provider_of_type("anthropic")
        svc = self._service(registry, "anthropic")

        with pytest.raises(HTTPException) as exc_info:
            await svc._resolve_target(
                _make_identity_request(), _make_auth_context(allowed_models=["other"]),
                "m", api="openai-completions",
            )
        assert exc_info.value.status_code == 403
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_nonexistent_model_is_404_not_wrong_api(self, registry, mock_get):
        """Existence before protocol: an unknown model answers 404 from
        resolve_model, before the registry lookup the gate reads."""
        svc = _build_service(models={}, providers={}, registry=registry)
        with pytest.raises(HTTPException) as exc_info:
            await svc._resolve_target(
                _make_identity_request(), _make_auth_context(), "ghost", api="anthropic-messages")
        assert exc_info.value.status_code == 404
        mock_get.assert_not_called()


def _provider_of_type(provider_type):
    """A stand-in provider instance of the given protocol (entry carries type)."""
    return SimpleNamespace(identity=None, entry=parse_provider(
        {"type": provider_type, "base_url": "https://x.example.com"}))


# ===================================================================
# _build_identity_headers / _extract_passthrough_headers
# ===================================================================

def _make_identity_request(headers=None, request_id="req-1", project_name="proj"):
    """Mock request with real dict headers and a typed RequestContext."""
    request = MagicMock()
    request.headers = dict(headers or {})
    request.state = SimpleNamespace(
        request_context=RequestContext(request_id=request_id, project_name=project_name)
    )
    return request


def _make_identity_provider(identity=None, provider_name="glm"):
    return SimpleNamespace(identity=identity, provider_name=provider_name)


def _build_identity_service():
    return BaseService(_make_config_manager(), _registry())


class TestExtractPassthroughHeaders:
    """Full forward minus the denylist (core/header_policy.py)."""

    def test_client_headers_forwarded_verbatim(self):
        """Everything not denylisted goes up with the client's own spelling."""
        svc = _build_identity_service()
        request = _make_identity_request({
            "user-agent": "Kilo-Code/7.5.5",
            "x-session-id": "ses_01ab",
            "x-session-affinity": "ses_01ab",
            "x-kilocode-mode": "code",
            "anthropic-beta": "interleaved-thinking-2025-05-14",
            "accept": "application/json",
        })
        assert svc._extract_passthrough_headers(request) == {
            "user-agent": "Kilo-Code/7.5.5",
            "x-session-id": "ses_01ab",
            "x-session-affinity": "ses_01ab",
            "x-kilocode-mode": "code",
            "anthropic-beta": "interleaved-thinking-2025-05-14",
            "accept": "application/json",
        }

    def test_credential_headers_dropped(self):
        """Client credentials never reach the upstream — only the router's key goes."""
        svc = _build_identity_service()
        request = _make_identity_request({
            "authorization": "Bearer nnp-v1-x",
            "proxy-authorization": "Basic x",
            "cookie": "a=b",
            "x-api-key": "sk-client",
            "api-key": "sk-client",
            "x-goog-api-key": "sk-client",
            "user-agent": "oc/1.0",
        })
        assert svc._extract_passthrough_headers(request) == {"user-agent": "oc/1.0"}

    def test_transport_headers_dropped(self):
        """Hop-by-hop / framing values are stale: the router re-serializes the body."""
        svc = _build_identity_service()
        request = _make_identity_request({
            "host": "router:8777",
            "content-length": "123",
            "content-type": "application/json",
            "content-encoding": "gzip",
            "connection": "keep-alive",
            "transfer-encoding": "chunked",
            "te": "trailers",
            "upgrade": "h2c",
            "keep-alive": "timeout=5",
            "expect": "100-continue",
            "accept-encoding": "br, zstd",
        })
        assert svc._extract_passthrough_headers(request) == {}

    def test_topology_headers_dropped(self):
        """Reverse-proxy headers would leak internal IPs of the lab."""
        svc = _build_identity_service()
        request = _make_identity_request({
            "x-forwarded-for": "10.10.1.5",
            "x-forwarded-host": "internal.local",
            "x-real-ip": "10.10.1.5",
            "forwarded": "for=10.10.1.5",
            "true-client-ip": "10.10.1.5",
            "cf-connecting-ip": "10.10.1.5",
            "cdn-loop": "cdn1",
        })
        assert svc._extract_passthrough_headers(request) == {}

    def test_denylist_is_case_insensitive(self):
        svc = _build_identity_service()
        request = _make_identity_request({"X-Api-Key": "sk-client", "HOST": "router"})
        assert svc._extract_passthrough_headers(request) == {}

    def test_none_request_returns_empty(self):
        assert _build_identity_service()._extract_passthrough_headers(None) == {}


class TestBuildIdentityHeaders:

    def test_provider_without_identity_returns_none(self):
        svc = _build_identity_service()
        request = _make_identity_request({"user-agent": "oc/1.0"})
        assert svc._build_identity_headers(_make_identity_provider(None), request) is None
        assert svc._build_identity_headers(SimpleNamespace(), request) is None

    def test_passthrough_forwards_all_but_denylist(self):
        svc = _build_identity_service()
        request = _make_identity_request({
            "user-agent": "oc/1.0",
            "x-custom": "yes",
            "authorization": "Bearer nnp-v1-x",
        })
        result = svc._build_identity_headers(_make_identity_provider("passthrough"), request)
        assert result == {"user-agent": "oc/1.0", "x-custom": "yes"}

    def test_passthrough_with_only_denied_headers_returns_none(self):
        svc = _build_identity_service()
        request = _make_identity_request({"authorization": "Bearer nnp-v1-x"})
        result = svc._build_identity_headers(_make_identity_provider("passthrough"), request)
        assert result is None

    def test_none_request_returns_none_for_passthrough(self):
        svc = _build_identity_service()
        assert svc._build_identity_headers(_make_identity_provider("passthrough"), None) is None
