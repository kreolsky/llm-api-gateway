"""Unit tests for src/services/rerank_service.py."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from src.core.config_schema import parse_config, parse_provider
from src.core.context import AuthContext, RequestContext
from src.core.usage_db import RequestStats
from src.providers import ProviderRegistry
from src.services.rerank_service import RerankService

MODELS = {"rerank/model": {"provider": "rr", "provider_model_name": "upstream-rr"}}
PROVIDERS = {"rr": {"type": "openai", "base_url": "https://api.example.com"}}
RERANK_RESPONSE = {
    "model": "upstream-rr",
    "object": "list",
    "usage": {"prompt_tokens": 169, "total_tokens": 169},
    "results": [{"index": 1, "relevance_score": 0.99}, {"index": 0, "relevance_score": 0.01}],
}


def _make_auth_context(allowed_models=None):
    return AuthContext(allowed_models=allowed_models or [], allowed_endpoints=[])


def _make_service():
    cm = MagicMock()
    cm.get_config.return_value = parse_config({"models": MODELS, "providers": PROVIDERS})
    return RerankService(cm, MagicMock(spec=ProviderRegistry))


def _make_request(body: bytes):
    request = MagicMock()
    request.state = SimpleNamespace(
        request_context=RequestContext(request_id="req-rr", project_name="proj"),
        request_stats=RequestStats(endpoint="rerank"),
    )
    request.json = AsyncMock(side_effect=lambda: json.loads(body))
    request.headers = {}
    return request


def _stub_provider():
    provider = SimpleNamespace(identity=None, entry=parse_provider(PROVIDERS["rr"]))
    provider.rerank = AsyncMock(return_value=RERANK_RESPONSE)
    return provider


def _body(model="rerank/model"):
    return json.dumps({"model": model, "query": "q", "documents": ["a", "b"]}).encode()


class TestRerankDispatch:

    @pytest.mark.asyncio
    async def test_returns_backend_json_and_records_usage(self):
        service = _make_service()
        request = _make_request(_body())
        provider = _stub_provider()
        with patch.object(service.registry, "get", return_value=provider):
            response = await service.rerank(request, _make_auth_context())

        assert isinstance(response, JSONResponse)
        assert json.loads(response.body) == RERANK_RESPONSE
        args = provider.rerank.call_args.args
        assert args[0]["documents"] == ["a", "b"]
        assert args[1] == "upstream-rr"
        stats = request.state.request_stats
        assert stats.has_usage is True
        assert stats.prompt_tokens == 169
        assert stats.model_id == "rerank/model"
        assert stats.provider_name == "rr"

    @pytest.mark.asyncio
    async def test_refused_key_is_denied_before_existence(self):
        """A key without the model gets 403 even for a model that does not exist."""
        service = _make_service()
        provider = _stub_provider()
        with patch.object(service.registry, "get", return_value=provider):
            for model in ("rerank/model", "rerank/missing"):
                with pytest.raises(HTTPException) as exc_info:
                    await service.rerank(_make_request(_body(model)),
                                         _make_auth_context(allowed_models=["other/model"]))
                assert exc_info.value.status_code == 403
        provider.rerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_malformed_json_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            await _make_service().rerank(_make_request(b"{not json"), _make_auth_context())
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_client_user_agent_reaches_provider(self):
        service = _make_service()
        request = _make_request(_body())
        request.headers = {"user-agent": "Kilo-Code/7.5.5", "authorization": "Bearer nnp-v1-x"}
        provider = _stub_provider()
        provider.identity = "passthrough"
        with patch.object(service.registry, "get", return_value=provider):
            await service.rerank(request, _make_auth_context())

        assert provider.rerank.call_args.kwargs["extra_headers"] == {"user-agent": "Kilo-Code/7.5.5"}
