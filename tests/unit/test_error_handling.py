"""Unit tests for the error handling system (ErrorType, create_error)."""

from unittest.mock import patch

import pytest
from fastapi import HTTPException

from src.core.error_handling.envelope import enrich_stats_from_envelope
from src.core.error_handling.error_handler import create_error
from src.core.error_handling.error_types import ErrorType

# ---------------------------------------------------------------------------
# Anthropic error conversion (/v1/messages response shape)
# ---------------------------------------------------------------------------

class TestAnthropicErrorConversion:
    """One converter (core/error_handling/anthropic.py) serves the HTTP
    handlers and the SSE failure frame — status -> Anthropic error type."""

    @pytest.mark.parametrize("status_code, expected_type", [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (413, "request_too_large"),
        (429, "rate_limit_error"),
        (503, "overloaded_error"),
        (529, "overloaded_error"),
        (500, "api_error"),
        (502, "api_error"),
        (418, "api_error"),
        (None, "api_error"),
    ])
    def test_status_to_type_map(self, status_code, expected_type):
        from src.core.error_handling.anthropic import anthropic_error_type
        assert anthropic_error_type(status_code) == expected_type

    def test_every_error_type_status_maps_without_raising(self):
        """Walking EVERY ErrorType status through the converter: the handler
        renders any router error on /v1/messages, so an exotic status must
        never raise — worst case it is an api_error."""
        from src.core.error_handling.anthropic import anthropic_error_type
        for member in ErrorType:
            assert isinstance(anthropic_error_type(member.status_code), str)

    def test_payload_shape(self):
        from src.core.error_handling.anthropic import anthropic_error_payload
        assert anthropic_error_payload(401, "bad key") == {
            "type": "error",
            "error": {"type": "authentication_error", "message": "bad key"},
        }

    def test_from_envelope_keeps_message(self):
        from src.core.error_handling.anthropic import anthropic_error_from_envelope
        out = anthropic_error_from_envelope(429, {
            "error": {"code": 429, "message": "rate limited",
                      "metadata": {"error_code": "provider_http_error"}}})
        assert out["error"]["type"] == "rate_limit_error"
        assert out["error"]["message"] == "rate limited"


# ---------------------------------------------------------------------------
# ErrorType enum
# ---------------------------------------------------------------------------

class TestErrorTypeEnumValues:
    """Verify each enum member carries the correct (code, status_code, template)."""

    @pytest.mark.parametrize(
        "member, expected_code, expected_status, expected_template",
        [
            (ErrorType.MODEL_NOT_SPECIFIED, "model_not_specified", 400, "Model not specified in request"),
            (ErrorType.MISSING_REQUIRED_FIELD, "missing_required_field", 400, "Missing required field: {field_name}"),
            (ErrorType.WRONG_API, "wrong_api", 400, "Model '{model_id}' is served over {expected_path}"),
            (ErrorType.MISSING_API_KEY, "missing_api_key", 401, "API key missing"),
            (ErrorType.INVALID_API_KEY, "invalid_api_key", 401, "Invalid API key"),
            (ErrorType.MODEL_NOT_ALLOWED, "model_not_allowed", 403, "Model '{model_id}' is not available for your account"),
            (ErrorType.ENDPOINT_NOT_ALLOWED, "endpoint_not_allowed", 403, "Access to endpoint '{endpoint_path}' is not allowed"),
            (ErrorType.MODEL_NOT_FOUND, "model_not_found", 404, "Model '{model_id}' not found in configuration"),
            (ErrorType.PROVIDER_NOT_FOUND, "provider_not_found", 404, "Provider '{provider_name}' not found for model '{model_id}'"),
            (ErrorType.PROVIDER_CONFIG_ERROR, "provider_config_error", 500, "Provider configuration error: {error_details}"),
            (ErrorType.INTERNAL_SERVER_ERROR, "internal_server_error", 500, "Internal server error: {error_details}"),
            (ErrorType.PROVIDER_NETWORK_ERROR, "provider_network_error", 500, "Network error communicating with provider: {error_details}"),
            (ErrorType.SERVICE_UNAVAILABLE, "service_unavailable", 503, "Could not connect to service: {error_details}"),
        ],
    )
    def test_enum_member_attributes(self, member, expected_code, expected_status, expected_template):
        assert member.code == expected_code
        assert member.status_code == expected_status
        assert member.message_template == expected_template


class TestErrorTypeFormatMessage:
    def test_format_message_with_valid_kwargs(self):
        msg = ErrorType.MODEL_NOT_FOUND.format_message(model_id="gpt-4")
        assert msg == "Model 'gpt-4' not found in configuration"

    def test_format_message_with_multiple_kwargs(self):
        msg = ErrorType.PROVIDER_NOT_FOUND.format_message(
            provider_name="openai", model_id="gpt-4"
        )
        assert msg == "Provider 'openai' not found for model 'gpt-4'"

    def test_format_message_missing_kwargs_strips_placeholder_tokens(self):
        """A missing kwarg must not leak the raw `{...}` token to the client.

        metadata.error_code is the machine-read field; the message is
        human-only prose.
        """
        msg = ErrorType.SERVICE_UNAVAILABLE.format_message()
        assert msg == "Could not connect to service: "
        assert "{" not in msg

    def test_format_message_missing_kwarg_keeps_surrounding_text(self):
        msg = ErrorType.MODEL_NOT_FOUND.format_message()
        assert msg == "Model '' not found in configuration"

    def test_format_message_no_placeholders(self):
        msg = ErrorType.MODEL_NOT_SPECIFIED.format_message()
        assert msg == "Model not specified in request"


class TestErrorTypeCreateErrorDetail:
    def test_returns_correct_structure(self):
        detail = ErrorType.MODEL_NOT_SPECIFIED.create_error_detail()
        assert "error" in detail
        assert detail["error"]["code"] == 400
        assert detail["error"]["message"] == "Model not specified in request"

    def test_includes_formatted_message(self):
        detail = ErrorType.MODEL_NOT_FOUND.create_error_detail(model_id="gpt-4")
        assert detail["error"]["message"] == "Model 'gpt-4' not found in configuration"

    def test_always_includes_metadata_error_code(self):
        """metadata.error_code is always emitted (string classification).

        WHY: the numeric error.code carries the HTTP status; consumers that
        need the error class (stats enrichment among them) read
        metadata.error_code.
        """
        detail = ErrorType.INVALID_API_KEY.create_error_detail()
        assert detail["error"]["metadata"]["error_code"] == "invalid_api_key"

    def test_includes_provider_name_in_metadata_when_given(self):
        detail = ErrorType.PROVIDER_NOT_FOUND.create_error_detail(
            provider_name="openai", model_id="gpt-4"
        )
        assert detail["error"]["metadata"]["provider_name"] == "openai"
        assert detail["error"]["metadata"]["error_code"] == "provider_not_found"

    def test_metadata_without_provider_name_still_present(self):
        """No provider_name: metadata exists but carries no provider_name."""
        detail = ErrorType.MODEL_NOT_FOUND.create_error_detail(model_id="gpt-4")
        assert "metadata" in detail["error"]
        assert "provider_name" not in detail["error"]["metadata"]

    def test_metadata_when_provider_name_is_none(self):
        detail = ErrorType.MODEL_NOT_FOUND.create_error_detail(
            model_id="gpt-4", provider_name=None
        )
        assert "provider_name" not in detail["error"]["metadata"]


# ---------------------------------------------------------------------------
# create_error
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_logger():
    with patch("src.core.error_handling.error_handler.logger") as mock_log:
        yield mock_log


class TestCreateProviderHttpError:
    """create_provider_http_error must carry its own metadata.error_code."""

    def test_metadata_contains_provider_http_error_code(self, mock_logger):
        from src.core.error_handling.error_handler import create_provider_http_error

        exc = create_provider_http_error(
            status_code=429, message="rate limited",
            provider_name="openai", raw="upstream said no",
        )
        assert exc.status_code == 429
        assert exc.detail["error"]["metadata"]["error_code"] == "provider_http_error"
        assert exc.detail["error"]["metadata"]["provider_name"] == "openai"
        assert exc.detail["error"]["metadata"]["raw"] == "upstream said no"


class TestCreateError:
    def test_returns_http_exception_with_correct_status(self, mock_logger):
        exc = create_error(ErrorType.MODEL_NOT_SPECIFIED)
        assert isinstance(exc, HTTPException)
        assert exc.status_code == 400

    def test_detail_contains_error_structure(self, mock_logger):
        exc = create_error(ErrorType.MODEL_NOT_SPECIFIED)
        assert "error" in exc.detail
        assert exc.detail["error"]["code"] == 400

    def test_4xx_logs_warning_not_error(self, mock_logger):
        """4xx is a client error: WARNING, and log_type "warning" — auth already
        logged its own WARNING line; a second line at ERROR doubled every 401/403."""
        create_error(ErrorType.INVALID_API_KEY)
        mock_logger.error.assert_not_called()
        call_kwargs = mock_logger.warning.call_args
        assert call_kwargs[1]["extra"]["log_type"] == "warning"

    def test_5xx_logs_error(self, mock_logger):
        create_error(ErrorType.INTERNAL_SERVER_ERROR, error_details="oops")
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_called_once()
        assert mock_logger.error.call_args[1]["extra"]["log_type"] == "error"

    def test_context_kwargs_included_in_message(self, mock_logger):
        exc = create_error(ErrorType.MODEL_NOT_FOUND, model_id="gpt-4")
        assert "gpt-4" in exc.detail["error"]["message"]

    def test_original_exception_logged_with_exc_info(self, mock_logger):
        orig = ValueError("original")
        create_error(ErrorType.INTERNAL_SERVER_ERROR, original_exception=orig, error_details="fail")
        call_kwargs = mock_logger.error.call_args
        assert call_kwargs[1]["exc_info"] is True
        assert call_kwargs[1]["extra"]["original_exception"] == "original"


class TestCreateErrorAllTypes:
    def test_model_not_specified_returns_400(self, mock_logger):
        exc = create_error(ErrorType.MODEL_NOT_SPECIFIED)
        assert exc.status_code == 400
        assert "Model not specified" in exc.detail["error"]["message"]

    def test_model_not_allowed_returns_403(self, mock_logger):
        exc = create_error(ErrorType.MODEL_NOT_ALLOWED, model_id="gpt-4")
        assert exc.status_code == 403
        assert "gpt-4" in exc.detail["error"]["message"]

    def test_model_not_found_returns_404(self, mock_logger):
        exc = create_error(ErrorType.MODEL_NOT_FOUND, model_id="gpt-4")
        assert exc.status_code == 404
        assert "gpt-4" in exc.detail["error"]["message"]

    def test_provider_not_found_returns_404(self, mock_logger):
        exc = create_error(ErrorType.PROVIDER_NOT_FOUND, provider_name="openai", model_id="gpt-4")
        assert exc.status_code == 404
        assert "openai" in exc.detail["error"]["message"]
        assert "gpt-4" in exc.detail["error"]["message"]

    def test_provider_config_error_returns_500(self, mock_logger):
        exc = create_error(ErrorType.PROVIDER_CONFIG_ERROR, error_details="bad config")
        assert exc.status_code == 500
        assert "bad config" in exc.detail["error"]["message"]

    def test_provider_config_error_with_original_exception(self, mock_logger):
        orig = ValueError("original")
        create_error(ErrorType.PROVIDER_CONFIG_ERROR, original_exception=orig, error_details="bad config")
        call_kwargs = mock_logger.error.call_args
        assert call_kwargs[1]["extra"]["original_exception"] == "original"

    def test_provider_network_error_returns_500(self, mock_logger):
        exc = create_error(ErrorType.PROVIDER_NETWORK_ERROR, error_details="connection refused")
        assert exc.status_code == 500
        assert "connection refused" in exc.detail["error"]["message"]

    def test_missing_api_key_returns_401(self, mock_logger):
        exc = create_error(ErrorType.MISSING_API_KEY)
        assert exc.status_code == 401

    def test_invalid_api_key_returns_401(self, mock_logger):
        exc = create_error(ErrorType.INVALID_API_KEY)
        assert exc.status_code == 401

    def test_endpoint_not_allowed_returns_403(self, mock_logger):
        exc = create_error(ErrorType.ENDPOINT_NOT_ALLOWED, endpoint_path="/v1/secret")
        assert exc.status_code == 403
        assert "/v1/secret" in exc.detail["error"]["message"]

    def test_service_unavailable_returns_503(self, mock_logger):
        exc = create_error(ErrorType.SERVICE_UNAVAILABLE, error_details="redis down")
        assert exc.status_code == 503
        assert "redis down" in exc.detail["error"]["message"]

    def test_internal_server_error_returns_500(self, mock_logger):
        exc = create_error(ErrorType.INTERNAL_SERVER_ERROR, error_details="oops")
        assert exc.status_code == 500
        assert "oops" in exc.detail["error"]["message"]


# ---------------------------------------------------------------------------
# enrich_stats_from_envelope — the single error-envelope -> stats walk
# ---------------------------------------------------------------------------

from src.core.usage_db import RequestStats  # noqa: E402


class TestEnrichStatsFromEnvelope:
    """One extractor owns the {"error": {message, metadata}} walk so the HTTP
    handler and the stream processor cannot drift apart when the envelope
    gains a field."""

    def test_full_envelope_writes_all_three_fields(self):
        stats = RequestStats()
        enrich_stats_from_envelope(stats, {"error": {
            "message": "rate limited",
            "metadata": {"error_code": "provider_http_error", "provider_name": "kimi"},
        }})
        assert stats.error_code == "provider_http_error"
        assert stats.error_message == "rate limited"
        assert stats.provider_name == "kimi"

    def test_provider_name_never_overwrites_an_existing_value(self):
        stats = RequestStats(provider_name="already-set")
        enrich_stats_from_envelope(stats, {"error": {
            "message": "x",
            "metadata": {"error_code": "e", "provider_name": "upstream"},
        }})
        assert stats.provider_name == "already-set"

    def test_default_error_code_applies_when_metadata_missing(self):
        """Stream semantics: no metadata => coarse internal_server_error."""
        stats = RequestStats()
        enrich_stats_from_envelope(
            stats, {"error": {"code": 500, "message": "kaboom"}},
            default_error_code="internal_server_error",
        )
        assert stats.error_code == "internal_server_error"
        assert stats.error_message == "kaboom"

    def test_no_default_leaves_error_code_null(self):
        """HTTP-handler semantics: a plain detail keeps error_code NULL."""
        stats = RequestStats()
        enrich_stats_from_envelope(stats, {"error": {"code": 404, "message": "Not Found"}})
        assert stats.error_code is None
        assert stats.error_message == "Not Found"

    def test_metadata_error_code_wins_over_the_default(self):
        stats = RequestStats()
        enrich_stats_from_envelope(
            stats, {"error": {"message": "x", "metadata": {"error_code": "model_not_found"}}},
            default_error_code="internal_server_error",
        )
        assert stats.error_code == "model_not_found"

    def test_non_dict_metadata_is_ignored(self):
        stats = RequestStats()
        enrich_stats_from_envelope(
            stats, {"error": {"message": "x", "metadata": "garbage"}},
            default_error_code="internal_server_error",
        )
        assert stats.error_code == "internal_server_error"
        assert stats.error_message == "x"
        assert stats.provider_name == ""

    def test_empty_envelope_is_tolerated(self):
        stats = RequestStats()
        enrich_stats_from_envelope(stats, {})
        assert stats.error_code is None
        assert stats.error_message is None
        assert stats.provider_name == ""

    def test_overwrite_replaces_an_error_code_enriched_earlier(self):
        """Stream semantics: the mid-stream payload is the terminal error.

        A generic failure (no metadata) must still record
        internal_server_error even when something enriched the holder
        earlier in the request — this is what the pre-merge
        _apply_stream_error did unconditionally.
        """
        stats = RequestStats(error_code="model_not_found", error_message="stale")
        enrich_stats_from_envelope(
            stats, {"error": {"code": 500, "message": "connection reset"}},
            default_error_code="internal_server_error", overwrite=True,
        )
        assert stats.error_code == "internal_server_error"
        assert stats.error_message == "connection reset"

    def test_overwrite_clears_a_stale_message_when_the_envelope_has_none(self):
        stats = RequestStats(error_code="e", error_message="stale")
        enrich_stats_from_envelope(
            stats, {"error": {"code": 500}},
            default_error_code="internal_server_error", overwrite=True,
        )
        assert stats.error_message is None
        assert stats.error_code == "internal_server_error"

    def test_without_overwrite_an_existing_error_code_survives(self):
        """HTTP-handler semantics: best-effort enrichment never clobbers."""
        stats = RequestStats(error_code="model_not_found", error_message="kept")
        enrich_stats_from_envelope(
            stats, {"error": {"code": 500}}, default_error_code="internal_server_error",
        )
        assert stats.error_code == "model_not_found"
        assert stats.error_message == "kept"

    def test_overwrite_still_never_clobbers_provider_name(self):
        stats = RequestStats(provider_name="resolved")
        enrich_stats_from_envelope(
            stats, {"error": {"message": "x", "metadata": {"provider_name": "upstream"}}},
            default_error_code="internal_server_error", overwrite=True,
        )
        assert stats.provider_name == "resolved"

    def test_http_exception_handler_and_stream_share_the_one_extractor(self):
        """The seam: both call sites import the same function object."""
        from src.api.main import custom_http_exception_handler
        from src.services.chat_service import stream_processor as sp

        assert "enrich_stats_from_envelope" in custom_http_exception_handler.__globals__
        assert sp.enrich_stats_from_envelope is enrich_stats_from_envelope
