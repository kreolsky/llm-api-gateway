"""Standardized error types for the LLM Router."""

import re
from enum import Enum
from typing import Any

from fastapi import status

# A `{placeholder}` token in a template whose kwarg was not supplied.
_PLACEHOLDER_RE = re.compile(r"\{[^{}]*\}")


class ErrorType(Enum):
    """Enumeration of standard error types in the system."""

    # Validation Errors (400)
    MODEL_NOT_SPECIFIED = ("model_not_specified", status.HTTP_400_BAD_REQUEST, "Model not specified in request")
    MISSING_REQUIRED_FIELD = ("missing_required_field", status.HTTP_400_BAD_REQUEST, "Missing required field: {field_name}")
    INVALID_PARAMETER_VALUE = ("invalid_parameter_value", status.HTTP_400_BAD_REQUEST, "Invalid value for '{parameter_name}': {parameter_value}. Allowed values: {allowed_values}")
    WRONG_API = ("wrong_api", status.HTTP_400_BAD_REQUEST, "Model '{model_id}' is served over {expected_path}")

    # Authorization Errors (401)
    MISSING_API_KEY = ("missing_api_key", status.HTTP_401_UNAUTHORIZED, "API key missing")
    INVALID_API_KEY = ("invalid_api_key", status.HTTP_401_UNAUTHORIZED, "Invalid API key")

    # Permission Errors (403)
    MODEL_NOT_ALLOWED = ("model_not_allowed", status.HTTP_403_FORBIDDEN, "Model '{model_id}' is not available for your account")
    ENDPOINT_NOT_ALLOWED = ("endpoint_not_allowed", status.HTTP_403_FORBIDDEN, "Access to endpoint '{endpoint_path}' is not allowed")

    # Not Found Errors (404)
    MODEL_NOT_FOUND = ("model_not_found", status.HTTP_404_NOT_FOUND, "Model '{model_id}' not found in configuration")
    PROVIDER_NOT_FOUND = ("provider_not_found", status.HTTP_404_NOT_FOUND, "Provider '{provider_name}' not found for model '{model_id}'")

    # Server Errors (500)
    PROVIDER_CONFIG_ERROR = ("provider_config_error", status.HTTP_500_INTERNAL_SERVER_ERROR, "Provider configuration error: {error_details}")
    INTERNAL_SERVER_ERROR = ("internal_server_error", status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal server error: {error_details}")
    PROVIDER_NETWORK_ERROR = ("provider_network_error", status.HTTP_500_INTERNAL_SERVER_ERROR, "Network error communicating with provider: {error_details}")
    PROVIDER_INVALID_RESPONSE = ("provider_invalid_response", status.HTTP_502_BAD_GATEWAY, "Provider returned invalid response: {error_details}")

    # Service Unavailable (503)
    SERVICE_UNAVAILABLE = ("service_unavailable", status.HTTP_503_SERVICE_UNAVAILABLE, "Could not connect to service: {error_details}")
    PROVIDER_CONCURRENCY_LIMIT = ("provider_concurrency_limit", status.HTTP_503_SERVICE_UNAVAILABLE, "Concurrency limit reached for provider '{provider_name}'; retry later.")

    def __init__(self, code: str, status_code: int | None, message_template: str):
        self.code = code
        self.status_code = status_code
        self.message_template = message_template

    def format_message(self, **kwargs) -> str:
        """Format the error message with provided parameters.

        A missing kwarg strips the `{...}` tokens instead of returning the raw
        template — braces in a client-visible message are a placeholder leak.
        metadata.error_code is the machine-read field; the message is
        human-only prose.
        """
        try:
            return self.message_template.format(**kwargs)
        except KeyError:
            return _PLACEHOLDER_RE.sub("", self.message_template)

    def create_error_detail(self, **kwargs) -> dict[str, Any]:
        """Create standardized error detail dictionary in OpenRouter format.

        metadata.error_code (the string classification) is always emitted:
        the numeric error.code carries the HTTP status, and consumers such as
        the usage-stats enrichment read the class from metadata.error_code.
        """
        metadata: dict[str, Any] = {"error_code": self.code}
        if kwargs.get("provider_name"):
            metadata["provider_name"] = kwargs["provider_name"]
        return {
            "error": {
                "code": self.status_code,
                "message": self.format_message(**kwargs),
                "metadata": metadata,
            }
        }