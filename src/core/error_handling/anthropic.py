"""Anthropic-shaped error rendering for the /v1/messages route.

One converter shared by the HTTP exception handlers (src/api/main.py) and the
Anthropic SSE failure frame (stream_processor._frame_anthropic_error), so the
HTTP body and the mid-stream frame cannot drift. Router errors are produced
in the OpenRouter envelope everywhere (that is what the stats enrichment
reads); this module converts AFTER enrichment, at the response boundary.
"""

# HTTP status -> Anthropic error "type" ( Anthropic docs: errors have a
# `type` string and a `message`; 529 "overloaded_error" included — the
# Messages API's own overloaded status, mapped from our 503s too).
_STATUS_TO_ANTHROPIC_TYPE = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    503: "overloaded_error",
    529: "overloaded_error",
}


def anthropic_error_type(status_code: int | None) -> str:
    """The Anthropic error type string for an HTTP status (else api_error)."""
    if status_code is None:
        return "api_error"
    return _STATUS_TO_ANTHROPIC_TYPE.get(status_code, "api_error")


def anthropic_error_payload(status_code: int | None, message: str) -> dict:
    """The ``{"type": "error", "error": {"type", "message"}}`` body."""
    return {
        "type": "error",
        "error": {"type": anthropic_error_type(status_code), "message": message},
    }


def anthropic_error_from_envelope(status_code: int, envelope: dict) -> dict:
    """Convert an OpenRouter-shaped HTTPException detail (``{"error": {...}}``)
    into the Anthropic error body, keeping the envelope's message."""
    error = envelope.get("error") if isinstance(envelope, dict) else None
    message = str(error.get("message")) if isinstance(error, dict) and error.get("message") \
        else "Request failed"
    return anthropic_error_payload(status_code, message)
