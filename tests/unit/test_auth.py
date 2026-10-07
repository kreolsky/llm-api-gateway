"""Unit tests for src/core/auth.py — non-ASCII bearer tokens yield 401, not 500.

Driven through raw ASGI with raw header bytes: httpx cannot even express a
non-ASCII header value (it encodes as ASCII), while any real client (curl)
can put arbitrary bytes on the wire — Starlette decodes them latin-1, and the
token reaches get_api_key as a non-ASCII str.
"""

import json

import pytest
from fastapi import Depends, FastAPI, HTTPException, Request

from src.api.main import custom_http_exception_handler
from src.core.auth import get_api_key
from src.core.config_schema import parse_config
from src.core.context import AuthContext, request_context


def _config_manager_with_keys(keys: dict):
    class _CM:
        def get_config(self):
            return parse_config({"user_keys": {
                name: ({"api_key": key} if isinstance(key, str) else dict(key))
                for name, key in keys.items()
            }})
    return _CM()


@pytest.fixture
def app():
    """Minimal app wiring the REAL HTTPException handler so the envelope shape is asserted."""
    application = FastAPI()
    application.state.config_manager = _config_manager_with_keys({"lab": "nnp-v1-valid-key"})
    application.add_exception_handler(HTTPException, custom_http_exception_handler)

    @application.get("/check")
    async def check(request: Request, auth_context: AuthContext = Depends(get_api_key)):
        # AuthContext carries grants only; the resolved project name is read
        # from the request context it now solely owns.
        return {"user": request_context(request).user_id}

    return application


async def _drive(app, headers: list[tuple[bytes, bytes]], path: str = "/check") -> tuple[int, dict, bytes | None]:
    """Send GET <path> through raw ASGI; return (status, body-bytes, raised-or-None)."""
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": "GET", "scheme": "http",
        "path": path, "raw_path": path.encode(), "query_string": b"",
        "root_path": "", "headers": headers,
        "client": ("127.0.0.1", 123), "server": ("testserver", 80), "state": {},
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    raised = None
    try:
        await app(scope, receive, send)
    except Exception as e:  # noqa: BLE001 — the driver must observe ServerErrorMiddleware's re-raise
        raised = e

    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, body, raised


class TestNonAsciiBearer:

    @pytest.mark.asyncio
    async def test_non_ascii_bearer_is_401_envelope_not_500(self, app):
        """compare_digest on a non-ASCII str raises TypeError today — must be 401 + envelope."""
        status, body, _ = await _drive(app, [(b"authorization", b"Bearer nnp-v1-\xbf\xbf")])

        assert status == 401
        envelope = json.loads(body)
        assert envelope["error"]["code"] == 401
        assert envelope["error"]["metadata"]["error_code"] == "invalid_api_key"

    @pytest.mark.asyncio
    async def test_ascii_invalid_key_still_401(self, app):
        status, body, _ = await _drive(app, [(b"authorization", b"Bearer nnp-v1-wrong")])
        assert status == 401
        assert json.loads(body)["error"]["code"] == 401

    @pytest.mark.asyncio
    async def test_valid_key_passes(self, app):
        status, body, raised = await _drive(app, [(b"authorization", b"Bearer nnp-v1-valid-key")])
        assert raised is None
        assert status == 200
        assert json.loads(body) == {"user": "lab"}

    @pytest.mark.asyncio
    async def test_missing_key_401(self, app):
        status, body, _ = await _drive(app, [])
        assert status == 401
        assert json.loads(body)["error"]["metadata"]["error_code"] == "missing_api_key"


class TestXApiKey:
    """x-api-key as a client credential (the Anthropic SDK inside pi-ai sends
    it): accepted on every route when no Bearer is present, Bearer wins when
    both are sent."""

    @pytest.mark.asyncio
    async def test_valid_x_api_key_grants(self, app):
        status, body, raised = await _drive(app, [(b"x-api-key", b"nnp-v1-valid-key")])
        assert raised is None
        assert status == 200
        assert json.loads(body) == {"user": "lab"}

    @pytest.mark.asyncio
    async def test_invalid_x_api_key_is_401(self, app):
        status, body, _ = await _drive(app, [(b"x-api-key", b"nnp-v1-wrong")])
        assert status == 401
        assert json.loads(body)["error"]["metadata"]["error_code"] == "invalid_api_key"

    @pytest.mark.asyncio
    async def test_bearer_wins_over_x_api_key(self, app):
        """A valid Bearer plus a junk x-api-key authenticates via the Bearer."""
        status, body, raised = await _drive(app, [
            (b"authorization", b"Bearer nnp-v1-valid-key"),
            (b"x-api-key", b"junk"),
        ])
        assert raised is None
        assert status == 200
        assert json.loads(body) == {"user": "lab"}

    @pytest.mark.asyncio
    async def test_x_api_key_not_considered_when_bearer_present(self, app):
        """A missing/invalid Bearer does not fall back to x-api-key: the
        Bearer header declares the credential scheme being used."""
        status, body, _ = await _drive(app, [
            (b"authorization", b"Bearer nnp-v1-wrong"),
            (b"x-api-key", b"nnp-v1-valid-key"),
        ])
        assert status == 401
        assert json.loads(body)["error"]["metadata"]["error_code"] == "invalid_api_key"


def _guarded_app(key: str, allowed_endpoints: list[str] | None) -> FastAPI:
    """App whose /guarded route is protected by check_endpoint_access.

    allowed_endpoints=None leaves the key's list unset (admin: unrestricted).
    """
    project = {"api_key": key}
    if allowed_endpoints is not None:
        project["allowed_endpoints"] = allowed_endpoints

    application = FastAPI()
    application.state.config_manager = _config_manager_with_keys({"lab": project})
    application.add_exception_handler(HTTPException, custom_http_exception_handler)

    from src.core.auth import check_endpoint_access

    @application.get("/guarded")
    async def guarded(request: Request,
                      auth_context: AuthContext = Depends(check_endpoint_access("/guarded"))):
        return {"user": request_context(request).user_id}

    return application


class TestCheckEndpointAccess:
    """check_endpoint_access: empty list passes, mismatch 403s with user_id logged."""

    AUTH = [(b"authorization", b"Bearer nnp-v1-valid-key")]

    @pytest.mark.asyncio
    async def test_empty_list_is_unrestricted(self):
        app = _guarded_app("nnp-v1-valid-key", [])
        status, body, raised = await _drive(app, self.AUTH, path="/guarded")
        assert raised is None
        assert status == 200
        assert json.loads(body) == {"user": "lab"}

    @pytest.mark.asyncio
    async def test_unsupported_key_has_unrestricted_access(self):
        """No allowed_endpoints key at all behaves like the empty list."""
        app = _guarded_app("nnp-v1-valid-key", None)
        status, body, _ = await _drive(app, self.AUTH, path="/guarded")
        assert status == 200
        assert json.loads(body) == {"user": "lab"}

    @pytest.mark.asyncio
    async def test_matching_endpoint_passes(self):
        app = _guarded_app("nnp-v1-valid-key", ["/guarded"])
        status, _, _ = await _drive(app, self.AUTH, path="/guarded")
        assert status == 200

    @pytest.mark.asyncio
    async def test_mismatch_is_403_envelope_with_user_id(self):
        from unittest.mock import patch

        app = _guarded_app("nnp-v1-valid-key", ["/v1/other"])
        with patch("src.core.auth.logger") as mock_logger:
            status, body, _ = await _drive(app, self.AUTH, path="/guarded")

        assert status == 403
        envelope = json.loads(body)
        assert envelope["error"]["code"] == 403
        assert envelope["error"]["metadata"]["error_code"] == "endpoint_not_allowed"
        assert "{" not in envelope["error"]["message"]
        assert "/guarded" in envelope["error"]["message"]
        # user_id reaches the log from request_context — the single owner of
        # the resolved project name (get_api_key, the wrapped Depends, has
        # already attached it).
        log_extra = mock_logger.warning.call_args[1]["extra"]
        assert log_extra["auth"]["user_id"] == "lab"
