"""Unit tests for RequestLoggerMiddleware."""

from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from src.api.middleware import RequestLoggerMiddleware


@pytest.fixture
def app():
    """Create a FastAPI app with the middleware for testing."""
    app = FastAPI()
    app.add_middleware(RequestLoggerMiddleware)

    @app.get("/ok")
    async def ok_endpoint():
        return {"status": "ok"}

    @app.post("/echo")
    async def echo_endpoint(request: Request):
        body = await request.json()
        return body

    @app.get("/http-error")
    async def http_error_endpoint():
        raise HTTPException(status_code=403, detail="forbidden")

    @app.get("/http-error-structured")
    async def http_error_structured():
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": 400, "message": "bad request", "metadata": {}}}
        )

    @app.get("/unhandled-error")
    async def unhandled_error_endpoint():
        raise RuntimeError("boom")

    return app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestRequestLoggerMiddleware:

    @patch("src.api.middleware.logger")
    def test_injects_request_id(self, mock_logger, client):
        """Request ID is generated and set in state."""
        response = client.get("/ok")
        assert response.status_code == 200
        # X-Process-Time header is added
        assert "x-process-time" in response.headers

    @patch("src.api.middleware.logger")
    def test_x_process_time_header(self, mock_logger, client):
        """X-Process-Time header is a valid float."""
        response = client.get("/ok")
        process_time = float(response.headers["x-process-time"])
        assert process_time >= 0

    @patch("src.api.middleware.logger")
    def test_logs_request_and_response(self, mock_logger, client):
        """Middleware calls logger.info for request and response."""
        client.get("/ok")
        info_calls = mock_logger.info.call_args_list
        # At least two info calls: one for request, one for response
        assert len(info_calls) >= 2
        req_msg = info_calls[0].args[0]
        assert "Request: Incoming Request" in req_msg
        resp_msg = info_calls[-1].args[0]
        assert "Response: Outgoing Response" in resp_msg

    @patch("src.api.middleware.logger")
    def test_http_exception_returns_correct_status(self, mock_logger, client):
        """HTTPException endpoints return the correct HTTP status code."""
        response = client.get("/http-error")
        assert response.status_code == 403

    @patch("src.api.middleware.logger")
    def test_http_exception_structured_returns_correct_status(self, mock_logger, client):
        """HTTPException with structured detail returns correct status."""
        response = client.get("/http-error-structured")
        assert response.status_code == 400

    @patch("src.api.middleware.logger")
    def test_unhandled_exception_logged(self, mock_logger, client):
        """Unhandled exceptions are logged with 500 status."""
        response = client.get("/unhandled-error")
        assert response.status_code == 500
        mock_logger.error.assert_called()
        error_call = mock_logger.error.call_args
        assert "boom" in error_call.args[0]
        assert error_call.kwargs["status_code"] == 500

    @patch("src.api.middleware.logger")
    def test_user_id_defaults_to_unknown(self, mock_logger, client):
        """When project_name is not set, user_id defaults to 'unknown'."""
        client.get("/ok")
        info_calls = mock_logger.info.call_args_list
        req_kwargs = info_calls[0].kwargs
        assert req_kwargs["user_id"] == "unknown"

    @patch("src.api.middleware.logger")
    def test_middleware_does_not_intercept_body(self, mock_logger, client):
        """The middleware owns no body log — services log the parsed body."""
        mock_logger.is_debug_enabled.return_value = True
        client.post("/echo", json={"key": "value"})
        mock_logger.debug_data.assert_not_called()


# ---------------------------------------------------------------------------
# Endpoint names live on the routes; the middleware reads them there
# ---------------------------------------------------------------------------

from fastapi.routing import APIRoute  # noqa: E402

# The stored endpoint values (usage rows group by them — legacy strings).
EXPECTED_ENDPOINT_NAMES = {
    "/v1/chat/completions": "chat",
    "/v1/messages": "messages",
    "/v1/embeddings": "embeddings",
    "/v1/audio/transcriptions": "transcriptions",
    "/v1/models": "models",
    "/v1/models/{model_id:path}": "models",
    "/v1/capabilities": "capabilities",
}


class TestNamedRoutes:
    def test_every_recorded_route_carries_its_stored_endpoint_name(self):
        """A route without an explicit name lands in usage rows as a raw path.

        Walking app.routes (not the middleware's table copy) is what makes the
        NEXT route fail loudly here instead of silently in the dashboard.
        """
        from src.api.main import app

        skip = ("/health", "/stat", "/docs", "/openapi.json", "/favicon.ico")
        checked = 0
        for route in app.routes:
            if not isinstance(route, APIRoute):
                continue  # mounts (StaticFiles) and the router's own redirects
            if route.path.startswith(skip):
                continue
            assert route.path in EXPECTED_ENDPOINT_NAMES, (
                f"route {route.path!r} records usage but has no expected endpoint name — "
                f"add name= to the decorator and to EXPECTED_ENDPOINT_NAMES"
            )
            assert route.name == EXPECTED_ENDPOINT_NAMES[route.path], (
                f"route {route.path!r} must carry name={EXPECTED_ENDPOINT_NAMES[route.path]!r}"
            )
            checked += 1
        assert checked >= len(EXPECTED_ENDPOINT_NAMES)


class TestEndpointReadFromRoute:
    """stats.endpoint comes from scope["route"].name (populated by the router
    during the app call), falling back to the raw path when no route resolved."""

    @patch("src.api.middleware.schedule_flush")
    def test_endpoint_from_route_name(self, mock_flush):
        app = FastAPI()
        app.add_middleware(RequestLoggerMiddleware)

        @app.get("/somewhere", name="chat")
        async def somewhere():
            return {"ok": True}

        with TestClient(app) as client:
            client.get("/somewhere")

        stats = mock_flush.call_args.args[0]
        assert stats.endpoint == "chat"

    @patch("src.api.middleware.schedule_flush")
    def test_unresolved_route_keeps_raw_path(self, mock_flush):
        app = FastAPI()
        app.add_middleware(RequestLoggerMiddleware)

        with TestClient(app) as client:
            client.get("/does/not/exist")

        stats = mock_flush.call_args.args[0]
        assert stats.endpoint == "/does/not/exist"
