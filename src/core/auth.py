"""Authentication and authorization for the API gateway."""
# SYSTEM: auth — bearer authentication (constant-time key comparison) and per-key model access control
import hashlib
import hmac
from collections.abc import Mapping

from fastapi import Depends, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config_schema import KeyEntry
from .context import AuthContext, request_context
from .error_handling import ErrorType, create_error
from .logging import logger
from .usage_db import request_stats

bearer_scheme = HTTPBearer(auto_error=False)


async def get_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)
) -> AuthContext:
    """Authenticate the request and return the typed AuthContext.

    The client key is read from the Authorization Bearer header, falling back
    to ``x-api-key`` when no Bearer is present (the Anthropic SDK inside pi-ai
    sends its key there); Bearer wins when both are sent. Uses constant-time
    comparison to prevent timing attacks. Attaches project_name to the typed
    RequestContext on request.state.request_context (via with_project_name(...))
    for downstream handlers — RequestContext is the single owner of the resolved
    project name; AuthContext carries grants alone.
    """
    config_manager = request.app.state.config_manager
    config = config_manager.get_config()

    api_key = credentials.credentials if credentials else None
    # WHY: Bearer present means the client chose the Bearer scheme — a junk
    # x-api-key next to it must not authenticate, and a junk Bearer must not
    # fall back to a valid x-api-key (the failure names the scheme was used).
    if not api_key:
        api_key = request.headers.get("x-api-key")

    logger.debug("Authentication attempt", extra={
        "auth": {
            "has_api_key": api_key is not None,
            "request_path": str(request.url.path)
        }
    })

    if not api_key:
        _log_auth_failure(request, "missing API key", "missing_api_key")
        raise create_error(ErrorType.MISSING_API_KEY)

    found_project = _find_project(config.user_keys, api_key)
    if not found_project:
        # Enrich the stats holder before raising: the row will carry a
        # truncated SHA-256 hash of the presented (invalid) key.
        request_stats(request).api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:8]
        _log_auth_failure(request, "invalid API key", "invalid_api_key")
        raise create_error(ErrorType.INVALID_API_KEY)

    return _grant(request, found_project, config.user_keys[found_project])


def _grant(request: Request, project_name: str, key_entry: KeyEntry) -> AuthContext:
    """Attach the project to the request context and return the key's grants."""
    allowed_models = list(key_entry.allowed_models)
    allowed_endpoints = list(key_entry.allowed_endpoints)

    # SIDE EFFECT: attach project_name to the typed RequestContext read by
    # downstream handlers — the single owner of the resolved project name.
    # The accessor's placeholder covers a request that never passed through
    # the middleware (unit tests driving raw ASGI).
    request.state.request_context = request_context(request).with_project_name(project_name)

    logger.info("Authentication successful", extra={
        "auth": {
            "project_name": project_name,
            "allowed_models_count": len(allowed_models),
            "allowed_endpoints_count": len(allowed_endpoints),
            "request_path": str(request.url.path)
        }
    })

    return AuthContext(
        allowed_models=allowed_models,
        allowed_endpoints=allowed_endpoints,
    )


def _find_project(user_keys: Mapping[str, KeyEntry], api_key: str) -> str | None:
    """The project whose key matches api_key, or None."""
    for project_name, key_entry in user_keys.items():
        stored_key = key_entry.api_key
        # INVARIANT: constant-time comparison prevents timing attacks
        # WHY: compared as UTF-8 bytes — hmac.compare_digest rejects non-ASCII
        # str, so a non-ASCII bearer (raw bytes on the wire, latin-1-decoded by
        # Starlette) would raise TypeError ⇒ 500 instead of 401. Mirrors the
        # stat-key check in api/stat_routes.py verify_stat_key.
        if stored_key and hmac.compare_digest(stored_key.encode(), api_key.encode()):
            return project_name
    return None


def _log_auth_failure(request: Request, reason: str, error_type: str) -> None:
    """The WARNING line every refused authentication writes."""
    logger.warning(f"Authentication failed: {reason}", extra={
        "auth": {
            "error_type": error_type,
            "request_path": str(request.url.path)
        }
    })


def check_endpoint_access(endpoint_path: str):
    """Return a FastAPI dependency that checks if the user's key grants access to endpoint_path.

    Empty allowed_endpoints list means unrestricted access (default for admin keys).
    """
    async def endpoint_checker(
        request: Request,
        auth_context: AuthContext = Depends(get_api_key)
    ):
        # WHY: empty allowed_endpoints means unrestricted access (default for admin keys)
        if not auth_context.allowed_endpoints or endpoint_path in auth_context.allowed_endpoints:
            return auth_context

        # WHY: user_id is read from request_context, not AuthContext —
        # RequestContext is the single owner of the resolved project name,
        # and get_api_key (the Depends this checker wraps) has already
        # attached it by the time this body runs.
        logger.warning("Endpoint access denied", extra={
            "auth": {
                "error_type": "endpoint_not_allowed",
                "user_id": request_context(request).user_id,
                "endpoint_path": endpoint_path,
                "allowed_endpoints": auth_context.allowed_endpoints
            }
        })
        raise create_error(ErrorType.ENDPOINT_NOT_ALLOWED, endpoint_path=endpoint_path,
                           user_id=request_context(request).user_id)

    return endpoint_checker
