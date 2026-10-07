"""FastAPI application, lifespan management, and route definitions."""
# SYSTEM: api-app — FastAPI app, lifespan, routes, eager provider validation
import asyncio
import contextlib
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..core.auth import check_endpoint_access
from ..core.config_manager import ConfigManager
from ..core.config_schema import RouterConfig
from ..core.context import AuthContext, request_context
from ..core.error_handling import (
    anthropic_error_from_envelope,
    anthropic_error_payload,
    enrich_stats_from_envelope,
)
from ..core.logging import logger
from ..core.model_capabilities import CapabilitiesCache
from ..core.usage_db import close_db, drain_pending_flushes, init_db, request_stats
from ..providers import ProviderRegistry
from ..services.capabilities_refresh import capabilities_refresh_loop
from ..services.chat_service.chat_service import ChatService
from ..services.embedding_service import EmbeddingService
from ..services.messages_service import MessagesService
from ..services.model_service import ModelService
from ..services.transcription_service import TranscriptionService
from .middleware import RequestLoggerMiddleware
from .stat_page import STATIC_DIR, stat_page
from .stat_routes import router as stat_router

# WHY: the deploy host has no git and receives src/ by rsync, so deploy-server writes
# `git describe` into src/VERSION after the sync; read once, a restart picks up a new one.
_VERSION_FILE = Path(__file__).resolve().parents[1] / "VERSION"


def read_app_version(path: Path) -> str:
    """The deployed release string, or "dev" for a tree deploy-server never stamped."""
    if not path.is_file():
        return "dev"
    return path.read_text(encoding="utf-8").strip() or "dev"


APP_VERSION = read_app_version(_VERSION_FILE)


async def _validate_providers(config_manager: ConfigManager, registry: ProviderRegistry) -> None:
    """Eager validation: build & cache every configured provider (fail-fast).

    prepare + publish back to back — the same pair a config reload drives in
    its two phases. All failures are collected by ProviderRegistry.prepare into
    one RuntimeError so operators can fix multiple issues at once.
    """
    await registry.prepare(config_manager.get_config(), config_manager.settings)
    await registry.publish()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize ConfigManager, validate providers, and all services; tear down on shutdown."""
    config_manager = ConfigManager()
    app.state.config_manager = config_manager
    registry = ProviderRegistry()
    app.state.provider_registry = registry

    # ARCH: eager validation — fail fast on bad provider config / missing env keys.
    await _validate_providers(config_manager, registry)

    # ARCH: two-phase provider-cache reload. prepare runs pre-swap (it can
    # veto a broken providers.yaml by raising); publish runs post-swap, after
    # self.config is already the new one, so a provider removed by the reload
    # cannot be re-populated from the stale config (see the INVARIANT on
    # ProviderRegistry.publish).
    async def _prepare_on_reload(new_config: RouterConfig) -> None:
        """Pre-swap callback: stage the provider cache for the freshly loaded config."""
        await registry.prepare(new_config, config_manager.settings)

    async def _publish_on_reload(new_config: RouterConfig) -> None:
        """Post-swap callback: publish the staged cache, drain superseded pools."""
        await registry.publish()

    config_manager.add_reload_callback(_prepare_on_reload, name="provider_registry.prepare")
    config_manager.add_post_swap_callback(_publish_on_reload, name="provider_registry.publish")
    reload_task = config_manager.start_reloader_task()

    # ARCH: capabilities auto-cache. Loaded from disk so data is available even
    # when upstreams are down; refreshed in the background (never blocks startup,
    # never touches the network on the hot path).
    capabilities_cache = CapabilitiesCache(config_manager.settings.model_cache_path)
    capabilities_cache.load()

    app.state.model_service = ModelService(config_manager, registry, capabilities_cache)
    app.state.chat_service = ChatService(config_manager, registry)
    app.state.embedding_service = EmbeddingService(config_manager, registry)
    app.state.transcription_service = TranscriptionService(config_manager, registry)
    app.state.messages_service = MessagesService(config_manager, registry)

    await init_db(config_manager.settings.usage_db_path, app.state.model_service.get_pricing)

    capabilities_task = asyncio.create_task(capabilities_refresh_loop(config_manager, registry, capabilities_cache))

    yield

    reload_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await reload_task
    capabilities_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await capabilities_task
    # Close every provider-owned pool on shutdown (awaited so pools drain),
    # then drain pending usage flushes BEFORE the DB connection closes — an
    # in-flight _flush_row would otherwise race close_db() and silently no-op.
    await registry.aclose_all()
    await drain_pending_flushes()
    await close_db()

app = FastAPI(lifespan=lifespan)
app.mount("/stat/static", StaticFiles(directory=STATIC_DIR), name="stat_static")
app.include_router(stat_router)

# WHY: registered on Starlette's HTTPException, not FastAPI's subclass —
# router-raised 404/405 are the PARENT class, and a subclass-keyed handler
# silently leaves them on Starlette's default {"detail": ...}.
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """OpenRouter-compatible error shape + the single error-enrichment point.

    Writes error_code / error_message / provider_name into the per-request
    stats holder out of exc.detail (best-effort) via the ONE envelope
    extractor (core/error_handling/envelope.py), shared with the stream
    processor's mid-stream error path so the two cannot drift. Enrichment
    tolerates details without metadata.error_code: a plain string detail
    (unmatched-route 404s) writes only error_message and leaves error_code
    NULL — an error status with NULL error_code is an expected shape, the UI
    groups it under "—".

    ARCH: on /v1/messages the RENDERED body is Anthropic-shaped
    ({"type":"error","error":{"type","message"}}) — a native Messages client
    cannot parse the OpenRouter envelope. The conversion runs AFTER the
    stats enrichment read the envelope, so the usage row keeps the
    router-side error_code either way.
    """
    stats = request_stats(request)
    content = exc.detail
    if isinstance(content, dict) and "error" in content:
        enrich_stats_from_envelope(stats, content)
        if _is_messages_route(request):
            return JSONResponse(
                status_code=exc.status_code,
                content=anthropic_error_from_envelope(exc.status_code, content),
            )
        return JSONResponse(status_code=exc.status_code, content=content)
    if content is not None:
        stats.error_message = str(content)
    if _is_messages_route(request):
        return JSONResponse(
            status_code=exc.status_code,
            content=anthropic_error_payload(exc.status_code, str(content or "Request failed")),
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.status_code, "message": str(content)}}
    )


def _is_messages_route(request: Request) -> bool:
    """The Anthropic-shape boundary: router-made errors on POST /v1/messages
    are rendered for a native Messages client (Lore's pi-ai harness)."""
    return request.url.path == "/v1/messages"

app.add_middleware(RequestLoggerMiddleware)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """OpenRouter envelope for unhandled exceptions (ServerErrorMiddleware's layer).

    Starlette re-raises after this handler responds (uvicorn still logs the
    traceback), so the response is sent exactly once. Usage-stats flushing is
    unaffected: RequestLoggerMiddleware sits BELOW ServerErrorMiddleware, so
    its ``finally`` still records the row as a 500. The message is generic on
    purpose — the traceback lives in the server log, not the client response.
    No exc_info here on purpose: RequestLoggerMiddleware below already logs
    the full traceback with the request_id, and uvicorn logs the re-raise.
    """
    logger.error(
        f"Unhandled exception: {type(exc).__name__}: {exc}",
        request_id=request_context(request).request_id,
    )
    if _is_messages_route(request):
        return JSONResponse(
            status_code=500,
            content=anthropic_error_payload(500, "Internal server error"),
        )
    return JSONResponse(
        status_code=500,
        content={"error": {"code": 500,
                           "message": "Internal server error",
                           "metadata": {"error_code": "internal_server_error"}}},
    )


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": APP_VERSION}

@app.get("/v1/models", name="models")
async def list_models(
    auth_context: AuthContext = Depends(check_endpoint_access("/v1/models"))
):
    return await app.state.model_service.list_models(auth_context)


# ARCH: /v1/capabilities is the flat per-model reasoning map (one derivation
# with /v1/models, no drift) — a separate endpoint so OpenAI-compat clients
# parsing /v1/models never see an unexpected shape, while capability readers
# (Lore, MCP) get the map without per-model detail round-trips.
@app.get("/v1/capabilities", name="capabilities")
async def capabilities(
    auth_context: AuthContext = Depends(check_endpoint_access("/v1/capabilities"))
):
    return await app.state.model_service.capabilities(auth_context)


# ARCH: the endpoint string "/v1/models/{model_id:path}" differs from "/v1/models" —
# this is what allows granting access to the model list without access to a
# specific model's detail endpoint
@app.get("/v1/models/{model_id:path}", name="models")
async def retrieve_model(
    model_id: str,
    refresh: bool = False,
    auth_context: AuthContext = Depends(check_endpoint_access("/v1/models/{model_id:path}"))
):
    return await app.state.model_service.retrieve_model(model_id, auth_context, refresh=refresh)

@app.post("/v1/chat/completions", name="chat")
async def chat_completions(
    request: Request,
    auth_context: AuthContext = Depends(check_endpoint_access("/v1/chat/completions"))
):
    return await app.state.chat_service.chat_completions(request, auth_context)

# ARCH: the Anthropic Messages pass-through — native Messages bodies to
# anthropic-type providers (the protocol gate in _resolve_target refuses an
# openai-type model here with 400 wrong_api). Errors on this route render
# Anthropic-shaped (see custom_http_exception_handler).
@app.post("/v1/messages", name="messages")
async def messages(
    request: Request,
    auth_context: AuthContext = Depends(check_endpoint_access("/v1/messages"))
):
    return await app.state.messages_service.messages(request, auth_context)

@app.post("/v1/embeddings", name="embeddings")
async def create_embeddings(
    request: Request,
    auth_context: AuthContext = Depends(check_endpoint_access("/v1/embeddings"))
):
    return await app.state.embedding_service.create_embeddings(request, auth_context)

@app.post("/v1/audio/transcriptions", name="transcriptions")
async def create_transcription(
    request: Request,
    audio_file: UploadFile | None = File(None),
    file: UploadFile | None = File(None),
    model: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    language: str | None = Form(None),
    return_timestamps: bool | None = Form(False),
    auth_context: AuthContext = Depends(check_endpoint_access("/v1/audio/transcriptions"))
):
    return await app.state.transcription_service.create_transcription(
        request=request,
        audio_file=audio_file,
        file=file,
        auth_context=auth_context,
        model_id=model,
        response_format=response_format,
        temperature=temperature,
        language=language,
        return_timestamps=return_timestamps,
    )

@app.get("/stat/")
async def stat_dashboard(request: Request):
    # The page stays open even with STAT_API_KEY set: it is what prompts for
    # the key. /stat/static is a mount and cannot carry a dependency at all.
    # The /stat/api/* JSON endpoints live in stat_routes.py (stat_router).
    return await stat_page(request)
