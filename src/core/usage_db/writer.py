"""Writer side of usage tracking: schema, per-request holder, single-row flush.

ARCH: exactly one INSERT per request. The pure-ASGI RequestLoggerMiddleware
creates a per-request RequestStats holder in scope state, services only enrich
the holder (model, provider, tokens, errors), and the middleware flushes a
single row in a ``finally`` at the end of the request lifecycle — including
SSE streams, whose generator completes before the middleware returns.

Cost is computed at flush time from merged capabilities pricing, so historical
rows do not drift when tariffs change. The pricing source is injected by
init_db (a model_id -> pricing callable), so this core package never reaches
into the service layer for it.
"""

import asyncio
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import aiosqlite

from ..logging import logger
from . import _conn
from ._conn import get_connection

# ARCH: fire-and-forget flush tasks are tracked here so they are not
# garbage-collected before completion. Each task removes itself via a done
# callback that logs a WARNING on unexpected failure.
_usage_tasks: set[asyncio.Task] = set()

# model_id -> stored per-token pricing (or None when unpriced).
PricingLookup = Callable[[str], dict[str, Any] | None]

# Installed by init_db, cleared by close_db; None prices nothing.
_pricing_lookup: PricingLookup | None = None

# Stored error messages are truncated to this length.
_MAX_ERROR_MESSAGE = 500

# Additive migration: columns added by the usage-stats redesign, applied over
# an existing DB via ALTER TABLE (driven by PRAGMA table_info in init_db).
_MIGRATIONS: dict[str, str] = {
    "error_code": "TEXT",
    "error_message": "TEXT",
    "api_key_hash": "TEXT",
    "client_ip": "TEXT",
    "reasoning_tokens": "INTEGER NOT NULL DEFAULT 0",
    "cost_usd": "REAL",
    # Default asymmetry is intentional: the column default is 1 so that
    # pre-existing rows (which always had usage) read correctly, while
    # RequestStats.has_usage defaults to False in Python and is set to True
    # only inside set_usage().
    "has_usage": "INTEGER NOT NULL DEFAULT 1",
    "stream": "INTEGER NOT NULL DEFAULT 0",
    "cache_write_tokens": "INTEGER NOT NULL DEFAULT 0",
}


# ---------------------------------------------------------------------------
# Per-request holder
# ---------------------------------------------------------------------------

@dataclass
class RequestStats:
    """Mutable per-request stats, stored in scope["state"]["request_stats"].

    Created by the middleware, enriched by services/auth, flushed exactly once
    by the middleware at the end of the request.
    """

    endpoint: str = ""
    client_ip: str = ""
    model_id: str = ""
    provider_name: str = ""
    stream: bool = False
    has_usage: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    error_code: str | None = None
    error_message: str | None = None
    api_key_hash: str | None = None

    def set_usage(self, usage: dict[str, Any]) -> None:
        """Single token-extraction point for the provider usage shape.

        The only place that reads ``prompt_tokens_details.cached_tokens`` /
        ``cache_write_tokens`` and ``completion_tokens_details.reasoning_tokens``,
        so the stream and non-stream field sets cannot drift.
        """
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        self.prompt_tokens = int(usage.get("prompt_tokens") or 0)
        self.completion_tokens = int(usage.get("completion_tokens") or 0)
        self.cached_tokens = int(prompt_details.get("cached_tokens") or 0)
        self.cache_write_tokens = int(prompt_details.get("cache_write_tokens") or 0)
        self.reasoning_tokens = int(completion_details.get("reasoning_tokens") or 0)
        self.total_tokens = int(usage.get("total_tokens") or 0)
        self.has_usage = True

    def set_anthropic_usage(self, usage: dict[str, Any]) -> None:
        """Single token-extraction point for the Anthropic Messages usage shape.

        Anthropic reports ``input_tokens`` WITHOUT the cache components
        (matching Claude's prompt_tokens = input + cache_read + cache_creation
        convention), so the router's prompt_tokens sums all three to stay
        comparable with the OpenAI-wire rows; cached_tokens / cache_write_tokens
        carry the split for the cost formula.
        """
        input_tokens = int(usage.get("input_tokens") or 0)
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        cache_write = int(usage.get("cache_creation_input_tokens") or 0)
        self.prompt_tokens = input_tokens + cache_read + cache_write
        self.cached_tokens = cache_read
        self.cache_write_tokens = cache_write
        self.completion_tokens = int(usage.get("output_tokens") or 0)
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        self.has_usage = True


def request_stats(request: object | None) -> RequestStats:
    """Read the RequestStats set by middleware, or a fresh throwaway holder.

    Tolerates a missing request and a request that never passed through the
    middleware (unit tests, ASGI lifespan), so callers never branch on None.
    The throwaway is created per call — the holder is mutable, so a shared
    one would leak enrichment between requests.
    """
    if request is None:
        return RequestStats()
    stats = getattr(getattr(request, "state", None), "request_stats", None)
    return stats if isinstance(stats, RequestStats) else RequestStats()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

async def init_db(db_path: str, pricing_lookup: PricingLookup) -> None:
    """Open the usage DB, apply the schema and migrations, install pricing_lookup."""
    global _pricing_lookup
    _pricing_lookup = pricing_lookup
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    # WHY: installed through set_connection, never assigned directly — a
    # prior connection (a re-init, a leaked handle) is closed instead of
    # orphaned.
    conn = await aiosqlite.connect(db_path)
    # WHY: dashboard queries map rows by column name (dict(row)), so the SQL
    # aliases ARE the JSON keys; index access keeps working for the rest.
    conn.row_factory = aiosqlite.Row
    await _conn.set_connection(conn)
    await conn.execute("PRAGMA journal_mode=WAL")
    # WHY: WAL allows one writer at a time; busy_timeout makes a concurrent
    # writer wait instead of failing instantly with "database is locked" — and
    # the failure is swallowed by _flush_row, so usage events would vanish in
    # silence. The only other client this tolerates is an in-container
    # inspection connection; a host-side sqlite3 open of the file is not
    # survivable at all (it resets the WAL over VirtioFS) — preventing that is
    # the INVARIANT(data-loss) on the usage_data volume in docker-compose.yml.
    await conn.execute("PRAGMA busy_timeout=5000")
    await _apply_schema(conn)


async def _apply_schema(conn: aiosqlite.Connection) -> None:
    """Create usage_events, apply the additive migrations and the timestamp index."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS usage_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL,
            project_name TEXT NOT NULL,
            model_id TEXT NOT NULL,
            provider_name TEXT,
            endpoint TEXT NOT NULL,
            timestamp REAL NOT NULL,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            cached_tokens INTEGER NOT NULL DEFAULT 0,
            cache_write_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            duration_ms REAL,
            status_code INTEGER,
            error_code TEXT,
            error_message TEXT,
            api_key_hash TEXT,
            client_ip TEXT,
            reasoning_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd REAL,
            has_usage INTEGER NOT NULL DEFAULT 1,
            stream INTEGER NOT NULL DEFAULT 0
        )
    """)
    # Additive migration over an existing DB: existing rows are kept and read
    # with NULL/default values in the new columns.
    cursor = await conn.execute("PRAGMA table_info(usage_events)")
    existing = {row[1] for row in await cursor.fetchall()}
    for column, decl in _MIGRATIONS.items():
        if column not in existing:
            await conn.execute(
                f"ALTER TABLE usage_events ADD COLUMN {column} {decl}")
    # The request log is ORDER BY timestamp DESC; the existing composite index
    # starts with project_name, so it cannot serve that scan.
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_events(timestamp)")
    await conn.commit()


async def close_db() -> None:
    global _pricing_lookup
    _pricing_lookup = None
    await _conn.set_connection(None)


# ---------------------------------------------------------------------------
# Flush
# ---------------------------------------------------------------------------

# INVARIANT: stored pricing is USD PER TOKEN (OpenRouter's convention).
# Why: model_info.yaml is hand-curated — a "per 1M tokens" value would silently
# inflate every recorded cost by 10^6, and costs are frozen at write time so
# the error would be invisible and permanent. Pinned by tests.
def _compute_cost_usd(stats: RequestStats, pricing_lookup: PricingLookup | None) -> float | None:
    """Write-time cost from merged capabilities pricing; None when unpriced.

    (prompt - cached - cache_write) * prompt + cached * input_cache_read +
    cache_write * input_cache_write + completion * completion.
    A missing input_cache_read/input_cache_write falls back to the prompt
    rate — stored pricing only contains the keys the upstream actually sent,
    and treating an absent cache rate as free would systematically
    under-report cost. A price that is not a number also returns None: an
    unpriced row, never a lost one.
    """
    if stats.prompt_tokens + stats.completion_tokens <= 0:
        return None
    if pricing_lookup is None:
        return None
    try:
        pricing = pricing_lookup(stats.model_id)
    except Exception:
        # A broken pricing source degrades to an unpriced row, never a lost one.
        return None
    if not isinstance(pricing, dict) or not pricing:
        return None
    prompt_price = pricing.get("prompt")
    completion_price = pricing.get("completion")
    if prompt_price is None or completion_price is None:
        return None
    cache_read_price = pricing.get("input_cache_read", prompt_price)
    cache_write_price = pricing.get("input_cache_write", prompt_price)
    # WHY: a price that is not a number degrades to an unpriced row, never a
    # lost one — int*str is string repetition, a float+str sum raises
    # TypeError, and _flush_row calls this OUTSIDE its try, so the whole
    # usage row would vanish. bool is excluded explicitly: it is an int
    # subclass, so True would price as a real 1.0.
    if any(isinstance(p, bool) or not isinstance(p, (int, float))
           for p in (prompt_price, completion_price, cache_read_price, cache_write_price)):
        return None
    non_cached = max(stats.prompt_tokens - stats.cached_tokens - stats.cache_write_tokens, 0)
    return (non_cached * prompt_price
            + stats.cached_tokens * cache_read_price
            + stats.cache_write_tokens * cache_write_price
            + stats.completion_tokens * completion_price)


async def _flush_row(
    stats: RequestStats,
    *,
    request_id: str,
    project_name: str,
    duration_ms: float,
    status_code: int,
) -> None:
    """Insert the single usage_events row for a finished request.

    NOT NULL fallbacks: a 401 knows no project/model/provider, so the flush
    substitutes "unknown" / "" — without this the INSERT raises and the
    failure is swallowed below, i.e. errors would silently not be recorded.
    """
    cost_usd = _compute_cost_usd(stats, _pricing_lookup)
    error_message = stats.error_message[:_MAX_ERROR_MESSAGE] if stats.error_message else None
    try:
        conn = get_connection()
        if conn is None:
            return
        await conn.execute(
            """INSERT INTO usage_events
               (request_id, project_name, model_id, provider_name, endpoint,
                timestamp, prompt_tokens, completion_tokens, cached_tokens,
                cache_write_tokens, reasoning_tokens, total_tokens, duration_ms,
                status_code, error_code, error_message, api_key_hash, client_ip,
                cost_usd, has_usage, stream)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                request_id,
                project_name or "unknown",
                stats.model_id or "",
                stats.provider_name or "",
                stats.endpoint,
                time.time(),
                stats.prompt_tokens,
                stats.completion_tokens,
                stats.cached_tokens,
                stats.cache_write_tokens,
                stats.reasoning_tokens,
                stats.total_tokens,
                duration_ms,
                status_code,
                stats.error_code,
                error_message,
                stats.api_key_hash,
                stats.client_ip or "",
                cost_usd,
                1 if stats.has_usage else 0,
                1 if stats.stream else 0,
            ),
        )
        await conn.commit()
    except Exception as e:
        # WHY: the flush runs on every request; a sustained DB outage would
        # otherwise flood logs. Throttle to a summary once per minute.
        global _last_usage_error_logged
        now = time.time()
        if now - _last_usage_error_logged >= _USAGE_ERROR_LOG_INTERVAL:
            _last_usage_error_logged = now
            logger.error(
                f"Failed to record usage: {e}",
                extra={
                    "log_type": "error",
                    "error_type": "usage_recording_error",
                    "request_id": request_id,
                    "model_id": stats.model_id,
                    "endpoint": stats.endpoint,
                },
                exc_info=True,
            )


# Last wall-clock timestamp at which a flush error was logged at full detail.
# Used to throttle logging during a DB outage.
_last_usage_error_logged: float = 0.0
_USAGE_ERROR_LOG_INTERVAL: float = 60.0


def _on_usage_done(task: asyncio.Task) -> None:
    """Done callback: discard the task and log WARNING on failure."""
    _usage_tasks.discard(task)
    exc = task.exception()
    if exc is not None:
        logger.warning(
            f"Usage recording task failed: {exc}",
            extra={
                "log_type": "warning",
                "error_type": "usage_task_failure",
                "error_message": str(exc),
            },
            exc_info=exc,
        )


async def drain_pending_flushes(drain_timeout: float = 10.0) -> None:
    """Await every pending flush task, bounded by drain_timeout — the shutdown path.

    Called by the app lifespan between provider-pool close and close_db() so an
    in-flight flush cannot race the connection close. Gathers a COPY of the
    set: each task's done callback discards it from _usage_tasks, so awaiting
    tasks while iterating the live set would race its own completion. A flush
    is one INSERT + commit; anything still pending after the timeout is logged
    at WARNING and abandoned — a stuck flush must never block shutdown.
    """
    pending = [t for t in list(_usage_tasks) if not t.done()]
    if not pending:
        return
    _, leftovers = await asyncio.wait(pending, timeout=drain_timeout)
    if leftovers:
        logger.warning(
            f"{len(leftovers)} usage flush task(s) still pending after {drain_timeout}s; abandoning",
            extra={
                "log_type": "warning",
                "error_type": "usage_drain_timeout",
                "pending_count": len(leftovers),
            },
        )


def schedule_flush(
    stats: RequestStats,
    *,
    request_id: str,
    project_name: str,
    duration_ms: float,
    status_code: int,
) -> asyncio.Task | None:
    """Fire-and-forget single-row flush (tracked in _usage_tasks).

    Returns the created task, or None when no event loop is running. The
    shutdown lifespan drains the tracked set via drain_pending_flushes()
    BEFORE close_db(); a flush scheduled after that drain sees a None
    connection and no-ops — a bounded loss window at process exit, accepted.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    task = loop.create_task(_flush_row(
        stats,
        request_id=request_id,
        project_name=project_name,
        duration_ms=duration_ms,
        status_code=status_code,
    ))
    _usage_tasks.add(task)
    task.add_done_callback(_on_usage_done)
    return task
