"""Unit tests for the src/core/usage_db package — holder, flush, cost, migration, queries.

Patch targets point at the SUBMODULE the caller resolves the name from, never
the package re-export in __init__ (an alias patch is inert; see the WHY in
src/core/usage_db/__init__.py).
"""

import asyncio
import contextlib
import hashlib
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import httpx
import pytest
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pytest_asyncio import fixture as asyncio_fixture

from src.api.stat_routes import verify_stat_key
from src.core import usage_db
from src.core.config_schema import parse_config
from src.core.usage_db import RequestStats, writer


@pytest.fixture(autouse=True)
def reset_tasks():
    writer._usage_tasks.clear()
    yield
    writer._usage_tasks.clear()


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "usage.db")


# Pricing the writer resolves at flush time (per token, like stored pricing).
_PRICES = {
    "m": {"prompt": 1e-6, "completion": 2e-6},
    "m1": {"prompt": 1e-6, "completion": 2e-6},
}


@asyncio_fixture
async def db(db_path):
    await usage_db.init_db(db_path, _PRICES.get)
    yield usage_db.get_connection()
    await usage_db.close_db()


async def fetch_rows(conn, sql="SELECT * FROM usage_events", params=()):
    cursor = await conn.execute(sql, params)
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in await cursor.fetchall()]


async def drain_tasks():
    tasks = list(writer._usage_tasks)
    if tasks:
        await asyncio.gather(*tasks)


async def flush(stats, *, request_id="r", project_name="p", duration_ms=1.0,
                status_code=200):
    await writer._flush_row(
        stats, request_id=request_id, project_name=project_name,
        duration_ms=duration_ms, status_code=status_code,
    )


def pricing_lookup(pricing):
    """A pricing source that answers `pricing` for every model."""
    return lambda model_id: pricing


# ---------------------------------------------------------------------------
# Connection setter (_conn is the seam between writer and queries)
# ---------------------------------------------------------------------------

class TestConnectionSetter:
    """set_connection is the single write point for the shared connection and
    closes any prior one, so a re-init can never orphan an open handle."""

    @pytest.mark.asyncio
    async def test_reinit_closes_prior_connection(self, db_path):
        await usage_db.init_db(db_path, _PRICES.get)
        first = usage_db.get_connection()
        with patch.object(first, "close", wraps=first.close) as close_spy:
            await usage_db.init_db(db_path, _PRICES.get)
            close_spy.assert_called_once()
        second = usage_db.get_connection()
        assert second is not first
        await usage_db.close_db()
        assert usage_db.get_connection() is None

    @pytest.mark.asyncio
    async def test_failing_close_still_installs_the_new_connection(self, db_path):
        """A prior close that raises must not abort the swap: bailing would
        leave the dead handle installed AND orphan the fresh one the caller
        already opened — the exact leak set_connection exists to prevent."""
        await usage_db.init_db(db_path, _PRICES.get)
        first = usage_db.get_connection()
        with patch.object(first, "close", side_effect=RuntimeError("close failed")):
            await usage_db.init_db(db_path, _PRICES.get)
        second = usage_db.get_connection()
        assert second is not first
        # The fresh handle is live, not a corpse left behind by the failed close.
        cursor = await second.execute("SELECT 1")
        assert tuple(await cursor.fetchone()) == (1,)
        await usage_db.close_db()


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

_OLD_SCHEMA = """
    CREATE TABLE usage_events (
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
        total_tokens INTEGER NOT NULL DEFAULT 0,
        duration_ms REAL,
        status_code INTEGER
    )
"""

_NEW_COLUMNS = {
    "error_code", "error_message", "api_key_hash", "client_ip",
    "reasoning_tokens", "cost_usd", "has_usage", "stream", "cache_write_tokens",
}


class TestMigration:

    @pytest.mark.asyncio
    async def test_adds_new_columns_to_existing_db(self, db_path):
        conn = await aiosqlite.connect(db_path)
        await conn.execute(_OLD_SCHEMA)
        await conn.execute(
            """INSERT INTO usage_events
               (request_id, project_name, model_id, endpoint, timestamp,
                prompt_tokens, completion_tokens, cached_tokens, total_tokens)
               VALUES ('old', 'p', 'm', 'chat', 1.0, 10, 5, 0, 15)"""
        )
        await conn.commit()
        await conn.close()

        await usage_db.init_db(db_path, _PRICES.get)
        conn = usage_db.get_connection()
        cursor = await conn.execute("PRAGMA table_info(usage_events)")
        columns = {row[1] for row in await cursor.fetchall()}
        assert columns >= _NEW_COLUMNS

    @pytest.mark.asyncio
    async def test_existing_rows_kept_with_defaults(self, db_path):
        conn = await aiosqlite.connect(db_path)
        await conn.execute(_OLD_SCHEMA)
        await conn.execute(
            """INSERT INTO usage_events
               (request_id, project_name, model_id, endpoint, timestamp,
                prompt_tokens, completion_tokens, cached_tokens, total_tokens)
               VALUES ('old', 'p', 'm', 'chat', 1.0, 10, 5, 0, 15)"""
        )
        await conn.commit()
        await conn.close()

        await usage_db.init_db(db_path, _PRICES.get)
        rows = await fetch_rows(usage_db.get_connection())
        assert len(rows) == 1
        row = rows[0]
        # Default asymmetry: pre-existing rows always had usage → has_usage 1.
        assert row["has_usage"] == 1
        assert row["reasoning_tokens"] == 0
        assert row["stream"] == 0
        assert row["cost_usd"] is None
        assert row["error_code"] is None
        assert row["client_ip"] is None

    @pytest.mark.asyncio
    async def test_fresh_db_has_all_columns(self, db):
        cursor = await db.execute("PRAGMA table_info(usage_events)")
        columns = {row[1] for row in await cursor.fetchall()}
        assert columns >= _NEW_COLUMNS

    @pytest.mark.asyncio
    async def test_timestamp_index_created(self, db_path):
        conn = await aiosqlite.connect(db_path)
        await conn.execute(_OLD_SCHEMA)
        await conn.commit()
        await conn.close()

        await usage_db.init_db(db_path, _PRICES.get)
        cursor = await usage_db.get_connection().execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_usage_ts'"
        )
        assert await cursor.fetchone() is not None


# ---------------------------------------------------------------------------
# RequestStats.set_usage — the single token-extraction point
# ---------------------------------------------------------------------------

class TestSetUsage:

    def test_extracts_tokens_and_details(self):
        stats = RequestStats()
        stats.set_usage({
            "prompt_tokens": 100,
            "completion_tokens": 40,
            "total_tokens": 140,
            "prompt_tokens_details": {"cached_tokens": 60},
            "completion_tokens_details": {"reasoning_tokens": 25},
        })
        assert stats.prompt_tokens == 100
        assert stats.completion_tokens == 40
        assert stats.cached_tokens == 60
        assert stats.reasoning_tokens == 25
        assert stats.total_tokens == 140
        assert stats.has_usage is True

    def test_missing_details_tolerated(self):
        stats = RequestStats()
        stats.set_usage({"prompt_tokens": 10, "total_tokens": 10})
        assert stats.cached_tokens == 0
        assert stats.reasoning_tokens == 0
        assert stats.has_usage is True

    def test_none_details_tolerated(self):
        stats = RequestStats()
        stats.set_usage({
            "prompt_tokens": 10,
            "prompt_tokens_details": None,
            "completion_tokens_details": None,
        })
        assert stats.cached_tokens == 0
        assert stats.reasoning_tokens == 0

    def test_defaults_has_usage_false(self):
        assert RequestStats().has_usage is False


class TestSetAnthropicUsage:
    """set_anthropic_usage — the single extraction point for the Anthropic
    Messages usage shape (input/output + cache read/creation)."""

    def test_full_arithmetic(self):
        stats = RequestStats()
        stats.set_anthropic_usage({
            "input_tokens": 25, "output_tokens": 12,
            "cache_read_input_tokens": 64, "cache_creation_input_tokens": 128,
        })
        assert stats.prompt_tokens == 25 + 64 + 128
        assert stats.cached_tokens == 64
        assert stats.cache_write_tokens == 128
        assert stats.completion_tokens == 12
        assert stats.total_tokens == 25 + 64 + 128 + 12
        assert stats.has_usage is True

    def test_missing_cache_keys_default_zero(self):
        stats = RequestStats()
        stats.set_anthropic_usage({"input_tokens": 5, "output_tokens": 2})
        assert stats.prompt_tokens == 5
        assert stats.cached_tokens == 0
        assert stats.cache_write_tokens == 0
        assert stats.total_tokens == 7
        assert stats.has_usage is True

    def test_none_usage_tolerated(self):
        stats = RequestStats()
        stats.set_anthropic_usage({})
        assert stats.has_usage is True
        assert stats.prompt_tokens == 0


class TestSetUsageCacheWrite:
    """set_usage also reads the OpenRouter-reported cache write."""

    def test_openrouter_cache_write_tokens_read(self):
        stats = RequestStats()
        stats.set_usage({
            "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
            "prompt_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 30},
        })
        assert stats.cached_tokens == 40
        assert stats.cache_write_tokens == 30

    def test_absent_details_leave_zero(self):
        stats = RequestStats()
        stats.set_usage({"prompt_tokens": 10, "total_tokens": 10})
        assert stats.cache_write_tokens == 0


class TestRequestStatsAccessor:

    def test_returns_holder_from_request_state(self):
        request = MagicMock()
        holder = RequestStats()
        request.state.request_stats = holder
        assert usage_db.request_stats(request) is holder

    def test_throwaway_when_missing(self):
        request = MagicMock(spec=Request)
        del request.state  # attribute error on access
        # MagicMock(spec) makes .state raise AttributeError → accessor must cope
        stats = usage_db.request_stats(request)
        assert isinstance(stats, RequestStats)

    def test_throwaway_is_fresh_per_call(self):
        request = MagicMock(spec=Request)
        first = usage_db.request_stats(request)
        first.model_id = "mutated"
        second = usage_db.request_stats(request)
        assert second.model_id == ""

    def test_none_request_returns_throwaway(self):
        stats = usage_db.request_stats(None)
        assert isinstance(stats, RequestStats)


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

class TestComputeCost:

    def test_priced_event(self):
        stats = RequestStats(prompt_tokens=1000, completion_tokens=200, cached_tokens=0)
        state = pricing_lookup({"prompt": 1e-6, "completion": 2e-6, "input_cache_read": 0.1e-6})
        cost = writer._compute_cost_usd(stats, state)
        assert cost == pytest.approx(1000 * 1e-6 + 200 * 2e-6)

    def test_cached_tokens_use_cache_rate(self):
        stats = RequestStats(prompt_tokens=1000, completion_tokens=0, cached_tokens=400)
        state = pricing_lookup({"prompt": 1e-6, "completion": 2e-6, "input_cache_read": 0.1e-6})
        cost = writer._compute_cost_usd(stats, state)
        assert cost == pytest.approx(600 * 1e-6 + 400 * 0.1e-6)

    def test_missing_input_cache_read_falls_back_to_prompt_rate(self):
        """An absent cache rate must NOT be treated as free."""
        stats = RequestStats(prompt_tokens=1000, completion_tokens=0, cached_tokens=400)
        state = pricing_lookup({"prompt": 1e-6, "completion": 2e-6})
        cost = writer._compute_cost_usd(stats, state)
        assert cost == pytest.approx(1000 * 1e-6)

    def test_unpriced_model_returns_none(self):
        stats = RequestStats(prompt_tokens=10, completion_tokens=5)
        assert writer._compute_cost_usd(stats, pricing_lookup(None)) is None

    def test_zero_token_event_returns_none(self):
        stats = RequestStats(prompt_tokens=0, completion_tokens=0)
        state = pricing_lookup({"prompt": 1e-6, "completion": 2e-6})
        assert writer._compute_cost_usd(stats, state) is None

    def test_no_pricing_lookup_returns_none(self):
        """Before init_db / after close_db nothing is priced."""
        stats = RequestStats(prompt_tokens=10, completion_tokens=5)
        assert writer._compute_cost_usd(stats, None) is None

    def test_get_pricing_failure_degrades_to_none(self):
        stats = RequestStats(prompt_tokens=10, completion_tokens=5)
        cost = writer._compute_cost_usd(stats, MagicMock(side_effect=RuntimeError("boom")))
        assert cost is None

    def test_missing_prompt_or_completion_rate_returns_none(self):
        stats = RequestStats(prompt_tokens=10, completion_tokens=5)
        assert writer._compute_cost_usd(stats, pricing_lookup({"prompt": 1e-6})) is None
        assert writer._compute_cost_usd(stats, pricing_lookup({"completion": 1e-6})) is None

    def test_pricing_is_per_token_not_per_million(self):
        # INVARIANT: stored pricing is USD per token. 1M tokens at $3/M must
        # cost $3 — a "per 1M" stored value would inflate this by 10^6.
        stats = RequestStats(prompt_tokens=1_000_000, completion_tokens=0)
        state = pricing_lookup({"prompt": 3e-6, "completion": 15e-6})
        assert writer._compute_cost_usd(stats, state) == pytest.approx(3.0)

    def test_non_numeric_price_returns_none(self):
        """A str/bool price degrades to an unpriced row, never a lost one:
        int*str is string repetition and a float+str sum raises TypeError —
        and _flush_row computes cost OUTSIDE its try, so the row would vanish."""
        stats = RequestStats(prompt_tokens=10, completion_tokens=5)
        assert writer._compute_cost_usd(
            stats, pricing_lookup({"prompt": "1e-7", "completion": 2e-6})) is None
        assert writer._compute_cost_usd(
            stats, pricing_lookup({"prompt": 1e-6, "completion": True})) is None

    def test_non_numeric_cache_rate_returns_none(self):
        """The prompt-rate fallback only applies when input_cache_read is
        ABSENT — a present-but-bad cache rate must be caught by the guard."""
        stats = RequestStats(prompt_tokens=10, completion_tokens=0, cached_tokens=4)
        state = pricing_lookup({"prompt": 1e-6, "completion": 2e-6,
                                "input_cache_read": "x"})
        assert writer._compute_cost_usd(stats, state) is None

    def test_cache_write_tokens_use_write_rate(self):
        """(prompt - cached - cache_write) * prompt + cached * read +
        cache_write * write + completion * completion."""
        stats = RequestStats(prompt_tokens=1000, completion_tokens=100,
                             cached_tokens=200, cache_write_tokens=300)
        state = pricing_lookup({"prompt": 1e-6, "completion": 2e-6,
                                "input_cache_read": 0.1e-6, "input_cache_write": 1.25e-6})
        cost = writer._compute_cost_usd(stats, state)
        assert cost == pytest.approx(500 * 1e-6 + 200 * 0.1e-6 + 300 * 1.25e-6
                                     + 100 * 2e-6)

    def test_missing_input_cache_write_falls_back_to_prompt_rate(self):
        """Same rule and WHY as input_cache_read: an absent cache-write rate
        must not be treated as free."""
        stats = RequestStats(prompt_tokens=1000, completion_tokens=0, cache_write_tokens=400)
        state = pricing_lookup({"prompt": 1e-6, "completion": 2e-6})
        cost = writer._compute_cost_usd(stats, state)
        assert cost == pytest.approx(1000 * 1e-6)

    def test_non_numeric_cache_write_rate_returns_none(self):
        stats = RequestStats(prompt_tokens=10, completion_tokens=0, cache_write_tokens=4)
        state = pricing_lookup({"prompt": 1e-6, "completion": 2e-6,
                                "input_cache_write": "x"})
        assert writer._compute_cost_usd(stats, state) is None


# ---------------------------------------------------------------------------
# Flush
# ---------------------------------------------------------------------------

class TestFlushRow:

    @pytest.mark.asyncio
    async def test_not_null_fallbacks_for_error_rows(self, db):
        """A 401 knows no project/model/provider — flush must substitute."""
        stats = RequestStats(
            endpoint="chat", client_ip="1.2.3.4",
            error_code="invalid_api_key", error_message="Invalid API key",
            api_key_hash="abcd1234",
        )
        await flush(stats, request_id="r1", project_name="", status_code=401)

        rows = await fetch_rows(db)
        assert len(rows) == 1
        row = rows[0]
        assert row["project_name"] == "unknown"
        assert row["model_id"] == ""
        assert row["provider_name"] == ""
        assert row["status_code"] == 401
        assert row["error_code"] == "invalid_api_key"
        assert row["error_message"] == "Invalid API key"
        assert row["api_key_hash"] == "abcd1234"
        assert row["client_ip"] == "1.2.3.4"
        assert row["has_usage"] == 0
        assert row["stream"] == 0
        assert row["cost_usd"] is None

    @pytest.mark.asyncio
    async def test_success_row_with_usage_and_stream(self, db):
        stats = RequestStats(
            endpoint="chat", model_id="m", provider_name="p", stream=True,
        )
        stats.set_usage({
            "prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140,
            "prompt_tokens_details": {"cached_tokens": 60},
            "completion_tokens_details": {"reasoning_tokens": 25},
        })
        await flush(stats, request_id="r2", project_name="proj", status_code=200)

        row = (await fetch_rows(db))[0]
        assert row["prompt_tokens"] == 100
        assert row["completion_tokens"] == 40
        assert row["cached_tokens"] == 60
        assert row["reasoning_tokens"] == 25
        assert row["total_tokens"] == 140
        assert row["has_usage"] == 1
        assert row["stream"] == 1
        assert row["cost_usd"] == pytest.approx(100 * 1e-6 + 40 * 2e-6)
        assert row["error_code"] is None
        assert row["error_message"] is None

    @pytest.mark.asyncio
    async def test_cache_write_tokens_column_written(self, db):
        stats = RequestStats(endpoint="messages", model_id="m", provider_name="p")
        stats.set_anthropic_usage({"input_tokens": 10, "output_tokens": 4,
                                   "cache_read_input_tokens": 5,
                                   "cache_creation_input_tokens": 2})
        await flush(stats, request_id="r-cw", project_name="proj", status_code=200)

        row = (await fetch_rows(db))[0]
        assert row["cache_write_tokens"] == 2
        assert row["cached_tokens"] == 5
        assert row["prompt_tokens"] == 17

    @pytest.mark.asyncio
    async def test_error_message_truncated_to_500(self, db):
        stats = RequestStats(error_code="internal_server_error", error_message="x" * 600)
        await flush(stats, status_code=500)
        row = (await fetch_rows(db))[0]
        assert row["error_message"] == "x" * 500

    @pytest.mark.asyncio
    async def test_non_numeric_price_still_inserts_unpriced_row(self, db):
        """A bad price loses the cost, not the row: cost_usd NULL, usage kept."""
        stats = RequestStats(endpoint="chat", model_id="m", provider_name="p")
        stats.set_usage({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        with patch("src.core.usage_db.writer._pricing_lookup",
                   pricing_lookup({"prompt": 1e-7, "completion": "x"})):
            await flush(stats, status_code=200)

        row = (await fetch_rows(db))[0]
        assert row["prompt_tokens"] == 10
        assert row["has_usage"] == 1
        assert row["cost_usd"] is None

    @pytest.mark.asyncio
    async def test_no_connection_noop(self, db_path):
        """After close_db (or before init) the flush is a silent no-op."""
        stats = RequestStats(endpoint="chat")
        with patch("src.core.usage_db.writer.logger") as mock_logger:
            await flush(stats)
        mock_logger.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_db_failure_logged_throttled(self, db):
        conn = MagicMock()
        conn.execute = AsyncMock(side_effect=RuntimeError("db down"))
        with patch("src.core.usage_db.writer.get_connection", return_value=conn), \
             patch("src.core.usage_db.writer.logger") as mock_logger:
            await flush(RequestStats(endpoint="chat"))
        mock_logger.error.assert_called_once()


class TestScheduleFlush:

    @pytest.mark.asyncio
    async def test_task_tracked_and_discarded_on_success(self):
        with patch("src.core.usage_db.writer._flush_row", new=AsyncMock()):
            task = writer.schedule_flush(
                RequestStats(endpoint="chat"), request_id="r",
                project_name="p", duration_ms=1.0, status_code=200,
            )
            assert task is not None
            assert task in writer._usage_tasks
            await task
            assert task not in writer._usage_tasks

    @pytest.mark.asyncio
    async def test_failing_task_logs_warning(self):
        with patch("src.core.usage_db.writer._flush_row",
                   new=AsyncMock(side_effect=RuntimeError("boom"))), \
             patch("src.core.usage_db.writer.logger") as mock_logger:
            task = writer.schedule_flush(
                RequestStats(endpoint="chat"), request_id="r",
                project_name="p", duration_ms=1.0, status_code=200,
            )
            try:
                await task
            except RuntimeError:
                pass
            await asyncio.sleep(0)
        assert task not in writer._usage_tasks
        mock_logger.warning.assert_called_once()

    def test_no_running_loop_returns_none(self):
        task = writer.schedule_flush(
            RequestStats(endpoint="chat"), request_id="r",
            project_name="p", duration_ms=1.0, status_code=200,
        )
        assert task is None


class TestDrainPendingFlushes:
    """Shutdown drain: pending flush tasks complete BEFORE close_db()."""

    @pytest.mark.asyncio
    async def test_drain_awaits_pending_flush(self, db):
        """A gated _flush_row finishes only after release; drain must wait for it.

        Two tasks are scheduled so one completes (its done callback mutates
        _usage_tasks) while drain is still awaiting the other — a drain that
        iterated the live set would race its own completion.
        """
        entered = asyncio.Event()
        gate = asyncio.Event()

        async def slow_flush(*a, **k):
            entered.set()
            await gate.wait()

        async def fast_flush(*a, **k):
            return None

        with patch("src.core.usage_db.writer._flush_row", new=slow_flush):
            gated = writer.schedule_flush(
                RequestStats(endpoint="chat"), request_id="r-gated",
                project_name="p", duration_ms=1.0, status_code=200,
            )
            await asyncio.wait_for(entered.wait(), timeout=1.0)
        with patch("src.core.usage_db.writer._flush_row", new=fast_flush):
            writer.schedule_flush(
                RequestStats(endpoint="chat"), request_id="r-fast",
                project_name="p", duration_ms=1.0, status_code=200,
            )
            await asyncio.sleep(0)  # let the fast task finish (done callback fires)

            drained = asyncio.create_task(writer.drain_pending_flushes(drain_timeout=1.0))
            await asyncio.sleep(0.05)
            assert not drained.done(), "drain returned while a flush was pending"

            gate.set()
            await asyncio.wait_for(drained, timeout=1.0)
        assert gated.done() and not gated.cancelled()

    @pytest.mark.asyncio
    async def test_drain_no_pending_returns_at_once(self, db):
        await asyncio.wait_for(writer.drain_pending_flushes(drain_timeout=1.0), timeout=1.0)

    @pytest.mark.asyncio
    async def test_drain_timeout_logs_warning_and_abandons(self, db):
        """A stuck flush cannot block shutdown: WARNING, then abandon."""
        entered = asyncio.Event()

        async def stuck_flush(*a, **k):
            entered.set()
            await asyncio.Event().wait()

        with patch("src.core.usage_db.writer._flush_row", new=stuck_flush), \
             patch("src.core.usage_db.writer.logger") as mock_logger:
            stuck = writer.schedule_flush(
                RequestStats(endpoint="chat"), request_id="r",
                project_name="p", duration_ms=1.0, status_code=200,
            )
            await asyncio.wait_for(entered.wait(), timeout=1.0)

            await asyncio.wait_for(writer.drain_pending_flushes(drain_timeout=0.05), timeout=1.0)
            mock_logger.warning.assert_called_once()

        stuck.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stuck


# ---------------------------------------------------------------------------
# Summary query
# ---------------------------------------------------------------------------

async def seed_summary_rows(db):
    ok1 = RequestStats(endpoint="chat", model_id="m1", provider_name="p1", stream=True)
    ok1.set_usage({"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130,
                   "prompt_tokens_details": {"cached_tokens": 50}})
    ok2 = RequestStats(endpoint="chat", model_id="m2", provider_name="p2")
    ok2.set_usage({"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110})
    err_no_usage = RequestStats(endpoint="chat", error_code="invalid_api_key")
    await flush(ok1, request_id="r1", project_name="alice", status_code=200)  # m1: priced
    await flush(ok2, request_id="r2", project_name="bob", status_code=200)  # m2: unpriced
    await flush(err_no_usage, request_id="r3", project_name="unknown", status_code=401)


class TestGetSummary:

    @pytest.mark.asyncio
    async def test_empty_selection_returns_zeros(self, db):
        result = await usage_db.get_summary([], [], None)
        totals = result["totals"]
        assert totals["requests"] == 0
        assert totals["errors"] == 0
        assert totals["error_rate"] == 0.0
        assert totals["cache_hit_rate"] == 0.0
        assert totals["unpriced"] == 0
        assert result["by_user"] == []
        assert result["by_model"] == []
        assert result["by_provider"] == []
        assert result["by_error_code"] == []

    @pytest.mark.asyncio
    async def test_totals_and_rates(self, db):
        await seed_summary_rows(db)
        result = await usage_db.get_summary([], [], None)
        totals = result["totals"]
        assert totals["requests"] == 3
        assert totals["errors"] == 1
        assert totals["error_rate"] == pytest.approx(1 / 3)
        assert totals["prompt_tokens"] == 200
        assert totals["cached_tokens"] == 50
        assert totals["completion_tokens"] == 40
        assert totals["total_tokens"] == 240
        # cache_hit_rate = Σcached / Σprompt; the has_usage=0 row contributes 0 to both sides
        assert totals["cache_hit_rate"] == pytest.approx(50 / 200)
        assert totals["unpriced"] == 1  # ok2 had usage but no pricing
        # cost sums only priced rows
        assert totals["cost_usd"] == pytest.approx(100 * 1e-6 + 30 * 2e-6)

    @pytest.mark.asyncio
    async def test_error_grouping_includes_null_bucket(self, db):
        err1 = RequestStats(endpoint="chat", error_code="invalid_api_key")
        err2 = RequestStats(endpoint="chat", error_code=None)  # e.g. a 422
        midstream = RequestStats(endpoint="chat", model_id="m", error_code="internal_server_error")
        await flush(err1, status_code=401)
        await flush(err2, status_code=422)
        await flush(midstream, status_code=200)  # mid-stream SSE failure
        await flush(RequestStats(endpoint="chat", model_id="m"), status_code=200)  # success

        result = await usage_db.get_summary([], [], None)
        codes = {r["error_code"]: r["count"] for r in result["by_error_code"]}
        assert codes == {None: 1, "invalid_api_key": 1, "internal_server_error": 1}
        # A status-200 row with error_code counts as an error
        assert result["totals"]["errors"] == 3

    @pytest.mark.asyncio
    async def test_user_model_provider_breakdowns(self, db):
        await seed_summary_rows(db)
        result = await usage_db.get_summary([], [], None)
        by_user = {r["user"]: r for r in result["by_user"]}
        assert set(by_user) == {"alice", "bob", "unknown"}
        assert by_user["alice"]["requests"] == 1
        assert by_user["alice"]["total_tokens"] == 130

        by_model = {r["model"]: r for r in result["by_model"]}
        assert set(by_model) == {"m1", "m2", ""}

        by_provider = {r["provider"]: r for r in result["by_provider"]}
        assert set(by_provider) == {"p1", "p2", ""}

    @pytest.mark.asyncio
    async def test_user_and_model_filters(self, db):
        await seed_summary_rows(db)
        result = await usage_db.get_summary(["alice"], ["m1"], None)
        assert result["totals"]["requests"] == 1
        assert result["totals"]["prompt_tokens"] == 100

    @pytest.mark.asyncio
    async def test_days_filter(self, db):
        await seed_summary_rows(db)
        conn = usage_db.get_connection()
        await conn.execute("UPDATE usage_events SET timestamp = ? WHERE request_id = 'r1'",
                           (time.time() - 10 * 86400,))
        await conn.commit()
        result = await usage_db.get_summary([], [], 7)
        assert result["totals"]["requests"] == 2
        result_all = await usage_db.get_summary([], [], None)
        assert result_all["totals"]["requests"] == 3

    @pytest.mark.asyncio
    async def test_by_day_breakdown(self, db):
        await seed_summary_rows(db)
        result = await usage_db.get_summary([], [], None)
        assert len(result["by_day"]) == 1
        day = result["by_day"][0]
        assert day["requests"] == 3
        assert day["errors"] == 1


# ---------------------------------------------------------------------------
# Requests query
# ---------------------------------------------------------------------------

async def seed_request_log(db):
    rows = [
        # (request_id, project, model, provider, status, error_code, ts_offset)
        ("r1", "alice", "m1", "p1", 200, None, 0),
        ("r2", "bob", "m2", "p2", 401, "invalid_api_key", 10),
        ("r3", "alice", "m1", "p1", 200, "internal_server_error", 20),  # mid-stream
        ("r4", "carol", "m3", "p3", 422, None, 30),
    ]
    now = time.time()
    for request_id, project, model, provider, status, error_code, offset in rows:
        stats = RequestStats(endpoint="chat", model_id=model, provider_name=provider,
                             error_code=error_code)
        await flush(stats, request_id=request_id, project_name=project, status_code=status)
        # Past timestamps (higher offset = more recent) keep newest-first order
        # while staying excludable by a days=0 cutoff.
        await db.execute("UPDATE usage_events SET timestamp = ? WHERE request_id = ?",
                         (now - (100 - offset), request_id))
    await db.commit()


class TestGetRequests:

    @pytest.mark.asyncio
    async def test_newest_first_and_pagination(self, db):
        await seed_request_log(db)
        result = await usage_db.get_requests(
            [], [], [], "all", "", "", None, limit=2, offset=0)
        assert result["total"] == 4
        assert [r["request_id"] for r in result["requests"]] == ["r4", "r3"]

        page2 = await usage_db.get_requests(
            [], [], [], "all", "", "", None, limit=2, offset=2)
        assert [r["request_id"] for r in page2["requests"]] == ["r2", "r1"]

    @pytest.mark.asyncio
    async def test_row_shape(self, db):
        await seed_request_log(db)
        result = await usage_db.get_requests([], [], [], "all", "", "", None)
        row = result["requests"][0]
        assert set(row) == {
            "id", "request_id", "timestamp", "project_name", "model_id",
            "provider_name", "endpoint", "stream", "prompt_tokens",
            "completion_tokens", "cached_tokens", "reasoning_tokens",
            "total_tokens", "cost_usd", "duration_ms", "status_code",
            "error_code", "error_message", "api_key_hash", "client_ip",
        }
        assert isinstance(row["stream"], bool)

    @pytest.mark.asyncio
    async def test_status_error_includes_midstream(self, db):
        await seed_request_log(db)
        result = await usage_db.get_requests([], [], [], "error", "", "", None)
        ids = {r["request_id"] for r in result["requests"]}
        assert ids == {"r2", "r3", "r4"}

    @pytest.mark.asyncio
    async def test_status_ok_excludes_errors(self, db):
        await seed_request_log(db)
        result = await usage_db.get_requests([], [], [], "ok", "", "", None)
        ids = {r["request_id"] for r in result["requests"]}
        assert ids == {"r1"}

    @pytest.mark.asyncio
    async def test_error_code_none_sentinel_matches_null(self, db):
        await seed_request_log(db)
        result = await usage_db.get_requests([], [], [], "all", "none", "", None)
        # Only the 422 (error row with NULL error_code) matches
        ids = {r["request_id"] for r in result["requests"]}
        assert ids == {"r4"}

    @pytest.mark.asyncio
    async def test_error_code_exact_match(self, db):
        await seed_request_log(db)
        result = await usage_db.get_requests([], [], [], "all", "invalid_api_key", "", None)
        ids = {r["request_id"] for r in result["requests"]}
        assert ids == {"r2"}

    @pytest.mark.asyncio
    async def test_request_id_search(self, db):
        await seed_request_log(db)
        result = await usage_db.get_requests([], [], [], "all", "", "r3", None)
        assert [r["request_id"] for r in result["requests"]] == ["r3"]

    @pytest.mark.asyncio
    async def test_provider_filter(self, db):
        await seed_request_log(db)
        result = await usage_db.get_requests([], [], ["p1"], "all", "", "", None)
        ids = {r["request_id"] for r in result["requests"]}
        assert ids == {"r1", "r3"}

    @pytest.mark.asyncio
    async def test_user_model_days_filters(self, db):
        await seed_request_log(db)
        result = await usage_db.get_requests(["alice"], ["m1"], [], "all", "", "", None)
        ids = {r["request_id"] for r in result["requests"]}
        assert ids == {"r1", "r3"}

        result = await usage_db.get_requests([], [], [], "all", "", "", 0)
        assert result["total"] == 0


# ---------------------------------------------------------------------------
# Chart series + distinct filters (get_usage_data / get_distinct_*)
# ---------------------------------------------------------------------------

class TestUsageDataAndDistinct:

    @pytest.mark.asyncio
    async def test_distinct_users_and_models(self, db):
        await seed_request_log(db)
        assert await usage_db.get_distinct_users() == ["alice", "bob", "carol"]
        assert await usage_db.get_distinct_models() == ["m1", "m2", "m3"]

    @pytest.mark.asyncio
    async def test_usage_data_groups_by_user_model_day_and_zero_fills(self, db):
        """Two users, two models, two days: each series is aligned over the
        union of dates, missing (user, model, day) cells are zero."""
        now = time.time()
        a1 = RequestStats(endpoint="chat", model_id="m1", provider_name="p")
        a1.set_usage({"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
                      "prompt_tokens_details": {"cached_tokens": 4}})
        b1 = RequestStats(endpoint="chat", model_id="m2", provider_name="p")
        b1.set_usage({"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8})
        await flush(a1, request_id="r1", project_name="alice", status_code=200)
        await flush(b1, request_id="r2", project_name="bob", status_code=200)
        # Push r1 to yesterday: m1's series needs a zero for today, m2's for
        # yesterday.
        await db.execute("UPDATE usage_events SET timestamp = ? WHERE request_id = 'r1'",
                         (now - 86400,))
        await db.commit()

        result = await usage_db.get_usage_data([], [], None)
        series = {s["model"]: s for s in result["series"]}
        assert set(series) == {"m1", "m2"}
        assert len(series["m1"]["dates"]) == 2
        # dates are the shared union, sorted ascending (yesterday first)
        assert series["m1"]["prompt"] == [10, 0]
        assert series["m1"]["cached"] == [4, 0]
        assert series["m2"]["prompt"] == [0, 7]

    @pytest.mark.asyncio
    async def test_usage_data_user_filter_narrows_series(self, db):
        a1 = RequestStats(endpoint="chat", model_id="m1", provider_name="p")
        a1.set_usage({"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
        b1 = RequestStats(endpoint="chat", model_id="m2", provider_name="p")
        b1.set_usage({"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8})
        await flush(a1, request_id="r1", project_name="alice", status_code=200)
        await flush(b1, request_id="r2", project_name="bob", status_code=200)

        result = await usage_db.get_usage_data(["alice"], [], None)
        assert [s["model"] for s in result["series"]] == ["m1"]

    @pytest.mark.asyncio
    async def test_usage_data_days_filter_excludes_old_rows(self, db):
        a1 = RequestStats(endpoint="chat", model_id="m1", provider_name="p")
        a1.set_usage({"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
        await flush(a1, request_id="r1", project_name="alice", status_code=200)
        await db.execute("UPDATE usage_events SET timestamp = ? WHERE request_id = 'r1'",
                         (time.time() - 10 * 86400,))
        await db.commit()

        assert await usage_db.get_usage_data([], [], 7) == {"series": []}
        recent = await usage_db.get_usage_data([], [], None)
        assert recent["series"] != []

    @pytest.mark.asyncio
    async def test_no_connection_returns_empty_shapes(self):
        await usage_db.close_db()
        assert await usage_db.get_distinct_users() == []
        assert await usage_db.get_distinct_models() == []
        assert await usage_db.get_usage_data([], [], None) == {"series": []}


# ---------------------------------------------------------------------------
# End-to-end rows through the middleware + exception handler
# ---------------------------------------------------------------------------

def build_stats_app() -> FastAPI:
    from src.api.main import custom_http_exception_handler
    from src.api.middleware import RequestLoggerMiddleware
    from src.core.usage_db import request_stats

    app = FastAPI()
    app.add_middleware(RequestLoggerMiddleware)
    app.add_exception_handler(HTTPException, custom_http_exception_handler)

    @app.get("/ok", name="ok")
    async def ok():
        return {"status": "ok"}

    @app.get("/boom-dict")
    async def boom_dict():
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": 401, "message": "Invalid API key",
                              "metadata": {"error_code": "invalid_api_key"}}},
        )

    @app.get("/boom-string")
    async def boom_string():
        raise HTTPException(status_code=404, detail="Not Found")

    @app.get("/needs-int")
    async def needs_int(n: int):
        return {"n": n}

    @app.get("/provider-error")
    async def provider_error():
        raise HTTPException(
            status_code=429,
            detail={"error": {"code": 429, "message": "rate limited",
                              "metadata": {"error_code": "provider_http_error",
                                           "provider_name": "openai"}}},
        )

    @app.get("/stream")
    async def stream(request: Request):
        from src.services.chat_service.stream_processor import process_stream

        async def provider_stream():
            yield b'data: {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}\n\n'
            raise HTTPException(
                status_code=500,
                detail={"error": {"code": 500, "message": "upstream died mid-stream",
                                  "metadata": {"error_code": "internal_server_error"}}},
            )

        stats = request_stats(request)
        stats.model_id = "m"
        stats.provider_name = "p"
        stats.stream = True

        async def body():
            async for chunk in process_stream(
                    provider_stream(), "m", "r", "u", "p", stats=stats):
                yield chunk

        return StreamingResponse(body(), media_type="text/event-stream")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


class TestMiddlewareEndToEndRows:
    """Each request through the middleware produces exactly one row."""

    async def _get(self, app, path):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path)

    @pytest.mark.asyncio
    async def test_success_row(self, db):
        app = build_stats_app()
        resp = await self._get(app, "/ok")
        assert resp.status_code == 200
        await drain_tasks()
        rows = await fetch_rows(db)
        assert len(rows) == 1
        assert rows[0]["status_code"] == 200
        # endpoint now comes from the route's explicit name (see middleware);
        # a resolved-but-nameless synthetic route would record the handler name
        assert rows[0]["endpoint"] == "ok"

    @pytest.mark.asyncio
    async def test_skip_list_produces_no_row(self, db):
        app = build_stats_app()
        await self._get(app, "/health")
        await drain_tasks()
        assert await fetch_rows(db) == []

    @pytest.mark.asyncio
    async def test_handler_4xx_with_dict_detail(self, db):
        app = build_stats_app()
        resp = await self._get(app, "/boom-dict")
        assert resp.status_code == 401
        await drain_tasks()
        rows = await fetch_rows(db)
        assert len(rows) == 1
        row = rows[0]
        assert row["status_code"] == 401
        assert row["error_code"] == "invalid_api_key"
        assert row["error_message"] == "Invalid API key"
        assert row["project_name"] == "unknown"

    @pytest.mark.asyncio
    async def test_provider_error_enriches_provider_name(self, db):
        app = build_stats_app()
        resp = await self._get(app, "/provider-error")
        assert resp.status_code == 429
        await drain_tasks()
        row = (await fetch_rows(db))[0]
        assert row["error_code"] == "provider_http_error"
        assert row["provider_name"] == "openai"

    @pytest.mark.asyncio
    async def test_string_detail_404_error_code_null(self, db):
        app = build_stats_app()
        resp = await self._get(app, "/boom-string")
        assert resp.status_code == 404
        await drain_tasks()
        row = (await fetch_rows(db))[0]
        assert row["status_code"] == 404
        assert row["error_code"] is None
        assert row["error_message"] == "Not Found"

    @pytest.mark.asyncio
    async def test_422_bypasses_handler_error_code_null(self, db):
        app = build_stats_app()
        resp = await self._get(app, "/needs-int")
        assert resp.status_code == 422
        await drain_tasks()
        row = (await fetch_rows(db))[0]
        assert row["status_code"] == 422
        assert row["error_code"] is None

    @pytest.mark.asyncio
    async def test_midstream_sse_error_with_partial_usage(self, db):
        app = build_stats_app()
        resp = await self._get(app, "/stream")
        assert resp.status_code == 200
        await drain_tasks()
        row = (await fetch_rows(db))[0]
        # HTTP status stayed 200, but the row is marked as an error
        assert row["status_code"] == 200
        assert row["error_code"] == "internal_server_error"
        assert "upstream died" in row["error_message"]
        # Partial usage captured before the failure
        assert row["prompt_tokens"] == 10
        assert row["completion_tokens"] == 5
        assert row["has_usage"] == 1
        assert row["stream"] == 1


# ---------------------------------------------------------------------------
# Auth key hash enrichment
# ---------------------------------------------------------------------------

class TestAuthKeyHashEnrichment:

    def _auth_request(self, stats):
        from starlette.requests import Request as StarletteRequest

        app = FastAPI()
        app.state.config_manager = MagicMock()
        app.state.config_manager.get_config.return_value = parse_config({
            "user_keys": {"proj": {"api_key": "nnp-v1-real"}}
        })
        scope = {"type": "http", "method": "GET", "path": "/",
                 "headers": [], "app": app, "state": {}}
        request = StarletteRequest(scope)
        request.state.request_stats = stats
        return request

    @pytest.mark.asyncio
    async def test_invalid_key_writes_truncated_hash(self):
        from fastapi.security import HTTPAuthorizationCredentials

        from src.core.auth import get_api_key

        stats = RequestStats()
        request = self._auth_request(stats)
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="nnp-v1-wrong")
        with pytest.raises(HTTPException) as exc_info:
            await get_api_key(request, credentials)
        assert exc_info.value.status_code == 401
        expected = hashlib.sha256(b"nnp-v1-wrong").hexdigest()[:8]
        assert stats.api_key_hash == expected

    @pytest.mark.asyncio
    async def test_missing_key_leaves_hash_none(self):
        from src.core.auth import get_api_key

        stats = RequestStats()
        request = self._auth_request(stats)
        with pytest.raises(HTTPException):
            await get_api_key(request, None)
        assert stats.api_key_hash is None

    @pytest.mark.asyncio
    async def test_valid_key_leaves_hash_none(self):
        from fastapi.security import HTTPAuthorizationCredentials

        from src.core.auth import get_api_key

        stats = RequestStats()
        request = self._auth_request(stats)
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="nnp-v1-real")
        result = await get_api_key(request, credentials)
        # AuthContext carries grants; the resolved name lives on RequestContext.
        assert result.allowed_models == []
        assert result.allowed_endpoints == []
        assert request.state.request_context.project_name == "proj"
        assert stats.api_key_hash is None


# ---------------------------------------------------------------------------
# STAT_API_KEY guard on the /stat/api/* routes
# ---------------------------------------------------------------------------

class TestStatApiKeyGuard:

    def _app(self, stat_api_key: str) -> FastAPI:
        from src.api.main import custom_http_exception_handler
        from src.core.config_manager import Settings

        app = FastAPI()
        app.state.config_manager = SimpleNamespace(settings=Settings(stat_api_key=stat_api_key))
        app.add_exception_handler(HTTPException, custom_http_exception_handler)

        @app.get("/stat/api/thing")
        async def thing(_: None = Depends(verify_stat_key)):
            return {"ok": True}

        return app

    @pytest.mark.asyncio
    async def test_open_when_key_unset(self):
        transport = httpx.ASGITransport(app=self._app(""))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/stat/api/thing")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_missing_header_rejected_when_set(self):
        transport = httpx.ASGITransport(app=self._app("sekrit"))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/stat/api/thing")
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == 401

    @pytest.mark.asyncio
    async def test_wrong_header_rejected_when_set(self):
        transport = httpx.ASGITransport(app=self._app("sekrit"))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/stat/api/thing", headers={"X-Stat-Key": "wrong"})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_correct_header_accepted(self):
        transport = httpx.ASGITransport(app=self._app("sekrit"))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/stat/api/thing", headers={"X-Stat-Key": "sekrit"})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
