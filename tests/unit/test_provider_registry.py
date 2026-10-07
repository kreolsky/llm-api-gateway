"""Unit tests for src/providers/__init__.py — provider registry & cache."""

from unittest.mock import AsyncMock, patch

import pytest

from src.core.config_manager import Settings
from src.core.config_schema import PROVIDER_TYPES, ConfigError, parse_config, parse_provider
from src.providers import PROVIDER_CLASSES, ProviderRegistry, _build_provider
from src.services.base import PROTOCOL_BY_PROVIDER_TYPE


async def _prepare(registry: ProviderRegistry, config: dict, settings: Settings) -> None:
    """ProviderRegistry.prepare over a YAML-shaped dict, parsed like the loader does."""
    await registry.prepare(parse_config(config), settings)


@pytest.fixture
def registry():
    """A fresh registry per test — nothing staged, nothing published."""
    return ProviderRegistry()


def _make_config():
    return {"type": "openai", "base_url": "https://api.example.com", "api_key_env": "TEST_API_KEY"}


def _settings() -> Settings:
    return Settings()


def test_every_provider_type_has_a_class_and_a_protocol():
    """parse_provider accepts exactly PROVIDER_TYPES, so the factory and the
    protocol gate must cover exactly that set — a type with no class is a
    KeyError at build, a type with no protocol is a 400 on every request."""
    assert set(PROVIDER_CLASSES) == set(PROVIDER_TYPES)
    assert set(PROTOCOL_BY_PROVIDER_TYPE) == set(PROVIDER_TYPES)


class TestBuildProviderName:
    """The factory passes the providers.yaml key down as provider_name."""

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-123"}, clear=False)
    def test_instance_gets_real_provider_name(self):
        """_build_provider("glm", ...) → provider_name "glm": logs and startup
        errors name the actual backend, not the shared type literal."""
        instance = _build_provider("glm", parse_provider(_make_config()), Settings())
        assert instance.provider_name == "glm"


class TestGetProviderInstance:
    """A lookup, never a build: the published cache is the only source."""

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-123"}, clear=False)
    @pytest.mark.asyncio
    async def test_returns_the_published_instance(self, registry):
        """Every configured provider resolves to the instance publish put in
        the cache, and repeated lookups return that same object."""
        config = {"providers": {"alpha": _make_config(), "beta": _make_config()}}
        await _seed_cache(registry, config, Settings())

        for name in config["providers"]:
            first = registry.get(name)
            assert first is registry._published[name]
            assert first is registry.get(name)

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-123"}, clear=False)
    @pytest.mark.asyncio
    async def test_miss_raises_provider_not_found_and_builds_nothing(self, registry):
        """The principal that must be REFUSED: a name absent from the published
        cache. A lazy build here is what used to resurrect a provider the
        operator had just deleted (see the INVARIANT on ProviderRegistry.get)."""
        from fastapi import HTTPException
        await _seed_cache(registry, {"providers": {"alpha": _make_config()}}, Settings())

        with pytest.raises(HTTPException) as exc_info:
            registry.get("gone")
        assert exc_info.value.status_code == 404
        assert "gone" not in registry._published

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-123"}, clear=False)
    @pytest.mark.asyncio
    async def test_unknown_type_is_refused_before_prepare(self, registry):
        """An unknown provider type is refused by parse_config, so it never
        reaches the cache — the lookup is not where the type is validated."""
        with pytest.raises(ConfigError) as exc_info:
            await _prepare(registry, 
                {"providers": {"alpha": {"type": "unknown", "base_url": "https://x.example.com"}}},
                Settings(),
            )
        assert "alpha" in str(exc_info.value)
        assert registry._staged is None


class TestAcloseAll:

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-123"}, clear=False)
    @pytest.mark.asyncio
    async def test_async_close_awaits_aclose_before_clear(self, registry):
        """aclose_all awaits every aclose then clears the cache."""
        await _seed_cache(registry, {"providers": {"alpha": _make_config()}}, Settings())
        inst = registry._published["alpha"]
        inst.aclose = AsyncMock()
        await registry.aclose_all()
        inst.aclose.assert_awaited_once()
        assert registry._published == {}

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-123"}, clear=False)
    @pytest.mark.asyncio
    async def test_async_close_suppresses_one_failure(self, registry):
        """A failing aclose does not block the rest from closing."""
        await _seed_cache(registry, {"providers": {"bad": _make_config(), "good": _make_config()}}, Settings())
        bad = registry._published["bad"]
        good = registry._published["good"]
        bad.aclose = AsyncMock(side_effect=RuntimeError("boom"))
        good.aclose = AsyncMock()
        await registry.aclose_all()  # must not raise
        bad.aclose.assert_awaited_once()
        good.aclose.assert_awaited_once()
        assert registry._published == {}


async def _seed_cache(registry, config, settings):
    """prepare + publish back to back — what startup and a full reload do."""
    await _prepare(registry, config, settings)
    await registry.publish()


class TestPreparePublishProviderCache:
    """Two-phase cache rebuild: prepare stages pre-swap, publish swaps post-swap."""

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-123"}, clear=False)
    @pytest.mark.asyncio
    async def test_prepare_stages_without_swapping(self, registry):
        """prepare builds and stages the new instances; the live cache is
        untouched until publish runs."""
        config = {"providers": {"ok": _make_config()}}
        await _seed_cache(registry, config, Settings())
        old_instance = registry._published["ok"]

        await _prepare(registry, {"providers": {"renamed": _make_config()}}, Settings())

        assert "renamed" in registry._staged
        assert "renamed" not in registry._published
        assert registry._published["ok"] is old_instance

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-123"}, clear=False)
    @pytest.mark.asyncio
    async def test_publish_swaps_staged_and_drains_old(self, registry):
        """publish swaps the staged cache in and closes the previous pools in
        the background (drain task tracked until done)."""
        import asyncio
        await _seed_cache(registry, {"providers": {"ok": _make_config()}}, Settings())
        old_instance = registry._published["ok"]
        old_instance.aclose = AsyncMock()

        await _prepare(registry, {"providers": {"ok": _make_config()}}, Settings())
        await registry.publish()

        assert registry._published["ok"] is not old_instance
        assert registry._staged is None
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        old_instance.aclose.assert_awaited_once()

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-123"}, clear=False)
    @pytest.mark.asyncio
    async def test_prepare_failure_retains_cache_and_lists_all(self, registry):
        """On failure the live cache is retained, nothing stays staged, and all
        failures are reported (collect-all fail-fast)."""
        await _seed_cache(registry, {"providers": {"ok": _make_config()}}, Settings())
        old_instance = registry._published["ok"]

        bad = {
            "broken-a": {"type": "openai", "base_url": "https://a.example.com", "api_key_env": "MISSING_A"},
            "broken-b": {"type": "openai", "base_url": "https://b.example.com", "api_key_env": "MISSING_B"},
        }
        with patch.dict("os.environ", {}, clear=True), pytest.raises(RuntimeError) as exc_info:
            await _prepare(registry, {"providers": bad}, Settings())
        msg = str(exc_info.value)
        assert "broken-a" in msg
        assert "broken-b" in msg
        assert registry._published["ok"] is old_instance
        assert registry._staged is None

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-123"}, clear=False)
    @pytest.mark.asyncio
    async def test_publish_without_prepare_is_noop(self, registry):
        """publish with nothing staged leaves the live cache alone."""
        await _seed_cache(registry, {"providers": {"ok": _make_config()}}, Settings())
        instance = registry._published["ok"]
        await registry.publish()
        assert registry._published["ok"] is instance

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-123"}, clear=False)
    @pytest.mark.asyncio
    async def test_removed_provider_cannot_be_resurrected_around_publish(self, registry):
        """A request that resolved its provider_config before the reload cannot
        put the removed provider back — on either side of the publish. Before,
        the instance it built landed in the old cache and went out with it;
        now the lookup refuses outright."""
        from fastapi import HTTPException
        await _seed_cache(registry, 
            {"providers": {"kept": _make_config(), "doomed": _make_config()}}, Settings())

        # New config drops "doomed"; prepare stages the shrunken cache.
        await _prepare(registry, {"providers": {"kept": _make_config()}}, Settings())

        # Mid-reload, the stale resolution still finds the live (old) instance.
        assert registry.get("doomed") is registry._published["doomed"]

        await registry.publish()

        # Post-publish, the same stale resolution is refused instead of rebuilt.
        with pytest.raises(HTTPException):
            registry.get("doomed")
        assert "doomed" not in registry._published
        assert "kept" in registry._published

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-123"}, clear=False)
    @pytest.mark.asyncio
    async def test_superseded_stage_is_closed_not_leaked(self, registry):
        """A stage whose reload never reached publish is closed when the next
        prepare replaces it — one httpx pool per provider would leak otherwise."""
        await _seed_cache(registry, {"providers": {"ok": _make_config()}}, Settings())

        await _prepare(registry, {"providers": {"vetoed": _make_config()}}, Settings())
        stranded = registry._staged["vetoed"]
        stranded.aclose = AsyncMock()

        await _prepare(registry, {"providers": {"next": _make_config()}}, Settings())

        stranded.aclose.assert_awaited_once()
        assert set(registry._staged) == {"next"}

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-123"}, clear=False)
    @pytest.mark.asyncio
    async def test_publish_drain_task_tracked_until_done(self, registry):
        """The background close task is referenced until it finishes.

        asyncio only keeps a weak reference to a running task: an unreferenced
        ensure_future result can be garbage-collected mid-flight, dropping the
        pool closes it was carrying. Mirrors the _usage_tasks pattern in
        src/core/usage_db/writer.py.
        """
        import asyncio
        await _seed_cache(registry, {"providers": {"ok": _make_config()}}, Settings())
        old_instance = registry._published["ok"]

        release = asyncio.Event()

        async def slow_aclose(drain_timeout=None):
            await release.wait()

        old_instance.aclose = slow_aclose

        await _prepare(registry, {"providers": {"ok": _make_config()}}, Settings())
        await registry.publish()
        assert registry._drain_tasks, "drain task must be tracked while pools close"
        tracked = next(iter(registry._drain_tasks))
        assert not tracked.done()

        release.set()
        await asyncio.wait_for(tracked, timeout=2)
        await asyncio.sleep(0)
        assert not registry._drain_tasks, "done drain task must discard itself"


class TestInstanceReuse:
    """An unchanged provider entry carries the LIVE instance into the staged
    cache instead of building a new one (see the INVARIANT in
    ProviderRegistry.prepare)."""

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-123"}, clear=False)
    @pytest.mark.asyncio
    async def test_unchanged_config_reuses_instance_and_pool(self, registry):
        """A reload whose providers.yaml entries are equal (fresh dicts, same
        values — e.g. triggered by a models.yaml edit) keeps the SAME
        instances, and the reused pools are NOT closed after publish."""
        import asyncio
        settings = Settings()
        await _seed_cache(registry, 
            {"providers": {"alpha": _make_config(), "beta": _make_config()}}, settings)
        old_alpha = registry._published["alpha"]
        old_beta = registry._published["beta"]

        await _seed_cache(registry, 
            {"providers": {"alpha": _make_config(), "beta": _make_config()}}, settings)

        assert registry._published["alpha"] is old_alpha
        assert registry._published["beta"] is old_beta
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert old_alpha.pool._closed is False
        assert old_beta.pool._closed is False
        assert not old_alpha.pool.client.is_closed

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-123"}, clear=False)
    @pytest.mark.asyncio
    async def test_changed_entry_rebuilds_only_that_provider(self, registry):
        """A changed entry gets a fresh instance (old one drained); the
        untouched sibling is carried over as the same object."""
        import asyncio
        settings = Settings()
        await _seed_cache(registry, 
            {"providers": {"alpha": _make_config(), "beta": _make_config()}}, settings)
        old_alpha = registry._published["alpha"]
        old_beta = registry._published["beta"]
        old_alpha.aclose = AsyncMock()

        changed = {"type": "openai", "base_url": "https://changed.example.com",
                   "api_key_env": "TEST_API_KEY"}
        await _seed_cache(registry, 
            {"providers": {"alpha": changed, "beta": _make_config()}}, settings)

        assert registry._published["alpha"] is not old_alpha
        assert registry._published["beta"] is old_beta
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        old_alpha.aclose.assert_awaited_once()

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-123"}, clear=False)
    @pytest.mark.asyncio
    async def test_removed_provider_is_drained_and_404s(self, registry):
        """A name dropped from the new config is closed and its lookup 404s —
        reuse is keyed on the name being present in the NEW config, so it can
        never resurrect a deleted provider."""
        import asyncio

        from fastapi import HTTPException
        await _seed_cache(registry, 
            {"providers": {"kept": _make_config(), "doomed": _make_config()}}, Settings())
        doomed = registry._published["doomed"]
        doomed.aclose = AsyncMock()

        await _seed_cache(registry, {"providers": {"kept": _make_config()}}, Settings())

        await asyncio.sleep(0)
        await asyncio.sleep(0)
        doomed.aclose.assert_awaited_once()
        with pytest.raises(HTTPException) as exc_info:
            registry.get("doomed")
        assert exc_info.value.status_code == 404

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-123"}, clear=False)
    @pytest.mark.asyncio
    async def test_added_provider_is_built_and_served(self, registry):
        """A name present only in the new config is built and immediately
        resolvable after publish."""
        await _seed_cache(registry, {"providers": {"alpha": _make_config()}}, Settings())

        await _seed_cache(registry, 
            {"providers": {"alpha": _make_config(), "beta": _make_config()}}, Settings())

        assert registry.get("beta") is registry._published["beta"]

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-123"}, clear=False)
    @pytest.mark.asyncio
    async def test_different_settings_object_forces_rebuild(self, registry):
        """Sameness requires the SAME Settings object, not an equal one:
        settings are frozen per process, so a different object means a new
        process configuration and every provider must be rebuilt."""
        settings = Settings()
        await _seed_cache(registry, {"providers": {"alpha": _make_config()}}, settings)
        old_alpha = registry._published["alpha"]

        await _seed_cache(registry, {"providers": {"alpha": _make_config()}}, Settings())

        assert registry._published["alpha"] is not old_alpha

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-123"}, clear=False)
    @pytest.mark.asyncio
    async def test_failed_prepare_keeps_reused_live_instance_open(self, registry):
        """A vetoed reload must not close the live instances it reused into
        the failed temp cache — they are still serving traffic from the
        published cache. Also proves reuse ran: 'good' survives with the env
        var stripped, because it was never rebuilt."""
        settings = Settings()
        await _seed_cache(registry, 
            {"providers": {"good": _make_config(), "ok": _make_config()}}, settings)
        good = registry._published["good"]

        broken = {"type": "openai", "base_url": "https://broken.example.com",
                  "api_key_env": "MISSING_KEY"}
        with patch.dict("os.environ", {}, clear=True), pytest.raises(RuntimeError) as exc_info:
            await _prepare(registry, 
                {"providers": {"good": _make_config(), "broken": broken}}, settings)

        assert "broken" in str(exc_info.value)
        assert "good" not in str(exc_info.value)
        assert good.pool._closed is False
        assert registry._published["good"] is good
        assert registry._staged is None

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-123"}, clear=False)
    @pytest.mark.asyncio
    async def test_superseded_stage_does_not_close_reused_live_instance(self, registry):
        """A stage holding REUSED instances (a reload that never reached
        publish) is closed on supersede minus those instances — the live
        cache still owns them."""
        settings = Settings()
        await _seed_cache(registry, {"providers": {"alpha": _make_config()}}, settings)
        live_alpha = registry._published["alpha"]

        # Stage 1 reuses the live alpha; the reload is vetoed before publish.
        await _prepare(registry, {"providers": {"alpha": _make_config()}}, settings)
        assert registry._staged["alpha"] is live_alpha

        # Stage 2 supersedes stage 1 with a changed entry.
        changed = {"type": "openai", "base_url": "https://changed.example.com",
                   "api_key_env": "TEST_API_KEY"}
        await _prepare(registry, {"providers": {"alpha": changed}}, settings)

        assert live_alpha.pool._closed is False
        assert registry._staged["alpha"] is not live_alpha
