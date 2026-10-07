"""Provider registry with instance caching keyed by provider name."""
# SYSTEM: provider-registry — provider instances cached by name, drained on reload
import asyncio

from ..core.config_manager import Settings
from ..core.config_schema import ProviderEntry, RouterConfig
from ..core.error_handling import ErrorType, create_error
from .anthropic import AnthropicProvider
from .base import Provider

# ARCH: the factory is a {type: class} dict — a provider speaks exactly one
# protocol, and the dict's keys are exactly PROVIDER_TYPES (parse_provider
# validated `type` against it; tests/unit/test_provider_registry.py pins the
# two lists together).
PROVIDER_CLASSES: dict[str, type[Provider]] = {
    "openai": Provider,
    "anthropic": AnthropicProvider,
}


def _build_provider(
    provider_name: str,
    entry: ProviderEntry,
    settings: Settings,
) -> Provider:
    """Pure factory: return a new instance. No caching.

    The entry's `type` was validated by parse_config, so the lookup always
    resolves.
    """
    return PROVIDER_CLASSES[entry.type](entry, settings, provider_name=provider_name)


async def _gather_closes(coros) -> None:
    """Await all close coroutines, suppressing individual failures."""
    await asyncio.gather(*coros, return_exceptions=True)


# ARCH: cache key is the provider name (the dict key in providers.yaml).
# Each cached instance owns its own httpx pool. The cache is rebuilt in TWO
# phases on startup and config reload: prepare stages the next cache —
# REUSING every live instance whose providers.yaml entry is unchanged and
# building the rest (pre-swap, fail-fast) — and publish swaps it in and
# drains the superseded pools (post-swap). On prepare failure the old cache
# is retained. One registry is built in lifespan and handed to the services
# and the capabilities refresh; tests build their own.
class ProviderRegistry:
    """The published provider instances, the staged next generation, and their drain."""

    def __init__(self) -> None:
        self._published: dict[str, Provider] = {}
        # Staged by prepare, consumed by publish. None means nothing is
        # pending publication.
        self._staged: dict[str, Provider] | None = None
        # ARCH: serializes the two reload phases so a prepare cannot stage over a
        # publish that is mid-swap. Lookups do not take it — they are a plain dict read.
        self._lock = asyncio.Lock()
        # ARCH: background drain tasks are tracked here so they are not garbage-
        # collected before completion (asyncio holds only weak references to tasks).
        # Same pattern as _usage_tasks in src/core/usage_db/writer.py; each task
        # discards itself via a done callback. No failure logging here:
        # _gather_closes runs with return_exceptions=True, so the task never raises.
        self._drain_tasks: set[asyncio.Task] = set()

    # INVARIANT: the published cache is the ONLY source of provider instances —
    # a lookup miss is an error, never a lazy build.
    # Why: callers resolve the provider entry from the config and then await, so a
    # reload can publish between the two. A lazy build would take that caller's
    # stale entry and insert a provider the operator just deleted into
    # the freshly published cache, where it would serve traffic until the next
    # reload. prepare builds every configured provider up front
    # (startup and every reload), so a miss can only mean "not in the live
    # config" — which is exactly PROVIDER_NOT_FOUND.
    def get(self, provider_name: str) -> Provider:
        """Return the published provider instance for provider_name.

        Instances are cached by provider_name and built only by prepare. An
        entry left unchanged by a reload keeps its live instance (see the
        INVARIANT on _live_instance_if_unchanged); changed entries are picked
        up on the next rebuild.
        """
        cached = self._published.get(provider_name)
        if cached is None:
            raise create_error(
                ErrorType.PROVIDER_NOT_FOUND, provider_name=provider_name, model_id="unknown"
            )
        return cached

    def _live_instance_if_unchanged(
        self,
        provider_name: str,
        entry: ProviderEntry,
        settings: Settings,
    ) -> Provider | None:
        """Return the published instance when it can be carried into the staged
        cache as-is, else None.

        # INVARIANT: an unchanged provider entry is carried into the staged cache
        # as the SAME instance, never rebuilt.
        # Why: the instance owns its ProviderPool, and the pool owns the
        # semaphore. Rebuilding on an unrelated reload (all four watched files
        # trigger this callback, so a models.yaml edit qualifies) resets the
        # semaphore, letting the old pool's in-flight requests plus the new
        # pool's fresh slots exceed max_concurrent together for the drain window
        # (up to stream_read_timeout). Sameness is value equality on the
        # parsed ProviderEntry AND the same Settings object (frozen at construction,
        # never swapped without a restart — identity is enough). The name is
        # looked up in the NEW config's iteration, so a provider the operator
        # deleted is never carried over.
        """
        live = self._published.get(provider_name)
        if live is None:
            return None
        if live.settings is not settings or live.entry != entry:
            return None
        return live

    def _stale_stage_values(self, superseded: dict[str, Provider]) -> list[Provider]:
        """Values of a superseded stage that are NOT (by identity) in the
        published cache — the ones a failed/superseded prepare must close.

        # INVARIANT: a superseded stage is closed MINUS its reused instances.
        # Why: with instance reuse a staged cache can hold the very objects the
        # published cache holds; closing them on supersede would flip their
        # pool's _closed flag and permanently 503 a live provider (see
        # acquire_slot in pool.py).
        """
        live_ids = {id(inst) for inst in self._published.values()}
        return [inst for inst in superseded.values() if id(inst) not in live_ids]

    def _build_stage(
        self, config: RouterConfig, settings: Settings
    ) -> tuple[dict[str, Provider], list[Provider], list[str]]:
        """Stage one entry per configured provider: the live instance when the
        entry is unchanged (see _live_instance_if_unchanged), a fresh build
        otherwise. Returns (temp, fresh, errors); fresh lists only the instances
        THIS stage built — the ones a failed prepare must close.
        """
        temp: dict[str, Provider] = {}
        fresh: list[Provider] = []
        errors: list[str] = []
        for provider_name, entry in config.providers.items():
            reused = self._live_instance_if_unchanged(provider_name, entry, settings)
            if reused is not None:
                temp[provider_name] = reused
                continue
            try:
                built = _build_provider(provider_name, entry, settings)
            except Exception as e:
                errors.append(f"  - {provider_name}: {e}")
                continue
            temp[provider_name] = built
            fresh.append(built)
        return temp, fresh, errors

    async def prepare(self, config: RouterConfig, settings: Settings) -> None:
        """Build and stage the next provider cache (phase 1 of the reload).

        Stages a temp dict for every configured provider under the lock:
        unchanged entries carry their live instance, new and changed entries are
        built. If any provider fails to build, a RuntimeError listing all
        failures is raised and neither the live cache nor the staged slot is
        touched (the partially-built instances are closed so a failed prepare
        leaks nothing; reused live instances are NOT closed — they still serve
        traffic from the published cache). On success the temp dict is staged
        for publish — the live cache still serves traffic until the publish.
        A stage that was never published (a reload vetoed after this callback)
        is closed before being replaced, minus any instance the published
        cache still holds.
        """
        async with self._lock:
            superseded = self._staged
            self._staged = None
            temp, fresh, errors = self._build_stage(config, settings)
            if errors:
                # WHY: doomed holds only instances THIS prepare built, plus the
                # non-reused values of the superseded stage — a reused live
                # instance sitting in temp must survive the veto; it is still
                # serving traffic from the published cache. Both own open httpx
                # pools, so a failed prepare (e.g. one misconfigured provider)
                # leaks nothing.
                doomed = list(fresh)
                if superseded is not None:
                    doomed += self._stale_stage_values(superseded)
                await _gather_closes([inst.aclose() for inst in doomed])
                joined = "\n".join(errors)
                raise RuntimeError(
                    f"Provider validation failed; refusing to start:\n{joined}"
                )
            self._staged = temp
            if superseded is not None:
                # WHY: a previous prepare whose reload never reached publish (a
                # later pre-swap callback vetoed it) left open httpx pools staged;
                # replacing the stage without closing them leaks one pool per
                # provider per vetoed reload. Reused instances are excluded — the
                # published cache still owns them.
                await _gather_closes(
                    [inst.aclose() for inst in self._stale_stage_values(superseded)]
                )

    # INVARIANT: publish runs only AFTER ConfigManager has swapped
    # self.config, and stays the FIRST post-swap callback.
    # Why: config and cache are two views of the same providers, and whichever is
    # published second defines the window. Publishing the cache first would strand
    # a provider the new config still lists but the new cache no longer holds, and
    # every request resolving it in that window would get a spurious 404 lasting
    # until the swap. Publishing after the swap inverts the window to a provider
    # the new config has just ADDED, which is not yet in the cache — reachable only
    # if a coroutine runs between the swap and this callback, and reload_config
    # leaves no suspension point there. Lookups never build (see the INVARIANT
    # on get), so a deleted provider cannot be served from a rebuilt cache.
    async def publish(self) -> None:
        """Swap the staged cache in and drain the superseded pools (phase 2 of the reload).

        No-op when nothing is staged (e.g. a publish without a successful
        prepare). Instances the new cache REPLACED or dropped are closed in the
        background, bounded by their drain timeout, so live SSE streams finish
        intact.
        """
        async with self._lock:
            staged = self._staged
            if staged is None:
                return
            self._staged = None
            old_cache = self._published
            self._published = staged

        # INVARIANT: publish drains old-minus-published BY IDENTITY, never the
        # whole old cache.
        # Why: a reused instance is a value in BOTH dicts; closing it here would
        # set its pool's _closed flag under itself and every later request would
        # get a permanent 503 from acquire_slot (see pool.py) — the reuse in
        # prepare would have resurrected a dead pool. Only instances absent from
        # the published cache are superseded.
        published_ids = {id(inst) for inst in staged.values()}
        coros = [
            inst.aclose() for inst in old_cache.values()
            if id(inst) not in published_ids
        ]
        if coros:
            task = asyncio.ensure_future(_gather_closes(coros))
            self._drain_tasks.add(task)
            task.add_done_callback(self._drain_tasks.discard)

    async def aclose_all(self) -> None:
        """Await every provider's pool close (staged included, deduped by
        identity), then clear the cache.

        Used on shutdown so pools are drained gracefully before the loop stops.
        Close exceptions are suppressed via return_exceptions so one failing close
        never blocks the rest.
        """
        staged, self._staged = self._staged, None
        # WHY: dedupe by identity — with instance reuse the staged cache can hold
        # the very instances the published cache holds; each pool is closed once.
        seen: set[int] = set()
        coros = []
        for inst in (*self._published.values(), *(staged.values() if staged else ())):
            if id(inst) not in seen:
                seen.add(id(inst))
                coros.append(inst.aclose())
        self._published = {}
        await _gather_closes(coros)
