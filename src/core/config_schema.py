"""Typed router config: frozen entries parsed ONCE from the loaded YAML.

ARCH: ``parse_config`` is the single place a raw YAML dict becomes the typed
``RouterConfig`` every reader consumes — startup and every hot reload run it,
so the request path never re-validates and never sees a malformed entry.
Two validation classes, matching what a hot reload may do to a running router:

  soft — warn, never veto a reload: the per-model ``reasoning_effort`` block,
      ``model_info`` entries/keys and non-numeric pricing values (dropped),
      and dangling cross-file references — a model whose ``provider`` is not
      a providers.yaml key, a key whose ``allowed_models`` names an unknown
      model (kept; the request path answers them);
  hard — ``ConfigError`` listing every bad entry: startup refuses to start, a
      reload keeps the previous config: provider ``type``, ``identity``,
      ``reasoning_dialect``, static ``headers:``, ``max_concurrent``, and
      non-mapping entries.

Env-dependent checks (``base_url``, the ``api_key_env`` variable) stay at
provider construction — they belong to the process, not to the YAML.
"""
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .header_policy import FORBIDDEN_STATIC_HEADERS
from .logging import logger


class ConfigError(ValueError):
    """A config the router refuses to run: one message per bad entry."""


# ---------------------------------------------------------------------------
# reasoning_effort policy (models.yaml)
# ---------------------------------------------------------------------------

# Wire locations a policy may name; the first is the OpenAI dialect and the
# default. The second is the OpenRouter-style reasoning.effort nesting.
EFFORT_PARAMS = ("reasoning_effort", "reasoning.effort")
DEFAULT_EFFORT_PARAM = "reasoning_effort"
# Keys a reasoning_effort block may carry; anything else is a typo and drops the
# whole block (see parse_effort_policy).
EFFORT_BLOCK_KEYS = frozenset({"allowed", "default", "param"})


def read_effort_param(body: Mapping[str, Any], param: str) -> Any:
    """Read a wire location out of a body-shaped dict; None when absent.

    One reader for both dialects, so every consumer (the client-value gate, the
    options-conflict guard) covers the same set — EFFORT_PARAMS is the only list.
    """
    if param == "reasoning_effort":
        return body.get("reasoning_effort")
    reasoning = body.get("reasoning")
    return reasoning.get("effort") if isinstance(reasoning, dict) else None


def parse_effort_policy(model_config: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Validate a model's reasoning_effort block.

    Returns ``(policy, None)`` for a valid block or no block at all, and
    ``(None, reason)`` for a malformed one. ``(None, None)`` means the key is
    absent: no policy, not an error.
    """
    if not isinstance(model_config, dict) or "reasoning_effort" not in model_config:
        return None, None
    options = model_config.get("options")
    if isinstance(options, dict):
        # INVARIANT: BOTH dialects are checked, not just the OpenAI one.
        # Why: options merges into the outgoing body wholesale, so an
        # options.reasoning.effort next to an injected reasoning_effort ships
        # the double declaration this guard exists to prevent.
        conflicts = [p for p in EFFORT_PARAMS if read_effort_param(options, p) is not None]
        if conflicts:
            return None, f"'options.{conflicts[0]}' is also set (options wins at merge time)"
    block = model_config["reasoning_effort"]
    if not isinstance(block, dict):
        return None, "reasoning_effort is not a mapping"
    unknown = sorted(set(block) - EFFORT_BLOCK_KEYS)
    if unknown:
        return None, f"unknown key(s) {unknown}; allowed: {sorted(EFFORT_BLOCK_KEYS)}"
    allowed = block.get("allowed")
    if (not isinstance(allowed, list) or not allowed
            or not all(isinstance(v, str) for v in allowed)):
        return None, "'allowed' must be a non-empty list of strings"
    default = block.get("default")
    if default is not None and (not isinstance(default, str) or default not in allowed):
        return None, f"'default' ({default!r}) must be one of {allowed}"
    param = block.get("param", DEFAULT_EFFORT_PARAM)
    if param not in EFFORT_PARAMS:
        return None, f"'param' ({param!r}) must be one of: {', '.join(EFFORT_PARAMS)}"
    return {"allowed": list(allowed), "default": default, "param": param}, None


# ---------------------------------------------------------------------------
# reasoning_dialect (providers.yaml)
# ---------------------------------------------------------------------------

# Wire dialects a provider entry may name; the first is the default because a
# third-party OpenAI-compat gateway must work with no config change.
REASONING_DIALECTS = ("openai", "deepseek", "openrouter")
DEFAULT_REASONING_DIALECT = "openai"

# ARCH: one protocol per provider type — a model's protocol is its provider's
# type (`openai` -> /v1/chat/completions, `anthropic` -> /v1/messages), so a
# dual-protocol vendor gets two provider entries, one per protocol. The
# factory mapping type -> class lives in providers/__init__.py and must cover
# exactly this tuple.
PROVIDER_TYPES = ("openai", "anthropic")


def validate_reasoning_dialect(value: Any) -> None:
    """Raise ConfigError unless ``value`` names a known dialect, so a typo can
    never reach the funnel as a silent ``openai``."""
    if value not in REASONING_DIALECTS:
        raise ConfigError(
            f"Unknown reasoning_dialect: {value!r} "
            f"(expected one of: {', '.join(REASONING_DIALECTS)})."
        )


def _validate_static_headers(headers: Mapping[Any, Any]) -> None:
    """Static `headers:` validation — only operator-authored entries are
    checked (the provider adds its code-owned Content-Type default after)."""
    for name, value in headers.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(value, str):
            raise ConfigError(
                f"headers entries must be 'name: value' strings, got {name!r}: {value!r}.")
        if name.lower() in FORBIDDEN_STATIC_HEADERS:
            raise ConfigError(
                f"headers may not set {name!r} (credential headers come from api_key_env; "
                f"transport/hop-by-hop headers are owned by the router).")


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EffortPolicy:
    allowed: tuple[str, ...]
    default: str | None
    param: str


@dataclass(frozen=True)
class ProviderEntry:
    """One providers.yaml entry. Compared by value: the registry reuses a live
    instance whose entry is unchanged across a reload."""
    type: str
    base_url: str | None = None
    api_key_env: str | None = None
    headers: Mapping[str, str] | None = None
    proxy: str | None = None
    # ARCH: identity is opt-in. `passthrough`: every client header goes
    # upstream verbatim minus the denylist (core/header_policy.py),
    # assembled per request by the service layer via extra_headers.
    # Unset: no forwarding — plain gateway headers only.
    identity: str | None = None
    reasoning_dialect: str = DEFAULT_REASONING_DIALECT
    max_concurrent: int | None = None


@dataclass(frozen=True)
class ModelEntry:
    provider: str | None
    provider_model_name: str | None = None
    options: dict[str, Any] | None = None
    params: Any = None
    is_hidden: bool = False
    effort_policy: EffortPolicy | None = None


@dataclass(frozen=True)
class KeyEntry:
    api_key: str = ""
    allowed_models: tuple[str, ...] = ()
    allowed_endpoints: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouterConfig:
    providers: Mapping[str, ProviderEntry]
    models: Mapping[str, ModelEntry]
    user_keys: Mapping[str, KeyEntry]
    # Already the stored capability shape (see core/model_capabilities).
    model_info: Mapping[str, dict[str, Any]]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _require_mapping(raw: Any, what: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigError(f"{what} is not a mapping.")
    return raw


def parse_provider(raw: Any) -> ProviderEntry:
    """Parse one providers.yaml entry; ConfigError on the first hard problem."""
    raw = _require_mapping(raw, "entry")
    provider_type = raw.get("type")
    if provider_type not in PROVIDER_TYPES:
        raise ConfigError(
            f"Unknown provider type: {provider_type!r} (expected one of: {', '.join(PROVIDER_TYPES)}).")
    identity = raw.get("identity")
    if identity not in (None, "passthrough"):
        raise ConfigError(f"Unknown identity profile: {identity!r} (expected 'passthrough').")
    dialect = raw.get("reasoning_dialect")
    if dialect is not None:
        validate_reasoning_dialect(dialect)
    headers = raw.get("headers") or {}
    _validate_static_headers(_require_mapping(headers, "headers"))
    max_concurrent = raw.get("max_concurrent")
    # bool is an int subclass: `true` would pass isinstance and build a
    # Semaphore(1), and a quoted "3" silently disables the gate — neither
    # typo is auto-correctable, so the value refuses to parse.
    if max_concurrent is not None and (isinstance(max_concurrent, bool)
                                       or not isinstance(max_concurrent, int)
                                       or max_concurrent <= 0):
        raise ConfigError(
            f"max_concurrent must be a positive int, got {max_concurrent!r}.")
    return ProviderEntry(
        type=provider_type,
        base_url=raw.get("base_url"),
        api_key_env=raw.get("api_key_env"),
        headers=dict(headers) or None,
        proxy=raw.get("proxy"),
        identity=identity,
        reasoning_dialect=dialect if dialect is not None else DEFAULT_REASONING_DIALECT,
        max_concurrent=max_concurrent,
    )


def parse_model(model_id: str, raw: Any) -> ModelEntry:
    """Parse one models.yaml entry. An invalid reasoning_effort block is SOFT.

    Why soft: models.yaml is hot-reloaded every 5s; a hard raise there
    kills a running router on a typo. Dropping (not just warning) makes
    "ignore the whole block" real for every downstream consumer — the
    funnel and /v1/models never see a malformed block. A model carrying
    BOTH an options effort key and a reasoning_effort block gets the same
    verdict: options wins at merge time (providers/base.py
    _apply_model_config), so the block would advertise a policy the
    upstream never sees. A typo'd key drops the block too — a misspelled
    ``param`` would otherwise send the effort to the wrong wire location
    while /v1/models still advertised the policy.
    """
    raw = _require_mapping(raw, "entry")
    policy, reason = parse_effort_policy(raw)
    if reason is not None:
        logger.warning(
            f"models.yaml '{model_id}': ignoring reasoning_effort block — {reason}",
            extra={"config": {"model": model_id}},
        )
    return ModelEntry(
        provider=raw.get("provider"),
        provider_model_name=raw.get("provider_model_name"),
        options=raw.get("options"),
        params=raw.get("params"),
        is_hidden=bool(raw.get("is_hidden", False)),
        effort_policy=None if policy is None else EffortPolicy(
            allowed=tuple(policy["allowed"]), default=policy["default"], param=policy["param"]),
    )


def _parse_key(raw: Any) -> KeyEntry:
    raw = _require_mapping(raw, "entry")
    return KeyEntry(
        api_key=raw.get("api_key") or "",
        allowed_models=tuple(raw.get("allowed_models") or ()),
        allowed_endpoints=tuple(raw.get("allowed_endpoints") or ()),
    )


# Allowed top-level keys per model_info entry (normalized schema; see
# config/model_info.yaml header). Used by the soft validation below.
_MODEL_INFO_KEYS = {
    "name", "description", "context_length", "max_completion_tokens",
    "is_moderated", "architecture", "supported_parameters", "reasoning", "pricing",
    "compat",
}
_MODEL_INFO_ARCH_KEYS = {
    "input_modalities", "output_modalities", "tokenizer", "instruct_type",
}


def _parse_pricing(model_id: str, pricing: Any) -> dict[str, float] | None:
    """Normalize one model_info pricing block to floats.

    PyYAML loads ``1e-7`` (exponent, no dot) as the STRING ``"1e-7"`` while
    the dotted form is a float — and the usage writer multiplies prices with
    token counts, so a str price would record string repetition as ``cost_usd``
    or lose the whole row to a TypeError. int, float and numeric strings
    become floats; a bool, unparsable, non-finite or negative value is
    dropped with a warning naming the model and key. Returns None for a
    non-mapping ``pricing`` so the caller drops the whole block.
    """
    if not isinstance(pricing, dict):
        logger.warning(
            f"model_info entry '{model_id}' has a non-mapping pricing, ignoring",
            extra={"config": {"model_info_key": model_id}},
        )
        return None
    parsed: dict[str, float] = {}
    for key, value in pricing.items():
        try:
            # bool is an int subclass: float(True) == 1.0 would silently
            # price a typo'd `prompt: true` as a real rate.
            price = None if isinstance(value, bool) else float(value)
        except (TypeError, ValueError):
            price = None
        # float() also accepts "nan"/"inf"; neither, nor a negative rate, is a price.
        if price is None or not math.isfinite(price) or price < 0:
            logger.warning(
                f"model_info entry '{model_id}' pricing key '{key}' has a "
                f"invalid value {value!r} (expected a finite, non-negative number), ignoring it",
                extra={"config": {"model_info_key": model_id, "pricing_key": key}},
            )
            continue
        parsed[key] = price
    return parsed


def _parse_compat(model_id: str, compat: Any) -> dict[str, bool | int | str] | None:
    """Normalize one model_info compat block: string keys -> scalar values.

    ``compat`` carries pi-ai compatibility flags (/v1/models renders it as
    stored for anthropic-type models). Manual-only and hand-authored, so it is
    soft-validated like the other model_info keys: a non-mapping drops the
    whole block; a non-string key or a non-scalar value (list/dict/None) drops
    just that entry — each with a warning naming the model. Returns None for a
    non-mapping so the caller drops the whole key.
    """
    if not isinstance(compat, dict):
        logger.warning(
            f"model_info entry '{model_id}' has a non-mapping compat, ignoring",
            extra={"config": {"model_info_key": model_id}},
        )
        return None
    parsed: dict[str, bool | int | str] = {}
    for key, value in compat.items():
        if not isinstance(key, str) or not isinstance(value, (bool, int, str)):
            logger.warning(
                f"model_info entry '{model_id}' compat key {key!r} has a non-scalar "
                f"value {value!r} (expected bool/str/int), ignoring it",
                extra={"config": {"model_info_key": model_id, "compat_key": key}},
            )
            continue
        parsed[key] = value
    return parsed


def _normalize_model_info_entry(model_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Normalize one KEPT model_info entry: pricing to floats, compat to
    scalars. Returns a COPY whenever anything changed — the raw dict belongs
    to the caller and is never mutated.
    """
    if "pricing" in entry:
        parsed = _parse_pricing(model_id, entry["pricing"])
        entry = dict(entry)
        if parsed is None:
            entry.pop("pricing")
        else:
            entry["pricing"] = parsed
    if "compat" in entry:
        parsed_compat = _parse_compat(model_id, entry["compat"])
        entry = dict(entry)
        if parsed_compat:
            entry["compat"] = parsed_compat
        else:
            # None (non-mapping, already warned) or {} (every value was
            # dropped with its own warning) — either way nothing to render.
            entry.pop("compat")
    return entry


def _parse_model_info(model_info: Any, models: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Soft-validate model_info: warn on unknown keys and orphan entries.

    Non-fatal (model_info is optional). Warns when an entry has no
    matching model in models.yaml, or uses keys outside the normalized
    schema — both indicate a stale or mistyped catalog. A non-mapping entry
    is dropped. Pricing values are normalized to floats via _parse_pricing
    (PyYAML can hand a numeric string where a number is meant).
    """
    if not isinstance(model_info, dict):
        return {}
    kept: dict[str, dict[str, Any]] = {}
    for model_id, entry in model_info.items():
        if not isinstance(entry, dict):
            logger.warning(
                f"model_info entry '{model_id}' is not a mapping, ignoring",
                extra={"config": {"model_info_key": model_id}},
            )
            continue
        kept[model_id] = entry
        if model_id not in models:
            logger.warning(
                f"model_info entry '{model_id}' has no matching model in models.yaml",
                extra={"config": {"model_info_key": model_id}},
            )
        unknown = set(entry) - _MODEL_INFO_KEYS
        if unknown:
            logger.warning(
                f"model_info entry '{model_id}' has unknown keys: {sorted(unknown)}",
                extra={"config": {"model_info_key": model_id, "unknown_keys": sorted(unknown)}},
            )
        arch = entry.get("architecture")
        if isinstance(arch, dict):
            arch_unknown = set(arch) - _MODEL_INFO_ARCH_KEYS
            if arch_unknown:
                logger.warning(
                    f"model_info entry '{model_id}'.architecture has unknown keys: {sorted(arch_unknown)}",
                    extra={"config": {"model_info_key": model_id, "unknown_keys": sorted(arch_unknown)}},
                )
        kept[model_id] = _normalize_model_info_entry(model_id, entry)
    return kept


def _parse_section(section: str, raw: Any, parse_one, errors: list[str]) -> dict[str, Any]:
    """Parse every entry of one section, collecting hard errors instead of
    stopping at the first — an operator fixes them all in one pass."""
    parsed: dict[str, Any] = {}
    for name, entry in (raw or {}).items():
        try:
            parsed[name] = parse_one(name, entry)
        except ConfigError as e:
            errors.append(f"  - {section}.{name}: {e}")
    return parsed


def _warn_dangling_references(
        providers: Mapping[str, ProviderEntry],
        models: Mapping[str, ModelEntry],
        user_keys: Mapping[str, KeyEntry]) -> None:
    """Soft cross-file check: warn when models.yaml or user_keys.yaml name an
    entry another file does not have (the model_info orphan warning's class —
    nothing dropped, no veto; the request path answers the dangling reference).

    WHY: a section's references are checked only when that section is
    non-empty — an empty providers/models section never reaches parse_config in
    production (startup's _assert_config_complete and the reload's
    _missing_sections reject it first), so warning per entry there would
    only re-describe the missing section — every model/key would dangle.
    """
    if providers:
        for model_id, entry in models.items():
            if entry.provider is not None and entry.provider not in providers:
                logger.warning(
                    f"models.yaml '{model_id}' references unknown provider "
                    f"'{entry.provider}' (not a key of providers.yaml)",
                    extra={"config": {"model": model_id, "provider": entry.provider}},
                )
    if models:
        for key_id, entry in user_keys.items():
            unknown = [m for m in entry.allowed_models if m not in models]
            if unknown:
                logger.warning(
                    f"user_keys.yaml '{key_id}' allows unknown model(s): "
                    f"{', '.join(unknown)} (not in models.yaml)",
                    extra={"config": {"user_key": key_id, "unknown_models": unknown}},
                )


def parse_config(raw: Mapping[str, Any]) -> RouterConfig:
    """Turn the loaded YAML sections into a RouterConfig.

    Raises ConfigError listing every hard problem (see the module docstring);
    soft problems are logged and the offending piece dropped.
    """
    errors: list[str] = []
    for section in ("providers", "models", "user_keys"):
        if not isinstance(raw.get(section) or {}, dict):
            errors.append(f"  - {section}: section is not a mapping")
    if errors:
        raise ConfigError("Invalid configuration:\n" + "\n".join(errors))
    providers = _parse_section("providers", raw.get("providers"),
                               lambda _name, entry: parse_provider(entry), errors)
    models = _parse_section("models", raw.get("models"), parse_model, errors)
    user_keys = _parse_section("user_keys", raw.get("user_keys"),
                               lambda _name, entry: _parse_key(entry), errors)
    if errors:
        raise ConfigError("Invalid configuration:\n" + "\n".join(errors))
    _warn_dangling_references(providers, models, user_keys)
    return RouterConfig(
        providers=providers,
        models=models,
        user_keys=user_keys,
        model_info=_parse_model_info(raw.get("model_info"), models),
    )
