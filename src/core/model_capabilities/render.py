"""Serialization and merge of the STORED capability form.

The stored schema is the single normalized shape both capability layers share
(manual model_info.yaml and the auto-cache). This module owns stored ->
response rendering and the deep merge where the manual layer wins (the
precedence INVARIANT lives on the package docstring).
"""
from decimal import Decimal, InvalidOperation

# Full set of OpenRouter pricing keys emitted in responses; missing -> "0".
# Public: normalizers.py reads it across the module boundary to decide which
# pricing keys a normalized upstream entry carries.
PRICING_KEYS = (
    "prompt",
    "completion",
    "request",
    "image",
    "web_search",
    "internal_reasoning",
    "input_cache_read",
    "input_cache_write",
)


def _format_price(value) -> str:
    """Format a per-token price as a plain decimal string, never exponential.

    e.g. 4.35e-07 -> "0.000000435", 0 -> "0", 0.001 -> "0.001".
    # WHY: a raw float like 4.35e-07 serializes to "4.35e-07" in JSON, which
    # breaks clients that parse the price with a non-exponent float parser.
    """
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return "0"
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def _render_pricing(pricing: dict) -> dict[str, str]:
    """Render a stored pricing dict (numbers) into the full OpenRouter key set (strings)."""
    result: dict[str, str] = {}
    for key in PRICING_KEYS:
        result[key] = _format_price(pricing.get(key, 0))
    return result


def _build_modality(input_modalities, output_modalities) -> str:
    """Build the OpenRouter-style modality string, e.g. 'text+image->text'."""
    inp = "+".join(input_modalities) if input_modalities else ""
    out = "+".join(output_modalities) if output_modalities else ""
    if inp and out:
        return f"{inp}->{out}"
    return inp or out or ""


def render_capabilities(stored: dict) -> dict:
    """Serialize the STORED capability form into the response shape.

    Single source of truth for all derived fields:
    ``architecture.modality``, ``supports_vision``, ``top_provider`` (mirrors
    context_length / max_completion_tokens / is_moderated), flat
    ``max_completion_tokens`` (already stored), string ``pricing`` with the full
    key set, and ``per_request_limits: null``.

    Empty input -> empty output, so models without any capability data keep the
    bare OpenAI object shape (no spurious fields).
    """
    if not stored:
        return {}

    out = dict(stored)

    arch = out.get("architecture")
    if isinstance(arch, dict) and arch:
        in_mods = arch.get("input_modalities") or []
        out_mods = arch.get("output_modalities") or []
        rendered_arch = dict(arch)
        modality = _build_modality(in_mods, out_mods)
        if modality:
            rendered_arch["modality"] = modality
        out["architecture"] = rendered_arch
        # supports_vision: convenience flag for harnesses that don't read input_modalities.
        out["supports_vision"] = "image" in in_mods

    # top_provider mirrors the flat fields (OpenRouter canonical location).
    top_provider: dict = {}
    if out.get("context_length") is not None:
        top_provider["context_length"] = out["context_length"]
    if out.get("max_completion_tokens") is not None:
        top_provider["max_completion_tokens"] = out["max_completion_tokens"]
    if out.get("is_moderated") is not None:
        top_provider["is_moderated"] = bool(out["is_moderated"])
    if top_provider:
        out["top_provider"] = top_provider

    pricing = out.get("pricing")
    if isinstance(pricing, dict):
        out["pricing"] = _render_pricing(pricing)

    out["per_request_limits"] = None
    return out


def merge_capabilities(base: dict | None, override: dict | None) -> dict:
    """Deep-merge two STORED-form dicts where ``override`` wins.

    Nested dicts recurse; lists and scalars are *replaced* by override (NOT
    concatenated — contrast with utils.deep_merge). This lets a manual
    ``architecture.input_modalities`` override the cache without duplication.
    """
    result: dict = dict(base) if isinstance(base, dict) else {}
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_capabilities(result[key], value)
        else:
            result[key] = value
    return result
