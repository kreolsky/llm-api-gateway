"""Forwarded-header policy: the denylist applied to client headers going upstream.

ARCH: `identity: passthrough` forwards ALL client headers upstream verbatim
(the client's own spelling — casing is part of a harness fingerprint, and the
source of headers is now one real agent, not a synthesized profile). A full
forward REQUIRES a denylist: without one the router would leak the client's
own credentials upstream and forward stale transport values for a body it has
itself re-serialized (model override).
"""
# SYSTEM: header-policy — denylist for client headers forwarded upstream

# INVARIANT: the denylist is fail-open — an unknown client header IS forwarded. Why:
# the deployment is a private lab with its own agents and one external server, where
# forwarding beats silently dropping an unknown harness header; the consequence is
# that this list MUST be re-audited whenever a new harness or reverse proxy is
# onboarded — a leaked header name cannot be caught any other way.

# Client credentials: only the router's own key (Authorization built from
# api_key_env) ever goes upstream. Anthropic-/Azure-/Google-style agents carry
# their key in these names instead of Authorization.
_CREDENTIAL_HEADERS = frozenset({
    "authorization",
    "proxy-authorization",
    "cookie",
    "x-api-key",
    "api-key",
    "x-goog-api-key",
})

# Transport / hop-by-hop: the router re-serializes the body (model override),
# so the client's framing values are stale and would break the
# request. accept-encoding: httpx would honor a client's br/zstd it cannot
# decode.
_TRANSPORT_HEADERS = frozenset({
    "host",
    "content-length",
    "content-type",
    "content-encoding",
    "connection",
    "transfer-encoding",
    "te",
    "upgrade",
    "keep-alive",
    "expect",
    "accept-encoding",
})

# Lab network topology: behind a reverse proxy these would leak internal IPs.
_TOPOLOGY_HEADERS = frozenset({
    "x-real-ip",
    "forwarded",
    "true-client-ip",
    "cf-connecting-ip",
    "cdn-loop",
})
_TOPOLOGY_PREFIXES = ("x-forwarded-",)

FORWARDED_HEADER_DENYLIST = _CREDENTIAL_HEADERS | _TRANSPORT_HEADERS | _TOPOLOGY_HEADERS
FORWARDED_HEADER_DENY_PREFIXES = _TOPOLOGY_PREFIXES

# Static `headers:` from providers.yaml is operator-authored, so it is held to
# a stricter standard than client input: the provider's own credential header is
# built from api_key_env (see the INVARIANT over Provider — Authorization for
# OpenAI-compat, x-api-key for Anthropic-protocol), so a static twin would
# collide with it; and per-request transport values are owned by the router
# itself (Content-Type defaults / multipart boundary popping).
# WHY not all of _CREDENTIAL_HEADERS: api_key_env can only emit those two names,
# so a static `api-key` / `cookie` is the ONLY way to reach a backend keyed on
# another header (Azure OpenAI's `api-key`) — and nothing router-owned collides.
FORBIDDEN_STATIC_HEADERS = frozenset({"authorization", "x-api-key"}) | _TRANSPORT_HEADERS
