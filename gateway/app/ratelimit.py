"""Per-client request limiting on /v1/chat/completions.

A brake on bulk abuse of donated compute, not a security boundary: a client
that rotates source addresses gets a fresh bucket for each. The contribution
gate (`REQUIRE_CONTRIBUTION`) is the real control; this exists so a lone
contributor's laptop doesn't get melted by one script in a loop.

Token bucket, in-process, no external state:
  * capacity = the whole per-minute allowance (a burst up front, then a
    steady drip) — a human asks a question and reads the answer; a script
    hammers, and after the burst it's throttled to the drip rate.
  * keyed by client IP, from X-Forwarded-For when a proxy is in front
    (Railway) and the socket address otherwise. Trusting XFF lets a client
    name a fake IP and get a fresh bucket per request — accepted deliberately,
    because the alternative (ignoring XFF) gives the *entire internet* one
    shared bucket behind a proxy, which is worse and silent.
"""
from __future__ import annotations

import time

from app.config import settings

# key -> (tokens, last_refill_monotonic)
_buckets: dict[str, tuple[float, float]] = {}


def client_key(headers, client_host: str | None) -> str:
    xff = headers.get("x-forwarded-for")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    return client_host or "unknown"


def _check(key: str, now: float) -> float:
    """One token-bucket step. Returns 0 if allowed, else seconds to wait."""
    rate = settings.rate_limit_requests_per_minute
    if rate <= 0:
        return 0.0
    capacity = float(rate)
    refill_per_second = rate / 60.0

    tokens, last = _buckets.get(key, (capacity, now))
    # Refill elapsed allowance, capped at one full burst.
    tokens = min(capacity, tokens + max(0.0, now - last) * refill_per_second)

    if tokens < 1.0:
        _buckets[key] = (tokens, now)
        return (1.0 - tokens) / refill_per_second
    _buckets[key] = (tokens - 1.0, now)
    return 0.0


def check_rate_limit(request, now: float | None = None) -> float:
    """Consume one request slot for this client. Returns 0.0 if allowed,
    else the number of seconds until a slot is available (for Retry-After).
    """
    key = client_key(request.headers,
                     request.client.host if request.client else None)
    return _check(key, now if now is not None else time.monotonic())