"""What a health pass is allowed to cost.

No database, no network, no embedding model. The pass is the one thing in the
gateway that runs unattended, against every registered row, on a timer -- so
its cost is paid by everybody else in the process whether they asked for it or
not. Measured on the live gateway on 2026-10-17: /health, a handler that
touches nothing, answering in 8.12-8.48s, because one pass was costing the
whole event loop minutes.

Three properties, each of which was once false:

1. A pass does not block the loop. (It resolved DNS synchronously and waited
   out a retry window with time.sleep.)
2. An unresolvable node costs one attempt, not a retry burst. (It inherited
   registration's 30s tolerance, 23 times per pass.)
3. A pass is concurrent but bounded. (It ran one node at a time, and the
   fan-out was whatever the registry happened to contain.)
"""
import asyncio
import sys
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app import health  # noqa: E402
from app.config import settings  # noqa: E402

FAILURES = []


def check(name: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL  {name}  (got {got!r}, want {want!r})")
    else:
        print(f"  ok    {name}")


REAL_RESPONSE = httpx.Response
REAL_TRANSPORT = httpx.AsyncBaseTransport
REAL_CLIENT = httpx.AsyncClient


def node(i: int, host: str) -> dict:
    return dict(id=i, endpoint_url=f"https://{host}/v1", api_key_ref=None,
                worker_token="w", model_name="m")


def fake_db(rows):
    """A pool whose calls actually yield, the way asyncpg's do.

    That detail is load-bearing. A fake that returns without awaiting makes
    the whole pass one uninterrupted block, which is exactly the condition
    these tests exist to detect -- so a fake like that would have hidden the
    bug rather than caught it.
    """
    class _Conn:
        async def fetch(self, sql, *a):
            await asyncio.sleep(0)
            return rows

        async def execute(self, sql, *a):
            await asyncio.sleep(0)

    class _Acquire:
        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *a):
            return False

    class _Pool:
        def acquire(self):
            return _Acquire()

    return mock.patch.object(health.db, "pool", lambda: _Pool())


def fake_nodes(rows, handler):
    """A client for the health loop only; `handler` answers a probe."""
    class Transport(REAL_TRANSPORT):
        async def handle_async_request(self, request):
            return await handler(request)

    def factory(*a, **k):
        return REAL_CLIENT(transport=Transport())

    return mock.patch.object(health.httpx, "AsyncClient", factory)


def ok_handler(delay=0.0):
    async def handler(request):
        if delay:
            await asyncio.sleep(delay)
        return REAL_RESPONSE(200, json={"data": [{"id": "m"}]})
    return handler


def run_pass(rows, handler, getaddrinfo=None):
    """One pass; returns (elapsed, peak_concurrent_probes, resolver_calls)."""

    async def go():
        import app.registry as registry

        peak = [0]
        in_flight = [0]
        resolver_calls = [0]
        real = registry.socket.getaddrinfo

        def counted(host, *a, **k):
            resolver_calls[0] += 1
            if getaddrinfo is not None:
                return getaddrinfo(host, real, *a, **k)
            return real(host, *a, **k)

        async def counting_handler(request):
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
            try:
                return await handler(request)
            finally:
                in_flight[0] -= 1

        patches = [mock.patch.object(registry.socket, "getaddrinfo", counted),
                   fake_nodes(rows, counting_handler),
                   fake_db(rows)]
        for p in patches:
            p.start()
        try:
            started = time.monotonic()
            await health.run_health_checks_once()
            return time.monotonic() - started, peak[0], resolver_calls[0]
        finally:
            for p in reversed(patches):
                p.stop()

    return asyncio.run(go())


# --- 1. an unresolvable node costs one attempt, not a retry burst -----------
#
# Production's own settings: a 1s initial backoff, an 8s cap and a 30s window.
# Three unresolvable nodes under those settings used to cost 90 seconds.

print("\nan unresolvable node costs one attempt, not the registration window")
NXDOMAIN = [node(i, f"gone-{i}.trycloudflare.com") for i in range(3)]


def nxdomain(host, real, *a, **k):
    import app.registry as registry
    if host.endswith(".trycloudflare.com"):
        raise registry.socket.gaierror("Name or service not known")
    return real(host, *a, **k)


settings_before = (settings.dns_resolve_max_wait_seconds,
                   settings.dns_resolve_initial_backoff_seconds,
                   settings.dns_resolve_max_backoff_seconds)
settings.dns_resolve_max_wait_seconds = 30.0
settings.dns_resolve_initial_backoff_seconds = 1.0
settings.dns_resolve_max_backoff_seconds = 8.0
try:
    elapsed, _, resolver_calls = run_pass(NXDOMAIN, ok_handler(), nxdomain)
finally:
    (settings.dns_resolve_max_wait_seconds,
     settings.dns_resolve_initial_backoff_seconds,
     settings.dns_resolve_max_backoff_seconds) = settings_before

# 90s was the pre-fix cost for exactly these three rows.
check("three unresolvable nodes cost well under a second, not 90",
      elapsed < 2.0, True)
check("each unresolvable node resolves once", resolver_calls, 3)


# --- 2. a pass does not block the loop -------------------------------------

print("\na pass leaves the loop free for everything else")
rows = [node(0, "127.0.0.1")]


async def loop_stays_responsive():
    ticks = [0]
    deadline = time.monotonic() + 0.3

    async def ticker():
        while time.monotonic() < deadline:
            ticks[0] += 1
            await asyncio.sleep(0.01)

    import app.registry as registry
    real = registry.socket.getaddrinfo

    def slow_dns(host, *a, **k):
        if host.endswith(".slow.invalid"):
            # The resolver itself taking its time is the other half of the
            # bug: a blocking getaddrinfo holds the thread for however long
            # it takes to give up, and inline it held the loop for the same
            # stretch. Modelled as a real sleep, not a raise.
            time.sleep(0.5)
            raise registry.socket.gaierror("Name or service not known")
        return real(host, *a, **k)

    with mock.patch.object(registry.socket, "getaddrinfo", slow_dns), \
         fake_nodes([node(0, "slow.invalid")], ok_handler()), \
         fake_db([node(0, "slow.invalid")]):
        t = asyncio.create_task(ticker())
        try:
            await health.run_health_checks_once()
        finally:
            await t
    check("a pass leaves the loop free", ticks[0] >= 10, True)


asyncio.run(loop_stays_responsive())


# --- 3. a pass is concurrent but bounded ------------------------------------

print("\na pass checks nodes concurrently, and the cap binds")
MANY = [node(i, f"127.0.0.{10 + i}") for i in range(12)]
elapsed, peak, _ = run_pass(MANY, ok_handler(delay=0.2))
# 12 nodes at 0.2s each: 2.4s run in series, 0.4s in two batches of the cap.
check("12 slow nodes finish in batches, not in series", elapsed < 1.2, True)
check("no more than the cap are probed at once", peak <= health.HEALTH_CHECK_CONCURRENCY, True)
check("the cap is actually reached", peak, health.HEALTH_CHECK_CONCURRENCY)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all health-loop tests passed")
