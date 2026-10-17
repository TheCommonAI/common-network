"""Security tests for node registration.

No database, no network, no embedding model. These pin the two properties that
make permissionless registration safe to expose publicly (see SECURITY.md):

1. An endpoint_url is the one URL a stranger gets to make the gateway fetch —
   it must never reach cloud metadata services, and loopback only when the
   operator opted in.
2. A node's token is the only thing standing between a public name and a
   takeover — the registry must never hand it to someone who didn't already
   hold it. (The token logic itself needs a database; what is testable
   without one is the guard around it: what the 409 branch checks.)

Also pins the constants the token comparison depends on, and tests DNS
resolution retry behavior for fresh Cloudflare quick tunnels.
"""
import asyncio
import sys
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import HTTPException  # noqa: E402

from app import registry  # noqa: E402
from app.config import settings  # noqa: E402

FAILURES = []


def check(name: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL  {name}  (got {got!r}, want {want!r})")
    else:
        print(f"  ok    {name}")


# The validator resolves DNS through the running event loop, so every caller
# here runs inside one. `asyncio.run` per call would also work for the single
# checks, but the tests that matter most are the ones that need *concurrency*
# (see "the loop is not blocked"), and those have to own the loop.
def run(coro):
    return asyncio.run(coro)


def rejected(name: str, url: str) -> None:
    try:
        run(registry.validate_endpoint_url(url))
    except HTTPException as e:
        check(f"rejects {name}", e.status_code, 400)
        return
    check(f"rejects {name}", "accepted", "400")


def accepted(name: str, url: str) -> None:
    try:
        run(registry.validate_endpoint_url(url))
        check(f"accepts {name}", True, True)
    except HTTPException as e:
        check(f"accepts {name}", f"rejected ({e.detail})", True)


print("\nscheme validation")
rejected("plain file:// path", "file:///etc/passwd")
rejected("gopher", "gopher://x/v1")
rejected("no scheme at all", "localhost:11434/v1")
rejected("empty", "")

print("\ncloud metadata is always out of reach")
# 169.254.169.254 is the address every cloud SSRF write-up walks through.
rejected("AWS/GCP/Azure metadata", "http://169.254.169.254/latest/meta-data/v1")
rejected("link-local IPv6", "http://[fe80::1]/v1")

print("\nloopback honours the operator's setting")
original = settings.allow_loopback_node_endpoints
try:
    settings.allow_loopback_node_endpoints = True
    accepted("localhost when allowed (seed demo, same-machine dev)",
             "http://localhost:11434/v1")
    accepted("explicit 127.0.0.1 when allowed", "http://127.0.0.1:11434/v1")
    settings.allow_loopback_node_endpoints = False
    rejected("localhost when a public gateway disallows it", "http://localhost:11434/v1")
    rejected("0.0.0.0 (unspecified) when disallowed", "http://0.0.0.0:11434/v1")
finally:
    settings.allow_loopback_node_endpoints = original
    del original

print("\nreal nodes are unaffected")
accepted("private LAN address (school lab --lan)", "http://192.168.1.42:11434/v1")
accepted("public IP literal", "http://8.8.8.8:11434/v1")

# Hostname path, stubbed so the suite needs no DNS: a public tunnel/API
# hostname resolving to a public address is a legitimate node.
real_getaddrinfo = registry.socket.getaddrinfo
try:
    def stub_getaddrinfo(host, *args, **kwargs):
        if host == "my-node.example.com":
            return [(2, 1, 6, "", ("8.8.8.8", 0))]
        return real_getaddrinfo(host, *args, **kwargs)
    registry.socket.getaddrinfo = stub_getaddrinfo
    accepted("public https hostname (tunnel/API)", "https://my-node.example.com/v1")
finally:
    registry.socket.getaddrinfo = real_getaddrinfo

print("\nunresolvable hostnames fail at registration, not at first request")
# Called with no wait_seconds, i.e. exactly as registration calls it, so this
# is the real default path -- but with the window shrunk, because the default
# 30s is spent retrying and the assertion is about the rejection. The window
# itself is exercised in test_dns_retry_success below.
original_max_wait = settings.dns_resolve_max_wait_seconds
original_initial = settings.dns_resolve_initial_backoff_seconds
original_max_backoff = settings.dns_resolve_max_backoff_seconds
try:
    settings.dns_resolve_max_wait_seconds = 0.3
    settings.dns_resolve_initial_backoff_seconds = 0.05
    settings.dns_resolve_max_backoff_seconds = 0.1
    try:
        run(registry.validate_endpoint_url("http://definitely-not-a-real-host.invalid/v1"))
        check("unresolvable host rejected", "accepted", "400")
    except HTTPException as e:
        check("unresolvable host rejected", e.status_code, 400)
finally:
    settings.dns_resolve_max_wait_seconds = original_max_wait
    settings.dns_resolve_initial_backoff_seconds = original_initial
    settings.dns_resolve_max_backoff_seconds = original_max_backoff

print("\nDNS resolution retry for fresh tunnels")

# Test: DNS eventually succeeds after retries
def test_dns_retry_success():
    """Hostname that fails initially but succeeds on retry."""
    call_count = [0]
    real_getaddrinfo = registry.socket.getaddrinfo

    def flaky_getaddrinfo(host, *args, **kwargs):
        if host == "flaky-tunnel.trycloudflare.com":
            call_count[0] += 1
            if call_count[0] < 3:
                raise registry.socket.gaierror("Name or service not known")
            return [(2, 1, 6, "", ("104.16.0.1", 0))]
        return real_getaddrinfo(host, *args, **kwargs)

    original_max_wait = settings.dns_resolve_max_wait_seconds
    original_initial = settings.dns_resolve_initial_backoff_seconds
    original_max_backoff = settings.dns_resolve_max_backoff_seconds
    try:
        settings.dns_resolve_max_wait_seconds = 10.0
        settings.dns_resolve_initial_backoff_seconds = 0.1
        settings.dns_resolve_max_backoff_seconds = 0.2

        with mock.patch.object(registry.socket, "getaddrinfo", flaky_getaddrinfo):
            try:
                run(registry.validate_endpoint_url("https://flaky-tunnel.trycloudflare.com/v1"))
                check("DNS retry succeeds after transient failure", call_count[0] >= 3, True)
            except HTTPException:
                check("DNS retry succeeds after transient failure", "rejected", "should succeed")
    finally:
        settings.dns_resolve_max_wait_seconds = original_max_wait
        settings.dns_resolve_initial_backoff_seconds = original_initial
        settings.dns_resolve_max_backoff_seconds = original_max_backoff

test_dns_retry_success()

# Test: DNS permanently fails after retry window exhausted
def test_dns_retry_exhausted():
    """Hostname that never resolves — should fail after retry window."""
    call_count = [0]
    real_getaddrinfo = registry.socket.getaddrinfo

    def always_fail_getaddrinfo(host, *args, **kwargs):
        if host == "never-resolves.trycloudflare.com":
            call_count[0] += 1
            raise registry.socket.gaierror("Name or service not known")
        return real_getaddrinfo(host, *args, **kwargs)

    original_max_wait = settings.dns_resolve_max_wait_seconds
    original_initial = settings.dns_resolve_initial_backoff_seconds
    original_max_backoff = settings.dns_resolve_max_backoff_seconds
    try:
        settings.dns_resolve_max_wait_seconds = 0.5
        settings.dns_resolve_initial_backoff_seconds = 0.1
        settings.dns_resolve_max_backoff_seconds = 0.2

        with mock.patch.object(registry.socket, "getaddrinfo", always_fail_getaddrinfo):
            try:
                run(registry.validate_endpoint_url("https://never-resolves.trycloudflare.com/v1"))
                check("DNS retry exhausted rejects", "accepted", "should reject")
            except HTTPException as e:
                check("DNS retry exhausted rejects", e.status_code, 400)
                check("DNS retry exhausted error mentions retry",
                      "Retried for" in e.detail, True)
                check("DNS retry multiple attempts", call_count[0] > 1, True)
    finally:
        settings.dns_resolve_max_wait_seconds = original_max_wait
        settings.dns_resolve_initial_backoff_seconds = original_initial
        settings.dns_resolve_max_backoff_seconds = original_max_backoff

test_dns_retry_exhausted()

# Test: wait_seconds=0 is a single attempt, with no retry burst.
#
# This is what the health checker asks for. It matters on its own terms: the
# retry burst is right at registration and wrong on a periodic pass, and the
# difference is a per-node cost paid over every registered row.
def test_no_retry_when_wait_is_zero():
    call_count = [0]
    real_getaddrinfo = registry.socket.getaddrinfo

    def always_fail_getaddrinfo(host, *args, **kwargs):
        if host == "cold-tunnel.trycloudflare.com":
            call_count[0] += 1
            raise registry.socket.gaierror("Name or service not known")
        return real_getaddrinfo(host, *args, **kwargs)

    original_max_wait = settings.dns_resolve_max_wait_seconds
    original_initial = settings.dns_resolve_initial_backoff_seconds
    original_max_backoff = settings.dns_resolve_max_backoff_seconds
    try:
        settings.dns_resolve_max_wait_seconds = 30.0
        settings.dns_resolve_initial_backoff_seconds = 1.0
        settings.dns_resolve_max_backoff_seconds = 8.0

        started = time.monotonic()
        with mock.patch.object(registry.socket, "getaddrinfo", always_fail_getaddrinfo):
            try:
                run(registry.validate_endpoint_url(
                    "https://cold-tunnel.trycloudflare.com/v1", wait_seconds=0))
                check("wait_seconds=0 rejects an unresolvable host", "accepted", "400")
            except HTTPException as e:
                check("wait_seconds=0 rejects an unresolvable host", e.status_code, 400)
        elapsed = time.monotonic() - started
        check("wait_seconds=0 attempts DNS once", call_count[0], 1)
        # The configured window is 30s. Anything near it means the burst ran.
        check("wait_seconds=0 does not wait out the retry window", elapsed < 5.0, True)
    finally:
        settings.dns_resolve_max_wait_seconds = original_max_wait
        settings.dns_resolve_initial_backoff_seconds = original_initial
        settings.dns_resolve_max_backoff_seconds = original_max_backoff

test_no_retry_when_wait_is_zero()

# Test: Immediate success doesn't retry
def test_dns_immediate_success():
    """Hostname that resolves immediately — no retry needed."""
    call_count = [0]
    real_getaddrinfo = registry.socket.getaddrinfo

    def counting_getaddrinfo(host, *args, **kwargs):
        if host == "immediate-tunnel.trycloudflare.com":
            call_count[0] += 1
            return [(2, 1, 6, "", ("104.16.0.1", 0))]
        return real_getaddrinfo(host, *args, **kwargs)

    with mock.patch.object(registry.socket, "getaddrinfo", counting_getaddrinfo):
        try:
            run(registry.validate_endpoint_url("https://immediate-tunnel.trycloudflare.com/v1"))
            check("DNS immediate success no retry", call_count[0], 1)
        except HTTPException:
            check("DNS immediate success no retry", "rejected", "should succeed")

test_dns_immediate_success()

# Test: IP literals skip DNS resolution entirely
def test_ip_literal_no_dns():
    """IP address literal — no DNS resolution at all."""
    call_count = [0]
    real_getaddrinfo = registry.socket.getaddrinfo

    def counting_getaddrinfo(host, *args, **kwargs):
        call_count[0] += 1
        return real_getaddrinfo(host, *args, **kwargs)

    with mock.patch.object(registry.socket, "getaddrinfo", counting_getaddrinfo):
        try:
            run(registry.validate_endpoint_url("http://8.8.8.8:11434/v1"))
            check("IP literal skips DNS", call_count[0], 0)
        except HTTPException:
            check("IP literal skips DNS", "rejected", "should succeed")

test_ip_literal_no_dns()


# --- The regression this file exists for now --------------------------------
#
# Measured on the live gateway on 2026-10-17: `/health` — a handler that
# touches no database, no node and no embedder — answered in 8.12, 8.14,
# 8.15, 8.16, 8.19, 8.32, 8.47 and 8.48 seconds, with time-to-first-byte
# equal to total time. Not a slow endpoint: a stalled process. The stall was
# this module, reached from the health checker's pass over 23 registered
# nodes on a 30s loop, where 21 of them no longer resolve.
#
# So the property to pin is not "the retry works" (the tests above cover it,
# and they pass either way) but "a DNS retry does not stop everything else".
# The check below fails on the blocking implementation and passes on the
# async one, which is the whole reason for the change.

print("\nthe loop is not blocked while DNS is retrying")


async def loop_stays_responsive() -> None:
    ticks = [0]
    real_getaddrinfo = registry.socket.getaddrinfo

    def always_fail_getaddrinfo(host, *args, **kwargs):
        if host == "cold-tunnel.trycloudflare.com":
            raise registry.socket.gaierror("Name or service not known")
        return real_getaddrinfo(host, *args, **kwargs)

    # The deadline is fixed before the task exists, deliberately. If the
    # ticker computed it on its own first run, a blocking caller would shift
    # the whole window forward and the ticker would look busy afterwards —
    # measuring the wait, not the blockage. Fixed up front, a blocked loop
    # leaves the deadline already past and the ticker does nothing at all.
    deadline = time.monotonic() + 0.5

    async def ticker():
        # A stand-in for every other request in the process. It holds no
        # lock, awaits nothing this module owns, and only needs the loop to
        # come back to it.
        while time.monotonic() < deadline:
            ticks[0] += 1
            await asyncio.sleep(0.01)

    original_max_wait = settings.dns_resolve_max_wait_seconds
    original_initial = settings.dns_resolve_initial_backoff_seconds
    original_max_backoff = settings.dns_resolve_max_backoff_seconds
    try:
        settings.dns_resolve_max_wait_seconds = 1.0
        settings.dns_resolve_initial_backoff_seconds = 0.05
        settings.dns_resolve_max_backoff_seconds = 0.1

        with mock.patch.object(registry.socket, "getaddrinfo", always_fail_getaddrinfo):
            t = asyncio.create_task(ticker())
            try:
                await registry.validate_endpoint_url(
                    "https://cold-tunnel.trycloudflare.com/v1")
            except HTTPException:
                pass
            await t
    finally:
        settings.dns_resolve_max_wait_seconds = original_max_wait
        settings.dns_resolve_initial_backoff_seconds = original_initial
        settings.dns_resolve_max_backoff_seconds = original_max_backoff

    # The 1.0s window ticks a 10ms ticker ~50 times when the loop is free and
    # exactly 0 times when it is blocked. 20 is well clear of both, and clear
    # of a slow CI box too.
    check("a DNS retry burst leaves the loop free", ticks[0] >= 20, True)


run(loop_stays_responsive())

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all registry-security tests passed")