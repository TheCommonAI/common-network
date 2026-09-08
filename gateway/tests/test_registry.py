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

Also pins the constants the token comparison depends on.
"""
import sys
from pathlib import Path

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


def rejected(name: str, url: str) -> None:
    try:
        registry.validate_endpoint_url(url)
    except HTTPException as e:
        check(f"rejects {name}", e.status_code, 400)
        return
    check(f"rejects {name}", "accepted", "400")


def accepted(name: str, url: str) -> None:
    try:
        registry.validate_endpoint_url(url)
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
rejected("private LAN blocked by default", "http://192.168.1.42:11434/v1")
settings.allowed_node_cidrs = '192.168.1.0/24'
accepted("explicit trusted LAN subnet", "http://192.168.1.42:11434/v1")
settings.allowed_node_cidrs = ''
rejected("plaintext public endpoint", "http://8.8.8.8:11434/v1")
accepted("public TLS endpoint", "https://8.8.8.8/v1")

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
try:
    registry.validate_endpoint_url("http://definitely-not-a-real-host.invalid/v1")
    check("unresolvable host rejected", "accepted", "400")
except HTTPException as e:
    check("unresolvable host rejected", e.status_code, 400)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all registry-security tests passed")