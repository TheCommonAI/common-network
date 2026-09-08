"""Access-control tests: credential parsing, the rate limiter, shipped defaults.

The contribution gate's database lookup needs a database; what is testable
without one is everything the gate trusts — how a credential is spelled, what
the limiter does under sustained fire — plus the shipped defaults themselves,
because "gated, with rate limiting" is the product posture and a future flip
should be a deliberate test-updating decision, not drift.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import ratelimit  # noqa: E402
from app.config import Settings, settings  # noqa: E402
from app.gateway import contributor_token  # noqa: E402
from app.upstream import resolve_api_key  # noqa: E402

FAILURES = []


def check(name: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL  {name}  (got {got!r}, want {want!r})")
    else:
        print(f"  ok    {name}")


print("\nshipped posture is pinned")
# Deliberately not `settings`: the running instance loads gateway/.env, which
# the local demo overrides (false/0) for its tokenless seed nodes. What is
# pinned here is what a deployment gets from the code alone.
code_defaults = Settings(_env_file=None)
check("contribution gate on by default", code_defaults.require_contribution, True)
check("rate limit on by default (req/min)", code_defaults.rate_limit_requests_per_minute, 20)

print("\ncredential parsing (what the gate trusts)")
check("X-Common-Node-Token (the CLI spelling)",
      contributor_token({"x-common-node-token": "tok-1"}), "tok-1")
check("Authorization: Bearer (OpenAI SDK spelling)",
      contributor_token({"authorization": "Bearer tok-2"}), "tok-2")
check("bearer lowercase scheme",
      contributor_token({"authorization": "bearer tok-3"}), "tok-3")
check("empty Bearer is no credential",
      contributor_token({"authorization": "Bearer "}), None)
check("no headers -> no credential", contributor_token({}), None)
check("whitespace is stripped",
      contributor_token({"x-common-node-token": "  tok-4  "}), "tok-4")

print("\nrate limiter: burst then throttle then refill")
original_rate = settings.rate_limit_requests_per_minute
try:
    settings.rate_limit_requests_per_minute = 6  # capacity 6, refill 0.1/s
    # A fresh client gets the full burst up front — a human asks, then reads.
    waits = [ratelimit._check("client-a", now=float(t)) for t in range(7)]
    check("first 6 requests pass", [w for w in waits[:6]], [0.0] * 6)
    check("7th is throttled with a wait", waits[6] > 0, True)
    # Half a minute later the bucket has fully refilled.
    check("refilled after idle", ratelimit._check("client-a", now=30.0), 0.0)
    # Two requests' worth of idle buys exactly two more.
    check("one more after 10s", ratelimit._check("client-a", now=40.0), 0.0)
    check("second one after 10s more", ratelimit._check("client-a", now=50.0), 0.0)
    # Back-to-back requests drain the bucket: the drip buys one token per ten
    # idle seconds, so a client that doesn't stop reading throttles.
    drains = [ratelimit._check("client-a", now=50.0) for _ in range(3)]
    check("back-to-back requests drain the refilled bucket",
          [w == 0.0 for w in drains], [True, True, False])
    # Buckets are per client: someone else's hammering doesn't throttle you.
    check("independent per client", ratelimit._check("client-b", now=50.5), 0.0)
    # 0 disables the limiter entirely.
    settings.rate_limit_requests_per_minute = 0
    check("rate 0 disables", ratelimit._check("client-c", now=0.0), 0.0)
finally:
    settings.rate_limit_requests_per_minute = original_rate

print("\nclient key: proxy header honoured, socket address otherwise")
class FakeClient:
    def __init__(self, host): self.host = host
check("X-Forwarded-For wins (behind Railway)",
      ratelimit.client_key({"x-forwarded-for": "203.0.113.9, 10.0.0.1"}, None),
      "203.0.113.9")
check("socket address without proxy",
      ratelimit.client_key({}, "192.168.1.5"), "192.168.1.5")
check("unknown when nothing is known",
      ratelimit.client_key({}, None), "unknown")

print("\ncredential references are allowlisted, not caller-chosen")
os.environ["TEST_FAKE_SECRET"] = "super-secret-value"
_orig_allowed = settings.allowed_api_key_refs
try:
    # The attack: register an endpoint you control, name any variable in the
    # gateway's environment, receive its value as a bearer token.
    settings.allowed_api_key_refs = ""
    check("nothing allowed by default -> no key resolved",
          resolve_api_key("TEST_FAKE_SECRET"), None)
    check("an unlisted name resolves to nothing",
          resolve_api_key("OPENROUTER_API_KEY"), None)

    settings.allowed_api_key_refs = "OPENROUTER_API_KEY"
    check("a name the operator listed but did not set -> None",
          resolve_api_key("OPENROUTER_API_KEY"), None)
    check("still refuses a name that is merely in the environment",
          resolve_api_key("TEST_FAKE_SECRET"), None)

    settings.allowed_api_key_refs = "TEST_FAKE_SECRET, OPENROUTER_API_KEY"
    check("an allowlisted, present name does resolve",
          resolve_api_key("TEST_FAKE_SECRET"), "super-secret-value")
    check("no reference at all -> None", resolve_api_key(None), None)
    check("empty reference -> None", resolve_api_key(""), None)
finally:
    settings.allowed_api_key_refs = _orig_allowed
    os.environ.pop("TEST_FAKE_SECRET", None)

check("shipped default allows no credential references", _orig_allowed, "")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all access-control tests passed")
