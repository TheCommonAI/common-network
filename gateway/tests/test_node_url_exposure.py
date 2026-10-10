"""Security: endpoint URLs must not be exposed publicly.

GET /nodes is public and must NOT include endpoint_url — those are Cloudflare
tunnel URLs pointing at contributors' Ollama instances, which have no
authentication. Publishing them lets anyone bypass the gateway.

GET /admin/nodes returns full node info including endpoint_url, but only
when authenticated with ADMIN_TOKEN.

No database required — this tests the models and route structure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import NodeOut, NodePublicOut, NodeRegisterOut  # noqa: E402

FAILURES = []


def check(name: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL  {name}  (got {got!r}, want {want!r})")
    else:
        print(f"  ok    {name}")


print("NodePublicOut (public GET /nodes) must NOT have endpoint_url")
check("endpoint_url absent from NodePublicOut",
      "endpoint_url" in NodePublicOut.model_fields, False)
check("id present in NodePublicOut",
      "id" in NodePublicOut.model_fields, True)
check("name present in NodePublicOut",
      "name" in NodePublicOut.model_fields, True)
check("model_name present in NodePublicOut",
      "model_name" in NodePublicOut.model_fields, True)
check("healthy present in NodePublicOut",
      "healthy" in NodePublicOut.model_fields, True)
check("domain_tags present in NodePublicOut",
      "domain_tags" in NodePublicOut.model_fields, True)
check("base_model present in NodePublicOut",
      "base_model" in NodePublicOut.model_fields, True)
check("adapter_ids present in NodePublicOut",
      "adapter_ids" in NodePublicOut.model_fields, True)

# Structural guard: the checks above name one field, and a rule enforced by
# naming one field only holds until somebody adds another. `endpoint_url` was
# banned by name; `tunnel_url` or `endpoint` or `address` would walk straight
# past it.
#
# So this bans the *shape* of the name instead. Field names are split on
# underscores and any token that means "where the node is" fails the suite --
# a future field cannot smuggle a URL into public output by being called
# something else.
#
# Token-wise rather than substring, so `curl_count` is not mistaken for a URL
# and `endpoint_url` is. False positives here cost a review; a false negative
# costs contributor machines, since these URLs point at unauthenticated Ollama
# instances and publishing one is exactly the bypass the gateway exists to
# prevent. If a token ever fires on a field that genuinely is not a location,
# the fix is to justify it in this list, not to delete the check.
LOCATION_TOKENS = {"url", "uri", "host", "hostname", "endpoint", "address", "tunnel"}


def location_shaped(field_name: str) -> str | None:
    return next((t for t in field_name.split("_") if t in LOCATION_TOKENS), None)


print("\nNo field on a public node shape may be named like a location")
for shape in (NodePublicOut, NodeRegisterOut):
    for field in shape.model_fields:
        check(f"{shape.__name__}.{field} is not location-shaped",
              location_shaped(field), None)

# The guard has to be able to fail, or it is decoration. These assert that a
# location-shaped name is *caught*, not which token caught it -- `next()`
# returns the first match, so `endpoint_url` reports 'endpoint', and pinning
# the exact token would test the iteration order rather than the guard.
def catches(name: str) -> bool:
    token = location_shaped(name)
    return token is not None and token in LOCATION_TOKENS


check("guard catches endpoint_url", catches("endpoint_url"), True)
check("guard catches a bare 'endpoint'", catches("endpoint"), True)
check("guard catches an invented tunnel field", catches("tunnel_host"), True)
check("guard catches a URL on a new noun", catches("inference_url"), True)
check("guard ignores an unrelated name", catches("curl_count"), False)
check("guard ignores a plain noun", catches("model_name"), False)

print("\nNodeOut (admin/internal) must have endpoint_url")
check("endpoint_url present in NodeOut",
      "endpoint_url" in NodeOut.model_fields, True)

print("\nNodeOut extends NodePublicOut (inheritance check)")
check("NodeOut is subclass of NodePublicOut",
      issubclass(NodeOut, NodePublicOut), True)

print("\nNodeRegisterOut (POST /nodes response) must NOT have endpoint_url")
check("endpoint_url absent from NodeRegisterOut",
      "endpoint_url" in NodeRegisterOut.model_fields, False)
check("node_token present in NodeRegisterOut",
      "node_token" in NodeRegisterOut.model_fields, True)

print("\nNodeRegisterOut extends NodePublicOut, not NodeOut")
check("NodeRegisterOut is subclass of NodePublicOut",
      issubclass(NodeRegisterOut, NodePublicOut), True)
check("NodeRegisterOut is NOT subclass of NodeOut",
      issubclass(NodeRegisterOut, NodeOut), False)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all endpoint-url exposure tests passed")
