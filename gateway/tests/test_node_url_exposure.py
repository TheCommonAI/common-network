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
