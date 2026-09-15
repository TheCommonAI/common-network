"""Tests for the reported client string on node registration.

No database, no network. What needs pinning is not the column write -- that is
one parameter in a statement -- but the rules around it, because `client` is
the first field the gateway stores that a stranger controls purely for our own
reporting:

1. It resolves from the body field, then the X-Common-Client header. The
   desktop sends both; a CLI that later sends only the header must still
   count, or the number it produces is silently wrong.
2. An absent client reads as "did not say" (None), never as an empty string.
   A row that says "" and a row that says nothing are the same fact and must
   not sort into two buckets.
3. Junk is truncated, never fatal. Registration is permissionless and the
   network's whole thesis; refusing a contributor over a malformed statistic
   would be the tail wagging the dog.
4. The model caps length, so an oversized field cannot reach the column.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402

from app.models import NodeCreate  # noqa: E402

FAILURES = []


def check(name: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL  {name}: got {got!r}, want {want!r}")
    else:
        print(f"  ok    {name}")


def resolve(body: str | None, header: str | None) -> str | None:
    """The expression in registry.register_node, kept in one place.

    Mirrored rather than imported: calling the endpoint means a database and
    an embedding model, and this line is the whole of the logic.
    """
    return (body or header or "").strip()[:64] or None


print("client resolves from the body field, then the header")
check("body only", resolve("common-desktop/0.1.0", None), "common-desktop/0.1.0")
check("header only", resolve(None, "common-cli/0.1.2"), "common-cli/0.1.2")
check("body wins over header", resolve("common-desktop/0.1.0", "curl/8"), "common-desktop/0.1.0")

print("\nsilence is None, not empty string")
check("neither sent", resolve(None, None), None)
check("empty body", resolve("", None), None)
check("whitespace only", resolve("   ", None), None)
check("empty header", resolve(None, ""), None)

print("\njunk is truncated, never fatal")
check("padding stripped", resolve("  common-desktop/0.1.0  ", None), "common-desktop/0.1.0")
check("over-long truncated to 64", len(resolve("x" * 500, None)), 64)
check("over-long header truncated", len(resolve(None, "y" * 500)), 64)

print("\nthe model caps the field before it reaches the column")
node = dict(name="n", endpoint_url="https://e.example/v1", model_name="m", capability_text="c")
check("64 chars accepted", NodeCreate(**node, client="c" * 64).client, "c" * 64)
try:
    NodeCreate(**node, client="c" * 65)
    check("65 chars rejected", "accepted", "ValidationError")
except ValidationError:
    check("65 chars rejected", "ValidationError", "ValidationError")

print("\nclient is optional -- every existing caller still validates")
check("omitted entirely", NodeCreate(**node).client, None)
check("explicit null", NodeCreate(**node, client=None).client, None)

print("\nclient is never exposed by GET /nodes")
from app.models import NodeOut  # noqa: E402
check("absent from the public node model", "client" in NodeOut.model_fields, False)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all client-reporting tests passed")
