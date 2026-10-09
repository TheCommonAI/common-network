"""Operators-view auth: unconfigured must 404, wrong token must 401.

The page shows endpoint URLs, client addresses and failure rates, so the gate
in front of it is a security boundary and gets pinned like one. The database
queries behind /admin/state need a database and aren't covered here; what is
covered is every path by which someone reaches it, plus the one piece of that
route that is pure data-shaping -- shape_clients, which runs over rows rather
than reaching for them.
"""
import asyncio
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import HTTPException  # noqa: E402

from app import admin  # noqa: E402
from app.config import settings  # noqa: E402

FAILURES = []


def check(name: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL  {name}  (got {got!r}, want {want!r})")
    else:
        print(f"  ok    {name}")


class FakeRequest:
    """Only what _require_admin touches."""
    def __init__(self, headers=None, query=None):
        self.headers = headers or {}
        self.query_params = query or {}


def status_of(request) -> int:
    """The HTTP status _require_admin raises, or 200 if it allows through."""
    try:
        admin._require_admin(request)
        return 200
    except HTTPException as e:
        return e.status_code


async def delete_status(request) -> int:
    """The status the delete route rejects with. Only ever called with a
    request that must be turned away -- reaching the database would mean the
    gate let someone through, so a connection error here is itself a failure
    and is reported as one rather than swallowed."""
    try:
        await admin.admin_delete_node(uuid4(), request)
        return 200
    except HTTPException as e:
        return e.status_code


original = settings.admin_token
try:
    print("\nunconfigured: the routes must not exist")
    settings.admin_token = ""
    # 404 rather than 401 on purpose: an operator who never set ADMIN_TOKEN
    # should look like a gateway without the feature, not one hiding a login.
    check("no token set -> 404 even with a guess",
          status_of(FakeRequest(query={"token": "guess"})), 404)
    check("no token set -> 404 with nothing",
          status_of(FakeRequest()), 404)

    print("\nconfigured: only the right password gets in")
    settings.admin_token = "s3cret-operators-token"
    check("correct token in query param (browser)",
          status_of(FakeRequest(query={"token": "s3cret-operators-token"})), 200)
    check("correct token in header (curl/scripts)",
          status_of(FakeRequest(headers={"x-common-admin-token": "s3cret-operators-token"})), 200)
    check("wrong token -> 401",
          status_of(FakeRequest(query={"token": "wrong"})), 401)
    check("missing token -> 401",
          status_of(FakeRequest()), 401)
    check("empty token -> 401",
          status_of(FakeRequest(query={"token": ""})), 401)
    # A prefix of the real password must not pass; compare_digest is length-safe
    # but a naive startswith() refactor would break exactly here.
    check("prefix of the real token -> 401",
          status_of(FakeRequest(query={"token": "s3cret"})), 401)
    check("header wins nothing when wrong -> 401",
          status_of(FakeRequest(headers={"x-common-admin-token": "nope"})), 401)
    # DELETE /admin/nodes/{id} removes a contributor's row without their node
    # token, so it is the most dangerous door here and must sit behind exactly
    # the same gate as the read-only page -- not a weaker one bolted on later.
    # The delete itself needs a database and isn't covered; the gate is.
    print("\nnode deletion is gated identically")
    settings.admin_token = ""
    check("delete: unconfigured -> 404",
          asyncio.run(delete_status(FakeRequest(query={"token": "guess"}))), 404)
    settings.admin_token = "s3cret-operators-token"
    check("delete: wrong token -> 401",
          asyncio.run(delete_status(FakeRequest(query={"token": "wrong"}))), 401)
    check("delete: missing token -> 401",
          asyncio.run(delete_status(FakeRequest())), 401)

    # GET /admin/nodes returns full node info including endpoint_url, which
    # is exactly what GET /nodes must NOT expose. Same gate as everything else.
    print("\n/admin/nodes (full node list with URLs) is gated identically")
    async def admin_nodes_status(request) -> int:
        try:
            await admin.admin_list_nodes(request)
            return 200
        except HTTPException as e:
            return e.status_code
    settings.admin_token = ""
    check("admin_nodes: unconfigured -> 404",
          asyncio.run(admin_nodes_status(FakeRequest(query={"token": "guess"}))), 404)
    settings.admin_token = "s3cret-operators-token"
    check("admin_nodes: wrong token -> 401",
          asyncio.run(admin_nodes_status(FakeRequest(query={"token": "wrong"}))), 401)
    check("admin_nodes: missing token -> 401",
          asyncio.run(admin_nodes_status(FakeRequest())), 401)
finally:
    settings.admin_token = original

print("\nshipped default is off")
check("admin_token defaults to empty (routes 404 until set)", original, "")

# shape_clients is the only part of /admin/state that can be pinned without a
# database, and it is where the two things that would silently corrupt the page
# live: null meaning "did not say" (never pre-labelled into a fake client name)
# and Postgres handing back Decimal/interval values that json can't serialise.
print("\nper-client rows -> the JSON the page renders")


def client_row(**over):
    """A row shaped like the aggregate query returns, with the boring columns
    filled in so each check can override just the one it is about."""
    row = dict(
        client="common-cli/0.1.2",
        nodes=3,
        healthy=2,
        legacy=1,
        first_seen=datetime(2026, 10, 1, 9, 30, tzinfo=timezone.utc),
        last_seen=datetime(2026, 10, 10, 11, 0, tzinfo=timezone.utc),
        avg_tenure_hours=Decimal("41.6666666666666667"),
        max_tenure_hours=Decimal("120.0"),
    )
    row.update(over)
    return row


shaped = admin.shape_clients([client_row()])
check("one row in -> one dict out", len(shaped), 1)
check("client passes through verbatim", shaped[0]["client"], "common-cli/0.1.2")
# Decimal -> float, one decimal place. The raw avg is 41.66..., and rounding is
# the point: this number answers "roughly how long has this machine been here",
# and a float with seventeen digits implies precision the registry doesn't have.
check("avg tenure: Decimal -> float, 1dp", shaped[0]["avg_tenure_hours"], 41.7)
check("avg tenure is a float, not Decimal",
      type(shaped[0]["avg_tenure_hours"]).__name__, "float")
check("whole-hour tenure keeps 1dp", shaped[0]["max_tenure_hours"], 120.0)
check("first_seen -> ISO 8601", shaped[0]["first_seen"], "2026-10-01T09:30:00+00:00")
check("counts pass through", (shaped[0]["nodes"], shaped[0]["healthy"], shaped[0]["legacy"]), (3, 2, 1))

# A node that registered before the column existed, or a client old enough not
# to send the string. Migration 008 is explicit that this is a different fact
# from any label we could invent, so it must survive the shaping as null and be
# labelled at the one place that renders it.
null_client = admin.shape_clients([client_row(client=None)])[0]
check("null client stays null, never pre-labelled", null_client["client"], None)
check("null client is still a dict key", "client" in null_client, True)

# A group with no timestamps at all: impossible today, since nodes.created_at is
# not null, but the row is shaped before anything validates it and a crash here
# would take down the whole operators page for one bad row.
empty = admin.shape_clients([client_row(
    first_seen=None, last_seen=None,
    avg_tenure_hours=None, max_tenure_hours=None,
)])[0]
check("no timestamps -> None, not AttributeError", empty["first_seen"], None)
check("no last_seen -> None", empty["last_seen"], None)
check("no tenure -> None, not 0.0", empty["avg_tenure_hours"], None)

check("empty result -> empty list", admin.shape_clients([]), [])
# Ordering is the query's job (nodes desc, client asc nulls last); shaping must
# not quietly re-sort and disagree with it.
order_in = ["a/1", None, "b/1"]
check("row order is preserved",
      [r["client"] for r in admin.shape_clients([client_row(client=c) for c in order_in])],
      order_in)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all operators-view auth tests passed")
