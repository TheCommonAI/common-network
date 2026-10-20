"""Distributed model groups, as the gateway sees them.

A group is one registered node whose coordinator says what state it is in
(migration 015). These tests pin the gateway side of that:

1. The registration model accepts the group fields, rejects values the gateway
   cannot show honestly, and leaves every single node exactly as it was.
2. GET /nodes shows the group fields, and still never shows endpoint_url.
3. The health loop never marks a restarting group healthy, whatever its probe
   answers, so a group in the middle of a restart is not routed to.
4. A request whose only candidate is a restarting group gets a 503 with
   Retry-After and a detail that names the group status. A healthy node
   elsewhere still answers, and a forced node name that is restarting is a 503
   rather than a 404.

No database, no network, no embedding model. The pool, the upstream and the
embedder are replaced, the same way tests/test_route_clock.py replaces them.
The SQL itself needs Postgres and is exercised separately against a local
database, not here.
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from app import compose, embedder, gateway, health, metrics, ratelimit, registry, upstream  # noqa: E402
from app.compose import PanelPlan  # noqa: E402
from app.config import settings  # noqa: E402
from app.models import NodeCreate, NodePublicOut  # noqa: E402
from app.router import ScoredNode  # noqa: E402

FAILURES: list[str] = []


def check(name: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL  {name}  (got {got!r}, want {want!r})")
    else:
        print(f"  ok    {name}")


def base_node(**extra) -> dict:
    """The fields a single node registration carries, for the request model."""
    return dict(name="gw-node", endpoint_url="https://203.0.113.5/v1",
                model_name="llama-70b", capability_text="a general model", **extra)


# --- 1. the registration model ---------------------------------------------

print("\na single node registers exactly as before")
legacy = NodeCreate(**base_node())
check("group_status defaults to null", legacy.group_status, None)
check("group_members_total defaults to null", legacy.group_members_total, None)
check("group_members_up defaults to null", legacy.group_members_up, None)

print("\na group coordinator can say what state it is in")
restarting = NodeCreate(**base_node(group_status="restarting",
                                    group_members_total=3, group_members_up=2))
check("restarting is accepted", restarting.group_status, "restarting")
check("member counts are kept", (restarting.group_members_total,
                                 restarting.group_members_up), (3, 2))
ready = NodeCreate(**base_node(group_status="ready",
                               group_members_total=3, group_members_up=3))
check("ready is accepted", ready.group_status, "ready")


def rejected_model(name: str, **fields) -> None:
    try:
        NodeCreate(**base_node(**fields))
    except ValidationError:
        check(f"rejects {name}", True, True)
        return
    check(f"rejects {name}", "accepted", "rejected")


rejected_model("a status the gateway does not know", group_status="degraded")
rejected_model("more members up than the group has", group_status="ready",
               group_members_total=2, group_members_up=3)
rejected_model("a negative member count", group_members_total=-1)
rejected_model("a negative members-up count", group_members_up=-1)

print("\nGET /nodes shows the group fields, and still not the endpoint")
public = NodePublicOut(
    id=uuid4(), name="gw-node", operator=None, model_name="llama-70b", region=None,
    cost_per_1k=0.0, avg_latency_ms=0, healthy=False, last_heartbeat=None,
    capability_text="a general model", group_status="restarting",
    group_members_total=3, group_members_up=2,
).model_dump()
check("group_status shown", public["group_status"], "restarting")
check("members total shown", public["group_members_total"], 3)
check("members up shown", public["group_members_up"], 2)
check("endpoint_url still absent", "endpoint_url" in public, False)


def public_row(**group) -> dict:
    """A row shaped the way `select *` returns it, for the row mapper."""
    from datetime import datetime, timezone
    return dict(
        id=uuid4(), name="gw-node", operator=None, endpoint_url="https://203.0.113.5/v1",
        model_name="llama-70b", region=None, cost_per_1k=0, avg_latency_ms=0,
        healthy=True, last_heartbeat=datetime(2026, 10, 20, tzinfo=timezone.utc),
        last_seen_healthy=None, capability_text="a general model",
        domain_tags=["general"], catalogue_id=None, node_token="t",
        **group,
    )


print("\nthe row mapper passes the group fields through, and nulls for singles")
mapped = registry._row_to_node_public_out(public_row(
    group_status="ready", group_members_total=4, group_members_up=4))
check("group_status mapped", mapped.group_status, "ready")
check("members up mapped", mapped.group_members_up, 4)
single = registry._row_to_node_public_out(public_row(
    group_status=None, group_members_total=None, group_members_up=None))
check("a single node maps to null group_status", single.group_status, None)
check("a single node maps to null members", single.group_members_total, None)


# --- 2. registration passes the group fields to the database ----------------

class _RecordingConn:
    """Answers the existence check with `existing` and records the write."""

    def __init__(self, existing: dict | None):
        self.existing = existing
        self.writes: list[tuple] = []

    async def fetchrow(self, sql, *args):
        if sql.lstrip().startswith("select node_token"):
            return self.existing
        self.writes.append(args)
        return public_row(
            group_status=args[14], group_members_total=args[15], group_members_up=args[16],
        )


class _Acquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *a):
        return False


class _Pool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _Acquire(self.conn)


async def no_validate(*a, **k):
    return None


async def register(conn: _RecordingConn, node: NodeCreate, token: str | None = None):
    with patch.object(registry, "validate_endpoint_url", no_validate), \
            patch.object(embedder, "embed", lambda text: [0.0]), \
            patch.object(registry.db, "pool", lambda: _Pool(conn)):
        return await registry.register_node(node, x_common_node_token=token,
                                            x_common_client=None)


print("\nregistration writes the group fields on a fresh name (insert path)")
conn = _RecordingConn(existing=None)
out = asyncio.run(register(conn, NodeCreate(**base_node(
    group_status="restarting", group_members_total=3, group_members_up=2))))
check("insert carries group_status", conn.writes[0][14], "restarting")
check("insert carries members total", conn.writes[0][15], 3)
check("insert carries members up", conn.writes[0][16], 2)
check("response carries group_status", out.group_status, "restarting")

print("\nregistration of a re-registered name keeps the group fields it sent")
existing = public_row()
existing["node_token"] = "stored-token"
conn = _RecordingConn(existing=existing)
asyncio.run(register(conn, NodeCreate(**base_node(
    group_status="ready", group_members_total=3, group_members_up=3)),
    token="stored-token"))
check("update carries group_status", conn.writes[0][14], "ready")
check("update carries members up", conn.writes[0][16], 3)

print("\na single node's registration sends nulls, so nothing is written over")
conn = _RecordingConn(existing=None)
asyncio.run(register(conn, NodeCreate(**base_node())))
check("single node insert: group_status null", conn.writes[0][14], None)
check("single node insert: members total null", conn.writes[0][15], None)
check("single node insert: members up null", conn.writes[0][16], None)


# --- 3. the health loop never marks a restarting group healthy --------------

REAL_ASYNC_CLIENT = httpx.AsyncClient
OK_MODEL_LIST = httpx.Response(200, json={"data": [{"id": "llama-70b"}]})


def probe_always_ok(request: httpx.Request) -> httpx.Response:
    # A coordinator whose runtime is down can still answer /models. That is the
    # case this guards, so the probe here is deliberately a success.
    return OK_MODEL_LIST


class _HealthConn:
    def __init__(self, rows, writes):
        self.rows = rows
        self.writes = writes

    async def fetch(self, sql, *args):
        await asyncio.sleep(0)
        return self.rows

    async def execute(self, sql, *args):
        await asyncio.sleep(0)
        self.writes.append(args)


def health_pass(rows: list[dict]) -> dict:
    """Run one pass over `rows`; returns {node id: the healthy value written}."""
    writes: list[tuple] = []
    conn = _HealthConn(rows, writes)

    async def go():
        with patch.object(health.db, "pool", lambda: _Pool(conn)), \
                patch.object(health.httpx, "AsyncClient",
                             lambda *a, **k: REAL_ASYNC_CLIENT(
                                 transport=httpx.MockTransport(probe_always_ok))):
            await health.run_health_checks_once()

    asyncio.run(go())
    # execute(sql, ok, id, endpoint_url): the first argument is the healthy value.
    return {args[1]: args[0] for args in writes}


def health_row(group_status):
    return dict(id=uuid4(), endpoint_url="https://203.0.113.5/v1", api_key_ref=None,
                worker_token="w" * 16, model_name="llama-70b", group_status=group_status)


print("\na restarting group is not healthy, even when its probe answers")
restarting_row = health_row("restarting")
written = health_pass([restarting_row])
check("restarting group written unhealthy", written[restarting_row["id"]], False)

print("\na ready group and a single node are healthy when their probe answers")
ready_row, single_row = health_row("ready"), health_row(None)
written = health_pass([ready_row, single_row])
check("ready group written healthy", written[ready_row["id"]], True)
check("single node written healthy", written[single_row["id"]], True)


# --- 4. routing: a request whose only candidate is restarting ---------------

class FakeRequest:
    def __init__(self, body: dict, headers: dict | None = None):
        self._body = body
        self.headers = headers or {}
        self.client = None

    async def json(self):
        return self._body


class FakeResponse:
    """Enough of an httpx.Response for the non-streaming relay."""
    status_code = 200
    headers = {"content-type": "application/json"}

    def __init__(self):
        self.extensions = {"_client": self}

    async def aread(self) -> bytes:
        return b'{"ok": true}'

    async def aclose(self) -> None:
        pass


async def no_op(*args, **kwargs):
    return None


async def fake_embed(text):
    return [0.1, 0.2, 0.3, 0.4]


def scored(name: str, score: float) -> ScoredNode:
    n = {"id": name, "name": name, "model_name": "llama-70b", "domain_tags": ["general"],
         "client": None}
    return ScoredNode(node=n, score=score, sim=score, cost_term=0.0, lat_term=0.0,
                      region_term=0.0, tag_term=0.0)


BODY = {"messages": [{"role": "user", "content": "hello"}]}


def call(healthy: list[dict], restarting: list[dict], *, headers=None,
         scored_nodes=None):
    """Run one request through the real route. Returns (status, headers, detail)
    for a refusal, or (200, response headers, None) for an answer."""
    async def fetch_healthy():
        return healthy

    async def fetch_restarting():
        return restarting

    async def forward(candidate, body, stream):
        return FakeResponse()

    async def go():
        try:
            response = await gateway.chat_completions(FakeRequest(BODY, headers))
            return response.status_code, dict(response.headers), None
        except HTTPException as exc:
            return exc.status_code, dict(exc.headers or {}), exc.detail

    with patch.object(settings, "require_contribution", False), \
            patch.object(ratelimit, "check_rate_limit", lambda request: 0.0), \
            patch.object(gateway, "_fetch_healthy_nodes", fetch_healthy), \
            patch.object(gateway, "_fetch_restarting_groups", fetch_restarting), \
            patch.object(embedder, "embed_async", fake_embed), \
            patch.object(gateway, "best_matched_domain", lambda *a, **k: None), \
            patch.object(gateway, "score_nodes",
                         lambda *a, **k: scored_nodes if scored_nodes is not None else []), \
            patch.object(compose, "plan_panel",
                         lambda *a, **k: PanelPlan(compose=False, reason="test")), \
            patch.object(upstream, "forward", forward), \
            patch.object(gateway, "_record_decision", no_op), \
            patch.object(gateway, "_update_latency", no_op), \
            patch.object(metrics, "record_request", lambda **k: None):
        return asyncio.run(go())


GROUP = {"id": "g1", "name": "dist-70b", "group_members_total": 3, "group_members_up": 2}

print("\nno healthy node, and the group is restarting: 503 with Retry-After")
status, headers, detail = call([], [GROUP], headers={})
check("status is 503", status, 503)
check("Retry-After is the configured hint",
      headers.get("Retry-After"), str(settings.group_restart_retry_after_seconds))
check("detail names the group status", "group_status=restarting" in detail, True)
check("detail names the group", "dist-70b" in detail, True)
check("detail says how many members are up", "2 of 3 members up" in detail, True)

print("\nthe Retry-After value comes from settings")
original = settings.group_restart_retry_after_seconds
try:
    settings.group_restart_retry_after_seconds = 7
    status, headers, _ = call([], [GROUP])
    check("Retry-After follows the setting", headers.get("Retry-After"), "7")
finally:
    settings.group_restart_retry_after_seconds = original

print("\nno healthy node and nothing restarting: the generic 503, no Retry-After")
status, headers, detail = call([], [])
check("status is 503", status, 503)
check("generic detail", detail, "no healthy nodes available")
check("no Retry-After on the generic 503", "Retry-After" in headers, False)

print("\na healthy node elsewhere still answers; the group is not the candidate")
healthy_node = scored("gen-node", 0.9)
status, headers, _ = call([healthy_node.node], [GROUP],
                          scored_nodes=[healthy_node])
check("request is answered, not refused", status, 200)
check("answered by the healthy node", headers.get("x-common-node"), "gen-node")

# Healthy nodes are present in these two, so the forced-node branch is what
# decides the answer, not the no-healthy-nodes check above it.
print("\na forced node name that is restarting is a 503, not a 404")
status, headers, detail = call([healthy_node.node], [GROUP],
                               headers={"X-Common-Node": "dist-70b"})
check("forced restarting group is 503", status, 503)
check("forced restarting group carries Retry-After", "Retry-After" in headers, True)
check("forced restarting detail names the group status",
      "group_status=restarting" in detail, True)

print("\na forced node name that is unknown is still a 404")
status, headers, detail = call([healthy_node.node], [GROUP],
                               headers={"X-Common-Node": "nobody"})
check("unknown forced node is 404", status, 404)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all distributed-group gateway tests passed")
