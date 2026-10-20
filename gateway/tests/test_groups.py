"""Distributed model groups on the gateway: registration, reporting, health, 503.

A group is one node whose coordinator reports its own state (migrations/015).
What is pinned here:

1. Registration accepts a group, and refuses one that cannot be true: a status
   without counts, up above total, no tags, or the generalist tag "general".
2. The coordinator's report is authenticated by the node's own token, and a
   'restarting' report takes the node out of routing at once.
3. The health pass keeps a node unroutable while its restarting report is fresh,
   even when /models answers, and lets it back once the report is stale or the
   coordinator says ready.
4. A request that finds no healthy node, while a group is restarting, gets 503
   with Retry-After and a detail that names the group.

No database, no network beyond an in-process transport, no embedding model.
The database is a fake that records each statement, so the SQL argument order
and the values that reach it are what is checked.
"""
import asyncio
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from app import db, gateway, health, registry  # noqa: E402
from app.config import settings  # noqa: E402
from app.groups import RESTARTING, is_restarting, restarting_detail  # noqa: E402
from app.models import GroupReport, NodeCreate, NodePublicOut  # noqa: E402

FAILURES = []


def check(name: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL  {name}  (got {got!r}, want {want!r})")
    else:
        print(f"  ok    {name}")


def raises(name: str, fn, exc=Exception) -> None:
    try:
        fn()
    except exc:
        check(name, True, True)
        return
    except Exception as e:  # noqa: BLE001
        check(name, f"{type(e).__name__}: {e}", exc.__name__)
        return
    check(name, "no error", exc.__name__)


NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
TOKEN = "node-token-0123456789abcdef"
NODE_ID = "11111111-1111-1111-1111-111111111111"


# --- a fake database that records what reaches it ---------------------------

class FakeConn:
    def __init__(self, pool):
        self.pool = pool

    async def fetch(self, sql, *args):
        self.pool.log.append(("fetch", sql, args))
        await asyncio.sleep(0)
        return self.pool.respond("fetch", sql, args)

    async def fetchrow(self, sql, *args):
        self.pool.log.append(("fetchrow", sql, args))
        await asyncio.sleep(0)
        return self.pool.respond("fetchrow", sql, args)

    async def fetchval(self, sql, *args):
        self.pool.log.append(("fetchval", sql, args))
        await asyncio.sleep(0)
        return self.pool.respond("fetchval", sql, args)

    async def execute(self, sql, *args):
        self.pool.log.append(("execute", sql, args))
        await asyncio.sleep(0)
        return self.pool.respond("execute", sql, args)


class FakePool:
    def __init__(self, respond):
        self.respond = respond
        self.log = []

    def acquire(self):
        pool = self

        class _Acquire:
            async def __aenter__(self):
                return FakeConn(pool)

            async def __aexit__(self, *a):
                return False

        return _Acquire()


def node_row(**over) -> dict:
    """A row shaped like `select *` from nodes, with every column the converters read."""
    row = dict(
        id=NODE_ID, name="group-a", operator="op", endpoint_url="https://10.0.0.5/v1",
        model_name="qwen-test:7b", region=None, cost_per_1k=0, avg_latency_ms=0,
        healthy=False, last_heartbeat=NOW, last_seen_healthy=None,
        capability_text="a model split across machines", domain_tags=["coding"],
        catalogue_id=None, node_token=TOKEN, worker_token="worker-token-0123456789", api_key_ref=None,
        paused=False, client=None, created_at=NOW, group_status=None,
        group_members_total=None, group_members_up=None, group_reported_at=None,
    )
    row.update(over)
    return row


# --- 1. the migration --------------------------------------------------------

print("migration 015 adds nullable group columns and no groups table")
MIGRATIONS = Path(__file__).resolve().parent.parent / "migrations"
names = sorted(p.name for p in MIGRATIONS.glob("*.sql"))
check("015 is the last migration", names[-1], "015_node_group.sql")
sql = (MIGRATIONS / "015_node_group.sql").read_text().lower()
for col in ("group_status", "group_members_total", "group_members_up", "group_reported_at"):
    check(f"adds {col} if not exists",
          f"alter table nodes add column if not exists {col}" in sql, True)
check("no groups table", "create table" in sql, False)
check("status constrained to ready|restarting",
      "check (group_status in ('ready', 'restarting'))" in sql, True)


# --- 2. registration: what a group may claim -------------------------------

print("\nregistration accepts a fully described group, under its model's tag")
ok = NodeCreate(name="g", endpoint_url="https://10.0.0.5/v1", model_name="m",
                capability_text="c", domain_tags=["coding"],
                group_status="ready", group_members_total=3, group_members_up=3)
check("ready group accepted", ok.group_status, "ready")
check("restarting group accepted", NodeCreate(
    name="g", endpoint_url="https://10.0.0.5/v1", model_name="m", capability_text="c",
    domain_tags=["coding"], group_status="restarting",
    group_members_total=3, group_members_up=2).group_members_up, 2)
plain = NodeCreate(name="s", endpoint_url="https://10.0.0.5/v1", model_name="m",
                   capability_text="c")
check("a single-machine node sends no group fields", plain.group_status, None)

print("\nregistration refuses a group that cannot be true")
base = dict(name="g", endpoint_url="https://10.0.0.5/v1", model_name="m", capability_text="c")
raises("group tagged general is refused",
       lambda: NodeCreate(**base, domain_tags=["general"], group_status="ready",
                          group_members_total=2, group_members_up=2), ValidationError)
raises("general is refused alongside other tags",
       lambda: NodeCreate(**base, domain_tags=["coding", "General"], group_status="ready",
                          group_members_total=2, group_members_up=2), ValidationError)
raises("group with no tags is refused",
       lambda: NodeCreate(**base, group_status="ready",
                          group_members_total=2, group_members_up=2), ValidationError)
raises("up above total is refused",
       lambda: NodeCreate(**base, domain_tags=["coding"], group_status="ready",
                          group_members_total=2, group_members_up=3), ValidationError)
raises("status without counts is refused",
       lambda: NodeCreate(**base, domain_tags=["coding"], group_status="ready"), ValidationError)
raises("counts without status is refused",
       lambda: NodeCreate(**base, domain_tags=["coding"],
                          group_members_total=2, group_members_up=2), ValidationError)
raises("a 'degraded' status is refused",
       lambda: NodeCreate(**base, domain_tags=["coding"], group_status="degraded",
                          group_members_total=2, group_members_up=1), ValidationError)


# --- 3. the coordinator's report --------------------------------------------

print("\nthe report is validated the same way")
check("GroupReport accepts a restarting report",
      GroupReport(group_status="restarting", group_members_total=2,
                  group_members_up=1).group_status, "restarting")
raises("GroupReport refuses up above total",
       lambda: GroupReport(group_status="ready", group_members_total=1,
                           group_members_up=2), ValidationError)
raises("GroupReport refuses zero members",
       lambda: GroupReport(group_status="ready", group_members_total=0,
                           group_members_up=0), ValidationError)

print("\nthe public view carries the group fields and still no endpoint")
check("group_status on NodePublicOut", "group_status" in NodePublicOut.model_fields, True)
check("member counts on NodePublicOut",
      "group_members_total" in NodePublicOut.model_fields
      and "group_members_up" in NodePublicOut.model_fields, True)
check("endpoint_url still absent from NodePublicOut",
      "endpoint_url" in NodePublicOut.model_fields, False)


def report_call(token, report, row=None):
    """Run PUT /nodes/{id}/group against the fake, return (result-or-exc, pool)."""
    row = row if row is not None else node_row(group_status=report.group_status,
                                               group_members_total=report.group_members_total,
                                               group_members_up=report.group_members_up)

    def respond(kind, sql, args):
        if kind == "fetchval":
            return TOKEN
        if kind == "fetchrow":
            return row
        return None

    pool = FakePool(respond)
    with mock.patch.object(db, "pool", lambda: pool):
        try:
            out = asyncio.run(registry.report_group(
                uuid.UUID(NODE_ID), report, x_common_node_token=token))
        except HTTPException as e:
            out = e
    return out, pool


print("\nreporting needs the node's own token")
rep = GroupReport(group_status="ready", group_members_total=2, group_members_up=2)
out, pool = report_call(None, rep)
check("no token -> 404", isinstance(out, HTTPException) and out.status_code, 404)
check("no write without a token", [c for c in pool.log if c[0] == "fetchrow"], [])
out, pool = report_call("wrong-token", rep)
check("wrong token -> 404", isinstance(out, HTTPException) and out.status_code, 404)
check("no write with a wrong token", [c for c in pool.log if c[0] == "fetchrow"], [])
out, pool = report_call(TOKEN, rep)
check("right token -> report stored and returned", out.group_status, "ready")
check("returned counts are the reported ones", (out.group_members_total, out.group_members_up), (2, 2))

print("\na restarting report takes the node out of routing at once")
rr = GroupReport(group_status="restarting", group_members_total=2, group_members_up=1)
out, pool = report_call(TOKEN, rr, row=node_row(group_status="restarting",
                                                group_members_total=2, group_members_up=1))
writes = [c for c in pool.log if c[0] == "fetchrow"]
check("one update statement", len(writes), 1)
check("sets healthy false when restarting",
      "then false" in writes[0][1].replace("\n", " ") if writes else False, True)
check("restarting value reaches the statement",
      writes[0][2][1:5] if writes else None, ("restarting", 2, 1, RESTARTING))
check("the restart is recorded as a timestamped report",
      "group_reported_at = now()" in (writes[0][1] if writes else ""), True)
check("the returned node is not healthy yet", out.healthy, False)


# --- 4. the restart window ---------------------------------------------------

print("\na restarting report counts only while fresh")
ttl = 1800.0
check("fresh restarting is restarting",
      is_restarting("restarting", NOW - timedelta(seconds=60), NOW, ttl), True)
check("stale restarting stops counting",
      is_restarting("restarting", NOW - timedelta(seconds=ttl + 1), NOW, ttl), False)
check("ready is never restarting",
      is_restarting("ready", NOW, NOW, ttl), False)
check("no status is never restarting",
      is_restarting(None, NOW, NOW, ttl), False)
check("an undated restarting report does not count",
      is_restarting("restarting", None, NOW, ttl), False)

print("\nthe health pass keeps a restarting group unroutable, even when /models answers")
REAL_RESPONSE = httpx.Response
REAL_TRANSPORT = httpx.AsyncBaseTransport
REAL_CLIENT = httpx.AsyncClient


def health_pass(rows):
    """One real health pass over `rows`; every /models probe answers 200 with the model."""
    class Transport(REAL_TRANSPORT):
        async def handle_async_request(self, request):
            return REAL_RESPONSE(200, json={"data": [{"id": "qwen-test:7b"}]})

    pool = FakePool(lambda kind, sql, args: rows if kind == "fetch" else None)

    def factory(*a, **k):
        return REAL_CLIENT(transport=Transport())

    async def go():
        with mock.patch.object(db, "pool", lambda: pool), \
                mock.patch.object(health.httpx, "AsyncClient", factory):
            await health.run_health_checks_once()

    asyncio.run(go())
    updates = [c for c in pool.log if c[0] == "execute" and "update nodes" in c[1]]
    return [c[2][0] for c in updates]  # the first argument: healthy


def at(seconds_ago):
    return datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)


check("fresh restarting group stays unhealthy",
      health_pass([node_row(group_status="restarting", group_reported_at=at(60))]), [False])
check("stale restarting group is health-checked again",
      health_pass([node_row(group_status="restarting",
                            group_reported_at=at(ttl + 600))]), [True])
check("ready group passing its probe is healthy",
      health_pass([node_row(group_status="ready", group_reported_at=at(60))]), [True])
check("a single-machine node is unaffected",
      health_pass([node_row(group_status=None)]), [True])


# --- 5. registration writes the group fields --------------------------------

print("\nregistration writes the group fields, and a group re-registering as single clears them")
async def _no_dns(url, **kwargs):
    return None


def register(node: NodeCreate, existing: bool):
    def respond(kind, sql, args):
        if kind == "fetchrow" and "select node_token" in sql:
            return {"node_token": TOKEN} if existing else None
        if kind == "fetchrow":
            return node_row(group_status=node.group_status,
                            group_members_total=node.group_members_total,
                            group_members_up=node.group_members_up,
                            domain_tags=node.domain_tags)
        return None

    pool = FakePool(respond)
    with mock.patch.object(db, "pool", lambda: pool), \
            mock.patch.object(registry, "validate_endpoint_url", _no_dns), \
            mock.patch.object(registry.embedder, "embed", lambda text: [0.0] * settings.embed_dim):
        out = asyncio.run(registry.register_node(
            node, x_common_node_token=TOKEN if existing else None, x_common_client=None))
    writes = [c for c in pool.log if c[0] == "fetchrow" and "select node_token" not in c[1]]
    return out, writes


group = NodeCreate(name="group-a", endpoint_url="https://10.0.0.5/v1", model_name="qwen-test:7b",
                   capability_text="c", domain_tags=["coding"],
                   group_status="ready", group_members_total=3, group_members_up=3)
out, writes = register(group, existing=False)
check("fresh group registration writes the statement", len(writes), 1)
# The group values are the last three bound arguments of both statements.
check("group values reach the insert", writes[0][2][-3:], ("ready", 3, 3))
check("the response carries the group", (out.group_status, out.group_members_total), ("ready", 3))

out, writes = register(group, existing=True)
check("re-registration writes the group too", writes[0][2][-3:], ("ready", 3, 3))

single = NodeCreate(name="group-a", endpoint_url="https://10.0.0.5/v1", model_name="qwen-test:7b",
                    capability_text="c", domain_tags=["coding"])
out, writes = register(single, existing=True)
check("re-registering without group fields sends null, not the old group",
      writes[0][2][-3:], (None, None, None))


# --- 6. the 503 for a request with no healthy node ---------------------------

print("\nno healthy node + a restarting group -> 503 with Retry-After, naming the group")
restarting_row = node_row(name="group-a", group_status="restarting", group_members_total=3,
                          group_members_up=2, group_reported_at=at(30))


def detail_for(rows):
    return restarting_detail(rows, NOW)


text = restarting_detail([dict(restarting_row, group_reported_at=NOW - timedelta(seconds=30))], NOW)
check("detail names the group", "'group-a'" in (text or ""), True)
check("detail names the model", "qwen-test:7b" in (text or ""), True)
check("detail gives the member count", "2 of 3 members up" in (text or ""), True)
check("detail says when to retry",
      f"Retry after {settings.group_retry_after_seconds}s" in (text or ""), True)
check("no restarting group -> no detail", detail_for([]), None)
check("a stale restarting group -> no detail",
      restarting_detail([dict(restarting_row, group_reported_at=NOW - timedelta(days=1))], NOW), None)


class FakeRequest:
    headers: dict = {}
    client = None

    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def chat_with(rows):
    """POST /v1/chat/completions with no healthy node in the database; return the exception raised."""
    def respond(kind, sql, args):
        if kind == "fetch" and "healthy = true" in sql:
            return []
        if kind == "fetch" and "group_status = 'restarting'" in sql:
            return rows
        return None

    pool = FakePool(respond)
    saved = (settings.require_contribution, settings.rate_limit_requests_per_minute)
    settings.require_contribution = False
    settings.rate_limit_requests_per_minute = 0
    try:
        with mock.patch.object(db, "pool", lambda: pool):
            asyncio.run(gateway.chat_completions(FakeRequest({"messages": [
                {"role": "user", "content": "hello"}]})))
    except HTTPException as e:
        return e
    finally:
        settings.require_contribution, settings.rate_limit_requests_per_minute = saved
    return None


e = chat_with([restarting_row])
check("a restarting group answers 503", e.status_code if e else None, 503)
check("503 carries Retry-After", (e.headers or {}).get("Retry-After") if e else None,
      str(settings.group_retry_after_seconds))
check("503 detail names the group", "restarting" in (e.detail if e else ""), True)

e = chat_with([])
check("with no group, the generic 503 is unchanged",
      (e.status_code, e.detail) if e else None, (503, "no healthy nodes available"))
check("generic 503 has no Retry-After", (e.headers if e else None), None)


print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all group tests passed")
