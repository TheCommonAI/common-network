"""The operators view: what /dashboard deliberately doesn't show.

/dashboard is public and stays that way — legibility is the pitch, and a
visitor should be able to see which machines answer their questions. This is
the other half: the things an operator needs and a visitor shouldn't get,
namely which nodes are failing and why, which requests errored, and how close
clients are to the rate limit.

Gated on ADMIN_TOKEN. Unset (the default) and every route here 404s, so a
deployment that never configures it cannot leak operational detail to someone
who guesses the path. 404 rather than 401 is deliberate: an unconfigured
gateway should look like one without the feature, not like one hiding a login.

Nothing here exposes request text, embeddings, or node tokens. An operator
needs to know a node is failing, not what anyone asked it.
"""
from __future__ import annotations

import secrets
import time
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app import db, ratelimit
from app.config import settings

router = APIRouter()

ADMIN_PATH = Path(__file__).parent / "static" / "admin.html"


def _hours(value) -> float | None:
    """Postgres avg()/max() over an interval come back as Decimal; JSON wants a
    float. One decimal place is the honest precision for "how long ago did this
    machine register"."""
    return round(float(value), 1) if value is not None else None


def shape_clients(rows) -> list[dict]:
    """Rows from the per-client query -> the JSON the operators page renders.

    Pulled out of the route so it can be pinned without a database, the same way
    test_client.py pins the client-resolution expression rather than calling the
    endpoint.

    `client` passes through as None rather than being pre-labelled here. Null
    means "did not say" -- a node registered before this column existed, or a
    client that predates sending the string -- and migration 008 is explicit
    that this is a different fact from any string we could invent. The label
    belongs in the one place that has to render it, not in the data.
    """
    return [
        {
            "client": r["client"],
            "nodes": r["nodes"],
            "healthy": r["healthy"],
            "legacy": r["legacy"],
            "first_seen": r["first_seen"].isoformat() if r["first_seen"] else None,
            "last_seen": r["last_seen"].isoformat() if r["last_seen"] else None,
            "avg_tenure_hours": _hours(r["avg_tenure_hours"]),
            "max_tenure_hours": _hours(r["max_tenure_hours"]),
        }
        for r in rows
    ]


def _require_admin(request: Request) -> None:
    """404 when unconfigured, 401 when the password is wrong.

    The password may arrive as a header (curl, scripts) or a query parameter
    (the browser opening the page). A query parameter puts it in the URL bar
    and in any proxy log — acceptable for a page an operator opens on their
    own machine, and the alternative is a login form with session cookies,
    which is a lot of machinery for a status page.
    """
    if not settings.admin_token:
        raise HTTPException(status_code=404, detail="Not Found")

    supplied = request.headers.get("x-common-admin-token") or \
        request.query_params.get("token") or ""
    if not secrets.compare_digest(supplied, settings.admin_token):
        raise HTTPException(status_code=401, detail="bad or missing admin token")


@router.get("/admin")
async def admin_page(request: Request):
    _require_admin(request)
    return FileResponse(ADMIN_PATH)


@router.delete("/admin/nodes/{node_id}")
async def admin_delete_node(node_id: UUID, request: Request):
    """Remove a node row the operator has no other way to remove.

    `DELETE /nodes/{id}` is the contributor's own exit and requires that
    node's X-Common-Node-Token. That is right -- a stranger must never be able
    to deregister someone else's machine by reading its name off `GET /nodes`
    -- but it leaves the gateway operator unable to clear a row whose token
    nobody holds any more: a lab machine reimaged at the end of term, a node
    registered before tokens existed, a laptop that was lent for an afternoon.
    Those rows stay in the registry forever and every count of "who is on the
    network" includes them.

    This is the one deliberately operator-only door, which is why it is here
    behind ADMIN_TOKEN and not on the public registry router.

    Safe for history: `decisions.chosen_node` is ON DELETE SET NULL (migration
    002), so past requests survive with the node reading "(deregistered)"
    rather than the decision vanishing with it.

    One id per call, and the name is echoed back, so removing the wrong
    machine requires getting a uuid wrong rather than a filter wrong. There is
    no bulk or pattern delete on purpose.
    """
    _require_admin(request)

    async with db.pool().acquire() as conn:
        row = await conn.fetchrow(
            "delete from nodes where id = $1 returning name, healthy", node_id,
        )
    if row is None:
        raise HTTPException(status_code=404, detail=f"no node with id {node_id}")
    return {"deleted": str(node_id), "name": row["name"], "was_healthy": row["healthy"]}


@router.get("/admin/state")
async def admin_state(request: Request):
    """Everything the page renders, in one round trip."""
    _require_admin(request)

    async with db.pool().acquire() as conn:
        nodes = await conn.fetch(
            """
            select id, name, operator, model_name, endpoint_url, region,
                   healthy, can_aggregate, domain_tags, avg_latency_ms,
                   last_heartbeat, last_seen_healthy, created_at,
                   (node_token is not null) as has_token, client
            from nodes
            order by healthy asc, name asc
            """
        )
        # Per-node reliability over the recent window. Left join so a node that
        # has never been chosen still appears, with zeroes rather than absent.
        reliability = await conn.fetch(
            """
            select n.id,
                   count(d.id)                                  as requests,
                   count(d.id) filter (where d.ok is false)      as failures,
                   avg(d.latency_ms) filter (where d.ok)         as avg_latency_ms
            from nodes n
            left join decisions d
              on d.chosen_node = n.id
             and d.created_at > now() - interval '24 hours'
            group by n.id
            """
        )
        totals = await conn.fetchrow(
            """
            select count(*)                                as requests,
                   count(*) filter (where ok is false)     as failures,
                   count(*) filter (where topology = 'panel')    as panels,
                   count(*) filter (where topology = 'degraded') as degraded,
                   coalesce(sum(checks_run), 0)            as checks_run,
                   coalesce(sum(checks_failed), 0)         as checks_failed,
                   coalesce(sum(disagreements), 0)         as disagreements
            from decisions
            where created_at > now() - interval '24 hours'
            """
        )
        recent_failures = await conn.fetch(
            """
            select d.created_at, d.latency_ms, d.topology, n.name as node
            from decisions d
            left join nodes n on n.id = d.chosen_node
            where d.ok is false
            order by d.created_at desc
            limit 20
            """
        )
        # Who is on the network, by the program that registered them.
        #
        # The app and the CLI deliberately share the gateway, the endpoints and
        # one ~/.common-network/identity.json, so `client` is the only thing
        # that separates them (migration 008). It is self-reported and trivially
        # forgeable -- a usage statistic, never an authorisation input, nothing
        # branches on it -- which is why this lives here and not on the public
        # /dashboard: it is a number about our own distribution, not a claim
        # about the network a visitor should act on.
        #
        # Tenure is computed over nodes still registered, because
        # `DELETE /nodes/{id}` removes the row: a machine that has left takes its
        # history with it. So this answers "how long have the app's current
        # contributors been here", and NOT "how long did the app's contributors
        # stay" -- the second needs an exit record the registry does not keep.
        client_rows = await conn.fetch(
            """
            select client,
                   count(*)                                   as nodes,
                   count(*) filter (where healthy)            as healthy,
                   count(*) filter (where node_token is null)  as legacy,
                   min(created_at)                            as first_seen,
                   max(coalesce(last_seen_healthy, last_heartbeat)) as last_seen,
                   avg(extract(epoch from now() - created_at)) / 3600 as avg_tenure_hours,
                   max(extract(epoch from now() - created_at)) / 3600 as max_tenure_hours
            from nodes
            group by client
            order by nodes desc, client asc nulls last
            """
        )

    rel = {r["id"]: r for r in reliability}
    node_rows = []
    for n in nodes:
        r = rel.get(n["id"])
        requests = (r["requests"] if r else 0) or 0
        failures = (r["failures"] if r else 0) or 0
        node_rows.append({
            "name": n["name"],
            "operator": n["operator"],
            "model": n["model_name"],
            # Operators need to see where a node actually points -- a wrong or
            # stale endpoint is the single most common cause of a node that
            # registered fine and never answers.
            "endpoint": n["endpoint_url"],
            "region": n["region"],
            "healthy": n["healthy"],
            "can_aggregate": n["can_aggregate"],
            "domain_tags": list(n["domain_tags"] or []),
            "avg_latency_ms": n["avg_latency_ms"],
            "last_heartbeat": n["last_heartbeat"].isoformat() if n["last_heartbeat"] else None,
            # Null here is the signal for pruning: this gateway has never seen
            # the machine answer, so nothing is lost by removing the row.
            "last_seen_healthy": n["last_seen_healthy"].isoformat() if n["last_seen_healthy"] else None,
            "created_at": n["created_at"].isoformat() if n["created_at"] else None,
            # Legacy nodes registered before tokens existed cannot be
            # re-registered safely by their owner and cannot grant access.
            "has_token": n["has_token"],
            # Which program registered this node. Null is "did not say" -- see
            # shape_clients. Shown per node so an operator can spot the odd one
            # out (a machine re-registered by a different path than the rest)
            # rather than only in aggregate.
            "client": n["client"],
            "requests_24h": requests,
            "failures_24h": failures,
            "failure_rate": (failures / requests) if requests else None,
            "measured_latency_ms": int(r["avg_latency_ms"]) if r and r["avg_latency_ms"] else None,
        })

    # Rate-limiter state is in-process, so this is this worker's view. Said
    # plainly on the page rather than presented as network-wide truth.
    now = time.monotonic()
    rate = settings.rate_limit_requests_per_minute
    buckets = []
    if rate > 0:
        for key, (tokens, last) in list(ratelimit._buckets.items()):
            refilled = min(float(rate), tokens + max(0.0, now - last) * (rate / 60.0))
            buckets.append({
                "client": key,
                "tokens_left": round(refilled, 1),
                "throttled": refilled < 1.0,
                "idle_seconds": round(now - last, 1),
            })
        buckets.sort(key=lambda b: b["tokens_left"])

    return {
        "version": "0.1.3",
        "release": "The Common Network Alpha",
        "config": {
            "require_contribution": settings.require_contribution,
            "rate_limit_per_minute": rate,
            "compose_mode": settings.compose_mode,
            "allow_loopback_node_endpoints": settings.allow_loopback_node_endpoints,
            "health_check_interval_seconds": settings.health_check_interval_seconds,
            "source_url": settings.source_url,
        },
        "totals_24h": {
            "requests": totals["requests"] or 0,
            "failures": totals["failures"] or 0,
            "panels": totals["panels"] or 0,
            "degraded": totals["degraded"] or 0,
            "checks_run": totals["checks_run"] or 0,
            "checks_failed": totals["checks_failed"] or 0,
            "disagreements": totals["disagreements"] or 0,
        },
        "nodes": node_rows,
        # Aggregate by registering program. See the query above for why tenure
        # only covers nodes still registered.
        "clients": shape_clients(client_rows),
        "recent_failures": [
            {
                "at": f["created_at"].isoformat() if f["created_at"] else None,
                "node": f["node"] or "(deregistered)",
                "latency_ms": f["latency_ms"],
                "topology": f["topology"],
            }
            for f in recent_failures
        ],
        "rate_limit_buckets": buckets,
    }
