import asyncio
import json
import logging

import httpx
from fastapi import HTTPException

from app import db
from app.config import settings
from app.mobile_transport import HEARTBEAT_LEASE_SECONDS, manager
from app.registry import validate_endpoint_url
from app.upstream import auth_headers

logger = logging.getLogger(__name__)

# How many nodes are checked at once.
#
# Bounded rather than unbounded, because endpoint_url is chosen by whoever
# registered: an unbounded pass would mean one outbound connection and one
# executor thread per registered row, so the number of hosts this gateway
# talks to simultaneously is something a stranger could raise by registering
# in a loop. Eight keeps a pass to a few seconds while leaving the fan-out a
# decision the gateway makes rather than one the registry makes for it.
HEALTH_CHECK_CONCURRENCY = 8


async def _check_one(client: httpx.AsyncClient, node: dict) -> bool:
    url = node["endpoint_url"].rstrip("/") + "/models"
    # Same credential rules as the forwarding path, via the same helper -- a
    # worker rejects an unauthenticated GET /v1/models, so health checks must
    # carry the worker token or every worker-backed node reads as down.
    #
    # The api_key_ref branch matters more here than when forwarding: this runs
    # every 30s against every registered node, so an unrestricted reference
    # would hand a credential to a hostile endpoint without any user ever
    # sending it a request.
    headers = auth_headers(node)
    headers.pop("Content-Type", None)  # a GET has no body to describe
    async def probe():
        # Re-validate the endpoint every pass, not just at registration. This
        # closes the DNS-rebinding window: a hostname that resolved somewhere
        # legitimate when it registered and resolves at a metadata service now
        # stops being health-checked green — the gateway never fetches it
        # again, because an unhealthy node is never routed to.
        #
        # Two things about *where* this call sits, both load-bearing:
        #
        # * Inside the wait_for below, so `health_check_timeout_seconds`
        #   actually bounds it. It used to sit outside, which meant the one
        #   step that could block had no cap at all.
        # * wait_seconds=0 — one attempt, no retry. The retry burst exists to
        #   tolerate DNS that has not propagated yet at *registration*; a
        #   periodic pass is not that moment, and paying for it per node per
        #   pass is what turned this loop into a stall. A node whose hostname
        #   does not resolve right now is a node that cannot be routed to
        #   right now, which is exactly what this pass is here to record.
        await validate_endpoint_url(node["endpoint_url"], wait_seconds=0)
        async with client.stream('GET', url, headers=headers, timeout=settings.health_check_timeout_seconds) as resp:
            if not 200 <= resp.status_code < 300:
                return False
            content = bytearray()
            async for chunk in resp.aiter_bytes():
                content.extend(chunk)
                if len(content) > 65536:
                    return False
            try:
                data = json.loads(content).get('data')
                return isinstance(data, list) and any(
                    isinstance(m, dict) and isinstance(m.get('id'), str)
                    and (not node.get('model_name') or m['id'] == node['model_name'])
                    for m in data)
            except (ValueError, AttributeError, UnicodeError):
                return False
    try:
        return await asyncio.wait_for(probe(), settings.health_check_timeout_seconds)
    except (HTTPException, httpx.HTTPError, asyncio.TimeoutError):
        return False



async def run_health_checks_once() -> None:
    async with db.pool().acquire() as conn:
        rows = await conn.fetch(
            "select id, endpoint_url, api_key_ref, worker_token, model_name, group_status "
            "from nodes where not paused and transport = 'http'"
        )

    async with httpx.AsyncClient() as client:
        # Concurrently, not one after another. A pass costs the sum of its
        # checks when run in series: at the 5s cap, 23 registered rows is over
        # a minute -- and `last_heartbeat`, which only this loop writes, is
        # then aged by however many dead nodes happen to sort above a live one
        # rather than by the 30s the interval promises. Concurrently a pass
        # costs the slowest single check, so what `last_heartbeat` says keeps
        # meaning what the field above claims it means.
        #
        # return_exceptions=True because one node raising must not abandon the
        # pass or leave the rest unrecorded. An unexpected error is treated as
        # not-healthy, matching what this returns for every failure mode it
        # does handle -- and it is recomputed next pass, so a transient one
        # cannot strand a node.
        gate = asyncio.Semaphore(HEALTH_CHECK_CONCURRENCY)

        async def check(row) -> bool:
            async with gate:
                return await _check_one(client, dict(row))

        results = await asyncio.gather(
            *(check(row) for row in rows), return_exceptions=True
        )

    async with db.pool().acquire() as conn:
        for row, ok in zip(rows, results):
            if not isinstance(ok, bool):
                logger.warning(
                    "health check for node %s raised: %r", row["id"], ok
                )
                ok = False
            # A distributed group that says it is restarting is not healthy,
            # whatever the probe says. The probe is /v1/models, and a coordinator
            # whose runtime is down may still answer it; without this a group
            # would be routed to in the middle of a restart. The group's own
            # coordinator clears `restarting` by re-registering as ready.
            if row.get("group_status") == "restarting":
                ok = False
            await conn.execute(
                # last_heartbeat records that we asked; last_seen_healthy
                # records that the node answered, and only moves when it
                # did. Keeping both is what lets a peer list tell a laptop
                # that shut its lid apart from a machine that has left --
                # by last_heartbeat alone every row ever registered looks
                # equally fresh, because this loop touches them all.
                """
                update nodes
                   set healthy = $1,
                       last_heartbeat = now(),
                       last_seen_healthy = case when $1 then now()
                                                else last_seen_healthy end
                 where id = $2 and not paused and endpoint_url = $3
                """,
                ok, row["id"], row["endpoint_url"],
            )

    await manager.expire_stale()
    async with db.pool().acquire() as conn:
        rows = await conn.fetch(
            "select id from nodes where not paused and transport = 'android_ws'"
        )
        for row in rows:
            live = manager.is_live(row["id"])
            await conn.execute(
                """
                update nodes
                   set healthy = $1,
                       last_heartbeat = now(),
                       last_seen_healthy = case when $1 then now()
                                                else last_seen_healthy end
                 where id = $2 and not paused and transport = 'android_ws'
                """,
                live, row["id"],
            )


async def mobile_lease_loop() -> None:
    while True:
        try:
            expired = await manager.expire_stale()
            if expired:
                async with db.pool().acquire() as conn:
                    await conn.execute(
                        "update nodes set healthy = false where id = any($1::uuid[]) "
                        "and transport = 'android_ws'",
                        expired,
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Android worker lease sweep failed")
        await asyncio.sleep(max(1, HEARTBEAT_LEASE_SECONDS // 12))


async def health_check_loop() -> None:
    while True:
        try:
            await run_health_checks_once()
        except Exception:
            pass
        await asyncio.sleep(settings.health_check_interval_seconds)
