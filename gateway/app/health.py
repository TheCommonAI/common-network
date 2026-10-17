import asyncio
import json
import logging

import httpx
from fastapi import HTTPException

from app import db
from app.config import settings
from app.registry import validate_endpoint_url
from app.upstream import auth_headers

logger = logging.getLogger(__name__)


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
            "select id, endpoint_url, api_key_ref, worker_token, model_name from nodes where not paused"
        )

    async with httpx.AsyncClient() as client:
        # Every node at once, not one after another. A pass costs the sum of
        # its checks when run in series: at the 5s cap, 23 registered rows is
        # over a minute -- and `last_heartbeat`, which only this loop writes,
        # is then aged by however many dead nodes happen to sort above a live
        # one rather than by the 30s the interval promises. Concurrently a
        # pass costs the slowest single check, so what `last_heartbeat` says
        # keeps meaning what the field above claims it means.
        #
        # return_exceptions=True because one node raising must not abandon the
        # pass or leave the rest unrecorded. An unexpected error is treated as
        # not-healthy, matching what this returns for every failure mode it
        # does handle -- and it is recomputed next pass, so a transient one
        # cannot strand a node.
        results = await asyncio.gather(
            *(_check_one(client, dict(row)) for row in rows), return_exceptions=True
        )

    async with db.pool().acquire() as conn:
        for row, ok in zip(rows, results):
            if not isinstance(ok, bool):
                logger.warning(
                    "health check for node %s raised: %r", row["id"], ok
                )
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


async def health_check_loop() -> None:
    while True:
        try:
            await run_health_checks_once()
        except Exception:
            pass
        await asyncio.sleep(settings.health_check_interval_seconds)
