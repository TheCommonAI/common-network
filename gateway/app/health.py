import asyncio
import json

import httpx
from fastapi import HTTPException

from app import db
from app.config import settings
from app.registry import validate_endpoint_url
from app.upstream import auth_headers


async def _check_one(client: httpx.AsyncClient, node: dict) -> bool:
    # Re-validate the endpoint every pass, not just at registration. This
    # closes the DNS-rebinding window: a hostname that resolved somewhere
    # legitimate when it registered and resolves at a metadata service now
    # stops being health-checked green — the gateway never fetches it again,
    # because an unhealthy node is never routed to.
    try:
        validate_endpoint_url(node["endpoint_url"])
    except HTTPException:
        return False

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
    except (httpx.HTTPError, asyncio.TimeoutError):
        return False



async def run_health_checks_once() -> None:
    async with db.pool().acquire() as conn:
        rows = await conn.fetch(
            "select id, endpoint_url, api_key_ref, worker_token, model_name from nodes where not paused"
        )

    async with httpx.AsyncClient() as client:
        for row in rows:
            ok = await _check_one(client, dict(row))
            async with db.pool().acquire() as conn:
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
