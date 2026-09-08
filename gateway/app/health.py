import asyncio

import httpx
from fastapi import HTTPException

from app import db
from app.config import settings
from app.registry import validate_endpoint_url
from app.upstream import auth_headers
from app.network import pinned_request


async def _check_one(client: httpx.AsyncClient, node: dict) -> bool:
    try:
        return await asyncio.wait_for(_check_one_inner(client, node), settings.health_check_timeout_seconds)
    except (asyncio.TimeoutError, httpx.HTTPError):
        return False


async def _check_one_inner(client: httpx.AsyncClient, node: dict) -> bool:
    # The same pinned destination and credential policy is used for chat.
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
    try:
        req = await pinned_request(client, "GET", url, headers=headers,
                                   timeout=settings.health_check_timeout_seconds)
        resp = await client.send(req, stream=True)
        # 2xx only. `< 500` counted 401 (bad credential), 403 and 404 (no such
        # route -- not an OpenAI-compatible server at all) as healthy, so a
        # node could be routed real traffic on the strength of a reply that
        # said "I cannot serve you".
        try:
            if resp.status_code != 200:
                return False
            data = bytearray()
            async for chunk in resp.aiter_bytes():
                data.extend(chunk)
                if len(data) > 65536:
                    return False
            import json
            payload = json.loads(data)
            return any(m.get('id') == node.get('model_name')
                       for m in payload.get('data', []) if isinstance(m, dict))
        except (ValueError, AttributeError):
            return False
        finally:
            await resp.aclose()
    except httpx.HTTPError:
        return False


async def run_health_checks_once() -> None:
    async with db.pool().acquire() as conn:
        rows = await conn.fetch(
            "select id, endpoint_url, model_name, api_key_ref, worker_token from nodes"
        )

    async with httpx.AsyncClient(follow_redirects=False, trust_env=False,
                               limits=httpx.Limits(max_keepalive_connections=0)) as client:
        for row in rows:
            ok = await _check_one(client, dict(row))
            async with db.pool().acquire() as conn:
                await conn.execute(
                    "update nodes set healthy = $1, last_heartbeat = now() where id = $2",
                    ok, row["id"],
                )


async def health_check_loop() -> None:
    while True:
        try:
            await run_health_checks_once()
        except Exception:
            pass
        await asyncio.sleep(settings.health_check_interval_seconds)
