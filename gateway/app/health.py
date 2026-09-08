import asyncio

import httpx
from fastapi import HTTPException

from app import db
from app.config import settings
from app.registry import validate_endpoint_url
from app.upstream import resolve_api_key


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
    # Same allowlist as the forwarding path. This call matters more than that
    # one: it runs every 30s against every registered node, so an unrestricted
    # api_key_ref would hand a credential to a hostile endpoint without any
    # user ever sending it a request.
    headers = {}
    key = resolve_api_key(node.get("api_key_ref"))
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        resp = await client.get(url, headers=headers, timeout=settings.health_check_timeout_seconds)
        # 2xx only. `< 500` counted 401 (bad credential), 403 and 404 (no such
        # route -- not an OpenAI-compatible server at all) as healthy, so a
        # node could be routed real traffic on the strength of a reply that
        # said "I cannot serve you".
        return 200 <= resp.status_code < 300
    except httpx.HTTPError:
        return False


async def run_health_checks_once() -> None:
    async with db.pool().acquire() as conn:
        rows = await conn.fetch("select id, endpoint_url, api_key_ref from nodes")

    async with httpx.AsyncClient() as client:
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
