import asyncio
import os

import httpx
from fastapi import HTTPException

from app import db
from app.config import settings
from app.registry import validate_endpoint_url


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
    headers = {}
    api_key_ref = node.get("api_key_ref")
    if api_key_ref:
        key = os.environ.get(api_key_ref)
        if key:
            headers["Authorization"] = f"Bearer {key}"
    try:
        resp = await client.get(url, headers=headers, timeout=settings.health_check_timeout_seconds)
        return resp.status_code < 500
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
