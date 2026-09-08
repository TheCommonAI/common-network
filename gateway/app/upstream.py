"""Bounded forwarding shared by routing and composition.

Gateway keys require exact approved HTTPS destinations. Worker credentials are
separate per-node secrets. All outgoing connections use the pinned-address
transport policy; callers own response/client lifetime for streaming replies.
"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx

from app.config import settings
from app.network import endpoint_identity, pinned_request


def resolve_api_key(api_key_ref: str | None, endpoint_url: str | None = None) -> str | None:
    """Resolve a key only for an operator-approved exact HTTPS endpoint."""
    if not api_key_ref:
        return None
    if not endpoint_url or not endpoint_url.startswith('https://'):
        return None
    try:
        destination = endpoint_identity(endpoint_url)
        allowed = settings.api_key_destinations.get(api_key_ref, [])
        if destination not in {endpoint_identity(url) for url in allowed}:
            return None
    except Exception:
        return None
    return os.environ.get(api_key_ref)


def auth_headers(node: dict) -> dict[str, str]:
    """Headers for a request to a node.

    Two credential sources, in priority order:

    * `worker_token` -- what a `common join` worker requires. A node that
      sends one is running our worker, and the token is what proves to a
      contributor's machine that a request came from this gateway rather than
      from anyone who read the endpoint URL out of GET /nodes.
    * `api_key_ref` -- an allowlisted environment variable, for nodes that
      front a third-party API needing its own key (see resolve_api_key).

    A node cannot need both: the first is our own worker, the second is
    somebody else's service. Worker token wins, because a node that has one is
    running our worker and will reject anything else.
    """
    headers = {"Content-Type": "application/json"}
    worker_token = node.get("worker_token")
    if worker_token:
        headers["Authorization"] = f"Bearer {worker_token}"
        return headers
    key = resolve_api_key(node.get("api_key_ref"), node.get("endpoint_url"))
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def chat_url(node: dict) -> str:
    return node["endpoint_url"].rstrip("/") + "/chat/completions"


async def forward(node: dict, body: dict[str, Any], stream: bool) -> httpx.Response:
    """Forward a request body to a node, rewriting only the model name.

    The caller owns closing both the response and the client stashed in
    `resp.extensions["_client"]` -- streaming responses have to outlive this
    function, so the client cannot be context-managed here.
    """
    outgoing = dict(body)
    outgoing["model"] = node["model_name"]
    client = httpx.AsyncClient(timeout=settings.forward_timeout_seconds,
                               follow_redirects=False, trust_env=False,
                               limits=httpx.Limits(max_keepalive_connections=0))
    try:
        req = await pinned_request(client, "POST", chat_url(node), json=outgoing,
                                   headers=auth_headers(node))
        resp = await client.send(req, stream=True)
        if 300 <= resp.status_code < 400:
            await resp.aclose()
            raise httpx.ConnectError("node redirects are not permitted")
        resp.extensions["_client"] = client
        resp.extensions["_deadline"] = time.monotonic() + settings.request_deadline_seconds
        if not stream:
            content = await read_limited(resp)
            original = resp
            resp = httpx.Response(original.status_code, headers=original.headers,
                                  content=content, request=req, extensions=original.extensions)
            await original.aclose()
        return resp
    except BaseException:
        await client.aclose()
        raise


async def iter_limited(resp):
    import asyncio
    total = 0
    deadline = resp.extensions.get('_deadline', time.monotonic() + settings.request_deadline_seconds)
    async with asyncio.timeout(max(0, deadline - time.monotonic())):
        async for chunk in resp.aiter_bytes():
            total += len(chunk)
            if total > settings.max_response_bytes:
                raise httpx.ReadError('node response exceeded the byte limit')
            yield chunk


async def read_limited(resp):
    chunks = []
    try:
        async for chunk in iter_limited(resp):
            chunks.append(chunk)
        return b''.join(chunks)
    except BaseException:
        await resp.aclose()
        raise
