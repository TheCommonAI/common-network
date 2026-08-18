"""Talking to nodes.

Shared by the single-route path (`gateway.py`) and the composition path
(`compose.py`) so that the credential-handling rule lives in exactly one place:
a node stores the *name* of an env var, never a key. A permissionless registry
that accepted raw keys would be a credential-harvesting endpoint.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from app.config import settings


def auth_headers(node: dict) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key_ref = node.get("api_key_ref")
    if api_key_ref:
        key = os.environ.get(api_key_ref)
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
    client = httpx.AsyncClient(timeout=settings.forward_timeout_seconds)
    req = client.build_request("POST", chat_url(node), json=outgoing, headers=auth_headers(node))
    resp = await client.send(req, stream=stream)
    resp.extensions["_client"] = client
    return resp
