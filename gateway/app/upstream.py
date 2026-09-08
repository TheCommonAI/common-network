"""Talking to nodes.

Shared by the single-route path (`gateway.py`) and the composition path
(`compose.py`) so that the credential-handling rule lives in exactly one place:
a node stores the *name* of an env var, never a key. A permissionless registry
that accepted raw keys would be a credential-harvesting endpoint.

Storing only the name is necessary but not sufficient: the *name* is still
chosen by whoever registers, so the allowlist below decides which names the
gateway will resolve at all. See `resolve_api_key`.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from app.config import settings


def resolve_api_key(api_key_ref: str | None) -> str | None:
    """The env var a node asked for, if this gateway permits that name.

    api_key_ref is attacker-chosen: registration is permissionless, so anyone
    can register an endpoint they control and name any variable in the
    gateway's environment. Whatever that variable holds would then be sent to
    them as a bearer token -- on the next health check, without needing a
    single user request to be routed there.

    So the reference is only honoured when the operator has explicitly listed
    the name in ALLOWED_API_KEY_REFS. The default is empty: no node may
    reference any credential until an operator deliberately allows one.
    """
    if not api_key_ref:
        return None
    allowed = {n.strip() for n in settings.allowed_api_key_refs.split(",") if n.strip()}
    if api_key_ref not in allowed:
        return None
    return os.environ.get(api_key_ref)


def auth_headers(node: dict) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = resolve_api_key(node.get("api_key_ref"))
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
