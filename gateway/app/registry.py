import asyncio
import ipaddress
import secrets
import socket
from urllib.parse import urlparse
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Request

from app import db, embedder, ratelimit
from app.credentials import token_digest, token_matches
from app.config import settings
from app.models import NodeCreate, NodeOut, NodeRegisterOut

router = APIRouter()


# Shared with outbound connections: the address is pinned at connect time.
from app.network import validate_endpoint_url, _resolved_addresses


# --- Registry ---------------------------------------------------------------

def _row_to_node_out(row, private=False) -> NodeOut:
    return NodeOut(
        id=row["id"],
        name=row["name"],
        operator=row["operator"],
        endpoint_url=row["endpoint_url"] if private or settings.public_node_endpoints else "",
        model_name=row["model_name"],
        region=row["region"],
        cost_per_1k=float(row["cost_per_1k"]),
        avg_latency_ms=row["avg_latency_ms"],
        healthy=row["healthy"],
        last_heartbeat=row["last_heartbeat"].isoformat() if row["last_heartbeat"] else None,
        capability_text=row["capability_text"],
        domain_tags=row["domain_tags"],
        catalogue_id=row["catalogue_id"],
    )


@router.post("/nodes", response_model=NodeRegisterOut)
async def register_node(node: NodeCreate, request: Request, x_common_node_token: str | None = Header(default=None)):
    """Register or re-register a node.

    Permissionless by design — see README "Scope": anyone can contribute a
    node, no shared password. The credential issued here is scoped to *this*
    node only, and since the name is how the network routes to a node, the
    name itself has to be protected the same way:

    * **Fresh name** → a new token is issued and returned to the registrant.
    * **Existing name** → accepted only from whoever holds that node's
      X-Common-Node-Token, and the stored token is returned only to them.
      Without this check, anyone could read node names from `GET /nodes`,
      POST the same name, and be handed the original registrant's token by
      the upsert — taking over the node's routing and gaining the right to
      delete it. Names are public; tokens must not be.
    * **Existing name, no stored token** → a row from before the node-token
      migration, which nobody holds a token for. First re-registration
      claims it and is issued a fresh token.
    """
    ratelimit.check_action(request, 'register', settings.registration_requests_per_minute)
    try:
        await asyncio.wait_for(asyncio.to_thread(validate_endpoint_url, node.endpoint_url), 5)
    except asyncio.TimeoutError:
        raise HTTPException(400, 'endpoint DNS lookup timed out') from None
    if node.api_key_ref:
        from app.upstream import resolve_api_key
        if not resolve_api_key(node.api_key_ref, node.endpoint_url):
            raise HTTPException(400, 'credential reference is not approved for this endpoint')

    new_token = secrets.token_urlsafe(24)
    async with db.pool().acquire() as conn:
        async with conn.transaction():
            # Serialise claims of the same public name, including legacy null tokens.
            await conn.execute('select pg_advisory_xact_lock(hashtextextended($1, 0))', node.name)
            existing = await conn.fetchrow("select node_token from nodes where name = $1", node.name)
            if existing is not None:
                stored = existing["node_token"]
                if stored is not None and not (
                    token_matches(stored, x_common_node_token)
                ):
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"a node named '{node.name}' is already registered and this isn't its "
                            f"token. Re-registering an existing node requires the X-Common-Node-Token "
                            f"issued when it was registered — otherwise anyone who read the name "
                            f"could take the node over. Pick a different name, or deregister the "
                            f"node first if it is yours."
                        ),
                    )
                vec = embedder.embed(node.capability_text)
                row = await conn.fetchrow(
                    """
                    update nodes set
                        operator = $2, endpoint_url = $3, model_name = $4,
                        api_key_ref = $5, capability_text = $6, capability_embed = $7,
                        region = $8, cost_per_1k = $9, domain_tags = $10,
                        catalogue_id = $11,
                        node_token = $12,
                        worker_token = $13, healthy = false
                    where name = $1
                    returning *
                    """,
                    node.name, node.operator, node.endpoint_url, node.model_name, node.api_key_ref,
                    node.capability_text, vec, node.region, node.cost_per_1k,
                    node.domain_tags, node.catalogue_id, token_digest(x_common_node_token if existing and existing["node_token"] else new_token), node.worker_token,
                )
            else:
                vec = embedder.embed(node.capability_text)
                row = await conn.fetchrow(
                    """
                    insert into nodes
                        (name, operator, endpoint_url, model_name, api_key_ref,
                         capability_text, capability_embed, region, cost_per_1k,
                         domain_tags, catalogue_id, node_token, worker_token, healthy)
                    values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, false)
                    returning *
                    """,
                    node.name, node.operator, node.endpoint_url, node.model_name, node.api_key_ref,
                    node.capability_text, vec, node.region, node.cost_per_1k,
                    node.domain_tags, node.catalogue_id, token_digest(x_common_node_token if existing and existing["node_token"] else new_token), node.worker_token,
                )

    out = _row_to_node_out(row, private=True)
    # node_token reaches only whoever proved they own the name: fresh inserts
    # (nobody owned it before) and token-holding re-registrations.
    #
    # worker_token is deliberately NOT returned. The node generated it and
    # already has it; echoing it would put a live credential in one more
    # response body for no one's benefit.
    return NodeRegisterOut(**out.model_dump(), node_token=x_common_node_token if existing and existing["node_token"] else new_token)


@router.get("/nodes", response_model=list[NodeOut])
async def list_nodes():
    async with db.pool().acquire() as conn:
        rows = await conn.fetch("select * from nodes order by created_at desc")
    return [_row_to_node_out(r) for r in rows]


@router.delete("/nodes/{node_id}")
async def delete_node(node_id: UUID, x_common_node_token: str | None = Header(default=None)):
    async with db.pool().acquire() as conn:
        result = await conn.execute(
            "delete from nodes where id = $1 and (node_token = $2 or (node_token not like 'sha256:%' and node_token = $3))", node_id, token_digest(x_common_node_token or ""), x_common_node_token,
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="node not found, or X-Common-Node-Token doesn't match")
    return {"deleted": str(node_id)}