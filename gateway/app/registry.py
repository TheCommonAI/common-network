import ipaddress
import secrets
import socket
from urllib.parse import urlparse
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException

from app import db, embedder
from app.config import settings
from app.models import NodeCreate, NodeOut, NodeRegisterOut

router = APIRouter()


# --- Endpoint validation ---------------------------------------------------

def _resolved_addresses(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Every address the endpoint host could resolve to.

    Literal IPs parse directly; hostnames are resolved now, at registration
    time. That leaves a DNS-rebinding window (a host that resolves somewhere
    safe for the registration check and somewhere unsafe for the request) —
    a known limit of Alpha, recorded here rather than papered over.
    """
    try:
        return [ipaddress.ip_address(host)]
    except ValueError:
        try:
            return [ipaddress.ip_address(info[4][0]) for info in socket.getaddrinfo(host, None)]
        except socket.gaierror:
            raise HTTPException(
                status_code=400,
                detail=f"endpoint_url host '{host}' does not resolve — the gateway health "
                       f"checker would flag this node dead on its first pass anyway.",
            )


def validate_endpoint_url(url: str) -> None:
    """A node's endpoint_url is where the gateway POSTs request bodies, and it
    is chosen by an unauthenticated stranger — registration is permissionless,
    which is the network's thesis. Left unvalidated it makes the gateway a
    server-side request forgery proxy: register `http://169.254.169.254/...`
    and the gateway fetches it for you and streams the response back.

    Two rules, chosen so every legitimate use keeps working:

    * **http(s) only** — httpx supports nothing else, so anything else fails
      later anyway; refusing it here says why.
    * **Never link-local.** That is where cloud metadata services live, and
      it is never a node: no Ollama instance or contributed endpoint sits on
      169.254.x.x. Rejected unconditionally, including on private LANs.

    Loopback is allowed only when the operator opts in
    (`ALLOW_LOOPBACK_NODE_ENDPOINTS`, default true) — the seed demo and
    same-machine dev both register `localhost:11434`, so the default keeps
    those working. A *publicly reachable* gateway should set it to false in
    its environment; see SECURITY.md.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(
            status_code=400,
            detail="endpoint_url must be an http:// or https:// URL the gateway can POST to.",
        )
    for addr in _resolved_addresses(parsed.hostname):
        if addr.is_link_local:
            raise HTTPException(
                status_code=400,
                detail="endpoint_url may not point at a link-local address — that range is "
                       "cloud metadata services (169.254.x.x / fe80::), never a real node.",
            )
        if (addr.is_loopback or addr.is_unspecified) and not settings.allow_loopback_node_endpoints:
            raise HTTPException(
                status_code=400,
                detail="this gateway does not accept loopback endpoints "
                       "(ALLOW_LOOPBACK_NODE_ENDPOINTS=false). Register a reachable address.",
            )


# --- Registry ---------------------------------------------------------------

def _row_to_node_out(row) -> NodeOut:
    return NodeOut(
        id=row["id"],
        name=row["name"],
        operator=row["operator"],
        endpoint_url=row["endpoint_url"],
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
async def register_node(node: NodeCreate, x_common_node_token: str | None = Header(default=None)):
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
    validate_endpoint_url(node.endpoint_url)
    vec = embedder.embed(node.capability_text)

    new_token = secrets.token_urlsafe(24)
    async with db.pool().acquire() as conn:
        existing = await conn.fetchrow("select node_token from nodes where name = $1", node.name)
        if existing is not None:
            stored = existing["node_token"]
            if stored is not None and not (
                x_common_node_token
                and secrets.compare_digest(stored.encode(), x_common_node_token.encode())
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
            row = await conn.fetchrow(
                """
                update nodes set
                    operator = $2, endpoint_url = $3, model_name = $4,
                    api_key_ref = $5, capability_text = $6, capability_embed = $7,
                    region = $8, cost_per_1k = $9, domain_tags = $10,
                    catalogue_id = $11,
                    node_token = coalesce(node_token, $12),
                    worker_token = $13
                where name = $1
                returning *
                """,
                node.name, node.operator, node.endpoint_url, node.model_name, node.api_key_ref,
                node.capability_text, vec, node.region, node.cost_per_1k,
                node.domain_tags, node.catalogue_id, new_token, node.worker_token,
            )
        else:
            row = await conn.fetchrow(
                """
                insert into nodes
                    (name, operator, endpoint_url, model_name, api_key_ref,
                     capability_text, capability_embed, region, cost_per_1k,
                     domain_tags, catalogue_id, node_token, worker_token)
                values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                returning *
                """,
                node.name, node.operator, node.endpoint_url, node.model_name, node.api_key_ref,
                node.capability_text, vec, node.region, node.cost_per_1k,
                node.domain_tags, node.catalogue_id, new_token, node.worker_token,
            )

    out = _row_to_node_out(row)
    # node_token reaches only whoever proved they own the name: fresh inserts
    # (nobody owned it before) and token-holding re-registrations.
    #
    # worker_token is deliberately NOT returned. The node generated it and
    # already has it; echoing it would put a live credential in one more
    # response body for no one's benefit.
    return NodeRegisterOut(**out.model_dump(), node_token=row["node_token"])


@router.get("/nodes", response_model=list[NodeOut])
async def list_nodes():
    async with db.pool().acquire() as conn:
        rows = await conn.fetch("select * from nodes order by created_at desc")
    return [_row_to_node_out(r) for r in rows]


@router.delete("/nodes/{node_id}")
async def delete_node(node_id: UUID, x_common_node_token: str | None = Header(default=None)):
    async with db.pool().acquire() as conn:
        result = await conn.execute(
            "delete from nodes where id = $1 and node_token = $2", node_id, x_common_node_token,
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="node not found, or X-Common-Node-Token doesn't match")
    return {"deleted": str(node_id)}