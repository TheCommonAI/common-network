from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.groups import RESERVED_GROUP_TAGS


# --- Registry ---

class NodeCreate(BaseModel):
    name: str
    operator: str | None = None
    endpoint_url: str
    model_name: str
    api_key_ref: str | None = None
    capability_text: str
    region: str | None = None
    cost_per_1k: float = 0
    domain_tags: list[str] | None = None
    catalogue_id: str | None = None

    # What the gateway must present to this node's worker on every request.
    # Supplied by the joiner rather than issued here: the worker has to be
    # running (with its token) before the endpoint being registered is worth
    # anything, so generating it gateway-side would need a second round trip.
    # Omitted by nodes that front a third-party API instead of a worker.
    worker_token: str | None = Field(default=None, min_length=16, max_length=128)

    # Which program is registering: "common-desktop/0.1.0" and, later, a
    # matching string from the CLI. Self-reported and unverifiable, so it is a
    # usage statistic and nothing else -- never an authorisation input, and
    # nothing in the gateway may branch on it. Length-capped because it lands
    # in a text column straight from a stranger; not otherwise validated,
    # since an unrecognised client is data, not an error.
    client: str | None = Field(default=None, max_length=64)

    # A distributed group's own report of its state (migrations/015). Optional:
    # a single-machine node sends none of these. Validated together below, so a
    # group is either fully described or not at all.
    group_status: Literal["ready", "restarting"] | None = None
    group_members_total: int | None = Field(default=None, ge=1, le=256)
    group_members_up: int | None = Field(default=None, ge=0, le=256)

    @model_validator(mode="after")
    def _check_group(self):
        # `or []`: a group that declares no tags at all must fail the same way as
        # one that declares only "general", not slip past the tag check as None.
        check_group_fields(self.group_status, self.group_members_total,
                           self.group_members_up, self.domain_tags or [])
        return self


def check_group_fields(group_status, members_total, members_up, domain_tags=None) -> None:
    """Reject a group report that cannot be true, before it reaches the database.

    Raises ValueError, which pydantic reports as a 422. Called with domain_tags
    only at registration, where the tags are declared. A later status report
    carries no tags, and the tags it was registered under still stand.
    """
    if group_status is None:
        if members_total is not None or members_up is not None:
            raise ValueError("group_members_total and group_members_up need a group_status")
        return
    if members_total is None or members_up is None:
        raise ValueError("a group report needs both group_members_total and group_members_up")
    if members_up > members_total:
        raise ValueError("group_members_up cannot exceed group_members_total")
    if domain_tags is not None:
        tags = {t.lower() for t in domain_tags}
        if not tags:
            raise ValueError("a distributed group must declare its model's own domain tag")
        if tags & RESERVED_GROUP_TAGS:
            raise ValueError(
                "a distributed group must register under its model's own domain tag, "
                "not 'general' (the generalist lane would route requests it cannot serve)"
            )


class GroupReport(BaseModel):
    """What a coordinator sends to PUT /nodes/{id}/group, authenticated by its node token."""
    group_status: Literal["ready", "restarting"]
    group_members_total: int = Field(ge=1, le=256)
    group_members_up: int = Field(ge=0, le=256)

    @model_validator(mode="after")
    def _check_group(self):
        check_group_fields(self.group_status, self.group_members_total, self.group_members_up)
        return self


class NodePublicOut(BaseModel):
    """Public node info returned by GET /nodes. No endpoint_url — that is
    internal to the gateway. Exposing tunnel URLs lets anyone bypass the
    gateway's rate limiting and contribution gate."""
    id: UUID
    name: str
    operator: str | None
    model_name: str
    region: str | None
    cost_per_1k: float
    avg_latency_ms: int
    healthy: bool
    last_heartbeat: str | None
    last_seen_healthy: str | None = None
    capability_text: str
    domain_tags: list[str] | None = None
    catalogue_id: str | None = None
    # Display only, never a routing input from these values alone (a 'restarting'
    # report does take the node out of routing; see app/groups.py). Null for a
    # single-machine node.
    group_status: str | None = None
    group_members_total: int | None = None
    group_members_up: int | None = None


class NodeOut(NodePublicOut):
    """Full node info including endpoint_url — only for internal use and
    admin endpoints. Never returned by the public GET /nodes."""
    endpoint_url: str


class NodeRegisterOut(NodePublicOut):
    # Only ever returned once, from POST /nodes -- the one credential needed
    # to deregister this specific node. Never included in GET /nodes (that
    # would let anyone deregister anyone).
    node_token: str


# --- Decisions ---

class DecisionOut(BaseModel):
    id: UUID
    chosen_node: UUID | None
    chosen_node_name: str | None
    score: float | None
    runner_up: UUID | None
    latency_ms: int | None
    ok: bool | None
    created_at: str
    matched_domain: str | None = None

    # --- Composition (v0.1.1) ---
    # 'single' | 'panel' | 'degraded'. Defaulted rather than optional so a row
    # written by v0.1 reads back as what it was, not as a gap.
    topology: str = "single"
    panel: list[str] | None = None
    aggregator_node_name: str | None = None
    compose_reason: dict | None = None
    # Verification counters, per decision rather than only in aggregate:
    # "how often does the checker actually fire" is the question that reveals
    # whether it still works. A fire rate that quietly falls to zero means the
    # extractor broke, not that the models got better.
    checks_run: int | None = None
    checks_failed: int | None = None
    disagreements: int | None = None


# --- OpenAI-compatible passthrough ---
# Deliberately untyped/loose (dict passthrough) — v0.1 forwards whatever the
# client sends and returns whatever the node returns, unchanged except for
# our transparency headers. Do not model the full OpenAI schema here.

ChatCompletionRequest = dict[str, Any]
