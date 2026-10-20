from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


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

    # Distributed model groups (migration 015). Sent by a group's coordinator
    # and nobody else; omitted by every single node. Self-reported, so display
    # only: the one routing consequence is that a `restarting` group is not
    # marked healthy and gets a 503 when it has no other candidate. Omitted
    # values leave what an earlier registration said in place (see registry.py).
    group_status: Literal["ready", "restarting"] | None = None
    group_members_total: int | None = Field(default=None, ge=0)
    group_members_up: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _members_up_within_total(self):
        if (self.group_members_up is not None and self.group_members_total is not None
                and self.group_members_up > self.group_members_total):
            raise ValueError("group_members_up cannot exceed group_members_total")
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
    # Distributed model groups, display only (migration 015). Null for every
    # single node.
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
