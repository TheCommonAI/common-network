from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, Field


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

    # --- On-demand specialists -------------------------------------------
    # What a node could still be fused from, and what it is running now.
    #
    # `base_model` is the weights underneath. This is load-bearing rather than
    # descriptive: while a node is serving a blend its model_name is the blend
    # tag, not a real model, so base_model is the only thing that survives the
    # swap and says what it can be fused *from*. It is a hint, not an
    # authority -- where a node has a catalogue_id the gateway prefers the base
    # that entry names, because that one is ours. Either way a NULL means
    # "unknown" and the rule is refuse to assign, never guess.
    #
    # `adapter_ids` is the set fused on top of it. Small on purpose: a blend is
    # a distinct Ollama model with its own residency slot, so a node holds one
    # blend at a time, and this array is that blend.
    #
    # Both are self-reported by a stranger, so both are bounded. An adapter id
    # is a lookup key, and truncating one would produce a key that matches
    # nothing -- so an over-long value is refused rather than trimmed, unlike
    # `client` above, which is a statistic and is cut to fit.
    base_model: str | None = Field(default=None, max_length=128)
    adapter_ids: list[Annotated[str, Field(max_length=64)]] | None = Field(
        default=None, max_length=8)


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
    # Which adapters this node is serving, and the base under them. Public
    # deliberately -- it is a capability description like domain_tags, not a
    # credential. It is what lets a reader see the network building a
    # specialist instead of only being told it happened.
    base_model: str | None = None
    adapter_ids: list[str] | None = None


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
