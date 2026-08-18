from typing import Any
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


class NodeOut(BaseModel):
    id: UUID
    name: str
    operator: str | None
    endpoint_url: str
    model_name: str
    region: str | None
    cost_per_1k: float
    avg_latency_ms: int
    healthy: bool
    last_heartbeat: str | None
    capability_text: str
    domain_tags: list[str] | None = None
    catalogue_id: str | None = None


class NodeRegisterOut(NodeOut):
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
