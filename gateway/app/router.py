from dataclasses import dataclass

import numpy as np

from app import embedder
from app.config import settings


@dataclass
class ScoredNode:
    node: dict
    score: float
    sim: float
    cost_term: float
    lat_term: float
    region_term: float
    tag_term: float = 0.0

    @property
    def topical_score(self) -> float:
        """Confidence this node actually matches the request's topic -- similarity
        and tag overlap only, excluding cost/latency. Those measure node quality,
        not topical fit, and diluted a threshold on the full blended `score` down
        to something that almost never triggered in practice (cost_term + lat_term
        alone can contribute up to 0.3 regardless of relevance)."""
        return settings.w_sim * self.sim + self.tag_term


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


# Tag embeddings are cheap to compute and reused across requests -- cache
# them in-process rather than re-embedding the same handful of domain tags
# on every call.
_tag_embed_cache: dict[str, list[float]] = {}


def _tag_embed(tag: str) -> list[float]:
    if tag not in _tag_embed_cache:
        _tag_embed_cache[tag] = embedder.embed(tag)
    return _tag_embed_cache[tag]


def _tag_overlap_score(request_embed: list[float], domain_tags: list[str] | None) -> float:
    """How well the request matches any of a node's structured domain tags.

    Still just cosine similarity (no new algorithm) -- applied to short tag
    strings instead of the node's free-text capability profile, so a node
    with the right tags gets credit even if its capability_text phrasing
    doesn't happen to overlap with the request's wording.
    """
    if not domain_tags:
        return 0.0
    return max(_cosine(request_embed, _tag_embed(t)) for t in domain_tags)


def domain_similarities(nodes: list[dict], request_embed: list[float]) -> list[tuple[str, float]]:
    """Every domain the network declares, ranked by match against this request.

    v0.1 only ever needed the single best domain, because a request went to a
    single node. Composition needs the whole ranking: whether a request spans
    two domains that *different* nodes are best at is exactly the question of
    whether there is anything worth composing (see app/compose.py).
    """
    all_tags: set[str] = set()
    for n in nodes:
        all_tags.update(n.get("domain_tags") or [])
    ranked = [(t, _cosine(request_embed, _tag_embed(t))) for t in all_tags]
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked


def best_matched_domain(nodes: list[dict], request_embed: list[float], floor: float = 0.3) -> str | None:
    """The single domain tag (across all candidate nodes) that best matches this
    request, for logging as decisions.matched_domain -- the demand signal that
    assignment reads back. None if nothing clears the floor."""
    ranked = domain_similarities(nodes, request_embed)
    if not ranked:
        return None
    best_tag, best_sim = ranked[0]
    return best_tag if best_sim >= floor else None


def score_nodes(
    nodes: list[dict],
    request_embed: list[float],
    region_hint: str | None = None,
) -> list[ScoredNode]:
    """Score every candidate node against a request embedding.

    Returns nodes sorted best-first. Kept as plain arithmetic (no learned
    weights) so a human can read `sim`/`cost_term`/`lat_term`/`tag_term` off a
    ScoredNode and see exactly why a node won — see Master Build Prompt
    "Routing algorithm" for the rationale.
    """
    if not nodes:
        return []

    max_cost = max((float(n["cost_per_1k"]) for n in nodes), default=0.0)
    max_lat = max((float(n["avg_latency_ms"]) for n in nodes), default=0.0)

    scored: list[ScoredNode] = []
    for n in nodes:
        sim = _cosine(request_embed, n["capability_embed"])

        norm_cost = (float(n["cost_per_1k"]) / max_cost) if max_cost > 0 else 0.0
        cost_term = settings.w_cost * (1 - norm_cost)

        norm_lat = (float(n["avg_latency_ms"]) / max_lat) if max_lat > 0 else 0.0
        lat_term = settings.w_lat * (1 - norm_lat)

        region_term = 0.0
        if region_hint and n.get("region") == region_hint:
            region_term = settings.region_bonus

        tag_term = settings.w_tag_overlap * _tag_overlap_score(request_embed, n.get("domain_tags"))

        score = settings.w_sim * sim + cost_term + lat_term + region_term + tag_term
        scored.append(ScoredNode(
            node=n, score=score, sim=sim, cost_term=cost_term,
            lat_term=lat_term, region_term=region_term, tag_term=tag_term,
        ))

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored
