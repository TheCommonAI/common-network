"""Tests for the composition gate.

The gate is where the v0.1 findings are actually enforced, so it gets tested
the way a finding deserves: each case below names the result it encodes.

No database, no network, no real embedding model. `app.embedder.embed` is
replaced with a toy four-dimensional space (maths / code / legal / writing) so
similarity is something the test states rather than something a 90MB model
decides. The logic under test is "given these similarities, compose or not" —
mixing in a real encoder would test sentence-transformers instead.

Run: `python tests/test_compose.py` from `gateway/`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import embedder, router  # noqa: E402

# --- A toy embedding space ------------------------------------------------

AXES = ["math", "code", "legal", "writing"]
VECTORS = {
    "math": [1, 0, 0, 0], "arithmetic": [1, 0, 0, 0], "reasoning": [0.8, 0.2, 0, 0],
    "code": [0, 1, 0, 0], "python": [0, 1, 0, 0],
    "legal": [0, 0, 1, 0], "south-australia-law": [0, 0, 1, 0],
    "writing": [0, 0, 0, 1], "general": [0.5, 0.5, 0.5, 0.5],
}


def fake_embed(text: str) -> list[float]:
    return VECTORS.get(text, [0.25, 0.25, 0.25, 0.25])


embedder.embed = fake_embed
router._tag_embed_cache.clear()

from app.compose import plan_panel  # noqa: E402
from app.config import settings  # noqa: E402
from app.router import score_nodes  # noqa: E402

FAILURES: list[str] = []


def check(name: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL  {name}  (got {got!r}, want {want!r})")
    else:
        print(f"  ok    {name}")


def node(name: str, tags: list[str], embed: list[float], *, aggregate: bool = False) -> dict:
    return {
        "id": name, "name": name, "domain_tags": tags, "capability_embed": embed,
        "cost_per_1k": 0, "avg_latency_ms": 100, "model_name": name,
        "endpoint_url": "http://x/v1", "can_aggregate": aggregate,
    }


def plan_for(nodes: list[dict], request: list[float], mode: str | None = None):
    return plan_panel(score_nodes(nodes, request), request, mode_override=mode)


MATHS = node("mathstral", ["math"], VECTORS["math"])
CODER = node("coder", ["code", "python"], VECTORS["code"])
LEGAL = node("cgla", ["legal", "south-australia-law"], VECTORS["legal"])
WRITER = node("writer", ["writing"], VECTORS["writing"])
GENERAL = node("generalist", ["general"], VECTORS["general"])

# A request that genuinely spans maths and legal, equally.
SPANNING = [0.707, 0, 0.707, 0]
# A request squarely about one thing.
PURE_MATH = [1, 0, 0, 0]


print("\ncomposing when it should")
plan = plan_for([MATHS, LEGAL, GENERAL], SPANNING)
check("spans two domains with different best nodes -> composes", plan.compose, True)
check("panel is the two specialists",
      sorted(m.name for m in plan.members), ["cgla", "mathstral"])
check("generalist aggregates rather than answering",
      plan.aggregator["name"] if plan.aggregator else None, "generalist")
check("aggregator is off-panel", plan.aggregator_in_panel, False)

print("\nthe domination gate")
# The Experiment 2 result, encoded. One node best at every matched domain means
# nothing on the network beats it at anything relevant, so composing can only
# dilute it. seam-findings.md §2.
OMNI = node("omni", ["math", "arithmetic", "legal", "south-australia-law"], VECTORS["general"])
plan = plan_for([OMNI, GENERAL], SPANNING)
check("one node best at every matched domain -> refuses", plan.compose, False)
check("and says why", "not dominated" in plan.reason, True)

print("\nsingle-domain requests")
plan = plan_for([MATHS, LEGAL, GENERAL], PURE_MATH)
check("one clear domain -> no panel", plan.compose, False)
check("reason names the single-domain gate", "single-domain" in plan.reason, True)

print("\ngeneralists cannot take a panel seat")
# `general` matches everything moderately by construction, so if it could seat a
# member then almost every request would look multi-domain and the gateway would
# pair a specialist with a generalist inside that specialist's own lane -- the
# Experiment 2 pairing, rebuilt as a default. seam-findings.md §2.
check("generalist alongside one specialist does not form a panel",
      plan_for([MATHS, GENERAL], PURE_MATH).compose, False)

print("\nnear-synonym tags on one node")
# Two tags that both match, both held by the same node, is not two domains.
SYNONYM = node("mathstral2", ["math", "arithmetic"], VECTORS["math"])
plan = plan_for([SYNONYM, GENERAL], PURE_MATH)
check("synonymous tags on one node -> no panel", plan.compose, False)
check("caught by the domination gate", "not dominated" in plan.reason, True)

print("\ntoo few nodes")
check("one node -> nothing to compose with", plan_for([MATHS], SPANNING).compose, False)

print("\nmode overrides")
check("never -> refuses", plan_for([MATHS, LEGAL, GENERAL], SPANNING, "never").compose, False)
check("always -> composes a single-domain request",
      plan_for([MATHS, LEGAL, GENERAL], PURE_MATH, "always").compose, True)
# 'always' relaxes the worth-it heuristics, not the finding. Composing a node
# with one it dominates was measured to lose; no header should enable it.
check("always still refuses a dominated panel",
      plan_for([OMNI, GENERAL], SPANNING, "always").compose, False)

print("\npanel size")
FOUR = [0.5, 0.5, 0.5, 0.5]
plan = plan_for([MATHS, CODER, LEGAL, WRITER, GENERAL], FOUR, "always")
check("capped at compose_max_panel",
      len(plan.members) <= settings.compose_max_panel, True)

print("\naggregator selection")
VOLUNTEER = node("volunteer", ["writing"], VECTORS["writing"], aggregate=True)
plan = plan_for([MATHS, LEGAL, VOLUNTEER], SPANNING)
check("can_aggregate node preferred",
      plan.aggregator["name"] if plan.aggregator else None, "volunteer")

# A three-machine lab on day one: both nodes are on the panel, nobody spare.
# Refusing here would mean composition never runs on a small network, so a
# member aggregates -- but the conflict of interest is recorded, not hidden.
plan = plan_for([MATHS, LEGAL], SPANNING)
check("small network still composes", plan.compose, True)
check("and flags the aggregator as a panel member", plan.aggregator_in_panel, True)

print("\nexplanation is always present")
for label, p in [("composed", plan_for([MATHS, LEGAL, GENERAL], SPANNING)),
                 ("refused", plan_for([MATHS, LEGAL, GENERAL], PURE_MATH))]:
    check(f"{label} plan carries a reason", bool(p.reason), True)
    check(f"{label} plan serialises", isinstance(p.as_dict(), dict), True)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all composition-gate tests passed")
