"""Tests for the composition gate.

The gate is where the v0.1 findings are actually enforced, so it gets tested
the way a finding deserves: each case below names the result it encodes.

No database, no network, no real embedding model. **Each domain tag gets its
own axis**, so a request vector states each tag's cosine similarity directly
and a test can say "this request is 0.72 about maths and 0.11 about code"
rather than hoping a real encoder produces that. The logic under test is
"given these similarities, compose or not"; using a real encoder would test
sentence-transformers instead.

An earlier version of this file used one-hot vectors shared between related
tags, which made several tags score *identically*. Real embeddings never tie
exactly, and the ties were quietly sending every case down the standard-
deviation-is-zero branch — the tests passed while exercising a branch that
cannot occur in production. Distinct axes make ties impossible.

Run: `python tests/test_compose.py` from `gateway/`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import embedder, router  # noqa: E402

# One axis per tag. Index into TAGS is the axis.
TAGS = ["math", "arithmetic", "legal", "south-australia-law", "code", "writing", "general"]
AXIS = {t: i for i, t in enumerate(TAGS)}


def unit(tag: str) -> list[float]:
    v = [0.0] * len(TAGS)
    v[AXIS[tag]] = 1.0
    return v


def fake_embed(text: str) -> list[float]:
    return unit(text) if text in AXIS else [0.3] * len(TAGS)


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


def request(**sims: float) -> list[float]:
    """A request vector whose cosine against each tag is what you asked for.

    Each tag is a distinct unit axis, so cos(request, tag_i) == request[i] once
    the vector is normalised — the numbers in the test are the similarities.
    """
    v = [sims.get(t, 0.05) for t in TAGS]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def node(name: str, tags: list[str], *, aggregate: bool = False, embed_tag: str | None = None) -> dict:
    return {
        "id": name, "name": name, "domain_tags": tags,
        "capability_embed": unit(embed_tag or tags[0]),
        "cost_per_1k": 0, "avg_latency_ms": 100, "model_name": name,
        "endpoint_url": "http://x/v1", "can_aggregate": aggregate,
    }


def plan_for(nodes: list[dict], req: list[float], mode: str | None = None):
    return plan_panel(score_nodes(nodes, req), req, mode_override=mode)


MATHS = node("mathstral", ["math"])
CODER = node("coder", ["code"])
LEGAL = node("cgla", ["legal"])
WRITER = node("writer", ["writing"])
GENERAL = node("generalist", ["general"])

# Five lanes declared, so the relative (standard-deviation) path is in play —
# the regime a real network is in.
FIVE_LANE = [MATHS, CODER, LEGAL, WRITER, GENERAL]

# Clearly about maths and law, and clearly not about code or writing.
SPANNING = request(math=0.72, legal=0.70, code=0.11, writing=0.09)
# Clearly about one thing.
PURE_MATH = request(math=0.80, arithmetic=0.30, legal=0.12, code=0.10, writing=0.08)
# Similar to everything, standing out against nothing. This is the shape of the
# off-topic queries an absolute floor waved through: measured against the real
# embedder, "write me a poem about the sea" scores 0.553 on `math` — higher
# than an actual arithmetic question's 0.537.
FLAT = request(math=0.53, legal=0.51, code=0.54, writing=0.52)


print("\ncomposing when it should")
plan = plan_for(FIVE_LANE, SPANNING)
check("spans two standout domains -> composes", plan.compose, True)
check("panel is the two specialists",
      sorted(m.name for m in plan.members), ["cgla", "mathstral"])
check("seated under the domains they lead",
      sorted(m.domain for m in plan.members), ["legal", "math"])
check("generalist aggregates rather than answering",
      plan.aggregator["name"] if plan.aggregator else None, "generalist")
check("aggregator is off-panel", plan.aggregator_in_panel, False)

print("\nrelevance is relative, not absolute")
# The measurement that drove this: cosine against a short tag sits ~0.42-0.64
# for everything, so on-topic and off-topic ranges overlap completely and no
# absolute floor separates them. What separates them is whether a domain stands
# out against the request's OWN profile.
check("a request similar to everything -> no panel", plan_for(FIVE_LANE, FLAT).compose, False)
check("...and says it is single-domain",
      "single-domain" in plan_for(FIVE_LANE, FLAT).reason, True)
check("a barely-tilted request -> no panel",
      plan_for(FIVE_LANE, request(math=0.55, legal=0.53, code=0.52, writing=0.51)).compose, False)
# High absolute similarity across the board must not compose. Under the old
# absolute floor every one of these cleared it.
check("uniformly HIGH similarity is still flat -> no panel",
      plan_for(FIVE_LANE, request(math=0.9, legal=0.9, code=0.9, writing=0.9)).compose, False)

print("\nthe domination gate")
# The Experiment 2 result, encoded. One node best at every matched domain means
# nothing on the network beats it at anything relevant, so composing can only
# dilute it. seam-findings.md §2.
OMNI = node("omni", ["math", "legal"], embed_tag="math")
plan = plan_for([OMNI, CODER, WRITER, GENERAL], SPANNING)
check("one node best at every matched domain -> refuses", plan.compose, False)
check("and says why", "not dominated" in plan.reason, True)

print("\nsingle-domain requests")
plan = plan_for(FIVE_LANE, PURE_MATH)
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
SYNONYM = node("mathstral2", ["math", "arithmetic"])
plan = plan_for([SYNONYM, CODER, WRITER, GENERAL],
                request(math=0.80, arithmetic=0.74, code=0.10, writing=0.08))
check("synonymous tags on one node -> no panel", plan.compose, False)
check("caught by the domination gate", "not dominated" in plan.reason, True)

print("\ntoo few nodes")
check("one node -> nothing to compose with", plan_for([MATHS], SPANNING).compose, False)

print("\nsmall networks fall back to the absolute floor")
# Two or three declared domains give no background to measure against, so the
# relative test is unavailable and this falls back to the absolute floor. A
# three-node lab composing a shade too eagerly beats a three-node lab that can
# never compose at all — and the domination gate still applies.
plan = plan_for([MATHS, LEGAL], SPANNING)
check("small network still composes", plan.compose, True)
check("and flags the aggregator as a panel member", plan.aggregator_in_panel, True)

print("\nmode overrides")
check("never -> refuses", plan_for(FIVE_LANE, SPANNING, "never").compose, False)
check("always -> composes a request the gates would decline",
      plan_for(FIVE_LANE, FLAT, "always").compose, True)
# 'always' relaxes the worth-it heuristics, not the finding. Composing a node
# with one it dominates was measured to lose; no header should enable it.
# The network here is OMNI plus a generalist, so OMNI leads every domain there
# is — no third lane for another node to pick up and make the panel look
# non-dominated. (With CODER present, 'always' reaches past the relevant
# domains, grabs `code`, and CODER legitimately leads it: that panel really
# isn't dominated, just pointless, which is what 'always' is for.)
check("always still refuses a dominated panel",
      plan_for([OMNI, GENERAL], SPANNING, "always").compose, False)

print("\npanel size")
plan = plan_for([MATHS, CODER, LEGAL, WRITER, GENERAL], FLAT, "always")
check("capped at compose_max_panel", len(plan.members) <= settings.compose_max_panel, True)

print("\naggregator selection")
VOLUNTEER = node("volunteer", ["writing"], aggregate=True)
plan = plan_for([MATHS, LEGAL, CODER, VOLUNTEER, GENERAL], SPANNING)
check("can_aggregate node preferred",
      plan.aggregator["name"] if plan.aggregator else None, "volunteer")

print("\nexplanation is always present")
for label, p in [("composed", plan_for(FIVE_LANE, SPANNING)),
                 ("refused", plan_for(FIVE_LANE, PURE_MATH))]:
    check(f"{label} plan carries a reason", bool(p.reason), True)
    check(f"{label} plan serialises", isinstance(p.as_dict(), dict), True)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all composition-gate tests passed")
