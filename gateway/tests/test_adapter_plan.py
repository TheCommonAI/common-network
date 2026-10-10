"""Tests for on-demand adapter planning.

The plan is a recommendation a node owner may act on, so what it says has to be
true: the right adapters for a region, on a base something is actually running,
one blend per node. Each case below names the rule it encodes.

No database, no network, no embedding model. **Every vector here is built so its
cosine against the thing it is compared to is the number written in the test**,
using the same distinct-axis trick as test_compose.py: a vector with `c` on axis
0 and `sqrt(1 - c^2)` on its own private axis has cosine exactly `c` against the
axis-0 unit vector, and the private axis guarantees no accidental similarity to
any other adapter. So "this adapter sits 0.62 from the cluster" is a fact of the
test rather than a hope about sentence-transformers.

That matters more here than in most suites because the numbers under test are
thresholds. A test that ran a real encoder would be measuring MiniLM, and its
verdict would change with the encoder version rather than with the code.

Run: `python tests/test_adapter_plan.py` from `gateway/`.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import demand  # noqa: E402
from app.adapters import (  # noqa: E402
    NODE_DISTANCE_FLOOR,
    blend_modelfile,
    blend_tag,
    detect_adapter_needs,
    nearest_node_similarity,
    plan_adapters,
    resolve_node_base,
    select_adapters,
)

FAILURES: list[str] = []


def check(name: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL  {name}  (got {got!r}, want {want!r})")
    else:
        print(f"  ok    {name}")


# --- building vectors with exact cosines ------------------------------------
#
# One axis for the cluster centroid (0), then one private axis per vector built.
# DIM is comfortably more than any test needs; the counter below asserts if it
# is ever exhausted. Section 7 also uses *explicit* axes for its second cluster
# and its nodes, chosen near the top of the range so the counter cannot reach
# them and hand some other vector an unintended similarity.

DIM = 256
QWEN = "Qwen/Qwen2.5-1.5B-Instruct"
GROUP = "qwen2.5-1.5b-r16-a32-7t"


def axis(i: int, dim: int = DIM) -> np.ndarray:
    v = np.zeros(dim)
    v[i] = 1.0
    return v


def at(similarity: float, axis_index: int, on_axis: int = 0) -> np.ndarray:
    """A vector whose cosine against `axis(on_axis)` is exactly `similarity`.

    `axis_index` must be unique per vector so two vectors never pick up a
    similarity to each other that the test did not ask for.
    """
    v = similarity * axis(on_axis) + (1.0 - similarity ** 2) ** 0.5 * axis(axis_index)
    return v / np.linalg.norm(v)


CENTROID = axis(0)

# Private axes are handed out by a counter, not derived from the id: `hash()` is
# randomised per process, so a test using it would not be reproducible, and two
# ids could collide onto one axis and pick up a similarity the test never asked
# for. A counter cannot collide before DIM is exhausted.
_PRIVATE = [1]


def next_private() -> int:
    _PRIVATE[0] += 1
    if _PRIVATE[0] >= DIM:
        raise AssertionError("ran out of private axes; raise DIM")
    return _PRIVATE[0]


def adapter_row(id, similarity, *, base=QWEN, group=GROUP, gguf_ref=None, **kw):
    """A row shaped like the `adapters` table, at a chosen cosine from CENTROID."""
    row = {
        "id": id,
        "display_name": id,
        "hf_repo": f"someone/{id}",
        "revision": "0" * 40,
        "gguf_ref": gguf_ref,
        "base_model": base,
        "blend_group": group,
        "domain_text": id,
        "domain_embed": at(similarity, next_private()),
        "size_mb": 71,
        "min_ram_gb": 2,
        "domain_tags": [id],
        "licence": None,
    }
    row.update(kw)
    return row


def node_row(name, similarity, *, base=None, healthy=True, adapter_ids=None,
             catalogue_id=None, private=None):
    return {
        "name": name,
        "capability_embed": at(similarity, private if private is not None
                               else next_private()),
        "base_model": base,
        "catalogue_id": catalogue_id,
        "adapter_ids": adapter_ids,
        "healthy": healthy,
    }


# A regression must read as a FAIL, not abort the suite. Several sections below
# index into a list the code is supposed to have filled -- `plan.assignments[0]`
# and friends -- and if the rule under test breaks, that list is empty and the
# index raises, killing the run before the failure summary prints. Mutation
# testing showed exactly that: removing the one-blend-per-node rule produced a
# traceback and no summary, which is a worse signal than the FAIL it should be.
#
# So those accesses go through `first(...)`, which yields MISSING instead of
# raising -- and MISSING absorbs whatever is done to it next (attribute, item,
# call, `in`, iteration) so the assertion underneath still runs and reports.
def first(seq, default=None):
    return seq[0] if seq else default


class _Missing:
    """An absent record that fails assertions instead of raising."""

    def __getattr__(self, _name):
        return self

    def __getitem__(self, _key):
        return self

    def __call__(self, *_args, **_kw):
        return self

    def __contains__(self, _key):
        return False

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False

    def __eq__(self, other):
        return other is self

    def __hash__(self):
        return id(self)


MISSING = _Missing()


def cluster(size, centroid) -> demand.UnservedCluster:
    return demand.UnservedCluster(size=size, nearest_domain=None,
                                  centroid=list(np.asarray(centroid, dtype=float)))


# ===========================================================================
print("=" * 62)
print("1. Cluster detection — and the centroid that used to be dropped")
print("=" * 62)

# Two tight groups and two singletons. The tight group has 3 members so it
# survives min_size=3; the pair and the singleton do not.
near = np.array([at(0.99, 3), at(0.98, 4), at(0.97, 5)])
vectors = np.vstack([near, axis(6), axis(6), axis(7)])

members = demand.cluster(vectors, 0.6, 3)
check("one cluster survives min_size", len(members), 1)
check("the tight group is the cluster", sorted(first(members, MISSING)), [0, 1, 2])

# The regression guard. `UnservedCluster.centroid` was computed inside analyse()
# and then dropped -- as_dict() never published it, so nothing outside could
# reach it. cluster_centroids() is what un-drops it, and this is what pins the
# value to the *mean* rather than to whichever member happened to come first.
centroids = demand.cluster_centroids(vectors, 0.6, 3)
check("cluster_centroids returns one pair", len(centroids), 1)
got_members, got_centroid = first(centroids, (None, None))
check("centroid's members match cluster()", sorted(got_members or []), [0, 1, 2])
check("centroid is the true mean",
      bool(np.allclose(got_centroid, vectors[got_members].mean(axis=0))), True)
check("centroid is not just the first member",
      bool(np.allclose(got_centroid, vectors[got_members[0]])), False)
check("centroid is unit-length-direction preserving",
      round(float(np.linalg.norm(got_centroid - vectors[got_members].mean(axis=0))), 12), 0.0)

unserved = demand.unserved_clusters(vectors, 0.6, 3)
check("unserved_clusters carries size", first(unserved, MISSING).size, 3)
check("unserved_clusters carries the same centroid",
      bool(np.allclose(first(unserved, MISSING).centroid, got_centroid)), True)
check("unserved_clusters does not publish a domain-tag verdict",
      first(unserved, MISSING).nearest_domain, None)

# ===========================================================================
print()
print("=" * 62)
print("2. Adapter selection")
print("=" * 62)

# Four adapters at known distances from CENTROID, all on one base and in one
# blend group, plus a fifth that is the *nearest of all* but on a different base.
# Making the wrong-base adapter the closest is the point: excluding it has to be
# the base filter's doing, not an accident of ordering.
same_base = [
    adapter_row("near", 0.90),
    adapter_row("mid", 0.75),
    adapter_row("low", 0.62),
    adapter_row("below", 0.55),
]
rows = same_base + [adapter_row("wrongbase", 0.95, base="Other/Base-7B")]

# `margin=1.0` (permissive) throughout this block on purpose: the four filters
# are independent, and each test below is about exactly one of them. These
# adapters sit 0.13-0.15 apart, so the default margin would truncate every set
# here to its anchor and the floor/cap tests would be measuring the margin
# instead of the thing they name. The margin has its own tests at the end of
# this section.
every = select_adapters(CENTROID, same_base, top_k=5, min_similarity=0.0, margin=1.0)
check("no floor, no cap -> the whole group, nearest first",
      [p.id for p in every], ["near", "mid", "low", "below"])
check("selection is ordered nearest-first",
      [round(p.similarity, 4) for p in every], [0.90, 0.75, 0.62, 0.55])

unfiltered = select_adapters(CENTROID, rows, top_k=5, min_similarity=0.0, margin=1.0)
check("without the filter the nearest adapter anchors the set alone",
      [p.id for p in unfiltered], ["wrongbase"])

picked = select_adapters(CENTROID, rows, top_k=5, min_similarity=0.0, margin=1.0,
                         base_models=[QWEN])
check("base_models excludes a nearer adapter on the wrong base",
      "wrongbase" in [p.id for p in picked], False)
check("base_models keeps the rest in order",
      [p.id for p in picked], ["near", "mid", "low", "below"])

capped = select_adapters(CENTROID, same_base, top_k=2, min_similarity=0.0, margin=1.0)
check("top_k caps the set", [p.id for p in capped], ["near", "mid"])

floored = select_adapters(CENTROID, same_base, top_k=5, min_similarity=0.65, margin=1.0)
check("min_similarity drops everything below it", [p.id for p in floored], ["near", "mid"])

check("nothing above the floor -> empty, not a best-effort pick",
      select_adapters(CENTROID, same_base, min_similarity=0.99), [])
check("no adapters at all -> empty", select_adapters(CENTROID, []), [])
check("empty centroid -> empty", select_adapters([], same_base), [])
check("an adapter with no embedding is skipped",
      [p.id for p in select_adapters(CENTROID, [adapter_row("x", 0.9, domain_embed=None)],
                                     min_similarity=0.0)], [])

# Weights: uniform across the members, and they sum to 1.
check("weights are uniform",
      [round(p.weight, 4) for p in every], [0.25, 0.25, 0.25, 0.25])
check("weights sum to 1", round(sum(p.weight for p in every), 12), 1.0)
check("a single-member set carries all the weight",
      [p.weight for p in select_adapters(CENTROID, [first(same_base, MISSING)])], [1.0])

# blend_group is the fusion precondition: averaging requires r, lora_alpha and
# target modules to agree, and the label is how that agreement is stated.
anchor_only = select_adapters(CENTROID, [
    adapter_row("anchor", 0.90, group=GROUP),
    adapter_row("solo", 0.88, group=None),
    adapter_row("foreign", 0.86, group="some-other-shape"),
], min_similarity=0.0)
check("a nearer adapter in another group is not absorbed",
      [p.id for p in anchor_only], ["anchor"])

null_anchor = select_adapters(CENTROID, [
    adapter_row("cannotjoin", 0.90, group=None),
    adapter_row("compatible", 0.88, group=GROUP),
], min_similarity=0.0)
check("a NULL-group anchor stays a single-adapter set",
      [p.id for p in null_anchor], ["cannotjoin"])

require_buildable = select_adapters(CENTROID, [
    adapter_row("hasgguf", 0.80, gguf_ref="https://example.invalid/a.gguf"),
    adapter_row("needsconvert", 0.90),
], min_similarity=0.0, require_buildable=True)
check("require_buildable drops an adapter with no GGUF",
      [p.id for p in require_buildable], ["hasgguf"])
check("buildable reflects gguf_ref, not the similarity",
      select_adapters(CENTROID, [rows[0]])[0].buildable, False)

# --- margin: membership is "as relevant as the anchor", not "relevant" -------
#
# The floor answers "is this adapter relevant at all". On its own it is not
# enough, and the failure is measured rather than hypothetical: on a
# code-debugging centroid the maths profiles score 0.61-0.67 and the code ones
# 0.76-0.78, so every one of them clears a 0.6 floor and `top_k=3` fills its
# third seat with a maths adapter on a code question. The margin is the second
# question -- is this as relevant as *the anchor* -- and these pin it.
margin_near = select_adapters(CENTROID, [
    adapter_row("anchor", 0.90),
    adapter_row("intie", 0.87),      # 0.03 below: a near-tie, keeps its seat
    adapter_row("pastit", 0.80),     # 0.10 below: drops out, and ends the walk
], top_k=5, min_similarity=0.0)
check("an adapter inside the margin joins the anchor",
      [p.id for p in margin_near], ["anchor", "intie"])

# The walk stops at the first pick past the margin because the rows are sorted
# descending -- so a *further* adapter can never re-enter after one drops out.
check("nothing past the margin re-enters behind a dropped one",
      [p.id for p in select_adapters(CENTROID, [
          adapter_row("anchor", 0.90),
          adapter_row("gap", 0.80),
          adapter_row("closer", 0.88),   # nearer than `gap`, sorted ahead of it
      ], top_k=5, min_similarity=0.0)], ["anchor", "closer"])

# The measured consequence, and the reason this rule changes behaviour rather
# than just tidying it: profiles inside one domain sit ~0.10 apart on a
# centroid in that domain (code writing 0.073, maths arithmetic 0.103, medical
# 0.182), so a one-adapter set is now the ordinary outcome on clean demand
# rather than the exception.
far_apart = select_adapters(CENTROID, [
    adapter_row("best", 0.80), adapter_row("second", 0.70),
], top_k=5, min_similarity=0.0)
check("far-apart profiles keep only the anchor", [p.id for p in far_apart], ["best"])
check("and that lone member carries the whole weight",
      [p.weight for p in far_apart], [1.0])

# A set is never emptied by the margin: the anchor is a member before the
# filter is consulted, so the worst case is one adapter, never zero.
check("the margin can never drop the anchor itself",
      len(select_adapters(CENTROID, same_base, min_similarity=0.0)), 1)

# What the margin does NOT do, recorded here so it is not mistaken for an
# oversight later: it cannot repair top-1. On a logic-puzzle centroid the maths
# adapter scores 0.699 against the reasoning adapter's 0.681 and stays the
# anchor; a margin restricts the rest of the set and leaves that choice alone.
check("the margin does not change who anchors the set",
      first(select_adapters(CENTROID, [
          adapter_row("nearest", 0.90), adapter_row("right", 0.86),
      ], min_similarity=0.0), MISSING).id, "nearest")

# ===========================================================================
print()
print("=" * 62)
print("3. blend_tag — determinism is the build-or-reuse logic")
print("=" * 62)

ids = ["math-12k", "math-pilot"]
weights = [0.5, 0.5]
tag = blend_tag(QWEN, ids, weights)
check("same inputs -> same tag", blend_tag(QWEN, ids, weights), tag)
check("adapter order does not change the tag",
      blend_tag(QWEN, list(reversed(ids)), weights), tag)
check("a different base is a different model", blend_tag("Other/Base-7B", ids, weights) != tag, True)
check("different weights are a different model",
      blend_tag(QWEN, ids, [0.7, 0.3]) != tag, True)
check("a different adapter set is a different model",
      blend_tag(QWEN, ["math-12k"], [1.0]) != tag, True)
check("the tag is stable across calls, not random",
      all(blend_tag(QWEN, ids, weights) == tag for _ in range(5)), True)
check("tag shape is blend-<8 hex>",
      (tag.startswith("blend-"), len(tag)), (True, 14))
check("default weights are uniform, so a bare call matches the uniform one",
      blend_tag(QWEN, ids), tag)

def raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False

check("an empty adapter set is refused", raises(lambda: blend_tag(QWEN, [], [])), True)
check("weights that do not line up are refused",
      raises(lambda: blend_tag(QWEN, ids, [1.0])), True)

# ===========================================================================
print()
print("=" * 62)
print("4. resolve_node_base — two namespaces, kept apart")
print("=" * 62)

# The catalogue's base_model convention is an Ollama pull-name; a node reports
# the Hugging Face repo id of its weights. They are never string-equal, so a
# catalogue value is only usable when it is already in the comparable namespace.
check("catalogue repo id wins over the node's own claim",
      resolve_node_base({"catalogue_id": "m1", "base_model": "Node/Says-This"},
                        {"m1": QWEN}), (QWEN, "catalogue"))
check("an Ollama pull-name is skipped, not stripped",
      resolve_node_base({"catalogue_id": "m1", "base_model": QWEN},
                        {"m1": "ollama:llama3.1:8b"}), (QWEN, "node"))
check("a pull-name with nothing else to fall back on resolves to nothing",
      resolve_node_base({"catalogue_id": "m1", "base_model": None},
                        {"m1": "ollama:llama3.1:8b"}), (None, None))
check("a self-reported base is used when the catalogue is silent",
      resolve_node_base({"catalogue_id": None, "base_model": QWEN}, {}), (QWEN, "node"))
check("a catalogue_id the map does not know falls through",
      resolve_node_base({"catalogue_id": "missing", "base_model": QWEN}, {}), (QWEN, "node"))
check("neither source knowing is (None, None), never a guess",
      resolve_node_base({"catalogue_id": None, "base_model": None}, {}), (None, None))
check("a missing key is tolerated", resolve_node_base({}, {}), (None, None))

# ===========================================================================
print()
print("=" * 62)
print("5. blend_modelfile")
print("=" * 62)

mf = blend_modelfile("qwen2.5:1.5b", "/tmp/adapters/math.gguf")
check("exactly FROM then ADAPTER, nothing else",
      mf, "FROM qwen2.5:1.5b\nADAPTER /tmp/adapters/math.gguf\n")
check("SYSTEM is omitted rather than emitted empty", "SYSTEM" in mf, False)
check("no URL can reach the build path", "://" in mf, False)
check("SYSTEM is appended last when given",
      blend_modelfile("qwen2.5:1.5b", "/tmp/a.gguf", system="You are terse."),
      "FROM qwen2.5:1.5b\nADAPTER /tmp/a.gguf\nSYSTEM You are terse.\n")
check("a URL in the base is refused",
      raises(lambda: blend_modelfile("https://example.invalid/b", "/tmp/a.gguf")), True)
check("a URL in the adapter ref is refused",
      raises(lambda: blend_modelfile("qwen2.5:1.5b", "https://example.invalid/a.gguf")), True)

# ===========================================================================
print()
print("=" * 62)
print("6. The nearest node, and when a cluster is adapter work")
print("=" * 62)

# This is the comparison that decides whether a region is already served, and
# it is new work: nothing before it compared a centroid against a node's
# capability embedding. demand.analyse() compares against domain *tags*, which
# is a different measurement and not a substitute.
check("the nearest node is returned with its similarity",
      nearest_node_similarity(CENTROID, [node_row("far", 0.20), node_row("close", 0.80)]),
      ("close", 0.80))
check("a node with no embedding is skipped even when it would be nearest",
      nearest_node_similarity(CENTROID, [
          dict(node_row("blank", 0.90), capability_embed=None),
          node_row("real", 0.40),
      ])[0], "real")
check("a tie breaks on name, not on row order",
      nearest_node_similarity(CENTROID, [node_row("bravo", 0.50, private=253),
                                         node_row("alpha", 0.50, private=254)])[0], "alpha")
check("no nodes at all -> (None, 0.0)",
      nearest_node_similarity(CENTROID, []), (None, 0.0))
check("an empty centroid -> (None, 0.0)",
      nearest_node_similarity([], [node_row("any", 0.90)]), (None, 0.0))

# The adapter that would be recommended for CENTROID. The two scores are 0.03
# apart on purpose: that is inside ADAPTER_MARGIN, so both keep their seats and
# this section can test what a *multi-adapter* need carries -- the set, the
# tag, the weights. A 0.15 gap (the first draft of this fixture) now yields a
# one-adapter need, which is the common real case and is tested in section 7
# rather than here.
mathlike = [adapter_row("math-12k", 0.89), adapter_row("math-pilot", 0.86)]

served_node = [node_row("codebox", 0.90)]
need_free = detect_adapter_needs([cluster(9, CENTROID)], served_node, mathlike)
check("a cluster already covered yields no need", need_free, [])

far_node = [node_row("codebox", 0.30)]
needs = detect_adapter_needs([cluster(9, CENTROID)], far_node, mathlike)
check("an uncovered cluster yields one need", len(needs), 1)
check("the need records the nearest node and how near",
      (first(needs, MISSING).nearest_node, round(first(needs, MISSING).nearest_node_similarity, 4)), ("codebox", 0.30))
check("the need names the adapters", first(needs, MISSING).adapter_ids, ["math-12k", "math-pilot"])
check("the need records the base", first(needs, MISSING).base_model, QWEN)
check("the need carries a tag", first(needs, MISSING).blend_tag, blend_tag(QWEN, ["math-12k", "math-pilot"],
                                                             [0.5, 0.5]))
check("the need says the weights are not fitted", "uniform" in first(needs, MISSING).note, True)

# The floor is a threshold, and it has to be the node comparison driving it --
# not the domain-tag bound UnservedCluster carries, which is a different thing.
check("a node exactly at the floor counts as covering",
      detect_adapter_needs([cluster(9, CENTROID)], [node_row("edge", NODE_DISTANCE_FLOOR)],
                           mathlike), [])
check("a node just below the floor does not",
      len(detect_adapter_needs([cluster(9, CENTROID)],
                               [node_row("edge", NODE_DISTANCE_FLOOR - 0.01)], mathlike)), 1)

# Real demand nothing is near *enough* to is still reported. Dropping it would
# make the plan read as though that region did not exist.
nobody = detect_adapter_needs([cluster(4, CENTROID)], [node_row("codebox", 0.10)],
                              [adapter_row("unrelated", 0.20)])
check("an unservable region is still reported", len(nobody), 1)
check("with no adapters selected", first(nobody, MISSING).adapter_ids, [])
check("and a note saying it needs a node, not a fusion",
      "needs a node, not a fusion" in first(nobody, MISSING).note, True)

# ===========================================================================
print()
print("=" * 62)
print("7. plan_adapters — assignment, one blend per node, reuse")
print("=" * 62)

needs_rows = [adapter_row("math-12k", 0.89), adapter_row("math-pilot", 0.86)]
uncovered = [cluster(9, CENTROID)]

# (a) no node reports the base at all — the honest answer, since every adapter
# row here is on QWEN and nothing in the network runs it.
plan = plan_adapters(uncovered, [node_row("otherbase", 0.20, base="Other/Base-7B")],
                     needs_rows)
check("no assignment when no node reports the base", len(plan.assignments), 0)
check("the refusal says which base was missing", "Other/Base-7B" in first(plan.unassigned, MISSING)["reason"] or
      QWEN in first(plan.unassigned, MISSING)["reason"], True)
check("the refusal names the base the adapters fit", QWEN in first(plan.unassigned, MISSING)["reason"], True)

# (b) a node with a NULL base is not a candidate either.
plan = plan_adapters(uncovered, [node_row("nobase", 0.20, base=None)], needs_rows)
check("a node with no base is not assignable", len(plan.assignments), 0)
check("and the reason is the missing base",
      "no node reports base" in first(plan.unassigned, MISSING)["reason"], True)

# (c) the happy path: one node on the right base.
plan = plan_adapters(uncovered, [node_row("qwenbox", 0.20, base=QWEN)], needs_rows)
check("the need is assigned", len(plan.assignments), 1)
a = first(plan.assignments, MISSING)
check("assigned to the node on that base", a.node, "qwenbox")
check("the base came from the node itself", a.base_source, "node")
check("the assignment carries the adapter set", a.adapter_ids, ["math-12k", "math-pilot"])
check("the assignment carries the tag", a.blend_tag,
      blend_tag(QWEN, ["math-12k", "math-pilot"], [0.5, 0.5]))
check("uniform weights reach the assignment", a.weights, [0.5, 0.5])
check("nothing to reuse yet", a.reused, False)
check("the note tells the operator to build and re-register",
      "build" in a.note and "re-register" in a.note, True)

# (d) the catalogue's base is preferred when it is usable.
plan = plan_adapters(uncovered,
                     [node_row("cataloguebox", 0.20, base="Node/Says-This",
                               catalogue_id="m1")],
                     needs_rows, catalogue_base_by_id={"m1": QWEN})
check("a catalogue-derived base is preferred over the node's claim",
      first(plan.assignments, MISSING).base_source, "catalogue")

# (e) reuse: the node already serves exactly this set, so there is nothing to
# build. This is what blend_tag's determinism buys.
plan = plan_adapters(uncovered,
                     [node_row("warm", 0.20, base=QWEN,
                               adapter_ids=["math-pilot", "math-12k"])],
                     needs_rows)
check("a node already serving the set is reused", first(plan.assignments, MISSING).reused, True)
check("and is told not to rebuild", "do not rebuild" in first(plan.assignments, MISSING).note, True)

# Note the order-independence: adapter_ids are compared sorted, so a node that
# reports the same set in a different order still counts as serving it.
reordered = plan_adapters(uncovered,
                          [node_row("warm", 0.20, base=QWEN,
                                    adapter_ids=["math-12k", "math-pilot"])],
                          needs_rows)
check("reuse ignores the order the node reported",
      first(reordered.assignments, MISSING).reused, True)

# (e2) the measured common case, not an edge case: profiles inside one domain
# sit ~0.10 apart on a centroid in that domain, so the margin leaves a
# one-adapter set. Everything downstream has to hold for a set of one.
apart_rows = [adapter_row("math-12k", 0.89), adapter_row("math-pilot", 0.74)]
plan = plan_adapters(uncovered, [node_row("qwenbox", 0.20, base=QWEN)], apart_rows)
solo = first(plan.assignments, MISSING)
check("a far-apart pair yields a one-adapter need", solo.adapter_ids, ["math-12k"])
check("a lone member carries the whole weight", solo.weights, [1.0])
check("the tag is the one-adapter tag",
      solo.blend_tag, blend_tag(QWEN, ["math-12k"], [1.0]))
check("which is not the two-adapter tag",
      solo.blend_tag != blend_tag(QWEN, ["math-12k", "math-pilot"], [0.5, 0.5]), True)

plan = plan_adapters(uncovered,
                     [node_row("warm", 0.20, base=QWEN, adapter_ids=["math-12k"])],
                     apart_rows)
check("a node serving the one-adapter blend is reused",
      first(plan.assignments, MISSING).reused, True)

# (f) one blend per node. Two different regions need two different sets, and
# there is only one node on the base.
#   cluster A (10 requests) -> alpha ; cluster B (5 requests) -> beta
# The two adapters are each 0.9 from their own region and orthogonal to the
# other's, so each region gets a genuinely different set.
SECOND_CENTROID = 240
alpha = {"id": "alpha", "display_name": "alpha", "hf_repo": "s/alpha", "revision": None,
         "gguf_ref": None, "base_model": QWEN, "blend_group": GROUP, "domain_text": "alpha",
         "domain_embed": list(at(0.90, 250, on_axis=0)), "size_mb": 1, "min_ram_gb": 1,
         "domain_tags": [], "licence": None}
beta = dict(alpha, id="beta", display_name="beta", hf_repo="s/beta",
            domain_embed=list(at(0.90, 251, on_axis=SECOND_CENTROID)))
# A node far from both centroids, so neither region is covered.
both_uncovered = [node_row("only", 0.0, base=QWEN, private=252)]
two = plan_adapters([cluster(10, axis(0)), cluster(5, axis(SECOND_CENTROID))], both_uncovered,
                    [alpha, beta])
check("two regions, one node -> one assignment", len(two.assignments), 1)
check("the bigger region wins", first(two.assignments, MISSING).requests, 10)
check("the second region is refused, not stacked on the same node",
      len(two.unassigned), 1)
check("and the refusal is the one-blend rule",
      "one blend per node" in first(two.unassigned, MISSING)["reason"], True)

# (g) a second node on the base takes the second region.
two_nodes = [node_row("only", 0.0, base=QWEN, private=252),
             node_row("spare", 0.0, base=QWEN, private=253)]
two = plan_adapters([cluster(10, axis(0)), cluster(5, axis(SECOND_CENTROID))], two_nodes,
                    [alpha, beta])
check("a second node on the base takes the second region", len(two.assignments), 2)
check("the two assignments are different blends",
      len({a.blend_tag for a in two.assignments}), 2)
check("each node is claimed once",
      sorted(a.node for a in two.assignments), ["only", "spare"])

# (h) healthy before unhealthy, when both are free.
unhealthy_first = plan_adapters(uncovered, [
    node_row("sick", 0.10, base=QWEN, healthy=False, private=252),
    node_row("well", 0.10, base=QWEN, healthy=True, private=253),
], needs_rows)
check("a healthy node is preferred over an unhealthy one on the same base",
      first(unhealthy_first.assignments, MISSING).node, "well")

# (i) a node running no blend is preferred over one that is, so a working
# specialist is not replaced without cause.
busy = plan_adapters(uncovered, [
    node_row("busy", 0.10, base=QWEN, adapter_ids=["something-else"], private=252),
    node_row("idle", 0.10, base=QWEN, private=253),
], needs_rows)
check("an idle node is preferred over one already running a blend",
      first(busy.assignments, MISSING).node, "idle")

# (j) a node already serving exactly this set outranks both, because it needs
# no build at all.
reuse_wins = plan_adapters(uncovered, [
    node_row("already", 0.10, base=QWEN, adapter_ids=["math-12k", "math-pilot"], private=252),
    node_row("idle", 0.10, base=QWEN, private=253),
], needs_rows)
check("a node already serving the set is chosen over an idle one",
      first(reuse_wins.assignments, MISSING).node, "already")
check("and marked reused", first(reuse_wins.assignments, MISSING).reused, True)

# (k) the cap counts assignments, and the plan says what it dropped.
capped = plan_adapters([cluster(10, axis(0)), cluster(5, axis(SECOND_CENTROID))], two_nodes,
                       [alpha, beta], max_assignments=1)
check("max_assignments caps the plan", len(capped.assignments), 1)
check("the dropped region says why",
      "already recommends 1 assignment" in first(capped.unassigned, MISSING)["reason"], True)

# (l) ordering is biggest-demand-first, so the same input always plans the same
# thing regardless of the order the clusters arrived in.
forward = plan_adapters([cluster(10, axis(0)), cluster(5, axis(SECOND_CENTROID))], two_nodes, [alpha, beta])
backward = plan_adapters([cluster(5, axis(SECOND_CENTROID)), cluster(10, axis(0))], two_nodes, [alpha, beta])
check("cluster input order does not change the plan",
      forward.as_dict()["assignments"], backward.as_dict()["assignments"])

# (m) covered regions are reported separately and never assigned.
mixed = plan_adapters([cluster(10, axis(0)), cluster(3, axis(SECOND_CENTROID))],
                      [node_row("only", 0.0, base=QWEN, private=252),
                       node_row("covers-second", 0.90, private=254, base=None)],
                      [alpha, beta])
check("a covered region is listed as covered", len(mixed.covered), 1)
check("and does not consume an assignment", len(mixed.assignments), 1)
check("clusters_considered counts every cluster", mixed.clusters_considered, 2)

# ===========================================================================
print()
print("=" * 62)
print("8. The published shape leaks nothing")
print("=" * 62)

# /adapters/plan reads `select * from nodes`, which brings back endpoint_url --
# live tunnel URLs to unauthenticated Ollama instances. None of it may reach the
# response. See also tests/test_node_url_exposure.py, which guards the same rule
# structurally on the public node models.
payload = plan_adapters(uncovered, [node_row("qwenbox", 0.20, base=QWEN)], needs_rows).as_dict()
check("the plan publishes only the documented keys",
      sorted(payload), ["assignments", "clusters_considered", "covered", "mode",
                        "unassigned"])

def any_location_shaped(obj, path="") -> str | None:
    tokens = {"url", "uri", "host", "hostname", "endpoint", "address", "tunnel"}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(t in k.split("_") for t in tokens):
                return f"{path}.{k}"
            found = any_location_shaped(v, f"{path}.{k}")
            if found:
                return found
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found = any_location_shaped(v, f"{path}[{i}]")
            if found:
                return found
    return None

check("no location-shaped key anywhere in the plan",
      any_location_shaped(payload), None)
check("no node row is published, only names",
      "endpoint_url" not in str(payload), True)

# The need/assignment shapes are the ones an operator acts on; pin their keys so
# a field cannot be added to the response without a test noticing.
need = first(detect_adapter_needs([cluster(9, CENTROID)], [node_row("codebox", 0.30)],
                                     needs_rows), MISSING)
pick = first(need.as_dict()["selected_adapters"], MISSING)
check("an adapter pick publishes no embedding",
      "domain_embed" in pick, False)
check("an adapter pick says whether a GGUF exists",
      "buildable" in pick, True)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all adapter-plan tests passed")
