"""On-demand specialists: which LoRA adapters sit nearest a demand cluster, and
which node could fuse them.

The catalogue answers "what specialists are known" -- a fixed list somebody
curated, and a node picks one off it when it joins. This module answers the
other direction: given a cluster of requests no node covers, which small public
LoRA adapters sit nearest that cluster, on which base, and which node on that
base could average them into a specialist that did not exist an hour ago.

**The gateway recommends. It never builds.** join/worker.py pins exactly one
model name at startup and /api/create is a deliberate 404, so there is no path
by which anything here could make a node load an adapter -- and there must not
be one. What this produces is a *plan*: a node name, an adapter set, a tag. The
node's owner runs `common adapters build` and re-registers on the result. That
is the same posture POST /assign and POST /demand/plan already have, and it is
the reason this file has no write path to anything but its own table.

The shape follows app/compose.py: pure functions over plain rows and vectors,
dataclasses with as_dict(), and one thin DB-reading endpoint at the bottom.
Every decision function takes pre-made vectors, so the tests can call them with
hand-built ones and no database, no network and no embedder.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from fastapi import APIRouter, Query

from app import db, embedder
from app.config import settings
from app.demand import load_unserved_clusters

router = APIRouter()


# Below this, a demand cluster counts as *already served* -- the network has a
# node whose declared capability is closer to this region than two requests in
# the same cluster are to each other.
#
# That tie to the clustering threshold is the argument for the number, not a fit
# to data: `cluster_threshold` (0.6, demand.analyse) is what makes a set of
# requests "the same region" in the first place, so a node that far from the
# centroid is a node covering that region. Anything lower and the test says
# every region is served; anything higher and it says none is.
#
# Measured, 9 synthetic demands against the 6 dev nodes. Nearest-node cosine:
#
#   maths word problems 0.616 | code debugging 0.638 | code writing 0.629
#   algebra             0.668 | translation    0.576 | maritime law 0.534
#   medical advice      0.515 | beekeeping      0.510 | logic puzzle 0.438
#
# So 0.6 splits these into "the maths and code nodes really do cover that" and
# "nothing here covers logic puzzles, medicine, translation or beekeeping" --
# which is the true state of that network. **Nine hand-written sentences is not
# a calibration.** Re-measure against real demand clusters before trusting it;
# it is a default that lets the feature fire, not a validated threshold.
NODE_DISTANCE_FLOOR = 0.6

# Below this, an adapter is not close enough to a demand region to be worth
# recommending for it. Separate constant from NODE_DISTANCE_FLOOR even though
# the values coincide: they are different comparisons (a demand region against a
# node's capability paragraph, and the same region against an adapter's domain
# profile) with different distributions, and they must be free to move apart.
#
# Measured on the same 9 demands against the 7 seeded adapters, best-adapter
# cosine, after the seed profiles were made topical rather than generic:
#
#   maths -> math-12k        0.891   | logic puzzle -> math-12k   0.464
#   algebra -> math-adaanchor 0.725  | translation -> reasoning   0.568
#   code writing -> code-r16 0.728   | beekeeping -> medical      0.508
#   medical -> medical       0.714   | maritime -> math-12k       0.554
#   code debugging -> code-r16 0.697
#
# True matches land 0.69-0.89 and mismatches 0.43-0.57, so 0.6 sits in the gap.
# Note it *loses* one true positive -- "logic puzzle" should reach `reasoning`
# and scores only 0.448 -- and that with the original generic one-line profiles
# math-12k was the nearest adapter for five of seven unrelated demands. So the
# profile wording is doing as much work as the threshold, and neither is
# validated. Report, do not trust.
ADAPTER_MIN_SIMILARITY = 0.6

# Nodes report the Hugging Face repo id of the weights they run, e.g.
# 'Qwen/Qwen2.5-1.5B-Instruct' -- the same namespace as adapters.base_model,
# and the identifier the adapter was fitted against.
#
# catalogue_models.base_model uses a *different* convention: an Ollama
# pull-name like 'ollama:llama3.1:8b', which says what to fetch rather than
# what the weights are. The two are never equal, so a node base resolved from
# the catalogue is only used when it is in the comparable namespace. See
# resolve_node_base.
OLLAMA_PULL_PREFIX = "ollama:"


def _cosine(a, b) -> float:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


# --- Adapters ---------------------------------------------------------------

@dataclass
class AdapterPick:
    """One adapter, and how well it matches a particular cluster."""
    id: str
    display_name: str
    hf_repo: str
    base_model: str
    similarity: float
    revision: str | None = None
    gguf_ref: str | None = None
    blend_group: str | None = None
    size_mb: int | None = None
    min_ram_gb: int | None = None
    domain_tags: list[str] | None = None
    licence: str | None = None

    # Share of the blend this adapter carries. Uniform by default, and the
    # default is a declared choice rather than a fitted one -- see the note in
    # select_adapters.
    weight: float = 0.0

    @property
    def buildable(self) -> bool:
        """Whether a GGUF is already published for this adapter.

        Not a gate on being recommended -- it is a statement about how much
        work the operator has to do. gguf_ref is a reference to *fetch from*,
        never the value that reaches a Modelfile (see blend_modelfile), so an
        adapter without one can still be recommended; the operator converts it
        first, out of band.
        """
        return bool(self.gguf_ref)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "base_model": self.base_model,
            "blend_group": self.blend_group,
            "similarity": round(self.similarity, 4),
            "weight": round(self.weight, 4),
            "size_mb": self.size_mb,
            "min_ram_gb": self.min_ram_gb,
            "domain_tags": self.domain_tags,
            "licence": self.licence,
            "gguf_ref": self.gguf_ref,
            "buildable": self.buildable,
        }


def _pick_from_row(row, similarity: float) -> AdapterPick:
    return AdapterPick(
        id=row["id"],
        display_name=row["display_name"],
        hf_repo=row["hf_repo"],
        base_model=row["base_model"],
        similarity=similarity,
        revision=row["revision"],
        gguf_ref=row["gguf_ref"],
        blend_group=row["blend_group"],
        size_mb=row["size_mb"],
        min_ram_gb=row["min_ram_gb"],
        domain_tags=row["domain_tags"],
        licence=row["licence"],
    )


def select_adapters(
    centroid,
    adapter_rows: list[dict],
    *,
    base_models: Sequence[str] | None = None,
    top_k: int = 3,
    min_similarity: float = ADAPTER_MIN_SIMILARITY,
    require_buildable: bool = False,
) -> list[AdapterPick]:
    """The adapters nearest a cluster's centroid, as one fusable set.

    Three filters, in order, and each one is a different kind of "no":

    * `base_models` -- a hard restriction on which bases are acceptable at all.
      A LoRA is meaningless on weights it was not fitted against, so an adapter
      on the wrong base is not a weaker recommendation, it is an invalid one.
    * `min_similarity` -- how close is close enough to be worth the operator's
      time. Below the floor nothing is recommended at all.
    * `require_buildable` -- whether the operator must not have to convert a
      GGUF themselves. Off by default, because refusing to *name* the right
      adapter for a cluster just because nobody has published its GGUF would
      hide the gap rather than report it.

    What comes back is a *set*, not a ranking, because a set is what gets fused
    and the set has to be internally consistent. The nearest adapter is the
    anchor -- it is the most relevant thing to this demand -- and the rest of
    the set is whatever else is both near enough and *compatible with the
    anchor*: same base, and a non-null blend_group matching the anchor's.

    That compatibility check is the whole reason blend_group exists. LoraHub
    averaging sums the A and B matrices across adapters and multiplies the sums,
    which is only a LoRA at all if every member agrees on r, lora_alpha and
    target modules. Of the seven adapters known for Qwen2.5-1.5B-Instruct, six
    agree exactly and the medical one (r=64, alpha=16, 4 targets) does not. An
    anchor with a NULL blend_group means a single-adapter set: that adapter
    alone, which is a perfectly good specialist, and never a pair.

    **Weights are uniform, and that is a declared default rather than a fitted
    one.** Stage 1 searched lambda on held-out items; this path has no labelled
    data at plan time, so there is nothing to search against. Uniform is the
    only honest choice available here, and it means the blend the network would
    build is not the same object as the blend Stage 1 measured -- the README
    says so, and nothing here should be read as carrying that result over.
    """
    vec = np.asarray(centroid, dtype=float)
    if vec.size == 0 or not adapter_rows:
        return []

    scored: list[AdapterPick] = []
    for row in adapter_rows:
        if row["domain_embed"] is None:
            continue
        if base_models is not None and row["base_model"] not in base_models:
            continue
        if require_buildable and not row["gguf_ref"]:
            continue
        sim = _cosine(vec, row["domain_embed"])
        if sim < min_similarity:
            continue
        scored.append(_pick_from_row(row, sim))

    if not scored:
        return []

    # Tie-break on id so the same inputs give the same set: two adapters scoring
    # identically must not be ordered by whatever the database happened to
    # return first, or the blend tag would drift between identical plans.
    scored.sort(key=lambda p: (-p.similarity, p.id))

    anchor = scored[0]
    members = [anchor]
    for pick in scored[1:]:
        if len(members) >= top_k:
            break
        if anchor.blend_group is None:
            break
        if pick.blend_group == anchor.blend_group and pick.base_model == anchor.base_model:
            members.append(pick)

    share = 1.0 / len(members)
    for pick in members:
        pick.weight = share
    return members


def blend_tag(base_model: str, adapter_ids: Sequence[str],
              weights: Sequence[float] | None = None) -> str:
    """The name of the fused model: `blend-<8 hex>`.

    **One implementation, here.** The CLI is stdlib-only and cannot import
    gateway code, so it uses the tag *verbatim from the plan response* rather
    than recomputing it -- a second copy would drift, and then a node would
    serve a model under a name the gateway does not recognise.

    Determinism is the build-or-reuse logic, not a nicety: the same adapter set
    on the same base yields the same tag, so a second cluster that lands on a
    node already serving that blend reuses it instead of rebuilding it.

    The weights are in the hash on purpose. Two builds over the same adapters
    with different lambda are two different models, and a tag that ignored the
    weights would let the second one silently reuse the first one's artifact.
    Rounded to 4dp so a float that differs in the last bit does not flip the
    name.
    """
    if not adapter_ids:
        raise ValueError("a blend needs at least one adapter")
    # `weights` is optional and means uniform, which is the same declared
    # default select_adapters uses. Spelled out here rather than left to the
    # callers, because the signature has always promised it and a default that
    # raised TypeError is the kind of thing a caller discovers in production.
    if weights is None:
        weights = [1.0 / len(adapter_ids)] * len(adapter_ids)
    elif len(weights) != len(adapter_ids):
        raise ValueError("weights must line up with adapter_ids, one per adapter")

    pairs = sorted(zip(adapter_ids, weights))
    payload = "|".join([base_model] + [f"{i}:{w:.4f}" for i, w in pairs])
    return "blend-" + hashlib.sha1(payload.encode()).hexdigest()[:8]


def blend_modelfile(base_model: str, adapter_ref: str, *, system: str | None = None) -> str:
    """The Modelfile that builds the blend, as a string. Written by the CLI, run
    by `ollama create` on the node's own machine; the gateway never sees it.

    Two things this function refuses to do, both deliberately:

    * **No URL may reach either line.** `adapter_ref` is a *local path* to an
      already-converted GGUF, and the plan response it comes from is data the
      gateway assembled. An `ADAPTER https://...` line would turn `ollama
      create` into a fetch of a remote artifact chosen by whatever ended up in
      the table -- a build path steered by a value, which is exactly the shape
      this project keeps out of the network. So any scheme is refused. Note the
      asymmetry: `gguf_ref` in the table *is* allowed to be a remote reference
      and is meant to be downloaded, but it is downloaded first and what is
      passed here is the resulting local file.
    * **No guessing about the base's name.** `base_model` here is whatever goes
      on the FROM line, which must be a name Ollama can already resolve on that
      machine -- a local tag or a GGUF path, not the Hugging Face repo id this
      table stores. Those are two names for the same weights and the caller is
      the only thing that knows the local one.

    SYSTEM is omitted rather than emitted empty, so a Modelfile without one does
    not overwrite the base's own system prompt with nothing.
    """
    for label, value in (("base_model", base_model), ("adapter_ref", adapter_ref)):
        if "://" in value:
            raise ValueError(
                f"{label} must be a local path, not a URL "
                f"(got {value!r}) -- a remote fetch must never be reachable "
                f"from the build path"
            )

    lines = [f"FROM {base_model}", f"ADAPTER {adapter_ref}"]
    if system:
        lines.append(f"SYSTEM {system}")
    return "\n".join(lines) + "\n"


# --- Nodes ------------------------------------------------------------------

def resolve_node_base(node: dict, catalogue_base_by_id: dict[str, str]) -> tuple[str | None, str | None]:
    """What base a node runs, and where that answer came from.

    Preference is catalogue-then-self-reported, for the reason the plan gives: a
    node's own `base_model` is a claim by a stranger, and where the catalogue
    knows better, ours wins.

    With one correction the plan did not have to state because it never met the
    data. The two columns are in *different namespaces*: nodes report the
    Hugging Face repo id of their weights, while catalogue_models.base_model
    holds an Ollama pull-name ('ollama:llama3.1:8b'). A node running
    Qwen2.5-1.5B-Instruct has weights whose repo id is
    'Qwen/Qwen2.5-1.5B-Instruct' and whose local pull-name might be
    'ollama:qwen2.5:1.5b-instruct' -- and those are never string-equal, so
    comparing them directly would make every catalogue-resolved node silently
    unmatchable against every adapter.

    So a catalogue value is only used when it is in the comparable namespace.
    A pull-name is skipped, not stripped: the prefix is a statement that the
    value names what to *fetch*, and inventing the repo id from it would be
    guessing at exactly the thing this function exists to be certain about.

    Returns `(None, None)` when neither source knows, and the rule downstream is
    refuse-to-assign, never guess -- which is also what happens today, since
    every catalogue entry has a NULL base_model and no node has a catalogue_id.
    This preference is inert against the current data; it is here so that a
    curator who fills the column gets the behaviour the plan describes.
    """
    catalogue_id = node.get("catalogue_id")
    if catalogue_id:
        from_catalogue = catalogue_base_by_id.get(catalogue_id)
        if from_catalogue and not from_catalogue.startswith(OLLAMA_PULL_PREFIX):
            return from_catalogue, "catalogue"

    self_reported = node.get("base_model")
    if self_reported:
        return self_reported, "node"

    return None, None


def nearest_node_similarity(centroid, node_rows: list[dict]) -> tuple[str | None, float]:
    """The node whose declared capability is closest to this cluster, and how
    close.

    This comparison is new work, and it is the one that decides whether a demand
    region is already served. `demand.analyse()` compares requests against
    declared domain *tags*; nothing until now compared a centroid against a
    node's `capability_embed`. So this is the measurement to use for "is this
    already covered", and demand.UnservedCluster.similarity_bound -- a bound
    against domain tags -- is not.

    Health is deliberately not a filter. A node that is momentarily unhealthy
    still exists in the network and its owner can bring it back; treating that
    as a gap would recommend builds that a restart makes pointless. Health does
    enter, one level up, when *choosing which node* should build a blend -- see
    plan_adapters.

    Ties break on name so the answer does not depend on row order.
    """
    vec = np.asarray(centroid, dtype=float)
    if vec.size == 0:
        return None, 0.0

    scored = [(row["name"], _cosine(vec, row["capability_embed"]))
              for row in node_rows if row.get("capability_embed") is not None]
    if not scored:
        return None, 0.0

    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[0]


# --- The plan ---------------------------------------------------------------

@dataclass
class AdapterNeed:
    """A cluster of demand no node covers, and what could cover it."""
    requests: int
    nearest_node: str | None = None
    nearest_node_similarity: float = 0.0
    selected_adapters: list[AdapterPick] = field(default_factory=list)
    base_model: str | None = None
    blend_tag: str | None = None
    note: str = ""

    @property
    def adapter_ids(self) -> list[str]:
        return [p.id for p in self.selected_adapters]

    @property
    def weights(self) -> list[float]:
        return [p.weight for p in self.selected_adapters]

    def as_dict(self) -> dict:
        """The centroid is not published. It is 384 floats of internal matching
        vector, and the same call demand.UnservedCluster made -- computed, then
        not returned -- is the right one here too."""
        return {
            "requests": self.requests,
            "nearest_node": self.nearest_node,
            "nearest_node_similarity": round(self.nearest_node_similarity, 4),
            "selected_adapters": [p.as_dict() for p in self.selected_adapters],
            "base_model": self.base_model,
            "blend_tag": self.blend_tag,
            "note": self.note,
        }


@dataclass
class AdapterAssignment:
    """One node told to fuse one set of adapters.

    `reused` is the determinism payoff: the node already serves exactly this
    adapter set on this base, so the same tag names a model that exists and
    there is nothing to build.
    """
    node: str
    base_model: str
    base_source: str
    adapter_ids: list[str]
    weights: list[float]
    blend_tag: str
    requests: int
    nearest_node_similarity: float
    reused: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "node": self.node,
            "base_model": self.base_model,
            "base_source": self.base_source,
            "adapter_ids": self.adapter_ids,
            "weights": [round(w, 4) for w in self.weights],
            "blend_tag": self.blend_tag,
            "requests": self.requests,
            "nearest_node_similarity": round(self.nearest_node_similarity, 4),
            "reused": self.reused,
            "note": self.note,
        }


@dataclass
class AdapterPlan:
    mode: str
    clusters_considered: int = 0
    covered: list[dict] = field(default_factory=list)
    assignments: list[AdapterAssignment] = field(default_factory=list)
    unassigned: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "clusters_considered": self.clusters_considered,
            "covered": self.covered,
            "assignments": [a.as_dict() for a in self.assignments],
            "unassigned": self.unassigned,
        }


def detect_adapter_needs(
    clusters: list,
    node_rows: list[dict],
    adapter_rows: list[dict],
    *,
    node_distance_floor: float = NODE_DISTANCE_FLOOR,
    min_similarity: float = ADAPTER_MIN_SIMILARITY,
    top_k: int = 3,
    base_models: Sequence[str] | None = None,
    require_buildable: bool = False,
) -> list[AdapterNeed]:
    """The clusters that are genuine adapter work, with what each would need.

    A cluster already close to an existing node is skipped: the network has a
    node for this, and the right answer to "we have no specialist" is not to
    fuse a second one next to a specialist that already exists. That test is
    `node_distance_floor` against the node's capability embedding, which is the
    measurement described in nearest_node_similarity -- not the domain-tag bound
    that UnservedCluster carries.

    A cluster nothing is near *enough* to is still returned, with an empty
    selection and a note. It is real demand the network cannot serve, and the
    useful thing to say about it is that no adapter helps -- which is a node
    problem, not an adapter problem. Dropping it would make the plan read as
    though that demand did not exist.
    """
    needs: list[AdapterNeed] = []

    for cluster in clusters:
        centroid = np.asarray(getattr(cluster, "centroid", []), dtype=float)
        node_name, node_sim = nearest_node_similarity(centroid, node_rows)

        if node_sim >= node_distance_floor:
            continue

        if centroid.size == 0:
            needs.append(AdapterNeed(
                requests=cluster.size, nearest_node=node_name,
                nearest_node_similarity=node_sim,
                note="cluster has no centroid, so no adapter could be matched "
                     "against it",
            ))
            continue

        picks = select_adapters(
            centroid, adapter_rows,
            base_models=base_models, top_k=top_k,
            min_similarity=min_similarity, require_buildable=require_buildable,
        )

        if not picks:
            needs.append(AdapterNeed(
                requests=cluster.size, nearest_node=node_name,
                nearest_node_similarity=node_sim,
                note="no known adapter is close enough to this cluster -- this "
                     "region needs a node, not a fusion",
            ))
            continue

        base = picks[0].base_model
        tag = blend_tag(base, [p.id for p in picks], [p.weight for p in picks])
        needs.append(AdapterNeed(
            requests=cluster.size, nearest_node=node_name,
            nearest_node_similarity=node_sim, selected_adapters=picks,
            base_model=base, blend_tag=tag,
            note=(
                f"{len(picks)} adapter(s) sit near this cluster on {base}; "
                f"weights are uniform because there is nothing at plan time to "
                f"fit them against"
            ),
        ))

    return needs


def plan_adapters(
    clusters: list,
    node_rows: list[dict],
    adapter_rows: list[dict],
    *,
    catalogue_base_by_id: dict[str, str] | None = None,
    node_distance_floor: float = NODE_DISTANCE_FLOOR,
    min_similarity: float = ADAPTER_MIN_SIMILARITY,
    top_k: int = 3,
    max_assignments: int = 3,
    base_models: Sequence[str] | None = None,
    require_buildable: bool = False,
) -> AdapterPlan:
    """Assign needs to nodes, at most one blend each.

    **One blend per node, enforced structurally rather than advised.** A blend
    is a distinct Ollama model with its own residency slot, and the keep_alive
    work makes residency sticky, so a node alternating between two blends keeps
    both resident -- roughly doubling what that node committed, on the 8-16 GB
    laptops this project targets. So a node is claimed by its first assignment
    and is not a candidate for the second. The only thing that would let a node
    hold two is a node that appears twice in `node_rows`, and the name is the
    claim key.

    Ordering is biggest-demand-first, because if only some of the plan can be
    built this week the largest unserved region should be the one that gets
    built. Within a need, candidate nodes rank:

      1. a node already serving *exactly this* adapter set -- nothing to build,
         and it is the node the demand is least likely to outgrow;
      2. healthy before unhealthy, so an operator's build lands somewhere that
         can immediately serve;
      3. a node running no blend before one that is, because replacing a working
         specialist to add a different one is a real cost;
      4. name, so the same inputs give the same plan.

    `max_assignments` caps the plan, and it counts *assignments* including
    reused ones rather than only builds. Reused assignments are cheap, but
    predictable beats clever here: a plan that spent the same budget differently
    depending on how many reuses it found would be harder to read than one that
    is simply the first N needs.
    """
    catalogue_base_by_id = catalogue_base_by_id or {}

    considered = len(clusters)
    covered: list[dict] = []
    for cluster in clusters:
        centroid = np.asarray(getattr(cluster, "centroid", []), dtype=float)
        name, sim = nearest_node_similarity(centroid, node_rows)
        if sim >= node_distance_floor:
            covered.append({"requests": cluster.size, "node": name,
                            "similarity": round(sim, 4)})

    needs = detect_adapter_needs(
        clusters, node_rows, adapter_rows,
        node_distance_floor=node_distance_floor, min_similarity=min_similarity,
        top_k=top_k, base_models=base_models, require_buildable=require_buildable,
    )

    # Resolve every node's base once, and keep the source with it so the plan
    # can say whether the base was the catalogue's answer or the node's own.
    candidates = []
    for row in node_rows:
        base, source = resolve_node_base(row, catalogue_base_by_id)
        adapter_ids = row.get("adapter_ids") or []
        candidates.append({
            "name": row["name"], "base": base, "source": source,
            "healthy": bool(row.get("healthy")),
            "adapter_ids": sorted(adapter_ids),
        })

    assignments: list[AdapterAssignment] = []
    unassigned: list[dict] = []
    claimed: set[str] = set()

    ordered = sorted(needs, key=lambda n: (-n.requests, n.blend_tag or ""))

    for need in ordered:
        if not need.selected_adapters or need.blend_tag is None or need.base_model is None:
            unassigned.append({
                "requests": need.requests,
                "blend_tag": need.blend_tag,
                "base_model": need.base_model,
                "adapter_ids": need.adapter_ids,
                "reason": need.note,
            })
            continue

        on_base = [c for c in candidates if c["base"] == need.base_model]
        if not on_base:
            unassigned.append({
                "requests": need.requests,
                "blend_tag": need.blend_tag,
                "base_model": need.base_model,
                "adapter_ids": need.adapter_ids,
                "reason": f"no node reports base {need.base_model} — the adapters "
                          f"fit weights nothing in the network is running",
            })
            continue

        free = [c for c in on_base if c["name"] not in claimed]
        if not free:
            unassigned.append({
                "requests": need.requests,
                "blend_tag": need.blend_tag,
                "base_model": need.base_model,
                "adapter_ids": need.adapter_ids,
                "reason": f"every node on {need.base_model} is already assigned "
                          f"another blend (one blend per node)",
            })
            continue

        if len(assignments) >= max_assignments:
            unassigned.append({
                "requests": need.requests,
                "blend_tag": need.blend_tag,
                "base_model": need.base_model,
                "adapter_ids": need.adapter_ids,
                "reason": f"plan already recommends {max_assignments} assignment(s)",
            })
            continue

        want = sorted(need.adapter_ids)
        free.sort(key=lambda c: (
            0 if c["adapter_ids"] == want else 1,
            0 if c["healthy"] else 1,
            1 if c["adapter_ids"] else 0,
            c["name"],
        ))
        chosen = free[0]
        reused = chosen["adapter_ids"] == want
        claimed.add(chosen["name"])

        assignments.append(AdapterAssignment(
            node=chosen["name"],
            base_model=need.base_model,
            base_source=chosen["source"],
            adapter_ids=need.adapter_ids,
            weights=need.weights,
            blend_tag=need.blend_tag,
            requests=need.requests,
            nearest_node_similarity=need.nearest_node_similarity,
            reused=reused,
            note=(
                f"{chosen['name']} already serves this blend — reuse it, do not "
                f"rebuild" if reused else
                f"build {need.blend_tag} on {chosen['name']} from "
                f"{len(need.adapter_ids)} adapter(s), then re-register the node on "
                f"the tag"
            ),
        ))

    return AdapterPlan(
        mode=settings.adapters_mode,
        clusters_considered=considered,
        covered=covered,
        assignments=assignments,
        unassigned=unassigned,
    )


# --- Seeding ----------------------------------------------------------------

async def seed_adapters_from_file() -> None:
    """Load catalogue/adapters.seed.yaml into the adapters table.

    Same convention as seed_catalogue_from_file, including the guarded delete at
    the end -- and this is precisely the function that would have been unsafe to
    share a table with. That seeder ends in
    `delete from catalogue_models where id <> all($1::text[])`; adapters living
    there would be wiped the first time somebody trimmed catalogue.seed.yaml.
    Different table, different file, same shape, no inherited wipe.

    No-ops unless adapters_mode is 'plan', so merging the PR cannot change what
    is in the database by itself.
    """
    if settings.adapters_mode != "plan":
        return
    if not settings.adapters_seed_on_startup:
        return

    path = Path(settings.adapters_seed_file)
    if not path.exists():
        return

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    adapters = data.get("adapters", [])

    for a in adapters:
        vec = embedder.embed(a["domain_text"])
        async with db.pool().acquire() as conn:
            await conn.execute(
                """
                insert into adapters
                    (id, display_name, hf_repo, gguf_ref, revision, base_model,
                     blend_group, domain_text, domain_embed, size_mb, min_ram_gb,
                     domain_tags, licence)
                values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                on conflict (id) do update set
                    display_name = excluded.display_name,
                    hf_repo = excluded.hf_repo,
                    gguf_ref = excluded.gguf_ref,
                    revision = excluded.revision,
                    base_model = excluded.base_model,
                    blend_group = excluded.blend_group,
                    domain_text = excluded.domain_text,
                    domain_embed = excluded.domain_embed,
                    size_mb = excluded.size_mb,
                    min_ram_gb = excluded.min_ram_gb,
                    domain_tags = excluded.domain_tags,
                    licence = excluded.licence
                """,
                a["id"], a["display_name"], a["hf_repo"], a.get("gguf_ref"),
                a.get("revision"), a["base_model"], a.get("blend_group"),
                a["domain_text"], vec, a.get("size_mb"), a.get("min_ram_gb"),
                a.get("domain_tags"), a.get("licence"),
            )

    # Guarded on a non-empty seed for the same reason the catalogue's is: an
    # unreadable or truncated file must not be able to empty the table.
    if adapters:
        async with db.pool().acquire() as conn:
            removed = await conn.fetch(
                "delete from adapters where id <> all($1::text[]) returning id",
                [a["id"] for a in adapters],
            )
        if removed:
            print(f"adapters: retired {len(removed)} entry(s) no longer in the "
                  f"seed file: {', '.join(r['id'] for r in removed)}")


# --- HTTP -------------------------------------------------------------------

def _row_to_adapter_out(row) -> dict:
    """A seeded adapter, as published.

    Nothing here is sensitive: repos, sizes and licences are public artifacts
    and the two refs are pointers to somebody else's download. No node URL can
    reach this shape -- it does not touch the nodes table at all.
    """
    return {
        "id": row["id"],
        "display_name": row["display_name"],
        "hf_repo": row["hf_repo"],
        "revision": row["revision"],
        "gguf_ref": row["gguf_ref"],
        "base_model": row["base_model"],
        "blend_group": row["blend_group"],
        "domain_text": row["domain_text"],
        "size_mb": row["size_mb"],
        "min_ram_gb": row["min_ram_gb"],
        "domain_tags": row["domain_tags"],
        "licence": row["licence"],
    }


@router.get("/adapters")
async def list_adapters():
    if settings.adapters_mode != "plan":
        return []
    async with db.pool().acquire() as conn:
        rows = await conn.fetch("select * from adapters order by id")
    return [_row_to_adapter_out(r) for r in rows]


@router.get("/adapters/plan")
async def adapters_plan(
    window_days: int = Query(default=30, ge=1, le=365),
    cluster_threshold: float = Query(default=0.6, gt=0, le=1),
    min_cluster_size: int = Query(default=3, ge=2, le=100),
    top_k: int = Query(default=3, ge=1, le=8),
    max_assignments: int = Query(default=3, ge=1, le=20),
):
    """What the network would build, if a node owner agreed to build it.

    Read-only and a recommendation, in that order of importance. Returns an
    empty plan unless adapters_mode is 'plan', so a merged PR is inert.

    `select * from nodes` deliberately: the adapter plan needs each node's
    capability embedding and base. What it does *not* do is publish those rows
    -- no node row reaches the response, only names, bases and similarities, so
    the tunnel URLs that `select *` brings back stay inside the gateway. See
    tests/test_node_url_exposure.py.
    """
    if settings.adapters_mode != "plan":
        return AdapterPlan(mode=settings.adapters_mode).as_dict()

    clusters = await load_unserved_clusters(window_days, cluster_threshold, min_cluster_size)

    async with db.pool().acquire() as conn:
        node_rows = [dict(r) for r in await conn.fetch("select * from nodes")]
        adapter_rows = [dict(r) for r in await conn.fetch("select * from adapters")]
        catalogue_rows = await conn.fetch("select id, base_model from catalogue_models")

    catalogue_base_by_id = {
        r["id"]: r["base_model"] for r in catalogue_rows if r["base_model"]
    }

    return plan_adapters(
        clusters, node_rows, adapter_rows,
        catalogue_base_by_id=catalogue_base_by_id,
        top_k=top_k, max_assignments=max_assignments,
    ).as_dict()
