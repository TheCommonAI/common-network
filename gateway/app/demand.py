"""What the network is missing, and what to install next.

v0.1 logged a request embedding for every routing decision and called it "the
seed of a future demand-analytics layer". This is that layer.

Two questions, deliberately kept separate because they have different standards
of evidence:

**1. Which declared domains are under-served?** Cheap and reliable: count
requests that matched a domain, count healthy nodes serving it, divide. This is
the signal `/assign` already used to pick a model for a joining machine; here it
is exposed directly so a contributor can see *why* before installing anything.

**2. Where is there demand the catalogue cannot serve at all?** This is the
Demand Vector Cloud idea from the architecture notes, and it is the harder and
more interesting one. When a request's embedding is far from every domain the
network declares, no amount of rebalancing existing specialists helps — the
capability does not exist. Those requests cluster in regions of the embedding
space nobody covers, and a cluster is the evidence that a *new* specialist is
warranted rather than one person's odd question.

The DVC theory in the architecture notes says the response to such a cluster is
to spawn an ephemeral model by interpolating nearby seed weights (Soup of
Experts). That is not what happens here, and pretending otherwise would be the
kind of gap between claim and code this project exists to avoid. What happens
here is: the cluster is reported, with its size and its distance from everything
we have, so a human can decide whether to add a catalogue entry. Weight merging
stays out of scope; the *demand signal it would need* is what this builds.

Clustering is a single-pass greedy pass over cosine similarity — no sklearn,
which is not a gateway dependency and would be a large one to add for this. It
is order-dependent and approximate, and that is acceptable: the output is a
prompt for human judgement, not an automated action.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app import db
from app.config import settings

router = APIRouter()


# A request whose best domain match is below this is treated as unserved —
# nothing the network declares is really about it. Same floor
# `best_matched_domain` uses to decide whether to record a domain at all, so
# "matched_domain is null" and "unserved" mean the same thing by construction.
UNSERVED_FLOOR = 0.3


@dataclass
class DomainGap:
    domain: str
    demand: int
    coverage: int
    gap: float
    recommended: dict | None = None

    def as_dict(self) -> dict:
        return {
            "domain": self.domain, "demand": self.demand, "coverage": self.coverage,
            "gap": round(self.gap, 3),
            "recommended": self.recommended,
        }


@dataclass
class UnservedCluster:
    """A region of request space the catalogue does not reach."""
    size: int
    nearest_domain: str | None
    nearest_similarity: float
    centroid: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "requests": self.size,
            "nearest_declared_domain": self.nearest_domain,
            "similarity_to_nearest": round(self.nearest_similarity, 3),
            "verdict": (
                "no specialist covers this — a new catalogue entry is warranted"
                if self.nearest_similarity < UNSERVED_FLOOR else
                "partially covered — an existing domain is adjacent"
            ),
        }


def _cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms
    return unit @ unit.T


def cluster(vectors: np.ndarray, threshold: float, min_size: int) -> list[list[int]]:
    """Greedy single-pass clustering by cosine similarity.

    Each unassigned vector opens a cluster and absorbs every other unassigned
    vector within `threshold` of it. Approximate and order-dependent by design:
    this feeds a human decision about what to install, and a rigorous clustering
    would not change that decision while adding a heavyweight dependency.
    """
    if len(vectors) == 0:
        return []
    sims = _cosine_matrix(vectors)
    assigned = np.zeros(len(vectors), dtype=bool)
    clusters: list[list[int]] = []

    # Densest first, so a real cluster isn't fragmented by an outlier that
    # happened to come first in the result set.
    order = np.argsort(-(sims >= threshold).sum(axis=1))

    for i in order:
        if assigned[i]:
            continue
        members = [int(j) for j in np.where((sims[i] >= threshold) & (~assigned))[0]]
        if not members:
            continue
        for j in members:
            assigned[j] = True
        if len(members) >= min_size:
            clusters.append(members)

    clusters.sort(key=len, reverse=True)
    return clusters


async def _catalogue_by_domain(conn) -> dict[str, list[dict]]:
    rows = await conn.fetch("select * from catalogue_models")
    by_domain: dict[str, list[dict]] = {}
    for r in rows:
        for tag in r["domain_tags"]:
            by_domain.setdefault(tag, []).append(dict(r))
    return by_domain


def _recommend(candidates: list[dict]) -> dict | None:
    """Pick which catalogue entry to suggest for a domain.

    Verified-in-lane first — a specialist measured to beat a frontier model at
    something is worth more to a panel than an unmeasured one — then smallest,
    because the machine that has not joined yet is more likely to be an old
    laptop than a workstation.
    """
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda m: (not m["verified_in_lane"],
                                               float(m["params_b"] or 0)))
    best = ranked[0]
    return {
        "catalogue_id": best["id"],
        "display_name": best["display_name"],
        "source": best["source"],
        "min_ram_gb": best["min_ram_gb"],
        "verified_in_lane": best["verified_in_lane"],
    }


async def analyse(window_days: int = 30, cluster_threshold: float = 0.6,
                  min_cluster_size: int = 3) -> dict:
    """The full picture: served domains, their gaps, and unserved demand."""
    async with db.pool().acquire() as conn:
        demand_rows = await conn.fetch(
            """
            select matched_domain, count(*) as n
            from decisions
            where matched_domain is not null
              and created_at > now() - make_interval(days => $1)
            group by matched_domain
            """,
            window_days,
        )
        coverage_rows = await conn.fetch(
            """
            select unnest(domain_tags) as tag, count(*) as n
            from nodes
            where healthy = true and domain_tags is not null
            group by tag
            """
        )
        unserved_rows = await conn.fetch(
            """
            select request_embed
            from decisions
            where matched_domain is null
              and request_embed is not null
              and created_at > now() - make_interval(days => $1)
            limit 5000
            """,
            window_days,
        )
        total = await conn.fetchval(
            "select count(*) from decisions where created_at > now() - make_interval(days => $1)",
            window_days,
        )
        catalogue = await _catalogue_by_domain(conn)

    demand = {r["matched_domain"]: r["n"] for r in demand_rows}
    coverage = {r["tag"]: r["n"] for r in coverage_rows}

    gaps: list[DomainGap] = []
    for domain in set(demand) | set(coverage) | set(catalogue):
        d, c = demand.get(domain, 0), coverage.get(domain, 0)
        # demand per node already serving it. +1 so an uncovered domain with
        # real demand ranks above a covered one, without dividing by zero.
        gaps.append(DomainGap(
            domain=domain, demand=d, coverage=c, gap=d / (c + 1),
            recommended=_recommend(catalogue.get(domain, [])) if c == 0 or d / (c + 1) > 1 else None,
        ))
    gaps.sort(key=lambda g: (-g.gap, -g.demand, g.domain))

    clusters: list[UnservedCluster] = []
    if unserved_rows:
        vectors = np.array([list(r["request_embed"]) for r in unserved_rows], dtype=float)
        for members in cluster(vectors, cluster_threshold, min_cluster_size):
            centroid = vectors[members].mean(axis=0)
            clusters.append(UnservedCluster(
                size=len(members),
                nearest_domain=None,
                # By construction every request here failed to clear the floor
                # against any declared domain, so the nearest similarity is
                # below it. Reported rather than recomputed per-tag: the useful
                # number is "how many people asked", not a third decimal place.
                nearest_similarity=UNSERVED_FLOOR,
                centroid=centroid.tolist(),
            ))

    unserved_total = len(unserved_rows)
    return {
        "window_days": window_days,
        "total_requests": total or 0,
        "unserved_requests": unserved_total,
        "unserved_fraction": round(unserved_total / total, 3) if total else 0.0,
        "domain_gaps": [g.as_dict() for g in gaps],
        "unserved_clusters": [c.as_dict() for c in clusters],
        "cold_start": (total or 0) == 0,
    }


# --- Install planning -----------------------------------------------------

def _fits(model: dict, ram_gb: float) -> bool:
    return float(model["min_ram_gb"]) <= ram_gb * settings.assignment_ram_headroom


async def plan_installs(machines: list[float], window_days: int = 30) -> dict:
    """What a set of machines should each install, so the result composes.

    `machines` is a list of available-RAM figures, one per machine.

    The rule that matters is the one this whole version is built on: a panel
    only pays when its members are non-dominated. Twenty machines all running
    the best model they can fit would produce twenty copies of one generalist —
    a network that cannot beat its own best node however large it gets. So this
    assigns each machine a *different* lane, filling the widest gap that machine
    can fit, and marks each assignment as taken before considering the next.

    Written for the concrete case of a school computer lab joining at once,
    where nobody wants to run twenty `common join` calls and hope the demand
    signal keeps up.
    """
    analysis = await analyse(window_days=window_days)

    async with db.pool().acquire() as conn:
        rows = await conn.fetch("select * from catalogue_models")
    catalogue = [dict(r) for r in rows]
    by_id = {m["id"]: m for m in catalogue}

    # Standing coverage counts, updated as we assign, so the plan spreads.
    assigned_count: dict[str, int] = {}
    for gap in analysis["domain_gaps"]:
        assigned_count[gap["domain"]] = gap["coverage"]

    excluded = {t.strip().lower() for t in settings.compose_excluded_domains.split(",") if t.strip()}
    demand_by_domain = {g["domain"]: g["demand"] for g in analysis["domain_gaps"]}

    def score_model(model: dict) -> float:
        """How much this machine running this model would help the network."""
        best = 0.0
        for tag in model["domain_tags"]:
            covered = assigned_count.get(tag, 0)
            d = demand_by_domain.get(tag, 0)
            # Cold start has no demand to read, so fall back to breadth of
            # coverage: an uncovered lane is worth filling on the argument that
            # a network with five lanes can compose and a network with one
            # cannot. Explicitly a declared default, not inferred demand.
            base = (d + 1) / (covered + 1)
            # Specialists over generalists. A generalist can never take a panel
            # seat (COMPOSE_EXCLUDED_DOMAINS), so a lab full of them cannot
            # compose at all -- but one is still needed to aggregate, which is
            # why the penalty is a discount rather than a ban.
            if tag.lower() in excluded:
                base *= 0.35
            best = max(best, base)
        if model["verified_in_lane"]:
            best *= 1.25
        # A second copy of a model already assigned adds redundancy but no
        # capability — two identical nodes are dominated by each other and the
        # gateway will never seat both on a panel. So a not-yet-assigned
        # specialist wins any close call. This is a strong discount rather than
        # a ban because duplicates are genuinely worth having once the distinct
        # lanes are filled: a lab where one machine is switched off should not
        # lose a lane entirely.
        if assigned_models.get(model["id"], 0):
            best *= 0.3 ** assigned_models[model["id"]]
        return best

    assigned_models: dict[str, int] = {}
    plan = []
    needs_aggregator = True
    for i, ram in enumerate(sorted(machines, reverse=True)):
        runnable = [m for m in catalogue if _fits(m, ram)]
        if not runnable:
            plan.append({
                "machine": i + 1, "available_ram_gb": ram, "catalogue_id": None,
                "reason": "no catalogue entry fits this machine, even the 3B fallback",
            })
            continue

        # The network needs exactly one aggregator before it needs a second
        # specialist: a panel with nobody to synthesise it degrades to passing
        # through a single member's answer.
        if needs_aggregator:
            generalists = [m for m in runnable
                           if any(t.lower() in excluded for t in m["domain_tags"])]
            if generalists:
                chosen = max(generalists, key=lambda m: float(m["params_b"] or 0))
                needs_aggregator = False
                assigned_models[chosen["id"]] = assigned_models.get(chosen["id"], 0) + 1
                for tag in chosen["domain_tags"]:
                    assigned_count[tag] = assigned_count.get(tag, 0) + 1
                plan.append({
                    "machine": i + 1, "available_ram_gb": ram,
                    "catalogue_id": chosen["id"], "display_name": chosen["display_name"],
                    "source": chosen["source"], "domain_tags": chosen["domain_tags"],
                    "role": "aggregator",
                    "reason": ("the network needs one node able to synthesise a panel's answers "
                               "before it needs a second specialist — this is the largest "
                               "generalist this machine can hold"),
                })
                continue

        chosen = max(runnable, key=score_model)
        assigned_models[chosen["id"]] = assigned_models.get(chosen["id"], 0) + 1
        for tag in chosen["domain_tags"]:
            assigned_count[tag] = assigned_count.get(tag, 0) + 1
        lane = max(chosen["domain_tags"],
                   key=lambda t: demand_by_domain.get(t, 0) if t.lower() not in excluded else -1)
        plan.append({
            "machine": i + 1, "available_ram_gb": ram,
            "catalogue_id": chosen["id"], "display_name": chosen["display_name"],
            "source": chosen["source"], "domain_tags": chosen["domain_tags"],
            "role": "specialist",
            "reason": (f"fills the '{lane}' lane"
                       + (f", which {demand_by_domain.get(lane, 0)} recent request(s) needed"
                          if demand_by_domain.get(lane) else " (no demand data yet — filling coverage)")),
        })

    # Whether the planned network can compose is a question about distinct
    # *models*, not distinct lane tags.
    #
    # Counting tags was wrong, and wrong in the direction that flatters the
    # plan: a single model declaring four tags (phi4-mini-reasoning declares
    # math, reasoning, logic and step-by-step) made four machines running that
    # one model look like four specialist lanes. Four copies of one model is
    # one specialist. The gateway's domination gate would correctly refuse to
    # compose them — so the plan was promising something the resulting network
    # provably cannot do. Found by standing the gateway up and reading the plan
    # it actually produced for six 8GB machines.
    specialist_models = {
        entry["catalogue_id"] for entry in plan
        if entry.get("role") == "specialist" and entry.get("catalogue_id")
    }
    lanes = {t for entry in plan if entry.get("domain_tags")
             for t in entry["domain_tags"] if t.lower() not in excluded}
    can_compose = len(specialist_models) >= settings.compose_min_domains and not needs_aggregator

    if can_compose:
        note = (
            "A panel can only beat its best member if the members are non-dominated — each "
            "better than the others at something. That is why this plan spreads machines "
            "across lanes instead of giving every machine the best model it can hold. "
            "See testing/seam-findings.md §2."
        )
    elif needs_aggregator:
        note = ("No machine in this set can hold a generalist to aggregate with. A panel with "
                "nobody to synthesise it degrades to passing through one member's answer.")
    else:
        # Almost always a hardware ceiling rather than a catalogue gap, so say
        # which, and say what would fix it. "can_compose: false" on its own
        # tells someone their lab won't work but not what to do about it.
        fits_anywhere = {m["id"] for m in catalogue
                         if any(_fits(m, ram) for ram in machines)
                         and not any(t.lower() in excluded for t in m["domain_tags"])}
        smallest_unreachable = sorted(
            (m for m in catalogue
             if m["id"] not in fits_anywhere
             and not any(t.lower() in excluded for t in m["domain_tags"])),
            key=lambda m: float(m["min_ram_gb"]),
        )
        need = smallest_unreachable[0] if smallest_unreachable else None
        note = (
            f"This set can only run {len(specialist_models)} distinct specialist "
            f"({', '.join(sorted(specialist_models)) or 'none'}), so it cannot compose — a panel "
            f"needs at least {settings.compose_min_domains} different specialists, and duplicates "
            f"of one model add redundancy but not capability."
        )
        if need:
            note += (f" The next specialist to become reachable is {need['display_name']}, which "
                     f"needs {need['min_ram_gb']}GB (so ~"
                     f"{need['min_ram_gb'] / settings.assignment_ram_headroom:.0f}GB available "
                     f"after headroom). One larger machine would unlock composition for the "
                     f"whole set.")

    return {
        "machines": len(machines),
        "plan": plan,
        "distinct_specialist_models": len(specialist_models),
        "distinct_specialist_lanes": len(lanes),
        "can_compose": can_compose,
        "note": note,
    }


# --- API ------------------------------------------------------------------

@router.get("/demand/gaps")
async def demand_gaps(window_days: int = Query(default=30, ge=1, le=365)):
    """What the network is short of, and what would fill it.

    Public and unauthenticated, like every other read endpoint here. Someone
    deciding whether to donate a machine should be able to see what that
    machine would actually be for, before installing anything.
    """
    return await analyse(window_days=window_days)


class PlanRequest(BaseModel):
    # One entry per machine: its available RAM in GB.
    machines: list[float] = Field(min_length=1, max_length=500)
    window_days: int = 30


@router.post("/demand/plan")
async def demand_plan(req: PlanRequest):
    """Given a set of machines, what should each one install?

    POST because the interesting case is a heterogeneous set — a lab where the
    teacher machine has 32GB and the student machines have 8GB — and that does
    not fit in a query string honestly.
    """
    if any(m <= 0 for m in req.machines):
        raise HTTPException(status_code=422, detail="every machine needs a positive RAM figure")
    return await plan_installs(req.machines, window_days=req.window_days)


@router.get("/demand/plan")
async def demand_plan_uniform(
    machines: int = Query(default=10, ge=1, le=500),
    ram_gb: float = Query(default=8.0, gt=0),
    window_days: int = Query(default=30, ge=1, le=365),
):
    """The identical-machines shortcut: `?machines=20&ram_gb=8`.

    Which is the school-lab case, where the machines really are identical.
    """
    return await plan_installs([ram_gb] * machines, window_days=window_days)
