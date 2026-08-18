import json

from fastapi import APIRouter, Query

from app import db
from app.models import DecisionOut

router = APIRouter()


@router.get("/decisions/recent", response_model=list[DecisionOut])
async def recent_decisions(
    limit: int = Query(default=50, le=500),
    topology: str | None = Query(default=None, description="Filter to 'single', 'panel' or 'degraded'"),
):
    """The routing log.

    Panel members are resolved to names via a left join on a lateral unnest, so
    a node that has since deregistered leaves a gap in the list rather than
    erasing the record that it answered. History outliving the node is the
    point of a log.
    """
    async with db.pool().acquire() as conn:
        rows = await conn.fetch(
            """
            select d.id, d.chosen_node, n.name as chosen_node_name,
                   d.score, d.runner_up, d.latency_ms, d.ok, d.created_at,
                   d.matched_domain, d.topology, d.aggregator_node,
                   a.name as aggregator_node_name, d.compose_reason,
                   d.checks_run, d.checks_failed, d.disagreements,
                   (select array_agg(pn.name order by ord)
                      from unnest(d.panel) with ordinality as p(pid, ord)
                      left join nodes pn on pn.id = p.pid) as panel_names
            from decisions d
            left join nodes n on n.id = d.chosen_node
            left join nodes a on a.id = d.aggregator_node
            where ($2::text is null or d.topology = $2)
            order by d.created_at desc
            limit $1
            """,
            limit, topology,
        )

    out = []
    for r in rows:
        reason = r["compose_reason"]
        if isinstance(reason, str):
            try:
                reason = json.loads(reason)
            except json.JSONDecodeError:
                reason = None
        out.append(DecisionOut(
            id=r["id"],
            chosen_node=r["chosen_node"],
            chosen_node_name=r["chosen_node_name"],
            score=float(r["score"]) if r["score"] is not None else None,
            runner_up=r["runner_up"],
            latency_ms=r["latency_ms"],
            ok=r["ok"],
            created_at=r["created_at"].isoformat(),
            matched_domain=r["matched_domain"],
            topology=r["topology"] or "single",
            panel=[p for p in (r["panel_names"] or []) if p] or None,
            aggregator_node_name=r["aggregator_node_name"],
            compose_reason=reason,
            checks_run=r["checks_run"],
            checks_failed=r["checks_failed"],
            disagreements=r["disagreements"],
        ))
    return out


@router.get("/decisions/composition")
async def composition_summary(window_days: int = Query(default=7, ge=1, le=365)):
    """How composition is actually behaving, in one call.

    Exists because the honest answer to "does composing beat single routing"
    is still *unmeasured* (see testing/compose-test), and the first thing
    anyone needs in order to measure it is how often each topology even runs,
    and what the verifier is catching when it does.

    Latency is reported per topology without adjustment. A panel is slower than
    a single node — it waits for the slowest of N specialists and then an
    aggregator on top — and that cost belongs next to any claim of quality
    gain, not in a footnote.
    """
    async with db.pool().acquire() as conn:
        rows = await conn.fetch(
            """
            select topology,
                   count(*) as n,
                   count(*) filter (where ok) as ok_count,
                   percentile_cont(0.5) within group (order by latency_ms) as median_latency_ms,
                   sum(coalesce(checks_run, 0)) as checks_run,
                   sum(coalesce(checks_failed, 0)) as checks_failed,
                   sum(coalesce(disagreements, 0)) as disagreements
            from decisions
            where created_at > now() - make_interval(days => $1)
            group by topology
            """,
            window_days,
        )

    by_topology = {}
    for r in rows:
        checks = int(r["checks_run"] or 0)
        by_topology[r["topology"] or "single"] = {
            "requests": r["n"],
            "succeeded": r["ok_count"],
            "median_latency_ms": int(r["median_latency_ms"]) if r["median_latency_ms"] else None,
            "arithmetic_checks_run": checks,
            "arithmetic_checks_failed": int(r["checks_failed"] or 0),
            # The number to watch. seam-findings.md measured overrides firing on
            # 42% of handoffs; a rate near zero here is far more likely to mean
            # the extractor stopped matching than that the models stopped
            # making arithmetic errors.
            "check_fire_rate": round(int(r["checks_failed"] or 0) / checks, 3) if checks else None,
            "cross_specialist_disagreements": int(r["disagreements"] or 0),
        }

    panel = by_topology.get("panel", {}).get("requests", 0)
    single = by_topology.get("single", {}).get("requests", 0)
    degraded = by_topology.get("degraded", {}).get("requests", 0)
    total = panel + single + degraded

    return {
        "window_days": window_days,
        "by_topology": by_topology,
        "composed_fraction": round(panel / total, 3) if total else 0.0,
        "caveat": (
            "Counts and latency only. This says how often the network composes and what the "
            "verifier caught — it does not say whether composing produced a better answer. "
            "Nothing here should be quoted as evidence for composition; run "
            "testing/compose-test for that."
        ),
    }
