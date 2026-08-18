"""Specialist catalogue: seeding, listing, and demand-aware assignment.

The catalogue is a curated list of known specialist models (catalogue/catalogue.seed.yaml).
Node onboarding assigns from it based on a machine's real hardware and the network's
current demand gaps, instead of an operator hand-declaring an arbitrary model.
"""
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import db, embedder
from app.config import settings

router = APIRouter()

# Cold start: no decisions data yet to compute real demand from. Fill these
# domains first, in this order -- not invented demand, just a declared
# default so the network doesn't stall on an empty decisions log.
COLD_START_DEFAULT_ORDER = ["code", "math", "general", "legal"]


async def seed_catalogue_from_file() -> None:
    if not settings.catalogue_seed_on_startup:
        return
    path = Path(settings.catalogue_seed_file)
    if not path.exists():
        return

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    models = data.get("models", [])

    for m in models:
        vec = embedder.embed(m["capability_text"])
        async with db.pool().acquire() as conn:
            await conn.execute(
                """
                insert into catalogue_models
                    (id, display_name, source, domain_tags, capability_text,
                     params_b, min_ram_gb, min_vram_gb, needs_gpu,
                     verified_in_lane, lane_benchmark, licence,
                     base_model, overlay)
                values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                on conflict (id) do update set
                    display_name = excluded.display_name,
                    source = excluded.source,
                    domain_tags = excluded.domain_tags,
                    capability_text = excluded.capability_text,
                    params_b = excluded.params_b,
                    min_ram_gb = excluded.min_ram_gb,
                    min_vram_gb = excluded.min_vram_gb,
                    needs_gpu = excluded.needs_gpu,
                    verified_in_lane = excluded.verified_in_lane,
                    lane_benchmark = excluded.lane_benchmark,
                    licence = excluded.licence,
                    base_model = excluded.base_model,
                    overlay = excluded.overlay
                """,
                m["id"], m["display_name"], m["source"], m["domain_tags"], m["capability_text"],
                m.get("params_b"), m["min_ram_gb"], m.get("min_vram_gb", 0), m.get("needs_gpu", False),
                m.get("verified_in_lane", False),
                _to_jsonb(m.get("lane_benchmark")), m.get("licence"),
                m.get("base_model"), _to_jsonb(m.get("overlay")),
            )

    # The seed file is the source of truth for what the network offers, so an
    # entry deleted from the YAML has to actually disappear. Upserting alone
    # would leave a retired model assignable forever -- and /assign reads the
    # table, not the file, so nobody would notice until a node pulled it.
    # Guarded on a non-empty seed: an unreadable or truncated file must not
    # be able to wipe the catalogue.
    if models:
        async with db.pool().acquire() as conn:
            removed = await conn.fetch(
                "delete from catalogue_models where id <> all($1::text[]) returning id",
                [m["id"] for m in models],
            )
        if removed:
            print(f"catalogue: retired {len(removed)} entry(s) no longer in the seed file: "
                  f"{', '.join(r['id'] for r in removed)}")


def _to_jsonb(value: Any) -> str | None:
    if value is None:
        return None
    import json
    return json.dumps(value)


@router.get("/catalogue")
async def list_catalogue():
    async with db.pool().acquire() as conn:
        rows = await conn.fetch("select * from catalogue_models order by id")
    return [_row_to_dict(r) for r in rows]


def _row_to_dict(row) -> dict:
    d = dict(row)
    d["added_at"] = d["added_at"].isoformat() if d.get("added_at") else None
    if d.get("lane_benchmark"):
        import json
        d["lane_benchmark"] = json.loads(d["lane_benchmark"]) if isinstance(d["lane_benchmark"], str) else d["lane_benchmark"]
    if d.get("params_b") is not None:
        d["params_b"] = float(d["params_b"])
    return d


class HardwareProfile(BaseModel):
    total_ram_gb: float
    available_ram_gb: float
    gpu_present: bool = False
    vram_gb: float = 0
    free_disk_gb: float | None = None
    cpu_cores: int | None = None
    os: str | None = None


class AssignRequest(BaseModel):
    hardware: HardwareProfile


class AssignResponse(BaseModel):
    catalogue_id: str
    display_name: str
    source: str
    domain_tags: list[str]
    capability_text: str
    reason: str
    cold_start: bool


def _runnable(model: dict, hw: HardwareProfile) -> bool:
    headroom_ram = hw.available_ram_gb * settings.assignment_ram_headroom
    if model["min_ram_gb"] > headroom_ram:
        return False
    if model["needs_gpu"] and not hw.gpu_present:
        return False
    if model["min_vram_gb"] and model["min_vram_gb"] > hw.vram_gb:
        return False
    return True


async def _domain_demand(conn) -> dict[str, int]:
    rows = await conn.fetch(
        """
        select matched_domain, count(*) as n
        from decisions
        where matched_domain is not null
          and created_at > now() - interval '30 days'
        group by matched_domain
        """
    )
    return {r["matched_domain"]: r["n"] for r in rows}


async def _domain_coverage(conn) -> dict[str, int]:
    rows = await conn.fetch(
        """
        select unnest(domain_tags) as tag, count(*) as n
        from nodes
        where healthy = true and domain_tags is not null
        group by tag
        """
    )
    return {r["tag"]: r["n"] for r in rows}


async def assign(req: AssignRequest) -> AssignResponse:
    async with db.pool().acquire() as conn:
        catalogue_rows = await conn.fetch("select * from catalogue_models")
        demand = await _domain_demand(conn)
        coverage = await _domain_coverage(conn)

    catalogue = [_row_to_dict(r) for r in catalogue_rows]
    runnable = [m for m in catalogue if _runnable(m, req.hardware)]
    if not runnable:
        raise HTTPException(status_code=422, detail="no catalogue model fits this machine's hardware")

    cold_start = sum(demand.values()) == 0

    if cold_start:
        for domain in COLD_START_DEFAULT_ORDER:
            candidates = [m for m in runnable if domain in m["domain_tags"]]
            if not candidates:
                continue
            # Prefer verified-in-lane, then smaller params (safer default fit).
            candidates.sort(key=lambda m: (not m["verified_in_lane"], m["params_b"] or 0))
            chosen = candidates[0]
            return AssignResponse(
                catalogue_id=chosen["id"], display_name=chosen["display_name"],
                source=chosen["source"], domain_tags=chosen["domain_tags"],
                capability_text=chosen["capability_text"],
                reason=(
                    f"Cold start — no demand data yet, so filling the network's declared "
                    f"default coverage order. '{domain}' isn't covered yet and this is the "
                    f"best {domain} specialist your hardware can run."
                ),
                cold_start=True,
            )
        # Nothing in the default order fits -- fall back to whatever's runnable.
        chosen = runnable[0]
        return AssignResponse(
            catalogue_id=chosen["id"], display_name=chosen["display_name"],
            source=chosen["source"], domain_tags=chosen["domain_tags"],
            capability_text=chosen["capability_text"],
            reason="Cold start, and no default-order domain fits this hardware — assigning the first runnable catalogue entry.",
            cold_start=True,
        )

    max_demand = max(demand.values()) if demand else 1
    scored = []
    for m in runnable:
        domain_scores = []
        for tag in m["domain_tags"]:
            d = demand.get(tag, 0)
            c = coverage.get(tag, 0)
            gap = d / (c + 1)
            domain_scores.append((tag, d, c, gap))
        if not domain_scores:
            continue
        best_tag, best_d, best_c, best_gap = max(domain_scores, key=lambda t: t[3])
        # Small bonus for verified-in-lane models -- prefer proven specialists
        # when gaps are otherwise close.
        score = best_gap + (0.1 if m["verified_in_lane"] else 0)
        scored.append((score, best_tag, best_d, best_c, m))

    if not scored:
        chosen = runnable[0]
        return AssignResponse(
            catalogue_id=chosen["id"], display_name=chosen["display_name"],
            source=chosen["source"], domain_tags=chosen["domain_tags"],
            capability_text=chosen["capability_text"],
            reason="No domain gap data available for any runnable model — assigning the first runnable catalogue entry.",
            cold_start=False,
        )

    scored.sort(key=lambda t: t[0], reverse=True)
    score, tag, d, c, chosen = scored[0]
    reason = (
        f"'{tag}' has the biggest coverage gap your hardware can fill: "
        f"{d} recent request(s) matched this domain against {c} healthy node(s) "
        f"currently serving it."
        + (" This model is verified to beat frontier models in this lane." if chosen["verified_in_lane"] else "")
    )
    return AssignResponse(
        catalogue_id=chosen["id"], display_name=chosen["display_name"],
        source=chosen["source"], domain_tags=chosen["domain_tags"],
        capability_text=chosen["capability_text"],
        reason=reason, cold_start=False,
    )


@router.post("/assign", response_model=AssignResponse)
async def assign_endpoint(req: AssignRequest):
    return await assign(req)
