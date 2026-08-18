import os
from pathlib import Path

import yaml

from app import db, embedder
from app.config import settings


async def seed_from_file() -> None:
    path = Path(settings.seed_file)
    if not path.exists():
        return

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    for node in data.get("nodes", []):
        vec = embedder.embed(node["capability_text"])
        async with db.pool().acquire() as conn:
            await conn.execute(
                """
                insert into nodes
                    (name, operator, endpoint_url, model_name, api_key_ref,
                     capability_text, capability_embed, region, cost_per_1k,
                     domain_tags, can_aggregate)
                values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                on conflict (name) do nothing
                """,
                node["name"], node.get("operator"), node["endpoint_url"], node["model_name"],
                node.get("api_key_ref"), node["capability_text"], vec,
                node.get("region"), node.get("cost_per_1k", 0),
                # Seeded nodes carried no domain tags before v0.1.1. That was
                # invisible while routing was similarity-only, but panel seats
                # are awarded by *declared* domain -- so a tagless node can
                # never join a panel. A demo network seeded from this file
                # would have silently never composed, which would have looked
                # exactly like composition not working.
                node.get("domain_tags"), node.get("can_aggregate", False),
            )
