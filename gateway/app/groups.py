"""Distributed model groups, as far as the gateway is concerned.

A group is one registered node whose coordinator reports its own state. The
gateway does not see the members, their addresses or their tokens (see
migrations/015). This module holds the few rules that reading that state needs,
kept free of database and HTTP so they can be tested on their own.
"""
from datetime import datetime, timedelta

from app.config import settings

READY = "ready"
RESTARTING = "restarting"
STATUSES = (READY, RESTARTING)

# Tags a group may not carry. "general" is the generalist lane in choose_primary
# and compose.py; a group registered under it would be picked for requests it
# cannot serve as well as for the model it actually runs.
RESERVED_GROUP_TAGS = frozenset({"general"})


def is_restarting(group_status: str | None, group_reported_at: datetime | None,
                  now: datetime, ttl_seconds: float | None = None) -> bool:
    """True while a coordinator's own 'restarting' report is still fresh.

    A stale report stops counting, and the health check decides again. A report
    with no timestamp cannot be dated, so it does not count either.
    """
    if group_status != RESTARTING or group_reported_at is None:
        return False
    ttl = settings.group_restart_ttl_seconds if ttl_seconds is None else ttl_seconds
    return now - group_reported_at <= timedelta(seconds=ttl)


def restarting_detail(rows: list[dict], now: datetime) -> str | None:
    """The 503 detail for a request that found no healthy node, if a group is restarting.

    None when no fresh restarting group exists, which leaves the caller's generic
    "no healthy nodes available" in place.
    """
    for r in rows:
        if is_restarting(r.get("group_status"), r.get("group_reported_at"), now):
            total = r.get("group_members_total")
            up = r.get("group_members_up")
            members = f" ({up} of {total} members up)" if total is not None and up is not None else ""
            return (
                f"distributed group '{r['name']}' for model '{r.get('model_name')}' is "
                f"restarting{members}. Its runtime is reloading, so it is not routable "
                f"until it reports ready. Retry after {settings.group_retry_after_seconds}s."
            )
    return None
