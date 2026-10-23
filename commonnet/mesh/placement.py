"""Where a joining peer should serve, and which peers a request should pass through.

Pure functions over lists of live peer records (dicts, see records.py) and a Manifest.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from fractions import Fraction

from commonnet.mesh.manifest import Manifest

BACKEND_SWITCH_PENALTY = 0.5  # replaying onto another quantisation drifts a little; keep the same backend when it costs nothing
PREFERRED_BACKEND_PENALTY = 0.5


class NoRoute(RuntimeError):
    """No chain of live peers covers the model. `boundary` is the farthest layer a chain reaches."""

    def __init__(self, boundary: int):
        super().__init__(f"no live peer serves layers from {boundary}")
        self.boundary = boundary


class NoWindow(RuntimeError):
    """No boundary-aligned window fits in the memory offered."""


@dataclass(frozen=True)
class Hop:
    window: tuple[int, int]
    primary: dict
    alternates: tuple = field(default_factory=tuple)


def _window(record: dict) -> tuple[int, int] | None:
    w = record.get("window")
    return None if w is None else (w[0], w[1])


def _serving(manifest: Manifest, live) -> list[dict]:
    out = []
    for r in live:
        w = _window(r)
        if r.get("state") == "up" and w is not None and manifest.is_window(*w):
            out.append(r)
    return out


def coverage(manifest: Manifest, live) -> list[int]:
    """For each stage, how many live peers serve it."""
    peers = _serving(manifest, live)
    counts = []
    for a, b in manifest.stages():
        counts.append(sum(1 for r in peers if _window(r)[0] <= a and b <= _window(r)[1]))
    return counts


def replica_counts(manifest: Manifest, live) -> dict[tuple[int, int], int]:
    """Live peers per exact window."""
    counts: dict[tuple[int, int], int] = {}
    for r in _serving(manifest, live):
        counts[_window(r)] = counts.get(_window(r), 0) + 1
    return counts


def choose_window(manifest: Manifest, live, gib: float, backend: str, me: str | None = None) -> tuple[int, int]:
    """The window a joining peer with `gib` of memory should serve.

    Stages nobody serves dominate the score, then a longer window (fewer hops), then a window some peer
    already serves with the fewest replicas, so that failover has an identical peer to go to.
    """
    others = [r for r in live if r.get("peer") != me]
    cov = coverage(manifest, others)
    replicas = replica_counts(manifest, others)
    stage_index = {s: i for i, s in enumerate(manifest.stages())}
    candidates = manifest.windows_fitting(gib, backend)
    if not candidates:
        raise NoWindow(f"{gib:.2f} GiB is too little for even one stage ({manifest.window_gib(*manifest.stages()[0], backend):.2f} GiB)")

    def key(w):
        score = sum(Fraction(1, 1 + cov[stage_index[s]]) for s in manifest.stages() if w[0] <= s[0] and s[1] <= w[1])
        existing = replicas.get(w, 0)
        return (-score, -(w[1] - w[0]), 0 if existing else 1, existing, w[0])

    return min(candidates, key=key)


def should_move(manifest: Manifest, live, me: str, *, gib: float | None = None,
                backend: str | None = None) -> tuple[int, int] | None:
    """A new window for `me`, only if that restores coverage at no cost to the stages it leaves.

    True when some stage has no live peer, every stage in my window has at least two, and I can hold a
    window that covers an uncovered stage. The caller applies the cooldown.
    """
    mine = next((r for r in live if r.get("peer") == me and r.get("state") == "up"), None)
    if mine is None or _window(mine) is None:
        return None
    cov = coverage(manifest, live)
    stages = manifest.stages()
    uncovered = [i for i, c in enumerate(cov) if c == 0]
    if not uncovered:
        return None
    w = _window(mine)
    if any(cov[i] < 2 for i, s in enumerate(stages) if w[0] <= s[0] and s[1] <= w[1]):
        return None
    gib = mine.get("gib", 0.0) if gib is None else gib
    backend = mine.get("backend", "") if backend is None else backend
    try:
        new = choose_window(manifest, live, gib, backend, me=me)
    except NoWindow:
        return None
    if new == w or not any(new[0] <= stages[i][0] and stages[i][1] <= new[1] for i in uncovered):
        return None
    return new


def _penalty(record: dict, now: float, previous_backend: str | None, prefer_backend: str | None) -> float:
    age = max(0.0, now - record.get("ts", now))
    penalty = min(age / 60.0, 1.0)
    if previous_backend is not None and record.get("backend") != previous_backend:
        penalty += BACKEND_SWITCH_PENALTY
    if prefer_backend is not None and record.get("backend") != prefer_backend:
        penalty += PREFERRED_BACKEND_PENALTY
    return penalty


def best_route(manifest: Manifest, live, prefer_backend: str | None = None, now: float | None = None) -> list[Hop]:
    """The chain of peers with the fewest hops, then the lowest penalty; each hop lists its identical-window replicas."""
    now = time.time() if now is None else now
    groups: dict[tuple[int, int], list[dict]] = {}
    for r in _serving(manifest, live):
        groups.setdefault(_window(r), []).append(r)
    end = manifest.num_layers
    # state: (boundary, backend of the previous hop) -> (hops, penalty, [(window, record), ...])
    best: dict[tuple[int, str | None], tuple[int, float, list]] = {(0, None): (0, 0.0, [])}
    reached = {0}
    for boundary in manifest.boundaries:
        for (b, prev), (hops, pen, path) in list(best.items()):
            if b != boundary:
                continue
            for window, peers in groups.items():
                if window[0] != boundary:
                    continue
                pick = min(peers, key=lambda r: (_penalty(r, now, prev, prefer_backend), r["peer"]))
                cost = (hops + 1, pen + _penalty(pick, now, prev, prefer_backend))
                state = (window[1], pick.get("backend"))
                if state not in best or cost < best[state][:2]:
                    best[state] = (cost[0], cost[1], path + [(window, pick)])
                reached.add(window[1])
    finals = [v for (b, _), v in best.items() if b == end]
    if not finals:
        raise NoRoute(max(reached))
    _, _, path = min(finals, key=lambda v: (v[0], v[1]))
    route = []
    for window, pick in path:
        others = [r for r in groups[window] if r["peer"] != pick["peer"]]
        others.sort(key=lambda r: (r.get("backend") != pick.get("backend"), -r.get("ts", 0.0), r["peer"]))
        route.append(Hop(window, pick, tuple(others)))
    return route
