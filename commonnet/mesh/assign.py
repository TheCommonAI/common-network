"""Whole-network layer assignment: the window every live peer serves, decided from all their records at once.

placement.py decides one joining peer's window from the peers it can see. This module decides the whole network
from the same signed records, so any coordinator that reads the same records computes the same plan.

A plan has two parts:

- a chain: windows that tile [0, num_layers) exactly, each on a stage boundary and each served by one peer. It is
  the route a request takes. No gaps, no overlaps.
- replicas: every other peer, on a window that fits it (a chain window when it can be one), so failover has a twin.

Among the plans that cover the model, the planner moves the fewest peers off their current windows (so a join or a
drop does not restart the others), then uses the fewest hops, then puts the fastest peers on the chain.
"""
from __future__ import annotations

from dataclasses import dataclass

from commonnet.mesh import placement
from commonnet.mesh.manifest import Manifest

MAX_STAGES = 16  # the planner tries every tiling of the stages: 2**(stages - 1) of them


class AssignmentError(RuntimeError):
    """The live peers cannot cover the model. The message names the layers nobody can hold."""


@dataclass(frozen=True)
class Plan:
    chain: tuple[tuple[int, int], ...]  # windows that tile [0, num_layers), in order
    windows: dict  # peer id -> (start, end), every peer that serves something (chain and replicas)
    idle: tuple[str, ...]  # live peers with no window: too little memory for any window of this model

    def to_dict(self) -> dict:
        return {"chain": [list(w) for w in self.chain],
                "windows": {p: list(w) for p, w in sorted(self.windows.items())},
                "idle": list(self.idle)}


def _window(record: dict) -> tuple[int, int] | None:
    w = record.get("window")
    return None if w is None else (w[0], w[1])


def _speed(record: dict) -> float:
    return float(record.get("speed") or 0.0)


def _fits(manifest: Manifest, record: dict, window) -> bool:
    return manifest.window_gib(*window, record["backend"]) <= float(record["gib"])


def _peers(live) -> list[dict]:
    """Live peers that can serve: up, with a backend and some memory. Fastest and largest first, then by id."""
    usable = [r for r in live if r.get("state") == "up" and r.get("backend") and float(r.get("gib") or 0.0) > 0]
    return sorted(usable, key=lambda r: (-_speed(r), -float(r["gib"]), r["peer"]))


def _tilings(boundaries) -> list[tuple[tuple[int, int], ...]]:
    """Every way to cut [0, L) at stage boundaries. Fewest windows first."""
    inner = boundaries[1:-1]
    out = []
    for mask in range(1 << len(inner)):
        cuts = (boundaries[0], *(b for k, b in enumerate(inner) if mask >> k & 1), boundaries[-1])
        out.append(tuple(zip(cuts, cuts[1:])))
    out.sort(key=len)
    return out


def _match(tiling, candidates: dict, seed: dict) -> dict | None:
    """Give every window of `tiling` its own peer (augmenting paths, starting from the seeded pairs).

    Seeded pairs stay put unless an augmenting path needs them elsewhere, which is how the plan avoids moves.
    Returns {window: peer index}, or None when some window cannot get a peer.
    """
    owner: dict[int, tuple] = {}
    match: dict[tuple, int] = {}
    for window, i in seed.items():
        match[window] = i
        owner[i] = window

    def augment(window, seen: set) -> bool:
        for i in candidates[window]:
            if i in seen:
                continue
            seen.add(i)
            if i not in owner or augment(owner[i], seen):
                match[window] = i
                owner[i] = window
                return True
        return False

    for window in tiling:
        if window not in match and not augment(window, set()):
            return None
    return match


def _assign(manifest: Manifest, tiling, peers: list[dict], previous: dict[int, tuple]):
    """(windows {peer index: window}, chain {window: peer index}) for one tiling, or None if it cannot be served."""
    candidates = {}
    for window in tiling:
        fit = [i for i, p in enumerate(peers) if _fits(manifest, p, window)]
        if not fit:
            return None
        fit.sort(key=lambda i: (previous.get(i) != window, -_speed(peers[i]), peers[i]["peer"]))
        candidates[window] = fit
    seed = {}
    for i in range(len(peers)):
        window = previous.get(i)
        if window in candidates and i in candidates[window] and window not in seed:
            seed[window] = i
    chain = _match(tiling, candidates, seed)
    if chain is None:
        return None
    windows = {i: window for window, i in chain.items()}
    for i, p in enumerate(peers):
        if i in windows:
            continue
        kept = previous.get(i)
        if kept is not None and _fits(manifest, p, kept):
            windows[i] = kept
            continue
        view = [{"peer": peers[j]["peer"], "window": windows[j], "state": "up"} for j in windows]
        try:
            windows[i] = placement.choose_window(manifest, view, float(p["gib"]), p["backend"])
        except placement.NoWindow:
            continue
    return windows, chain


def check_cover(manifest: Manifest, windows) -> None:
    """Raise AssignmentError unless `windows` tile [0, num_layers): boundary-aligned, no gaps, no overlaps."""
    ordered = sorted(tuple(w) for w in windows)
    position = 0
    for a, b in ordered:
        if not manifest.is_window(a, b):
            raise AssignmentError(f"window {a}-{b} does not start and end on a stage boundary")
        if a > position:
            raise AssignmentError(f"layers {position}-{a} have no window")
        if a < position:
            raise AssignmentError(f"layers {a}-{min(b, position)} are assigned twice")
        position = b
    if position != manifest.num_layers:
        raise AssignmentError(f"layers {position}-{manifest.num_layers} have no window")


def plan(manifest: Manifest, live, previous: dict | None = None) -> Plan:
    """The assignment for `live` peer records.

    `previous` maps peer id to the window it serves now. By default that is each record's own window, so a plan made
    from the current network keeps every peer where it is unless the model needs it moved. Raises AssignmentError
    when no assignment covers the model.
    """
    if len(manifest.stages()) > MAX_STAGES:
        raise AssignmentError(f"{len(manifest.stages())} stages is more than the planner searches ({MAX_STAGES})")
    peers = _peers(live)
    index = {p["peer"]: i for i, p in enumerate(peers)}
    current = {p["peer"]: _window(p) for p in peers} if previous is None else previous
    prev: dict[int, tuple] = {}
    for peer_id, window in current.items():
        if peer_id in index and window is not None and manifest.is_window(*window):
            prev[index[peer_id]] = (window[0], window[1])

    best = None
    for tiling in _tilings(manifest.boundaries):
        result = _assign(manifest, tiling, peers, prev)
        if result is None:
            continue
        windows, chain = result
        moved = sum(1 for i, window in prev.items() if windows.get(i) != window)
        chain_speed = sum(_speed(peers[i]) for i in chain.values())
        key = (moved, len(tiling), -chain_speed, tiling)
        if best is None or key < best[0]:
            best = (key, tiling, windows)
    if best is None:
        raise _uncovered(manifest, peers)
    _, tiling, windows = best
    result = Plan(
        chain=tiling,
        windows={peers[i]["peer"]: window for i, window in windows.items()},
        idle=tuple(sorted(p["peer"] for i, p in enumerate(peers) if i not in windows)),
    )
    check_cover(manifest, result.chain)
    return result


def changes(previous: dict, new: Plan) -> dict:
    """Peers in `previous` that now serve a different window (or none): the ones that would restart.

    Peers that are not in `previous` (joiners) are not listed; callers pass only peers that were live.
    """
    out = {}
    for peer_id, window in previous.items():
        if window is None:
            continue
        now = new.windows.get(peer_id)
        if now != tuple(window):
            out[peer_id] = (tuple(window), now)
    return out


def _spans(stages) -> str:
    merged: list[list[int]] = []
    for a, b in sorted(stages):
        if merged and merged[-1][1] == a:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return ", ".join(f"{a}-{b}" for a, b in merged)


def _uncovered(manifest: Manifest, peers: list[dict]) -> AssignmentError:
    name = manifest.model.get("name", manifest.id)
    total = manifest.num_layers
    if not peers:
        return AssignmentError(f"no live peers serve {name}")
    gap = [s for s in manifest.stages() if not any(_fits(manifest, p, s) for p in peers)]
    if gap:
        return AssignmentError(f"no live peer has the memory to hold layers {_spans(gap)} of {name} "
                               f"({len(peers)} peer(s) online)")
    reach = 0
    for end in manifest.boundaries[1:]:
        prefix = tuple(b for b in manifest.boundaries if b <= end)
        if any(_assign(manifest, t, peers, {}) is not None for t in _tilings(prefix)):
            reach = end
        else:
            break
    return AssignmentError(f"{len(peers)} live peer(s) can form a chain over layers 0-{reach} of {name}, but no "
                           f"chain of live peers covers layers {reach}-{total}; a peer with enough memory for "
                           f"layers {reach}-{total} (or a window starting at layer {reach}) is needed")
