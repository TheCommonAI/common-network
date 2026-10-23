"""The client's view of the layer network: find peers by gossip, build a route, build a transport that survives drop-outs.

    net = Network.from_spec("https://some-peer.trycloudflare.com")     # or a manifest file
    transport = net.transport()          # route right now; replicas for every hop; waits for a newcomer if a hop empties
    endpoints = net.endpoints()          # the BlockEndpoints of that route, for Client.generate
    Client4(checkpoint, transport, ...).generate(endpoints, prompt_ids, n)

There is no directory. The client keeps a table of signed peer records (commonnet.mesh), refreshed by asking a few
peers (and a seed when it knows no one) for theirs. A route is the shortest chain of live peers whose windows tile the
model; each hop also lists the other live peers with the identical window, which are the failover choices. When every
choice of a hop is dead, the transport asks this object for the peers now listed for that window.
"""
from __future__ import annotations

import json
import os
import random
import tempfile
import time
from pathlib import Path

from commonnet.client import BlockEndpoint
from commonnet.mesh.manifest import Manifest
from commonnet.mesh.placement import NoRoute, best_route, coverage
from commonnet.net.transport import HttpTransport

SEED_FANOUT = 3                 # live peers asked per refresh, before any seed
MAX_CACHED_PEERS = 50
CACHE_NAME = "peers.json"


class NetworkUnreachable(RuntimeError):
    """No seed and no known peer answered, so there is nothing to build a route from."""


class NoLayersServed(NoRoute):
    """No chain of live peers covers the model. The message says which layers nobody serves."""

    def __init__(self, boundary: int, message: str, uncovered: list[tuple[int, int]]):
        super().__init__(boundary)
        self.args = (message,)
        self.uncovered = uncovered


def _record_window(record: dict) -> tuple[int, int] | None:
    window = record.get("window")
    return None if not window else (window[0], window[1])


def _default_exchange(table, url: str, timeout: float) -> None:
    """Pull one peer's records (GET /peers). A client publishes no record of its own, so it does not push."""
    from commonnet.mesh.gossip import fetch_peers
    fetch_peers(table, url, timeout)


def _default_fetch_manifest(url: str, timeout: float):
    from commonnet.mesh.gossip import fetch_manifest
    return fetch_manifest(url, timeout)


class Network:
    def __init__(self, seeds=(), manifest=None, *, cache_dir=None, timeout: float = 5.0,
                 refresh_interval: float = 2.0, wait_s: float = 30.0, now=time.time, clock=time.monotonic,
                 exchange=None, table=None, fetch_manifest=None):
        self.seeds = [s.rstrip("/") for s in seeds]
        self._manifest = Manifest.load(manifest) if manifest is not None else None
        if self._manifest is not None:
            self._add_seeds(self._manifest.seeds)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self.refresh_interval = refresh_interval
        self.wait_s = wait_s
        self._now = now
        self._clock = clock
        self._exchange = exchange or _default_exchange
        self._fetch_manifest = fetch_manifest or _default_fetch_manifest
        self._table = table
        self._last_refresh: float | None = None
        self._hops: list = []
        self._windows: dict[str, tuple[int, int]] = {}
        self._backends: dict[str, str | None] = {}
        self._token_for = None
        self._cached = self._read_cache()

    # --- construction -------------------------------------------------------

    @classmethod
    def from_spec(cls, spec: str, **kw) -> "Network":
        """`spec` is a peer URL (the manifest is fetched from it) or a manifest file (it lists its own seeds)."""
        if spec.startswith(("http://", "https://")):
            return cls(seeds=[spec], **kw)
        manifest = Manifest.load(spec)
        return cls(seeds=list(manifest.seeds), manifest=manifest, **kw)

    def _add_seeds(self, urls) -> None:
        for url in urls:
            url = url.rstrip("/")
            if url not in self.seeds:
                self.seeds.append(url)

    # --- manifest and table -------------------------------------------------

    @property
    def manifest(self) -> Manifest:
        if self._manifest is None:
            self._manifest = self._load_manifest()
        return self._manifest

    def _load_manifest(self) -> Manifest:
        errors = []
        for url in self.seeds + self._cached:
            try:
                manifest = Manifest.load(self._fetch_manifest(url, self.timeout))
            except Exception as e:  # a bad seed is not fatal while another answers
                errors.append(f"{url}: {e}")
                continue
            self._add_seeds(manifest.seeds)
            return manifest
        raise NetworkUnreachable("could not fetch the network manifest from any seed"
                                 + (f" ({'; '.join(errors[-3:])})" if errors else " (no seeds given)"))

    @property
    def table(self):
        if self._table is None:
            from commonnet.mesh.gossip import PeerTable
            self._table = PeerTable(self.manifest.id, now=self._now)
        return self._table

    # --- cache --------------------------------------------------------------

    def _cache_path(self) -> Path | None:
        return self.cache_dir / CACHE_NAME if self.cache_dir else None

    def _read_cache(self) -> list[str]:
        path = self._cache_path()
        if path is None:
            return []
        try:
            data = json.loads(path.read_text())
            return [u for u in data.get("urls", []) if isinstance(u, str)][:MAX_CACHED_PEERS]
        except (OSError, ValueError, AttributeError):
            return []

    def _write_cache(self, urls: list[str]) -> None:
        path = self._cache_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".peers-")
            with os.fdopen(fd, "w") as f:
                json.dump({"network": self._manifest.id if self._manifest else None,
                           "urls": urls[:MAX_CACHED_PEERS]}, f)
            os.replace(tmp, path)
        except OSError:
            pass  # the cache is a convenience

    # --- refresh and route --------------------------------------------------

    def refresh(self, force: bool = False) -> int:
        """Ask a few peers for their records. Returns how many live peers are known afterwards.

        Throttled to one round per refresh_interval unless forced. Raises NetworkUnreachable only when nobody
        answered and the table holds no live peer.
        """
        if (not force and self._last_refresh is not None
                and self._clock() - self._last_refresh < self.refresh_interval):
            return len(self.table.live())
        table = self.table
        self._last_refresh = self._clock()
        known = [r["url"] for r in table.live() if r.get("url")]
        sources = random.sample(known, min(SEED_FANOUT, len(known)))
        tail = [u for u in self.seeds + self._cached if u not in sources]
        answered, errors = 0, []
        for url in sources + tail:
            if answered and url in tail:
                break  # seeds and cached peers are only for when the live peers did not answer
            try:
                self._exchange(table, url, self.timeout)
                answered += 1
            except Exception as e:
                errors.append(f"{url}: {e}")
        live = table.live()
        if not answered and not live:
            raise NetworkUnreachable("no seed or known peer answered"
                                     + (f" ({'; '.join(errors[-3:])})" if errors else " (no seeds given)"))
        self._write_cache([r["url"] for r in live if r.get("url")])
        return len(live)

    def route(self, prefer_backend: str | None = None) -> list:
        """The current chain of hops (mesh.placement.Hop). Refreshes once and retries before giving up."""
        manifest = self.manifest
        self.refresh()
        try:
            return best_route(manifest, self.table.live(), prefer_backend, self._now())
        except NoRoute:
            pass
        self.refresh(force=True)
        try:
            return best_route(manifest, self.table.live(), prefer_backend, self._now())
        except NoRoute as e:
            raise self._explain(e) from None

    def _explain(self, error: NoRoute) -> NoLayersServed:
        manifest, live = self.manifest, self.table.live()
        counts = coverage(manifest, live)
        stages = manifest.stages()
        missing = [stages[i] for i, c in enumerate(counts) if c == 0]
        online = len([r for r in live if r.get("state") == "up"])
        name = manifest.model.get("name", manifest.id)
        if missing:
            spans = ", ".join(f"{a}-{b}" for a, b in missing)
            why = f"no live peer serves layers {spans} of {name} ({online} peer(s) online)"
        else:
            why = (f"{online} peer(s) online, but no chain of them covers {name} past layer {error.boundary}; "
                   f"a peer starting at layer {error.boundary} is needed")
        return NoLayersServed(error.boundary, why + ". Run `common join --network` on a machine that can hold them.",
                              missing)

    # --- what the client uses ----------------------------------------------

    def _tokens(self, record: dict) -> str | None:
        return self._token_for(record) if self._token_for else record.get("token")

    def _fresh_for(self, name: str) -> list:
        """(url, token) of every live peer the network lists for this hop's window, same backend as the primary first."""
        try:
            self.refresh()
        except NetworkUnreachable:
            pass
        window = self._windows.get(name)
        if window is None:
            return []
        want = self._backends.get(name)
        records = [r for r in self.table.live()
                   if r.get("state") == "up" and r.get("url") and _record_window(r) == window]
        records.sort(key=lambda r: (r.get("backend") != want, -r.get("ts", 0.0), r["peer"]))
        return [(r["url"], self._tokens(r)) for r in records]

    def transport(self, *, token_for=None, prefer_backend: str | None = None, wait_s: float | None = None,
                  **transport_kwargs) -> HttpTransport:
        """A transport over the current route. Each hop's alternates are its replicas; an emptied hop waits for a newcomer."""
        self._token_for = token_for
        hops = self.route(prefer_backend)
        self._hops = hops
        self._windows = {f"hop{i}": h.window for i, h in enumerate(hops)}
        self._backends = {f"hop{i}": h.primary.get("backend") for i, h in enumerate(hops)}
        urls = {f"hop{i}": h.primary["url"] for i, h in enumerate(hops)}
        tokens = {f"hop{i}": self._tokens(h.primary) for i, h in enumerate(hops)}
        replicas = {f"hop{i}": [(r["url"], self._tokens(r)) for r in h.alternates] for i, h in enumerate(hops)}
        return HttpTransport(urls, tokens=tokens, replicas=replicas, on_exhausted=self._fresh_for,
                             wait_s=self.wait_s if wait_s is None else wait_s, **transport_kwargs)

    def endpoints(self) -> list[BlockEndpoint]:
        """The BlockEndpoints of the route transport() last built (it builds one if there is none)."""
        if not self._hops:
            self.transport()
        return [BlockEndpoint(f"hop{i}", h.window[0], h.window[1]) for i, h in enumerate(self._hops)]

    def plan(self, **kw):
        """(endpoints, transport) from one route, so the two always agree."""
        transport = self.transport(**kw)
        return self.endpoints(), transport

    def status(self) -> dict:
        """What /health shows: model, peers online and the per-stage coverage."""
        manifest = self.manifest
        live = self.table.live()
        return {"network": manifest.id, "model": manifest.model.get("name"), "peers": len(live),
                "stages": [{"layers": [a, b], "peers": c} for (a, b), c in zip(manifest.stages(), coverage(manifest, live))]}
