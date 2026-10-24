"""Peers learn about each other by swapping signed records. There is no directory.

PeerTable holds the newest verified record per peer. A peer's block server answers GET /peers,
POST /gossip and GET /manifest without authentication (the data is public and signed): handle_gossip,
peers_body and manifest_body are what those endpoints return. Gossiper pushes the table to a few
random peers and a seed every few seconds, and takes their tables back.
"""
from __future__ import annotations

import http.client
import json
import random
import socket
import threading
import time
import urllib.parse

from commonnet.mesh import records as rec
from commonnet.mesh.manifest import Manifest
from commonnet.net import dns

MAX_RECORDS = 1000
MAX_BODY_BYTES = 4 << 20
TOMBSTONE_SECONDS = 300.0
FORGET_SECONDS = 600.0
FAILURES_BEFORE_SKIP = 3
GOSSIP_TIMEOUT_SECONDS = 5.0
ROUND_SECONDS = (5.0, 10.0)
FANOUT = 3


class GossipError(OSError):
    pass


class PeerTable:
    def __init__(self, network_id: str, now=time.time):
        self.network = network_id
        self._now = now
        self._records: dict[str, dict] = {}
        self._lock = threading.Lock()

    def merge(self, record) -> bool:
        """Keep `record` if it is genuine, for this network, not from the future, and newer than what we hold."""
        if not rec.verify_record(record) or record["network"] != self.network:
            return False
        now = self._now()
        if record["ts"] > now + rec.FUTURE_SKEW_SECONDS or now - record["ts"] > FORGET_SECONDS:
            return False
        with self._lock:
            held = self._records.get(record["peer"])
            if held is not None and held["seq"] >= record["seq"]:
                return False
            self._records[record["peer"]] = dict(record)
            return True

    def merge_many(self, records) -> int:
        if not isinstance(records, list):
            return 0
        return sum(1 for r in records[:MAX_RECORDS] if self.merge(r))

    def get(self, peer: str) -> dict | None:
        with self._lock:
            held = self._records.get(peer)
            return None if held is None else dict(held)

    def live(self, now: float | None = None) -> list[dict]:
        now = self._now() if now is None else now
        with self._lock:
            return [dict(r) for r in self._records.values() if rec.is_live(r, now, self.network)]

    def export(self, limit: int = 200) -> list[dict]:
        """What we hand to a neighbour: fresh records (any state but leaving) and recent tombstones, newest first."""
        now = self._now()
        with self._lock:
            for peer in [p for p, r in self._records.items() if now - r["ts"] > FORGET_SECONDS]:
                del self._records[peer]
            keep = [dict(r) for r in self._records.values()
                    if (r["state"] != "leaving" and now - r["ts"] <= rec.LIVE_SECONDS)
                    or (r["state"] == "leaving" and now - r["ts"] <= TOMBSTONE_SECONDS)]
        keep.sort(key=lambda r: -r["ts"])
        return keep[:limit]

    def sample(self, k: int, exclude=(), rng=random) -> list[dict]:
        pool = [r for r in self.live() if r["peer"] not in exclude and r.get("url")]
        return rng.sample(pool, min(k, len(pool)))


# --- what a peer's server returns -------------------------------------------------------------

def handle_gossip(table: PeerTable, body) -> dict:
    """POST /gossip: take the sender's records, answer with ours."""
    if isinstance(body, dict):
        table.merge_many(body.get("records"))
    return {"records": table.export()}


def peers_body(table: PeerTable) -> dict:
    """GET /peers."""
    return {"records": table.export()}


def manifest_body(manifest: Manifest) -> dict:
    """GET /manifest."""
    return manifest.to_dict()


# --- talking to another peer ------------------------------------------------------------------

def _request(method: str, url: str, path: str, body: dict | None, timeout: float, context=None):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise GossipError(f"not an http(s) URL: {url!r}")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    base = parts.path.rstrip("/")
    cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    if context is not None and parts.scheme == "https":  # a direct peer: its certificate is pinned, not CA-signed
        conn = cls(parts.hostname, port, timeout=timeout, context=context)
    else:
        conn = cls(parts.hostname, port, timeout=timeout)
    try:
        # connect to the resolved address (public DNS fallback for brand-new quick-tunnel names)
        address = dns.resolve(parts.hostname, port)
        if address != parts.hostname:
            sock = socket.create_connection((address, port), timeout)
            if parts.scheme == "https":
                sock = conn._context.wrap_socket(sock, server_hostname=parts.hostname)
            conn.sock = sock
        data = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json"} if data is not None else {}
        conn.request(method, base + path, body=data, headers=headers)
        response = conn.getresponse()
        raw = response.read(MAX_BODY_BYTES + 1)
        if response.status != 200:
            raise GossipError(f"{method} {path} answered HTTP {response.status}")
        if len(raw) > MAX_BODY_BYTES:
            raise GossipError("reply is too large")
        return json.loads(raw.decode("utf-8"))
    except (OSError, http.client.HTTPException, ValueError) as e:
        if isinstance(e, GossipError):
            raise
        raise GossipError(f"{method} {url}{path}: {e}") from None
    finally:
        conn.close()


def exchange(table: PeerTable, url: str, timeout: float = GOSSIP_TIMEOUT_SECONDS, context=None) -> int:
    """Send our records to the peer at `url` (POST /gossip) and merge theirs. Returns how many were new to us."""
    reply = _request("POST", url, "/gossip", {"records": table.export()}, timeout, context)
    if not isinstance(reply, dict):
        raise GossipError("reply is not an object")
    return table.merge_many(reply.get("records"))


def fetch_peers(table: PeerTable, url: str, timeout: float = GOSSIP_TIMEOUT_SECONDS, context=None) -> int:
    """Pull only (GET /peers), for a client that does not publish itself."""
    reply = _request("GET", url, "/peers", None, timeout, context)
    if not isinstance(reply, dict):
        raise GossipError("reply is not an object")
    return table.merge_many(reply.get("records"))


def fetch_manifest(url: str, timeout: float = GOSSIP_TIMEOUT_SECONDS, context=None) -> Manifest:
    """GET /manifest from any peer."""
    return Manifest.from_dict(_request("GET", url, "/manifest", None, timeout, context))


def pinned_context(table: PeerTable, url: str):
    """The TLS context for gossiping with `url`: pinned to the fingerprint its record carries, or None for any
    other URL (tunnels and plain peers behave exactly as before)."""
    from commonnet.net import secure
    if not rec.is_direct_url(url):
        return None
    for r in table.export(limit=MAX_RECORDS):
        if r.get("url") == url and r.get("fp"):
            return secure.client_ssl_context(r["fp"])
    return None


def pinned_exchange(table: PeerTable, url: str, timeout: float = GOSSIP_TIMEOUT_SECONDS) -> int:
    """exchange() that pins a direct peer's certificate from its record. Same signature as exchange."""
    return exchange(table, url, timeout, pinned_context(table, url))


def pinned_fetch_peers(table: PeerTable, url: str, timeout: float = GOSSIP_TIMEOUT_SECONDS) -> int:
    return fetch_peers(table, url, timeout, pinned_context(table, url))


# --- the gossip loop --------------------------------------------------------------------------

class Gossiper:
    """Every few seconds: publish our record, swap tables with a few random peers and one seed.

    self_record_fn() returns this peer's freshly signed record (or None while it has nothing to say). A peer
    that fails three exchanges in a row is skipped until it answers again; its record simply ages out.
    """

    def __init__(self, table: PeerTable, manifest: Manifest, self_record_fn, seeds=(), *, exchange_fn=exchange,
                 fanout: int = FANOUT, interval=ROUND_SECONDS, rng=None):
        self.table = table
        self.manifest = manifest
        self.self_record_fn = self_record_fn
        self.seeds = [s for s in (list(seeds) or list(manifest.seeds))]
        self.exchange_fn = exchange_fn
        self.fanout = fanout
        self.interval = interval
        self.rng = rng or random.Random()
        self.failures: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def run_round(self) -> int:
        """One round, synchronously. Returns the number of exchanges that succeeded."""
        own = self.self_record_fn()
        if own is not None:
            self.table.merge(own)
        me, my_url = (own or {}).get("peer"), (own or {}).get("url")
        urls = [r["url"] for r in self.table.sample(self.fanout * 3, exclude={me} if me else (), rng=self.rng)
                if self.failures.get(r["url"], 0) < FAILURES_BEFORE_SKIP and r["url"] != my_url][:self.fanout]
        seeds = [s for s in self.seeds if s != my_url]
        if seeds:
            seed = self.rng.choice(seeds)
            if seed not in urls and (self.failures.get(seed, 0) < FAILURES_BEFORE_SKIP or not urls):
                urls.append(seed)
        done = 0
        for url in urls:
            try:
                self.exchange_fn(self.table, url, GOSSIP_TIMEOUT_SECONDS)
            except (OSError, ValueError):
                self.failures[url] = self.failures.get(url, 0) + 1
            else:
                self.failures[url] = 0
                done += 1
        return done

    def start(self) -> "Gossiper":
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="gossip", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=GOSSIP_TIMEOUT_SECONDS + 1)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_round()
            except Exception:  # a bad round must not end gossip
                pass
            self._stop.wait(self.rng.uniform(*self.interval))
