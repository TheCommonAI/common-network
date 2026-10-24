"""A small Kademlia DHT node over UDP that stores the existing signed peer records.

See README.md for the design. Everything here is off until something constructs and starts a DhtNode.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import random
import socket
import threading
import time
from collections import deque

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ..mesh.records import FUTURE_SKEW_SECONDS, LIVE_SECONDS, canonical, is_live, peer_id, verify_record, _b64d, _b64e
from .bootstrap import parse_addr
from .ids import ID_BITS, key_int, network_key, node_id, peer_key
from .store import StoreError, ValueStore
from .table import Contact, RoutingTable

MAX_PACKET = 1200          # bytes; below a typical path MTU so nothing fragments
MAX_INBOUND = 1500         # anything larger is dropped unread
REPLY_FLOOR = 300          # a reply may be this big even for a tiny request (a signed pong is ~230)
REQ_PAD = 1100             # requests that want a full-size reply are padded to this
REQUEST_TYPES = ("ping", "fn", "fv", "st")
RESPONSE_TYPES = {"ping": "pong", "fn": "nodes", "fv": "vals", "st": "ok"}
TOKEN_SECONDS = 300.0
MAX_NODES_PER_REPLY = 16
MAX_PAGES = 64
DUMMY_SIG = "A" * 88


def _encode(msg: dict) -> bytes:
    return json.dumps(msg, separators=(",", ":")).encode("utf-8")


def _is_peer(value) -> bool:
    return isinstance(value, str) and len(value) == 32 and all(c in "0123456789abcdef" for c in value)


def _is_hex(value, n: int) -> bool:
    return isinstance(value, str) and len(value) == n and all(c in "0123456789abcdef" for c in value)


def _good_addr(host, port) -> bool:
    if not isinstance(host, str) or type(port) is not int or not 0 < port < 65536:
        return False
    try:
        ip = ipaddress.IPv4Address(host)
    except ValueError:
        return False
    return not (ip.is_unspecified or ip.is_multicast or ip.is_reserved)


class _Bucket:
    """Token bucket."""

    def __init__(self, rate: float, burst: float, now: float):
        self.rate, self.burst, self.tokens, self.at = rate, burst, burst, now

    def take(self, now: float) -> bool:
        self.tokens = min(self.burst, self.tokens + (now - self.at) * self.rate)
        self.at = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class _Pending:
    __slots__ = ("addr", "expect", "data", "resp")

    def __init__(self, addr, expect, data):
        self.addr, self.expect, self.data, self.resp = addr, expect, data, None


class DhtNode:
    def __init__(self, identity, bind=("0.0.0.0", 0), bootstrap=(), network_id="", clock=time.time, now_rng=None,
                 *, k: int = 20, alpha: int = 3, rpc_timeout: float = 0.5, retries: int = 2,
                 rate=(100.0, 200.0), ip_rate=(1000.0, 2000.0), max_per_key: int = 256,
                 max_per_source: int = 128, refresh_seconds: float = 300.0, read_fanout: int = 8):
        self.identity = identity
        self.peer = identity.peer
        self.id = node_id(self.peer)
        self.network_id = network_id
        self.clock = clock
        self.rng = now_rng or random.Random()
        self.k, self.alpha = k, alpha
        self.rpc_timeout, self.retries = rpc_timeout, retries
        self.refresh_seconds = refresh_seconds
        self.read_fanout = read_fanout
        self._bind = tuple(bind)
        self._bootstrap = [parse_addr(a) for a in bootstrap]
        self.table = RoutingTable(self.id, k)
        self.store = ValueStore(clock, max_per_key, max_per_source)
        self.stats = {"sent": 0, "received": 0, "bad": 0, "rate_limited": 0, "lookups": 0}
        self._rate, self._ip_rate = rate, ip_rate
        self._buckets: dict = {}
        self._ip_buckets: dict = {}
        self._secret = os.urandom(16)
        self._tokens: dict = {}          # addr -> (token, monotonic time)
        self._pending: dict[str, _Pending] = {}
        self._cv = threading.Condition()
        self._tasks: deque = deque(maxlen=128)
        self._recent_verify: dict = {}
        self._wake = threading.Event()
        self._stopped = threading.Event()
        self._sock: socket.socket | None = None
        self._threads: list[threading.Thread] = []
        self._last_boot = 0.0
        self._seen_sigs: set = set()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        except OSError:
            pass
        sock.bind(self._bind)
        sock.settimeout(0.2)
        self._sock = sock
        self._stopped.clear()
        port = sock.getsockname()[1]
        for name, target in (("recv", self._recv_loop), ("maint", self._maint_loop)):
            t = threading.Thread(target=target, name=f"dht-{name}-{port}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stopped.set()
        self._wake.set()
        with self._cv:
            self._cv.notify_all()
        if self._sock is not None:  # wake the receive thread now instead of waiting out its timeout
            try:
                self._sock.sendto(b"", self._sock.getsockname())
            except OSError:
                pass
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(timeout=5)
        self._threads = []
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    @property
    def address(self) -> tuple:
        if self._sock is None:
            raise RuntimeError("node is not started")
        return self._sock.getsockname()

    # ------------------------------------------------------------------ public API
    def bootstrap(self, addrs=None) -> int:
        """Ping the bootstrap addresses, then look up our own id to fill the table. Returns contacts gained."""
        targets = self._bootstrap if addrs is None else [parse_addr(a) for a in addrs]
        self._last_boot = time.monotonic()
        before = len(self.table)
        own = self.address
        jobs = [(a, None, {"t": "ping"}) for a in targets if a != own]
        if jobs:
            self._rpc_many(jobs)
        if len(self.table):
            self._lookup(self.id)
        return max(0, len(self.table) - before)

    def add_contact(self, peer_id_: str, addr) -> None:
        """Offer a contact. It only enters the routing table once it answers a signed ping as that peer."""
        if not _is_peer(peer_id_) or peer_id_ == self.peer:
            return
        addr = parse_addr(addr)
        self._queue(("verify", addr, peer_id_))

    def announce(self, record: dict) -> int:
        """Store the caller's signed record under the network key and its own peer key. Returns replicas stored
        (STORE acknowledgements summed over both keys, counting this node when it is among the closest)."""
        if not verify_record(record):
            raise ValueError("record does not verify")
        if record["network"] != self.network_id:
            raise ValueError("record is for another network")
        if record["peer"] != self.peer:
            raise ValueError("a node announces only its own record")
        stored = 0
        for key in (network_key(self.network_id), peer_key(self.peer)):
            stored += self._store_on_closest(key, record)
        return stored

    def find_peers(self) -> list[dict]:
        """Live, verified records under the network key; newest seq per peer."""
        key = network_key(self.network_id)
        contacts = self._lookup(key_int(key))
        raw = self.store.get(key) + self._collect(key, contacts)
        now = self.clock()
        best: dict[str, dict] = {}
        for rec in raw:
            if not self._valid(rec):
                continue
            if not is_live(rec, now, self.network_id):
                continue
            cur = best.get(rec["peer"])
            if cur is None or rec["seq"] > cur["seq"]:
                best[rec["peer"]] = rec
        return [best[p] for p in sorted(best)]

    def lookup_peer(self, peer_id_: str) -> dict | None:
        if not _is_peer(peer_id_):
            return None
        key = peer_key(peer_id_)
        contacts = self._lookup(key_int(key))
        now = self.clock()
        best = None
        for rec in self.store.get(key) + self._collect(key, contacts):
            if not self._valid(rec) or rec["peer"] != peer_id_:
                continue
            if now - rec["ts"] > LIVE_SECONDS or rec["ts"] > now + FUTURE_SKEW_SECONDS:
                continue
            if best is None or rec["seq"] > best["seq"]:
                best = rec
        return best

    # ------------------------------------------------------------------ records
    def _valid(self, rec) -> bool:
        """verify_record with a small cache so a record seen in 20 replicas is checked once."""
        if not isinstance(rec, dict):
            return False
        sig = rec.get("sig")
        try:
            tag = (sig, json.dumps(rec, sort_keys=True))
        except (TypeError, ValueError):
            return False
        if tag in self._seen_sigs:
            return True
        if not verify_record(rec):
            return False
        if len(self._seen_sigs) > 20000:
            self._seen_sigs.clear()
        self._seen_sigs.add(tag)
        return True

    def _store_on_closest(self, key: bytes, record: dict) -> int:
        target = key_int(key)
        closest = self._lookup(target)
        stored = 0
        if len(closest) < self.k or (self.id ^ target) < (closest[-1].id ^ target):
            try:
                self.store.put(key, record, "local")
                stored += 1
            except StoreError:
                pass
        for attempt in range(2):
            need = [c for c in closest if self._fresh_token(c.addr) is None] if attempt == 0 else closest
            if attempt == 0 and need:
                self._rpc_many([(c.addr, c.peer, {"t": "ping"}) for c in need])
            jobs = [(c.addr, c.peer, {"t": "st", "k": key.hex(), "tok": self._fresh_token(c.addr) or "", "r": record})
                    for c in closest]
            resps = self._rpc_many(jobs)
            retry = []
            for c, r in zip(closest, resps):
                if r is not None and r["t"] == "ok":
                    stored += 1
                elif r is not None and r.get("why") == "token":
                    self._tokens.pop(c.addr, None)
                    retry.append(c)
            if not retry:
                break
            self._rpc_many([(c.addr, c.peer, {"t": "ping"}) for c in retry])
            closest = retry
        return stored

    def _collect(self, key: bytes, contacts: list[Contact]) -> list[dict]:
        """Page through FIND_VALUE on each contact until it has no more records."""
        out: list[dict] = []
        contacts = contacts[:self.read_fanout]
        after = {c.peer: "" for c in contacts}
        active = list(contacts)
        for _ in range(MAX_PAGES):
            if not active:
                break
            jobs = [(c.addr, c.peer, {"t": "fv", "k": key.hex(), "a": after[c.peer], "_pad": REQ_PAD}) for c in active]
            nxt = []
            for c, r in zip(active, self._rpc_many(jobs)):
                if r is None or r["t"] != "vals" or not isinstance(r.get("vals"), list):
                    continue
                vals = [v for v in r["vals"] if isinstance(v, dict)]
                out.extend(vals[:64])
                last = vals[-1].get("peer") if vals else None
                if r.get("more") is True and isinstance(last, str) and last > after[c.peer]:
                    after[c.peer] = last
                    nxt.append(c)
            active = nxt
        return out

    # ------------------------------------------------------------------ iterative lookup
    def _lookup(self, target: int) -> list[Contact]:
        """Iterative FIND_NODE (alpha in parallel). Returns up to k contacts that answered, closest first."""
        self.stats["lookups"] += 1
        cands: dict[str, Contact] = {c.peer: Contact(c.peer, c.id, c.addr) for c in self.table.closest(target, self.k)}
        responded: dict[str, Contact] = {}
        queried: set = set()
        failed: set = set()
        for _ in range(32):
            live = sorted((c for p, c in cands.items() if p not in failed), key=lambda c: c.id ^ target)[:self.k]
            todo = [c for c in live if c.peer not in queried][:self.alpha]
            if not todo:
                break
            jobs = [(c.addr, c.peer, {"t": "fn", "k": target.to_bytes(32, "big").hex(), "_pad": REQ_PAD}) for c in todo]
            for c, r in zip(todo, self._rpc_many(jobs)):
                queried.add(c.peer)
                if r is None or r["t"] != "nodes":
                    failed.add(c.peer)
                    existing = self.table.get(c.id)
                    if existing is not None:
                        self.table.fail(c.id)
                    continue
                responded[c.peer] = c
                for entry in (r.get("nodes") or [])[:MAX_NODES_PER_REPLY * 2]:
                    if (isinstance(entry, list) and len(entry) == 3 and _is_peer(entry[0]) and entry[0] != self.peer
                            and _good_addr(entry[1], entry[2]) and entry[0] not in cands):
                        cands[entry[0]] = Contact(entry[0], node_id(entry[0]), (entry[1], entry[2]))
        return sorted(responded.values(), key=lambda c: c.id ^ target)[:self.k]

    # ------------------------------------------------------------------ sending and receiving
    def _sendto(self, data: bytes, addr) -> None:
        """The single outgoing path (tests wrap it to simulate packet loss)."""
        assert len(data) <= MAX_PACKET, "packet too large"
        sock = self._sock
        if sock is None:
            return
        try:
            sock.sendto(data, addr)
            self.stats["sent"] += 1
        except OSError:
            pass

    def _rpc_many(self, jobs):
        """Send each (addr, expected peer or None, message) in parallel, retrying on silence. Responses or None."""
        if self._sock is None:
            raise RuntimeError("node is not started")
        pend: list[tuple[str, _Pending]] = []
        for addr, expect, msg in jobs:
            nonce = os.urandom(8).hex()
            m = dict(msg)
            pad = m.pop("_pad", 0)
            m.update(n=nonce, id=self.peer)
            if pad:
                m["p"] = ""
                m["p"] = "0" * max(0, pad - len(_encode(m)))
            pend.append((nonce, _Pending(tuple(addr), expect, _encode(m))))
        with self._cv:
            for nonce, p in pend:
                self._pending[nonce] = p
        try:
            for _ in range(self.retries + 1):
                for _, p in pend:
                    if p.resp is None:
                        self._sendto(p.data, p.addr)
                deadline = time.monotonic() + self.rpc_timeout
                with self._cv:
                    while not self._stopped.is_set() and any(p.resp is None for _, p in pend):
                        left = deadline - time.monotonic()
                        if left <= 0:
                            break
                        self._cv.wait(left)
                if self._stopped.is_set() or all(p.resp is not None for _, p in pend):
                    break
        finally:
            with self._cv:
                for nonce, _ in pend:
                    self._pending.pop(nonce, None)
        return [p.resp for _, p in pend]

    def _recv_loop(self) -> None:
        sock = self._sock
        while not self._stopped.is_set():
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                if self._stopped.is_set():
                    return
                time.sleep(0.01)
                continue
            try:
                self._on_packet(data, addr)
            except Exception:  # a hostile packet must never kill the loop
                self.stats["bad"] += 1

    def _on_packet(self, data: bytes, addr) -> None:
        self.stats["received"] += 1
        if len(data) > MAX_INBOUND:
            self.stats["bad"] += 1
            return
        try:
            msg = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            self.stats["bad"] += 1
            return
        if (not isinstance(msg, dict) or not _is_hex(msg.get("n"), 16) or not _is_peer(msg.get("id"))
                or not isinstance(msg.get("t"), str)):
            self.stats["bad"] += 1
            return
        kind = msg["t"]
        if kind in REQUEST_TYPES:
            self._on_request(msg, addr, len(data))
        elif kind in RESPONSE_TYPES.values() or kind == "err":
            self._on_response(msg, addr)
        else:
            self.stats["bad"] += 1

    def _allow(self, addr) -> bool:
        now = time.monotonic()
        for table, key, (rate, burst) in ((self._buckets, addr, self._rate), (self._ip_buckets, addr[0], self._ip_rate)):
            b = table.get(key)
            if b is None:
                if len(table) > 4096:
                    table.clear()
                b = table[key] = _Bucket(rate, burst, now)
            if not b.take(now):
                self.stats["rate_limited"] += 1
                return False
        return True

    def _on_response(self, msg: dict, addr) -> None:
        with self._cv:
            p = self._pending.get(msg["n"])
        if p is None or p.resp is not None or tuple(addr) != p.addr:
            return
        if p.expect is not None and msg["id"] != p.expect:
            return
        pub, sig = msg.get("pub"), msg.get("sig")
        try:
            raw = _b64d(pub)
            if len(raw) != 32 or peer_id(raw) != msg["id"] or not isinstance(sig, str):
                return
            Ed25519PublicKey.from_public_bytes(raw).verify(_b64d(sig), canonical(msg))
        except (InvalidSignature, ValueError, TypeError):
            return
        tok = msg.get("tok")
        if isinstance(tok, str) and len(tok) <= 64:
            self._tokens[p.addr] = (tok, time.monotonic())
        if msg["id"] != self.peer:
            evict = self.table.add(Contact(msg["id"], node_id(msg["id"]), p.addr), time.monotonic())
            if evict is not None:
                self._queue(("evict", evict))
        with self._cv:
            p.resp = msg
            self._cv.notify_all()

    def _fresh_token(self, addr):
        entry = self._tokens.get(addr)
        if entry and time.monotonic() - entry[1] < TOKEN_SECONDS * 0.8:
            return entry[0]
        return None

    def _token_for(self, ip: str, epoch: int | None = None) -> str:
        epoch = int(time.monotonic() // TOKEN_SECONDS) if epoch is None else epoch
        return hmac.new(self._secret, f"{ip}/{epoch}".encode(), hashlib.sha256).hexdigest()[:24]

    def _token_ok(self, ip: str, tok) -> bool:
        if not isinstance(tok, str):
            return False
        epoch = int(time.monotonic() // TOKEN_SECONDS)
        return any(hmac.compare_digest(tok, self._token_for(ip, e)) for e in (epoch, epoch - 1))

    def _reply(self, addr, req_len: int, kind: str, nonce: str, fields=None, items_key=None, items=(), more_key=None):
        budget = min(MAX_PACKET, max(REPLY_FLOOR, req_len))
        msg = {"t": kind, "n": nonce, "id": self.peer, "pub": self.identity.pub_b64, "tok": self._token_for(addr[0])}
        msg.update(fields or {})
        n = len(items)
        while True:
            if items_key:
                msg[items_key] = list(items[:n])
            if more_key:
                msg[more_key] = n < len(items)
            if n == 0 or len(_encode(dict(msg, sig=DUMMY_SIG))) <= budget:
                break
            n -= 1
        msg["sig"] = _b64e(self.identity._key.sign(canonical(msg)))
        self._sendto(_encode(msg), addr)

    def _nodes_near(self, target: int, n: int) -> list:
        return [[c.peer, c.addr[0], c.addr[1]] for c in self.table.closest(target, n)]

    def _on_request(self, msg: dict, addr, size: int) -> None:
        if not self._allow(addr):
            return
        kind, nonce = msg["t"], msg["n"]
        if msg["id"] != self.peer and self.table.by_addr(addr) is None:
            self._queue(("verify", tuple(addr), msg["id"]))
        if kind == "ping":
            self._reply(addr, size, "pong", nonce)
        elif kind == "fn":
            if not _is_hex(msg.get("k"), 64):
                return
            self._reply(addr, size, "nodes", nonce, items_key="nodes",
                        items=self._nodes_near(int(msg["k"], 16), MAX_NODES_PER_REPLY))
        elif kind == "fv":
            if not _is_hex(msg.get("k"), 64) or not isinstance(msg.get("a", ""), str) or len(msg.get("a", "")) > 32:
                return
            key = bytes.fromhex(msg["k"])
            after = msg.get("a", "")
            fields = {"nodes": self._nodes_near(int(msg["k"], 16), 3)} if after == "" else None
            self._reply(addr, size, "vals", nonce, fields, "vals", self.store.get(key, after), "more")
        elif kind == "st":
            self._on_store(msg, addr, size)

    def _on_store(self, msg: dict, addr, size: int) -> None:
        nonce = msg["n"]
        if not self._token_ok(addr[0], msg.get("tok")):
            return self._reply(addr, size, "err", nonce, {"why": "token"})
        if not _is_hex(msg.get("k"), 64):
            return self._reply(addr, size, "err", nonce, {"why": "key"})
        try:
            self.store.put(bytes.fromhex(msg["k"]), msg.get("r"), addr[0])
        except StoreError as e:
            return self._reply(addr, size, "err", nonce, {"why": str(e)[:60]})
        self._reply(addr, size, "ok", nonce)

    # ------------------------------------------------------------------ maintenance
    def _queue(self, task) -> None:
        if task[0] == "verify":
            now = time.monotonic()
            if now - self._recent_verify.get(task[1], -1e9) < 30.0:
                return
            if len(self._recent_verify) > 2048:
                self._recent_verify.clear()
            self._recent_verify[task[1]] = now
        self._tasks.append(task)
        self._wake.set()

    def _run_task(self, task) -> None:
        if task[0] == "verify":
            self._rpc_many([(task[1], task[2], {"t": "ping"})])
        elif task[0] == "evict":
            old: Contact = task[1]
            r = self._rpc_many([(old.addr, old.peer, {"t": "ping"})])[0]
            if r is None:
                self.table.remove(old.id)

    def run_maintenance(self, force_refresh: bool = False) -> None:
        """One maintenance pass (also what the background thread runs): tasks, expiry, refresh, re-bootstrap."""
        for _ in range(16):
            try:
                task = self._tasks.popleft()
            except IndexError:
                break
            self._run_task(task)
            if self._stopped.is_set():
                return
        self.store.prune()
        now = time.monotonic()
        stale = self.table.stale_buckets(now, 0.0 if force_refresh else self.refresh_seconds)
        if stale:
            idx = self.rng.choice(stale)
            self.table.touch_bucket(idx, now)
            low = 1 << idx
            target = self.id ^ (low | self.rng.getrandbits(idx) if idx else low)
            self._lookup(target)
        if not len(self.table) and self._bootstrap and now - self._last_boot > 20.0:
            self.bootstrap()

    def _maint_loop(self) -> None:
        if self._bootstrap:
            try:
                self.bootstrap()
            except Exception:
                pass
        while not self._stopped.is_set():
            self._wake.wait(1.0)
            self._wake.clear()
            if self._stopped.is_set():
                return
            try:
                self.run_maintenance()
            except Exception:
                self.stats["bad"] += 1
