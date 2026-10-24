"""A tiny in-memory rendezvous: peers announce their signed record, anyone lists the live ones for a network.

The gateway only stores and relays records; every reader still verifies the signature itself. Nothing is
persisted: a restart empties the directory and peers re-announce within seconds.

  POST /announce        a signed peer record (commonnet/mesh/records.py format)
  GET  /directory       up to 100 live records of one network
  POST /check           can the gateway open a TLS connection back to the CALLER's own ip:port?
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import ssl
import threading
import time

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

TTL = 90.0                 # a stored record is dropped this long after it arrived
LIVE_SECONDS = 60.0        # same window as commonnet.mesh.records.is_live
FUTURE_SKEW = 120.0
MAX_BODY = 4096
MAX_PER_NETWORK = 200
MAX_PER_SOURCE = 200
MAX_TOTAL = 5000
MAX_LIST = 100
ANNOUNCE_PER_MIN = 30
CHECK_PER_MIN = 10
CHECK_TIMEOUT = 3.0
STATES = ("loading", "up", "leaving")
SIGNED = ("v", "network", "peer", "pub", "url", "token", "window", "backend", "state", "gib", "seq", "ts")
OPTIONAL = ("fp", "fpsig", "dht")
BIND = b"commonnet-tls-cert-v1\n"

router = APIRouter()
_lock = threading.Lock()
_records: dict = {}        # (network, peer) -> {"rec", "at", "src"}
_hits: dict = {}           # (kind, ip) -> [timestamps]
# Tests set this so an X-Forwarded-For header can stand in for the caller; production never honours it.
TRUST_TEST_HEADER = os.environ.get("COMMON_GATEWAY_TEST") == "1"


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"), validate=True)


def caller_ip(request: Request) -> str:
    """The caller's address from the proxy header; never from a body. Uses the LAST entry (the one our own proxy
    appended), so a client cannot aim /check at a third party by sending its own X-Forwarded-For."""
    xff = request.headers.get("x-forwarded-for", "")
    if xff and (TRUST_TEST_HEADER or os.environ.get("RAILWAY_ENVIRONMENT")):
        first = xff.split(",")[-1].strip()
        if first:
            return first
    return request.client.host if request.client else "0.0.0.0"


def _limited(kind: str, ip: str, limit: int, now: float) -> bool:
    with _lock:
        if len(_hits) > 20000:
            for k in [k for k, v in _hits.items() if not v or now - v[-1] > 60]:
                del _hits[k]
        hits = [t for t in _hits.get((kind, ip), []) if now - t < 60.0]
        if len(hits) >= limit:
            _hits[(kind, ip)] = hits
            return True
        hits.append(now)
        _hits[(kind, ip)] = hits
        return False


def verify(rec) -> str | None:
    """None if the record is well formed and correctly signed, else the reason."""
    try:
        if not isinstance(rec, dict) or any(k not in rec for k in SIGNED):
            return "malformed"
        if set(rec) - set(SIGNED) - set(OPTIONAL) - {"sig"}:
            return "unknown field"
        if rec["v"] != 1 or rec["state"] not in STATES:
            return "bad version or state"
        for k in ("network", "peer", "pub", "backend", "state", "token"):
            if not isinstance(rec[k], str):
                return "malformed"
        url = rec["url"]
        if url is not None and (not isinstance(url, str) or len(url) > 512 or not url.startswith(("http://", "https://"))):
            return "bad url"
        if len(rec["token"]) > 256 or len(rec["network"]) > 128 or len(rec["backend"]) > 128:
            return "too long"
        ts = rec["ts"]
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or type(rec["seq"]) is not int:
            return "malformed"
        if "dht" in rec and (type(rec["dht"]) is not int or not 1 <= rec["dht"] <= 65535):
            return "bad dht"
        pub = _b64d(rec["pub"])
        if len(pub) != 32 or hashlib.sha256(pub).hexdigest()[:32] != rec["peer"]:
            return "peer does not match key"
        body = json.dumps({k: v for k, v in rec.items() if k != "sig"}, sort_keys=True,
                          separators=(",", ":")).encode()
        key = Ed25519PublicKey.from_public_bytes(pub)
        key.verify(_b64d(rec["sig"]), body)
        if ("fp" in rec) != ("fpsig" in rec):
            return "fp and fpsig come together"
        if "fp" in rec:
            fp = rec["fp"].replace(":", "").strip().lower()
            if len(fp) != 64 or any(c not in "0123456789abcdef" for c in fp):
                return "bad fp"
            key.verify(_b64d(rec["fpsig"]), BIND + fp.encode("ascii"))
        return None
    except (InvalidSignature, ValueError, TypeError, KeyError, AttributeError):
        return "bad signature"


def _expire(now: float) -> None:
    for k in [k for k, v in _records.items() if now - v["at"] > TTL]:
        del _records[k]


def _live(rec: dict, now: float) -> bool:
    return rec["state"] == "up" and now - rec["ts"] <= LIVE_SECONDS and rec["ts"] <= now + FUTURE_SKEW


@router.post("/announce")
async def announce(request: Request):
    now = time.time()
    ip = caller_ip(request)
    if _limited("announce", ip, ANNOUNCE_PER_MIN, now):
        return JSONResponse({"error": "slow down"}, status_code=429)
    raw = await request.body()
    if len(raw) > MAX_BODY:
        return JSONResponse({"error": "record too large"}, status_code=413)
    try:
        rec = json.loads(raw)
    except ValueError:
        return JSONResponse({"error": "not json"}, status_code=400)
    why = verify(rec)
    if why:
        return JSONResponse({"error": why}, status_code=400)
    if not _live(rec, now):
        return JSONResponse({"error": "record is not live"}, status_code=400)
    key = (rec["network"], rec["peer"])
    with _lock:
        _expire(now)
        old = _records.get(key)
        if old and old["rec"]["seq"] > rec["seq"]:
            return {"ok": True, "stale": True}
        if old is None:
            if len(_records) >= MAX_TOTAL:
                return JSONResponse({"error": "directory full"}, status_code=503)
            if sum(1 for k in _records if k[0] == rec["network"]) >= MAX_PER_NETWORK:
                return JSONResponse({"error": "network full"}, status_code=429)
            if sum(1 for v in _records.values() if v["src"] == ip) >= MAX_PER_SOURCE:
                return JSONResponse({"error": "too many peers from one address"}, status_code=429)
        _records[key] = {"rec": rec, "at": now, "src": ip}
    return {"ok": True}


@router.get("/directory")
def directory(network: str = ""):
    now = time.time()
    with _lock:
        _expire(now)
        out = [v["rec"] for k, v in _records.items() if k[0] == network and _live(v["rec"], now)]
    out.sort(key=lambda r: r["ts"], reverse=True)
    return out[:MAX_LIST]


def _probe(ip: str, port: int, fp: str | None) -> dict:
    out = _probe_one(ip, port, fp)
    out["ip"] = ip   # lets the caller confirm the gateway saw the address it expects
    return out


def _probe_one(ip: str, port: int, fp: str | None) -> dict:
    try:
        with socket.create_connection((ip, port), timeout=CHECK_TIMEOUT) as raw:
            raw.settimeout(CHECK_TIMEOUT)
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with ctx.wrap_socket(raw) as tls:
                der = tls.getpeercert(binary_form=True) or b""
    except (OSError, ssl.SSLError):
        return {"reachable": False, "reason": "no TLS answer on that port"}
    if fp and hashlib.sha256(der).hexdigest() != fp:
        return {"reachable": False, "reason": "certificate fingerprint differs"}
    return {"reachable": True}


@router.post("/check")
async def check(request: Request):
    now = time.time()
    ip = caller_ip(request)
    if _limited("check", ip, CHECK_PER_MIN, now):
        return JSONResponse({"error": "slow down"}, status_code=429)
    try:
        body = json.loads(await request.body() or b"{}")
        port = body["port"]
        fp = body.get("fp")
        if type(port) is not int or not 1024 <= port <= 65535:
            raise ValueError
        if fp is not None:
            fp = fp.replace(":", "").strip().lower()
            if len(fp) != 64 or any(c not in "0123456789abcdef" for c in fp):
                raise ValueError
    except (ValueError, KeyError, TypeError, AttributeError):
        return JSONResponse({"error": "need port (1024-65535) and optional sha256 fp"}, status_code=400)
    import asyncio
    return await asyncio.get_running_loop().run_in_executor(None, _probe, ip, port, fp)
