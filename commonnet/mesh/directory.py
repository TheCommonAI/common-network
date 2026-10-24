"""Client of the gateway's rendezvous directory (gateway/app/directory.py), so `common join` needs no typed address.

Everything fails soft: a gateway that is down, old (404/410) or odd returns [] / None / False and callers carry on
exactly as before. Records from the directory are verified like any gossip (signature, network, freshness); the gateway
is a mailbox, not an authority.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from commonnet.mesh import records as rec

DEFAULT_GATEWAY = "https://gateway-production-b820.up.railway.app"
GATEWAY_ENV = "COMMONNET_GATEWAY"
_OFF = ("off", "none", "no", "0", "false")
MAX_REPLY = 512 * 1024


def gateway_url(environ=None, default=DEFAULT_GATEWAY) -> str | None:
    """COMMONNET_GATEWAY if set ("off" disables), else the default. None means: do not use a gateway."""
    env = os.environ if environ is None else environ
    value = env.get(GATEWAY_ENV)
    if value is None:
        return default
    value = value.strip().rstrip("/")
    if not value or value.lower() in _OFF or not value.startswith(("http://", "https://")):
        return None
    return value


def _call(method: str, url: str, body=None, timeout: float = 5.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", "User-Agent": "commonnet"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read(MAX_REPLY + 1)
    if len(raw) > MAX_REPLY:
        raise ValueError("reply too large")
    return json.loads(raw)


def fetch(gateway: str | None, network: str, *, timeout: float = 5.0, now=None) -> list:
    """Verified, live records of `network` from the gateway's directory. [] on any problem."""
    if not gateway:
        return []
    try:
        data = _call("GET", f"{gateway}/directory?network={urllib.parse.quote(network)}", timeout=timeout)
    except (OSError, ValueError, urllib.error.URLError):
        return []
    if not isinstance(data, list):
        return []
    return [r for r in data[:100] if isinstance(r, dict) and rec.verify_record(r) and rec.is_live(r, now, network)]


def announce(gateway: str | None, record: dict, *, timeout: float = 5.0) -> bool:
    if not gateway or not record or not record.get("url"):
        return False
    try:
        return bool(_call("POST", f"{gateway}/announce", record, timeout=timeout).get("ok"))
    except (OSError, ValueError, urllib.error.URLError, AttributeError):
        return False


def check(gateway: str | None, port: int, fp: str | None = None, *, timeout: float = 10.0) -> dict | None:
    """The gateway's connect-back to our own address: {"reachable": bool, "ip": str, ...}, or None when it cannot say."""
    if not gateway:
        return None
    body = {"port": int(port)}
    if fp:
        body["fp"] = fp
    try:
        out = _call("POST", f"{gateway}/check", body, timeout=timeout)
    except (OSError, ValueError, urllib.error.URLError):
        return None
    return out if isinstance(out, dict) and isinstance(out.get("reachable"), bool) else None


def dht_contacts(records, own_peer: str | None = None, limit: int = 8) -> list:
    """(host, udp_port) DHT bootstrap contacts from directory records of directly reachable peers that publish `dht`."""
    from urllib.parse import urlsplit
    out = []
    for r in records:
        if r.get("peer") == own_peer or not r.get("dht") or not rec.is_direct_url(r.get("url")):
            continue
        addr = (urlsplit(r["url"]).hostname, int(r["dht"]))
        if addr not in out:
            out.append(addr)
    return out[:limit]
