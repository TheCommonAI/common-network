"""Public, text-free model/peer status and signed chat timing reports.

The directory still handles discovery. This is an observational view, never a request router.
State is bounded and in memory; fresh heartbeats rebuild it after a gateway restart.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi.responses import JSONResponse

WINDOW_SECONDS = 300
BUCKET_SECONDS = 10
HISTORY_SECONDS = 3600
LIVE_SECONDS = 60
MAX_BODY = 32768
MAX_PEERS = 5000
MAX_SESSIONS = 5000
MAX_PER_SOURCE = 200
BIND = b"commonnet-dashboard-v1\n"
COUNTERS_V1 = ("requests", "completed", "failed", "cancelled", "output_tokens", "decode_tokens", "decode_ms",
               "ttft_count", "ttft_ms", "total_ms")
COUNTERS_V2 = (*COUNTERS_V1, "stage_count", "stage_ms", "network_count", "network_ms")
COUNTERS = (*COUNTERS_V2, "speculative_drafted_tokens", "speculative_accepted_tokens")
TIME_COUNTERS = {"decode_ms", "ttft_ms", "total_ms", "stage_ms", "network_ms"}
PHASES = {"starting", "downloading", "loading", "up", "moving", "leaving", "stopped", "error"}
_lock = threading.Lock()
_peers = {}
_reports = {}
STARTED_AT = time.time()


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _layers(value):
    return value is None or (isinstance(value, list) and len(value) == 2
                             and all(type(v) is int for v in value) and 0 <= value[0] < value[1] <= 65536)


def _expire(now):
    for key in [k for k, value in _peers.items() if now - value["last_seen"] > HISTORY_SECONDS]:
        del _peers[key]
    for key in [k for k, value in _reports.items() if now - value["at"] > WINDOW_SECONDS + BUCKET_SECONDS]:
        del _reports[key]


def _remember(fields, now, source=None):
    key = (fields["network"], fields["peer"])
    old = _peers.get(key)
    stamp = min(fields["ts"], now)
    if old and stamp < old["last_seen"]:
        return
    if old is None and len(_peers) >= MAX_PEERS:
        return
    # A strict allowlist: no URLs, bearer tokens, public keys, IPs, signatures, or text.
    _peers[key] = {"peer": fields["peer"], "network": fields["network"],
                   "backend": fields["backend"], "gib": fields["gib"], "phase": fields["phase"],
                   "layers": fields["layers"], "assigned_layers": fields.get("assigned_layers"),
                   "last_seen": stamp, "source": source or (old or {}).get("source")}


def remember_directory(record, now):
    """Older nodes still appear, using already-verified directory announcements."""
    if not _layers(record.get("window")) or not _number(record.get("gib")) or not 0 <= record["gib"] <= 1e6:
        return
    try:
        record["backend"].encode("utf-8")
        record["network"].encode("utf-8")
    except (ValueError, AttributeError):
        return
    fields = {**record, "phase": "up", "layers": record["window"], "assigned_layers": record["window"]}
    with _lock:
        _expire(now)
        _remember(fields, now)


def validate(body, now):
    required = {"v", "network", "peer", "pub", "sig", "session", "seq", "ts", "phase", "backend", "gib",
                "layers", "assigned_layers", "buckets"}
    try:
        if not isinstance(body, dict) or set(body) != required or type(body["v"]) is not int or body["v"] not in (1, 2, 3):
            return "malformed report"
        if not isinstance(body["network"], str) or not 1 <= len(body["network"]) <= 128:
            return "bad network"
        if not isinstance(body["backend"], str) or not 1 <= len(body["backend"]) <= 32 or body["phase"] not in PHASES:
            return "bad node state"
        body["backend"].encode("utf-8")
        body["network"].encode("utf-8")
        if not _number(body["ts"]) or not now - LIVE_SECONDS <= body["ts"] <= now + 15:
            return "stale or future report"
        if type(body["seq"]) is not int or not 0 < body["seq"] < 2**63:
            return "bad sequence"
        if not isinstance(body["session"], str) or not re.fullmatch(r"[0-9a-f]{32}", body["session"]):
            return "bad session"
        if not _number(body["gib"]) or not 0 <= body["gib"] <= 1e6:
            return "bad memory"
        if not _layers(body["layers"]) or not _layers(body["assigned_layers"]):
            return "bad layers"
        if not isinstance(body["buckets"], list) or len(body["buckets"]) > WINDOW_SECONDS // BUCKET_SECONDS + 1:
            return "too many metric buckets"
        stamps = set()
        counters = COUNTERS if body["v"] == 3 else COUNTERS_V2 if body["v"] == 2 else COUNTERS_V1
        for bucket in body["buckets"]:
            if not isinstance(bucket, dict) or set(bucket) != {"ts", *counters}:
                return "bad metrics"
            stamp = bucket["ts"]
            if (type(stamp) is not int or stamp % BUCKET_SECONDS or stamp in stamps
                    or not now - WINDOW_SECONDS - BUCKET_SECONDS <= stamp <= now + 15):
                return "bad bucket timestamp"
            stamps.add(stamp)
            for name in counters:
                value = bucket[name]
                if not _number(value) or not 0 <= value <= 1e12 or (name not in TIME_COUNTERS and type(value) is not int):
                    return "bad metric counter"
            if (bucket["requests"] != bucket["completed"] + bucket["failed"] + bucket["cancelled"]
                    or bucket["ttft_count"] > bucket["completed"]
                    or bucket["decode_tokens"] != bucket["output_tokens"] - bucket["ttft_count"]):
                return "inconsistent metric counters"
            if body["v"] == 3 and bucket["speculative_accepted_tokens"] > bucket["speculative_drafted_tokens"]:
                return "inconsistent speculative counters"
        pub = base64.b64decode(body["pub"], validate=True)
        sig = base64.b64decode(body["sig"], validate=True)
        if len(pub) != 32 or hashlib.sha256(pub).hexdigest()[:32] != body["peer"]:
            return "peer does not match key"
        canonical = json.dumps({k: v for k, v in body.items() if k != "sig"}, sort_keys=True,
                               separators=(",", ":")).encode()
        Ed25519PublicKey.from_public_bytes(pub).verify(sig, BIND + canonical)
    except (InvalidSignature, ValueError, TypeError, KeyError, AttributeError):
        return "invalid signed report"
    return None


async def report(request):
    from . import directory
    now = time.time()
    source = directory.caller_ip(request)
    if source is None:
        return JSONResponse({"error": "cannot determine caller IP"}, status_code=400)
    if directory._limited("dashboard", source, 120, now):
        return JSONResponse({"error": "slow down"}, status_code=429)
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > MAX_BODY:
            return JSONResponse({"error": "report too large"}, status_code=413)
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return JSONResponse({"error": "not JSON"}, status_code=400)
    why = validate(body, now)
    if why:
        return JSONResponse({"error": why}, status_code=400)
    key = (body["network"], body["peer"], body["session"])
    with _lock:
        _expire(now)
        old = _reports.get(key)
        if old and old["seq"] >= body["seq"]:
            return {"ok": True, "stale": True}
        node_key = key[:2]
        if node_key not in _peers:
            if len(_peers) >= MAX_PEERS or sum(v.get("source") == source for v in _peers.values()) >= MAX_PER_SOURCE:
                return JSONResponse({"error": "dashboard full"}, status_code=429)
        if key not in _reports and len(_reports) >= MAX_SESSIONS:
            return JSONResponse({"error": "metrics full"}, status_code=429)
        buckets = dict(old["buckets"]) if old else {}
        for bucket in body["buckets"]:
            normalized = {**dict.fromkeys(COUNTERS, 0), **bucket}
            buckets[bucket["ts"]] = normalized
        buckets = {stamp: bucket for stamp, bucket in buckets.items() if stamp >= now - WINDOW_SECONDS}
        _reports[key] = {"seq": body["seq"], "at": now, "buckets": buckets}
        _remember(body, now, source)
    return {"ok": True}


def _summary(buckets):
    counts = {name: sum(b.get(name, 0) for b in buckets) for name in COUNTERS}
    decode, ttft = counts["decode_tokens"], counts["ttft_count"]
    return {"sample_count": counts["completed"], "failed_requests": counts["failed"],
            "cancelled_requests": counts["cancelled"], "output_tokens": counts["output_tokens"],
            "avg_tokens_per_second": round(decode * 1000 / counts["decode_ms"], 3) if decode and counts["decode_ms"] else None,
            "avg_ms_per_token": round(counts["decode_ms"] / decode, 3) if decode and counts["decode_ms"] else None,
            "avg_time_to_first_token_ms": round(counts["ttft_ms"] / ttft, 3) if ttft else None,
            "avg_stage_ms": round(counts["stage_ms"] / counts["stage_count"], 3) if counts["stage_count"] else None,
            "avg_network_ms": round(counts["network_ms"] / counts["network_count"], 3) if counts["network_count"] else None,
            "speculative_drafted_tokens": counts["speculative_drafted_tokens"],
            "speculative_accepted_tokens": counts["speculative_accepted_tokens"],
            "speculative_acceptance_rate": (
                round(counts["speculative_accepted_tokens"] / counts["speculative_drafted_tokens"], 4)
                if counts["speculative_drafted_tokens"] else None)}


def snapshot(networks_path, network=None):
    now = time.time()
    catalog = {}
    for path in networks_path.glob("*.json"):
        try:
            item = json.loads(path.read_text())
            if (isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
                    and isinstance(item.get("model"), dict) and isinstance(item.get("boundaries"), list)
                    and len(item["boundaries"]) >= 2 and all(type(b) is int for b in item["boundaries"])
                    and item["boundaries"][0] == 0
                    and all(a < b for a, b in zip(item["boundaries"], item["boundaries"][1:]))):
                catalog[item["id"]] = item
        except (OSError, ValueError):
            pass
    with _lock:
        _expire(now)
        peers = [dict(value) for value in _peers.values() if not network or value["network"] == network]
        buckets = {key: [b for stamp, b in report["buckets"].items() if stamp >= now - WINDOW_SECONDS]
                   for key, report in _reports.items() if not network or key[0] == network}
        reporters = {key[:2] for key, report in _reports.items() if now - report["at"] <= LIVE_SECONDS}
    by_node, by_model, model_nodes = defaultdict(list), defaultdict(list), defaultdict(list)
    for key, values in buckets.items():
        by_node[key[:2]].extend(values)
        by_model[key[0]].extend(values)
    for peer in peers:
        model_nodes[peer["network"]].append(peer)
    ids = set(catalog) | {p["network"] for p in peers} | {key[0] for key in buckets}
    if network:
        ids &= {network}
    models = []
    for ident in sorted(ids):
        definition = catalog.get(ident, {})
        model = definition.get("model", {})
        bounds = definition.get("boundaries", [])
        stages = list(zip(bounds, bounds[1:]))
        nodes = []
        for peer in sorted(model_nodes[ident], key=lambda p: p["peer"]):
            online = now - peer["last_seen"] <= LIVE_SECONDS and peer["phase"] not in ("leaving", "stopped", "error")
            metrics = by_node[(ident, peer["peer"])]
            nodes.append({key: peer[key] for key in ("peer", "backend", "gib", "phase", "layers", "assigned_layers")}
                         | {"online": online, "last_seen": peer["last_seen"], "metrics": _summary(metrics),
                            "telemetry_reporting": online and (ident, peer["peer"]) in reporters})
        windows = [n["layers"] for n in nodes if n["online"] and n["phase"] == "up" and n["layers"]
                   and n["layers"][0] in bounds and n["layers"][1] in bounds]
        reachable = {0}
        for a, b in sorted(windows):
            if a in reachable:
                reachable.add(b)
        ready = bool(bounds) and bounds[-1] in reachable
        for n in nodes:
            n["state"] = ("offline" if not n["online"] else "linking" if n["phase"] != "up"
                          else "unassigned" if not n["layers"] else "full" if ready else "partial")
            n["display_state"] = {"offline": "off", "linking": "lnk", "unassigned": "idl", "full": "ful", "partial": "grp"}[n["state"]]
        metrics = by_model[ident]
        model_summary = _summary(metrics)
        model_summary.pop("avg_stage_ms", None)  # stage averages belong to individual nodes
        model_counts = {name: sum(b.get(name, 0) for b in metrics) for name in COUNTERS}
        # network_ms is summed over participating stages, while completed is
        # counted once at each node that accepted the user's chat request.
        model_summary["avg_network_ms"] = (
            round(model_counts["network_ms"] / model_counts["completed"], 3)
            if model_counts["network_count"] and model_counts["completed"] else None)
        models.append({"network": ident, "model": model.get("name", ident), "num_layers": model.get("num_layers"),
                       "route_ready": ready, "online_nodes": sum(n["online"] for n in nodes),
                       "serving_nodes": sum(n["online"] and n["phase"] == "up" and bool(n["layers"]) for n in nodes),
                       "reporting_nodes": sum(n["telemetry_reporting"] for n in nodes),
                       "stages": [{"layers": [a, b], "peers": sum(x <= a and b <= y for x, y in windows)} for a, b in stages],
                       "metrics": model_summary, "nodes": nodes})
    return {"v": 1, "updated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "window_seconds": WINDOW_SECONDS, "bucket_seconds": BUCKET_SECONDS, "history_seconds": HISTORY_SECONDS,
            "history_since": STARTED_AT, "poll_after_seconds": 5, "metrics_source": "self_reported_local_chat_api",
            "online_nodes": sum(m["online_nodes"] for m in models),
            "serving_nodes": sum(m["serving_nodes"] for m in models),
            "reporting_nodes": sum(m["reporting_nodes"] for m in models),
            "metrics": _summary([b for values in buckets.values() for b in values]), "models": models}
