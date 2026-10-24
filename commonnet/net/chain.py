"""Bounded peer forwarding with token-only replies.

The authenticated caller supplies the route, including each hop's bearer credential
and (for direct peers) certificate pin. A peer sends those credentials only to their
specified HTTPS origin. Forwarding to private addresses is disabled unless explicitly
enabled for a local test network. No redirects or environment proxies are followed.
"""
from __future__ import annotations

import hashlib
import json
import re
from urllib.parse import urlsplit

from commonnet.net import auth, protocol
from commonnet.net.client import BlockClient, RemoteError, Unreachable

CAPABILITY = "chain_v1"
MAX_HOPS = 64
MAX_CHAIN_TOKENS = 4096
MAX_TOKEN_ID = 2 ** 32 - 1
MAX_FORWARD_CLIENTS = 128


class ChainUnavailable(RuntimeError):
    """A downstream hop is unavailable or has lost its session."""

    def __init__(self, layer: int, *, lost: bool = False):
        self.layer, self.lost = layer, lost
        super().__init__(f"chain hop {layer} {'lost its session' if lost else 'is unavailable'}")


def ids(value, *, empty: bool = False) -> list:
    if empty and value == []:
        return []
    result = protocol.check_ids(value)
    if len(result) > MAX_CHAIN_TOKENS or any(t > MAX_TOKEN_ID for t in result):
        raise protocol.ProtocolError("chain token IDs exceed the uint32 or sequence limit")
    return list(result)


def check_hop(value, *, allow_private: bool = False) -> dict:
    protocol._fields(value, "chain hop", ("url", "layers"), ("token", "pin"))
    url = value["url"]
    if not isinstance(url, str) or len(url) > 512:
        raise protocol.ProtocolError("chain hop needs an HTTPS origin")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise protocol.ProtocolError("invalid chain origin") from None
    if (parts.scheme not in (("http", "https") if allow_private else ("https",))
            or not parts.hostname or parts.username is not None or parts.password is not None
            or parts.path not in ("", "/") or parts.query or parts.fragment
            or (port is not None and not 1 <= port <= 65535)):
        raise protocol.ProtocolError("chain hop must be an HTTPS origin without credentials or a path")
    # Numeric private/loopback/metadata destinations are rejected before opening a
    # connection. Hostnames are also checked after resolution by BlockClient.
    import ipaddress
    try:
        address = ipaddress.ip_address(parts.hostname)
    except ValueError:
        address = None
    if address is not None and not allow_private and not address.is_global:
        raise protocol.ProtocolError("chain destination must be public")
    token = value.get("token")
    if token is not None:
        try:
            auth.check_token(token)
        except auth.TokenError:
            raise protocol.ProtocolError("invalid chain credential") from None
    pin = value.get("pin")
    if pin is not None:
        from commonnet.net.secure import normalize_fingerprint
        try:
            pin = normalize_fingerprint(pin)
        except ValueError:
            raise protocol.ProtocolError("invalid chain certificate pin") from None
    if address is not None and parts.scheme == "https" and pin is None and not allow_private:
        raise protocol.ProtocolError("direct chain peers require a certificate pin")
    return {"url": url.rstrip("/"), "layers": list(protocol.check_layers(value["layers"])),
            "token": token, "pin": pin}


def decode_request(obj, *, allow_private: bool = False) -> dict:
    protocol._fields(obj, "chain request", ("id", "start_pos", "layers", "route", "which"),
                     ("ids", "hidden", "sampling", "drafts", "truncate_to", "activation_codec"))
    if ("ids" in obj) == ("hidden" in obj):
        raise protocol.ProtocolError("chain request needs exactly one of IDs or hidden states")
    layers = protocol.check_layers(obj["layers"])
    route = obj["route"]
    if not isinstance(route, list) or len(route) >= MAX_HOPS:
        raise protocol.ProtocolError("chain route exceeds the hop limit")
    hops = [check_hop(h, allow_private=allow_private) for h in route]
    previous = layers[1]
    seen = set()
    for hop in hops:
        if hop["layers"][0] != previous or hop["url"] in seen:
            raise protocol.ProtocolError("chain route must advance through contiguous, distinct peers")
        previous = hop["layers"][1]
        seen.add(hop["url"])
    codec = obj.get("activation_codec", "float")
    if codec not in ("float", "int8_group_v1"):
        raise protocol.ProtocolError("unsupported chain activation codec")
    token_ids = ids(obj["ids"]) if "ids" in obj else None
    hidden = None
    if "hidden" in obj:
        if codec == "int8_group_v1":
            from commonnet.net.activation_codec import decode_int8_group
            hidden = decode_int8_group(obj["hidden"])
        else:
            hidden = protocol.decode_tensor(obj["hidden"])
    if token_ids is not None and layers[0] != 0:
        raise protocol.ProtocolError("token IDs enter only the first layer window")
    if hidden is not None and (hidden.shape[0] != 1 or hidden.shape[1] > MAX_CHAIN_TOKENS):
        raise protocol.ProtocolError("chain input needs batch one and a bounded sequence")
    if obj["which"] not in ("last", "all"):
        raise protocol.ProtocolError("chain which must be last or all")
    drafts = ids(obj["drafts"], empty=True) if "drafts" in obj else None
    seq = len(token_ids) if token_ids is not None else hidden.shape[1]
    if drafts is not None:
        if len(drafts) + 1 != seq or obj["which"] != "all" or obj.get("sampling") is not None:
            raise protocol.ProtocolError("verification needs greedy output and one more input than drafts")
        if token_ids is not None and drafts != token_ids[1:]:
            raise protocol.ProtocolError("drafts must match the input after its first token")
    cut = protocol.check_start_pos(obj["truncate_to"]) if "truncate_to" in obj else None
    pos = protocol.check_start_pos(obj["start_pos"])
    if cut is not None and cut != pos:
        raise protocol.ProtocolError("chain rollback must cut to this step's start position")
    return {"id": protocol.check_session_id(obj["id"]), "start_pos": pos,
            "layers": layers, "route": hops, "which": obj["which"], "ids": token_ids,
            "hidden": hidden, "sampling": protocol.check_sampling(obj.get("sampling")),
            "drafts": drafts, "truncate_to": cut, "activation_codec": codec}


def check_reply(obj) -> dict:
    protocol._fields(obj, "chain reply", ("ids",), ("accepted", "next"))
    result = {"ids": ids(obj["ids"])}
    if ("accepted" in obj) != ("next" in obj):
        raise protocol.ProtocolError("verification reply needs accepted and next together")
    if "accepted" in obj:
        accepted, nxt = obj["accepted"], obj["next"]
        if type(accepted) is not int or accepted < 0 or accepted + 1 != len(result["ids"]):
            raise protocol.ProtocolError("invalid accepted prefix length")
        ids([nxt])
        if nxt != result["ids"][-1]:
            raise protocol.ProtocolError("verification next token differs from the returned IDs")
        result.update(accepted=accepted, next=nxt)
    return result


def request_key(obj) -> bytes:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).digest()


def downstream_failure(error, hop, route) -> ChainUnavailable:
    # Keep the actual failed layer when the failure originated further downstream.
    # Only a bounded layer number is propagated, never the remote error/credential.
    text = str(error)
    found = re.search(r"chain hop (\d+)", text)
    layer = int(found.group(1)) if found else hop["layers"][0]
    allowed = {h["layers"][0] for h in route}
    if layer not in allowed:
        layer = hop["layers"][0]
    lost = (isinstance(error, RemoteError) and error.status in (404, 409)) or "lost its session" in text
    return ChainUnavailable(layer, lost=lost)


def run(server, obj) -> dict:
    """Compute locally once, then send the hidden tensor directly to the next peer."""
    req = decode_request(obj, allow_private=server.allow_chain_private)
    if req["layers"] != server.layers:
        raise protocol.ProtocolError("chain hop layer window differs from this server")
    if req["ids"] is not None and "ids_in" not in server.caps:
        raise NotImplementedError("first chain stage does not hold an embedding")
    if not req["route"] and "head_out" not in server.caps:
        raise NotImplementedError("final chain stage does not hold the output head")
    with server._lock:
        session = server._sessions.get(req["id"])
    if session is None:
        if req["start_pos"] != 0:
            from commonnet.net.server import UnknownSession
            raise UnknownSession("chain session is not open")
        server.open(req["id"])
        with server._lock:
            session = server._sessions[req["id"]]
    key = request_key(obj)
    with session.chain_lock:
        if session.chain_last is not None and session.chain_last[0] == key:
            return session.chain_last[1]
        final = not req["route"]
        out = server.forward(req["id"], req["hidden"], req["start_pos"], req["truncate_to"],
                             ids=req["ids"], output="ids" if final else None,
                             which=req["which"], sampling=req["sampling"])
        if final:
            if req["drafts"] is None:
                reply = {"ids": out}
            else:
                drafts = req["drafts"]
                accepted = 0
                while accepted < len(drafts) and drafts[accepted] == out[accepted]:
                    accepted += 1
                nxt = int(out[accepted])
                reply = {"ids": drafts[:accepted] + [nxt], "accepted": accepted, "next": nxt}
        else:
            hop = req["route"][0]
            if req["activation_codec"] == "int8_group_v1":
                from commonnet.net.activation_codec import encode_int8_group
                payload = encode_int8_group(out)
            else:
                payload = protocol.encode_tensor(out)
            onward = {"id": req["id"], "start_pos": req["start_pos"], "layers": hop["layers"],
                      "route": req["route"][1:], "which": req["which"], "hidden": payload}
            if req["activation_codec"] != "float":
                onward["activation_codec"] = req["activation_codec"]
            for field in ("sampling", "drafts", "truncate_to"):
                if req[field] is not None:
                    onward[field] = req[field]
            client_key = (hop["url"], hop["token"], hop["pin"])
            with server._lock:
                client = server._chain_clients.get(client_key)
                if client is None:
                    if len(server._chain_clients) >= MAX_FORWARD_CLIENTS:
                        oldest = server._chain_clients.pop(next(iter(server._chain_clients)))
                        oldest.close_connections()
                    client = server._chain_clients[client_key] = BlockClient(
                        token=hop["token"], pin=hop["pin"], retries=0,
                        public_only=not server.allow_chain_private,
                        max_request_bytes=server.max_request_bytes)
            try:
                reply = client.chain(hop["url"], onward)
            except (RemoteError, Unreachable) as exc:
                raise downstream_failure(exc, hop, req["route"]) from None
        reply = check_reply(reply)
        session.chain_last = (key, reply)
        return reply
