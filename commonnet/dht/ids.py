"""Key and node id derivation. All ids are 256-bit integers (sha256), compared by XOR distance."""
from __future__ import annotations

import hashlib

ID_BITS = 256


def _h(domain: bytes, text: str) -> bytes:
    return hashlib.sha256(b"commonnet-dht/" + domain + b"/" + text.encode("utf-8")).digest()


def node_id(peer: str) -> int:
    """The DHT position of a peer, derived from its (public-key-hash) peer id."""
    return int.from_bytes(_h(b"node", peer), "big")


def network_key(network_id: str) -> bytes:
    """Where every peer of one network stores its record."""
    return _h(b"net", network_id)


def peer_key(peer: str) -> bytes:
    """Where one peer's own record is stored, so it can be found by id."""
    return _h(b"peer", peer)


def key_int(key: bytes) -> int:
    return int.from_bytes(key, "big")


def distance(a: int, b: int) -> int:
    return a ^ b


def bucket_index(own: int, other: int) -> int:
    """0..255: position of the highest differing bit. Undefined (raises) for own == other."""
    d = own ^ other
    if d == 0:
        raise ValueError("a node has no bucket for itself")
    return d.bit_length() - 1
