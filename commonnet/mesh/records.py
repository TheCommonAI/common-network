"""Peer identity and signed peer records.

A record is a small JSON object a peer signs about itself (ed25519). Anyone can check it, nobody
can forge someone else's, and a record with a newer `seq` replaces an older one.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

RECORD_VERSION = 1
STATES = ("loading", "up", "leaving")
LIVE_SECONDS = 60.0
FUTURE_SKEW_SECONDS = 120.0
MAX_TOKEN_CHARS = 256
MAX_URL_CHARS = 512
SIGNED_FIELDS = ("v", "network", "peer", "pub", "url", "token", "window", "backend", "state", "gib", "seq", "ts")


class RecordError(ValueError):
    pass


def _b64e(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"), validate=True)


def peer_id(pub: bytes) -> str:
    """32 hex characters: the first half of sha256(public key)."""
    return hashlib.sha256(pub).hexdigest()[:32]


def canonical(record: dict) -> bytes:
    """The bytes that are signed: every field except `sig`, sorted, no spaces."""
    body = {k: v for k, v in record.items() if k != "sig"}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


class Identity:
    """A peer's keypair. The private key lives in a file readable only by its owner."""

    def __init__(self, key: Ed25519PrivateKey):
        self._key = key
        self.pub = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.pub_b64 = _b64e(self.pub)
        self.peer = peer_id(self.pub)
        self._last_seq = 0
        self._lock = threading.Lock()

    @classmethod
    def generate(cls) -> "Identity":
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def load_or_create(cls, path) -> "Identity":
        path = Path(path)
        if path.exists():
            try:
                raw = _b64d(path.read_text().strip())
                return cls(Ed25519PrivateKey.from_private_bytes(raw))
            except (OSError, ValueError):
                raise RecordError(f"cannot read the identity key at {path}; move it aside to make a new identity") from None
        path.parent.mkdir(parents=True, exist_ok=True)
        identity = cls.generate()
        raw = identity._key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                          serialization.NoEncryption())
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:  # another process made it first
            return cls.load_or_create(path)
        with os.fdopen(fd, "w") as f:
            f.write(_b64e(raw))
        return identity

    def sign_record(self, fields: dict, *, now: float | None = None) -> dict:
        """A signed record from the caller's fields.

        Needs network, url, token, window, backend, state and gib. `seq` and `ts` default to the
        current time; seq is kept strictly increasing for this identity.
        """
        now = time.time() if now is None else now
        with self._lock:
            seq = fields.get("seq")
            if seq is None:
                seq = max(int(now * 1000), self._last_seq + 1)
            self._last_seq = max(self._last_seq, int(seq))
        window = fields.get("window")
        record = {"v": RECORD_VERSION, "network": fields["network"], "peer": self.peer, "pub": self.pub_b64,
                  "url": fields.get("url"), "token": fields.get("token", ""),
                  "window": None if window is None else [int(window[0]), int(window[1])],
                  "backend": fields.get("backend", ""), "state": fields.get("state", "up"),
                  "gib": float(fields.get("gib", 0.0)), "seq": int(seq), "ts": float(fields.get("ts", now))}
        check_shape(record)
        record["sig"] = _b64e(self._key.sign(canonical(record)))
        return record


def check_shape(record: dict) -> None:
    """Raises RecordError unless every field has the right type and size. Does not check the signature."""
    if not isinstance(record, dict):
        raise RecordError("record is not an object")
    for key in SIGNED_FIELDS:
        if key not in record:
            raise RecordError(f"record has no {key}")
    if record["v"] != RECORD_VERSION:
        raise RecordError("unsupported record version")
    for key in ("network", "peer", "pub", "backend", "state", "token"):
        if not isinstance(record[key], str):
            raise RecordError(f"{key} must be a string")
    if record["state"] not in STATES:
        raise RecordError("unknown state")
    url = record["url"]
    if url is not None and (not isinstance(url, str) or len(url) > MAX_URL_CHARS
                            or not url.startswith(("http://", "https://"))):
        raise RecordError("url must be an http(s) URL or null")
    if len(record["token"]) > MAX_TOKEN_CHARS:
        raise RecordError("token is too long")
    window = record["window"]
    if window is not None and not (isinstance(window, list) and len(window) == 2
                                   and all(type(x) is int for x in window) and 0 <= window[0] < window[1]):
        raise RecordError("window must be null or [start, end]")
    if type(record["seq"]) is not int or record["seq"] < 0:
        raise RecordError("seq must be a non-negative integer")
    if isinstance(record["gib"], bool) or not isinstance(record["gib"], (int, float)):
        raise RecordError("gib must be a number")
    if isinstance(record["ts"], bool) or not isinstance(record["ts"], (int, float)):
        raise RecordError("ts must be a number")


def verify_record(record: dict) -> bool:
    """True if the record is well formed, `peer` is the hash of `pub`, and the signature holds."""
    try:
        check_shape(record)
        pub = _b64d(record["pub"])
        if len(pub) != 32 or peer_id(pub) != record["peer"]:
            return False
        sig = record.get("sig")
        if not isinstance(sig, str):
            return False
        Ed25519PublicKey.from_public_bytes(pub).verify(_b64d(sig), canonical(record))
        return True
    except (RecordError, InvalidSignature, ValueError, TypeError):
        return False


def is_live(record: dict, now: float | None = None, network: str | None = None) -> bool:
    """Up, recently re-signed and not dated in the future. Assumes the record was already verified."""
    now = time.time() if now is None else now
    if record.get("state") != "up":
        return False
    if network is not None and record.get("network") != network:
        return False
    ts = record.get("ts")
    if not isinstance(ts, (int, float)):
        return False
    return now - ts <= LIVE_SECONDS and ts <= now + FUTURE_SKEW_SECONDS
