"""Record storage for one DHT node: validated, capped, expiring."""
from __future__ import annotations

import json
import threading
from collections import Counter

from ..mesh.records import FUTURE_SKEW_SECONDS, LIVE_SECONDS, verify_record
from .ids import network_key, peer_key

MAX_RECORD_BYTES = 800  # encoded JSON; keeps one record plus message overhead inside one 1200-byte packet


class StoreError(ValueError):
    pass


def record_size(record: dict) -> int:
    return len(json.dumps(record, separators=(",", ":")).encode("utf-8"))


def check_record(record, key: bytes, now: float) -> dict:
    """Raises StoreError unless the record is acceptable under `key` right now. Returns the record."""
    if not isinstance(record, dict):
        raise StoreError("record is not an object")
    try:
        size = record_size(record)
    except (TypeError, ValueError):
        raise StoreError("record is not JSON") from None
    if size > MAX_RECORD_BYTES:
        raise StoreError("record too large")
    if not verify_record(record):
        raise StoreError("bad signature")
    if key not in (network_key(record["network"]), peer_key(record["peer"])):
        raise StoreError("record does not belong under this key")
    ts = record["ts"]
    if now - ts > LIVE_SECONDS:
        raise StoreError("stale record")
    if ts > now + FUTURE_SKEW_SECONDS:
        raise StoreError("record dated in the future")
    return record


class ValueStore:
    def __init__(self, clock, max_per_key: int = 256, max_per_source: int = 128):
        self._clock = clock
        self.max_per_key = max_per_key
        self.max_per_source = max_per_source
        self._data: dict[bytes, dict[str, tuple[dict, str]]] = {}  # key -> peer -> (record, source ip)
        self._per_source: Counter = Counter()
        self._lock = threading.Lock()

    def put(self, key: bytes, record, source: str) -> bool:
        """Store a record. Returns True if it is (now) held, raises StoreError if refused."""
        now = self._clock()
        check_record(record, key, now)
        with self._lock:
            self._prune_locked(now)
            bucket = self._data.setdefault(key, {})
            old = bucket.get(record["peer"])
            if old is not None:
                if old[0]["seq"] > record["seq"]:
                    raise StoreError("older than the stored record")
                if old[0]["seq"] == record["seq"]:
                    return True
                self._per_source[old[1]] -= 1
            else:
                if len(bucket) >= self.max_per_key:
                    raise StoreError("key is full")
            if self._per_source[source] >= self.max_per_source:
                if old is not None:
                    self._per_source[old[1]] += 1
                raise StoreError("too many values from this source")
            bucket[record["peer"]] = (record, source)
            self._per_source[source] += 1
            return True

    def get(self, key: bytes, after: str = "") -> list[dict]:
        """Live records under a key sorted by peer id, only those with peer id > `after`."""
        now = self._clock()
        with self._lock:
            self._prune_locked(now)
            bucket = self._data.get(key, {})
            return [bucket[p][0] for p in sorted(bucket) if p > after]

    def prune(self) -> None:
        with self._lock:
            self._prune_locked(self._clock())

    def _prune_locked(self, now: float) -> None:
        for key in list(self._data):
            bucket = self._data[key]
            for peer in [p for p, (r, _) in bucket.items() if now - r["ts"] > LIVE_SECONDS]:
                self._per_source[bucket[peer][1]] -= 1
                del bucket[peer]
            if not bucket:
                del self._data[key]
        for src in [s for s, n in self._per_source.items() if n <= 0]:
            del self._per_source[src]

    def count(self) -> int:
        with self._lock:
            return sum(len(b) for b in self._data.values())
