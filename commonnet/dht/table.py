"""Kademlia routing table: 256 k-buckets ordered least recently seen first, plus replacement caches."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .ids import ID_BITS, bucket_index


@dataclass
class Contact:
    peer: str
    id: int
    addr: tuple  # (host, port)
    last_seen: float = 0.0
    fails: int = 0


class RoutingTable:
    def __init__(self, own_id: int, k: int = 20):
        self.own = own_id
        self.k = k
        self._buckets: list[list[Contact]] = [[] for _ in range(ID_BITS)]
        self._cache: list[list[Contact]] = [[] for _ in range(ID_BITS)]
        self._touched = [0.0] * ID_BITS
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return sum(len(b) for b in self._buckets)

    def contacts(self) -> list[Contact]:
        with self._lock:
            return [c for b in self._buckets for c in b]

    def get(self, peer_id_int: int) -> Contact | None:
        if peer_id_int == self.own:
            return None
        with self._lock:
            for c in self._buckets[bucket_index(self.own, peer_id_int)]:
                if c.id == peer_id_int:
                    return c
        return None

    def by_addr(self, addr) -> Contact | None:
        with self._lock:
            for b in self._buckets:
                for c in b:
                    if c.addr == addr:
                        return c
        return None

    def add(self, contact: Contact, now: float) -> Contact | None:
        """Insert or refresh a verified contact.

        Returns None normally. If the bucket is full, the contact goes to the replacement cache and the
        least recently seen bucket member is returned so the caller can ping it (evict if it is dead).
        """
        if contact.id == self.own:
            return None
        idx = bucket_index(self.own, contact.id)
        with self._lock:
            bucket = self._buckets[idx]
            for i, c in enumerate(bucket):
                if c.id == contact.id:
                    c.addr, c.last_seen, c.fails = contact.addr, now, 0
                    bucket.append(bucket.pop(i))
                    self._touched[idx] = now
                    return None
            contact.last_seen = now
            if len(bucket) < self.k:
                bucket.append(contact)
                self._touched[idx] = now
                return None
            cache = self._cache[idx]
            cache[:] = [c for c in cache if c.id != contact.id]
            cache.append(contact)
            del cache[:-self.k]
            return bucket[0]

    def remove(self, contact_id: int) -> None:
        """Drop a dead contact and promote the newest replacement for its bucket, if any."""
        if contact_id == self.own:
            return
        idx = bucket_index(self.own, contact_id)
        with self._lock:
            bucket = self._buckets[idx]
            for i, c in enumerate(bucket):
                if c.id == contact_id:
                    del bucket[i]
                    if self._cache[idx]:
                        bucket.append(self._cache[idx].pop())
                    return
            self._cache[idx] = [c for c in self._cache[idx] if c.id != contact_id]

    def fail(self, contact_id: int, limit: int = 2) -> bool:
        """Note an unanswered request. Removes the contact once it has failed `limit` times in a row."""
        c = self.get(contact_id)
        if c is None:
            return False
        c.fails += 1
        if c.fails >= limit:
            self.remove(contact_id)
            return True
        return False

    def closest(self, target: int, n: int | None = None) -> list[Contact]:
        n = self.k if n is None else n
        with self._lock:
            return sorted((c for b in self._buckets for c in b), key=lambda c: c.id ^ target)[:n]

    def stale_buckets(self, now: float, older_than: float) -> list[int]:
        with self._lock:
            return [i for i in range(ID_BITS) if self._buckets[i] and now - self._touched[i] > older_than]

    def touch_bucket(self, idx: int, now: float) -> None:
        self._touched[idx] = now

    def check_invariants(self) -> None:
        """Raises AssertionError if the table is inconsistent. For tests."""
        with self._lock:
            seen = set()
            for idx, bucket in enumerate(self._buckets):
                assert len(bucket) <= self.k, "bucket over capacity"
                for c in bucket:
                    assert c.id != self.own, "own id in table"
                    assert bucket_index(self.own, c.id) == idx, "contact in the wrong bucket"
                    assert c.id not in seen, "duplicate contact"
                    seen.add(c.id)
                assert len(self._cache[idx]) <= self.k, "cache over capacity"
                for c in self._cache[idx]:
                    assert bucket_index(self.own, c.id) == idx
                    assert all(c.id != b.id for b in bucket), "cached contact also in bucket"
