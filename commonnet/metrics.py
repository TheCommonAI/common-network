"""Text-free chat timing and best-effort signed dashboard heartbeats."""
from __future__ import annotations

import json
import secrets
import threading
import time
import urllib.request

WINDOW_SECONDS = 300
BUCKET_SECONDS = 10
BIND = b"commonnet-dashboard-v1\n"
DEVICE_BIND = b"commonnet-device-v1\n"  # a device key vouching that a per-network peer key is on the same machine
COUNTERS = ("requests", "completed", "failed", "cancelled", "output_tokens", "decode_tokens", "decode_ms",
            "ttft_count", "ttft_ms", "total_ms", "stage_count", "stage_ms", "network_count", "network_ms",
            "speculative_drafted_tokens", "speculative_accepted_tokens", "served_tokens", "served_compute_ms")


class Metrics:
    def __init__(self, clock=time.time):
        self.clock = clock
        self._buckets = {}
        self._lock = threading.Lock()

    def _expire(self, now):
        for stamp in list(self._buckets):
            if stamp < now - WINDOW_SECONDS:
                del self._buckets[stamp]

    def record(self, *, outcome, count, first, last, started, ended,
               speculative_drafted_tokens=0, speculative_accepted_tokens=0):
        now = self.clock()
        with self._lock:
            self._expire(now)
            stamp = int(now // BUCKET_SECONDS) * BUCKET_SECONDS
            bucket = self._buckets.setdefault(stamp, {"ts": stamp, **dict.fromkeys(COUNTERS, 0)})
            bucket["requests"] += 1
            bucket[outcome] += 1
            if outcome == "completed":
                bucket["output_tokens"] += count
                bucket["total_ms"] += max(0, ended - started) * 1000
                if first is not None:
                    bucket["ttft_count"] += 1
                    bucket["ttft_ms"] += max(0, first - started) * 1000
                if count > 1:
                    bucket["decode_tokens"] += count - 1
                    bucket["decode_ms"] += max(0, last - first) * 1000
            if (type(speculative_drafted_tokens) is int and type(speculative_accepted_tokens) is int
                    and 0 <= speculative_accepted_tokens <= speculative_drafted_tokens):
                bucket["speculative_drafted_tokens"] += speculative_drafted_tokens
                bucket["speculative_accepted_tokens"] += speculative_accepted_tokens

    def record_stage(self, stage_ms, network_ms=None):
        """Record this node's compute and transit for one distributed model session."""
        if not isinstance(stage_ms, (int, float)) or stage_ms <= 0:
            return
        if network_ms is not None and (not isinstance(network_ms, (int, float)) or network_ms < 0):
            network_ms = None
        now = self.clock()
        with self._lock:
            self._expire(now)
            stamp = int(now // BUCKET_SECONDS) * BUCKET_SECONDS
            bucket = self._buckets.setdefault(stamp, {"ts": stamp, **dict.fromkeys(COUNTERS, 0)})
            bucket["stage_count"] += 1
            bucket["stage_ms"] += stage_ms
            if network_ms is not None:
                bucket["network_count"] += 1
                bucket["network_ms"] += network_ms

    def record_work(self, tokens, compute_ms):
        """Count one forward step this node computed for a route: positions through its layers, and the time."""
        if type(tokens) is not int or tokens <= 0 or not isinstance(compute_ms, (int, float)) or compute_ms < 0:
            return
        now = self.clock()
        with self._lock:
            self._expire(now)
            stamp = int(now // BUCKET_SECONDS) * BUCKET_SECONDS
            bucket = self._buckets.setdefault(stamp, {"ts": stamp, **dict.fromkeys(COUNTERS, 0)})
            bucket["served_tokens"] += tokens
            bucket["served_compute_ms"] += compute_ms

    def snapshot(self):
        with self._lock:
            self._expire(self.clock())
            return [{k: round(v, 3) if isinstance(v, float) else v for k, v in self._buckets[t].items()}
                    for t in sorted(self._buckets)]


class MeasuredStream:
    """Observe tokens at the local chat API, including queueing in time to first token."""
    def __init__(self, stream, metrics, started, clock=time.monotonic, speculative_stats=None):
        self.stream, self.metrics, self.started, self.clock = stream, metrics, started, clock
        self.speculative_stats = speculative_stats
        self.count, self.first, self.last, self.finished = 0, None, None, False

    def __iter__(self):
        return self

    def _finish(self, outcome):
        if not self.finished:
            self.finished = True
            stats = self.speculative_stats() if self.speculative_stats else None
            self.metrics.record(outcome=outcome, count=self.count, first=self.first, last=self.last,
                                started=self.started, ended=self.clock(),
                                speculative_drafted_tokens=getattr(stats, "drafted", 0),
                                speculative_accepted_tokens=getattr(stats, "accepted", 0))

    def __next__(self):
        try:
            token = next(self.stream)
        except StopIteration:
            self._finish("completed")
            raise
        except BaseException:
            self._finish("failed")
            raise
        now = self.clock()
        self.count += 1
        if self.first is None:
            self.first = now
        self.last = now
        return token

    def close(self):
        self._finish("cancelled")
        self.stream.close()


class Reporter:
    def __init__(self, identity, gateway, snapshot, metrics, *, interval=15, post=None, clock=time.time, device=None):
        self.identity, self.gateway, self.snapshot, self.metrics = identity, gateway, snapshot, metrics
        # The machine's network-independent key (the peer key changes with the model); defaults to the peer key.
        self.device = device or identity
        self._device_sig = self.device.sign_device_binding(identity.pub)
        self.interval, self.post, self.clock = interval, post or self._post, clock
        self.session = secrets.token_hex(16)
        self.seq = 0
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = None

    def payload(self):
        self.seq += 1
        return self.identity.sign_dashboard({"v": 4, **self.snapshot(), "session": self.session,
                                             "device": self.device.peer, "device_pub": self.device.pub_b64,
                                             "device_sig": self._device_sig,
                                             "seq": self.seq, "ts": self.clock(), "buckets": self.metrics.snapshot()})

    def _post(self, payload):
        request = urllib.request.Request(self.gateway + "/api/network-report", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json", "User-Agent": "common-dashboard"})
        with urllib.request.urlopen(request, timeout=3) as response:
            response.read(1024)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.post(self.payload())
            except Exception:
                pass  # observability must not interrupt serving or joining
            self._wake.wait(self.interval)
            self._wake.clear()
        try:
            self.post(self.payload())  # final leaving state, even when stop happened during an HTTP request
        except Exception:
            pass

    def start(self):
        self._thread = threading.Thread(target=self._run, name="dashboard-report", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=4)
