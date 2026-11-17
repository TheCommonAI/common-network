"""Cooperative process-wide budgets, shared by downloads and peer payloads.

These pace application work, not OS-wide traffic or instantaneous GPU utilization.
"""
from __future__ import annotations

import math
import os
import threading
import time


class Pacer:
    """Reserve time against one aggregate rate, including concurrent callers."""
    def __init__(self, rate, *, now=time.monotonic, sleep=time.sleep):
        self.rate = rate
        self.now, self.sleep = now, sleep
        self.next = 0.0
        self.lock = threading.Lock()

    def consume(self, amount):
        if not self.rate or amount <= 0:
            return
        with self.lock:
            self.next = max(self.next, self.now()) + amount / self.rate
            delay = self.next - self.now()
        if delay > 0:
            self.sleep(delay)


bandwidth = Pacer(None)
compute_percent = 100.0


def validate(args):
    for name in ("stage_gib", "cpu_threads", "compute_percent", "bandwidth_mbps"):
        value = getattr(args, name, None)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    percent = getattr(args, "compute_percent", None)
    if percent is not None and not 1 <= percent <= 100:
        raise ValueError("--compute-percent must be between 1 and 100")


def configure(args):
    global bandwidth, compute_percent
    validate(args)
    bandwidth = Pacer(args.bandwidth_mbps * 1_000_000 / 8 if args.bandwidth_mbps else None)
    compute_percent = args.compute_percent or 100.0
    if args.cpu_threads:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[name] = str(args.cpu_threads)
        import torch
        torch.set_num_threads(args.cpu_threads)


def idle_after(seconds):
    if compute_percent < 100:
        time.sleep(max(0, seconds) * (100 / compute_percent - 1))


class PacedResponse:
    """Wrap a model response without changing HTTP/retry/context-manager semantics."""
    def __init__(self, response):
        self.response = response

    def __getattr__(self, name):
        return getattr(self.response, name)

    def __enter__(self):
        self.response.__enter__()
        return self

    def __exit__(self, *args):
        return self.response.__exit__(*args)

    def read(self, size=-1):
        # Chunk even read-all calls, so large model shards do not bypass the budget.
        chunks = []
        remaining = size
        while remaining != 0:
            chunk = self.response.read(65536 if remaining < 0 else min(65536, remaining))
            if not chunk:
                break
            bandwidth.consume(len(chunk))
            chunks.append(chunk)
            if remaining > 0:
                remaining -= len(chunk)
        return b"".join(chunks)
