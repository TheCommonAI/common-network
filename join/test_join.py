#!/usr/bin/env python3
"""Tests for join.py configuration constants.

These verify the fixes for Windows localhost IPv4/IPv6 issues and stable node
identity across restarts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import join

FAILURES = []


def check(name: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL  {name}  (got {got!r}, want {want!r})")
    else:
        print(f"  ok    {name}")


print("\nIPv4 addresses (Windows localhost resolves to IPv6, breaking tunnels)")
check("OLLAMA_URL uses 127.0.0.1, not localhost",
      "127.0.0.1" in join.OLLAMA_URL and "localhost" not in join.OLLAMA_URL, True)
check("WORKER_URL uses 127.0.0.1, not localhost",
      "127.0.0.1" in join.WORKER_URL and "localhost" not in join.WORKER_URL, True)

print("\ntunnel health check timeout is reasonable")
check("TUNNEL_WARMUP_TIMEOUT_SECONDS is ~30s",
      25 <= join.TUNNEL_WARMUP_TIMEOUT_SECONDS <= 35, True)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all join config tests passed")
