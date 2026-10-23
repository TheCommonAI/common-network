"""Write a realistic, changing node status file, for developing a UI without a model or a network.

    python3 scripts/fake_status.py                       # to ~/.common-network/net/status.json, in a loop
    python3 scripts/fake_status.py --path /tmp/s.json --speed 4 --once
    python3 scripts/fake_status.py --scenario error      # starting -> error (no GPU)
    python3 scripts/fake_status.py --scenario crash      # goes up, then stops writing with a dead pid

Scenarios: normal (default; download 0->100%, loading, up, peers join and leave, a stage loses its last peer and a
spare moves in, chat API loads), error, crash. It uses the node's own StatusFile and blank_status, so the shape is
the real contract. Ctrl+C writes `stopped`.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from commonnet.node import StatusFile, blank_status, default_status_path  # noqa: E402

STAGES = [[0, 9], [9, 18], [18, 27], [27, 36]]
TOTAL = 4_394 << 20  # the Qwen3-8B 4-bit file, as the hub reports it


def peer(name, window, state="up", backend="mlx", me=False, age=1.0):
    return {"peer": name, "window": window, "backend": backend, "state": state, "age_s": age, "me": me}


def coverage(peers):
    return [sum(1 for p in peers if p["state"] == "up" and p["window"] and p["window"][0] <= a and b <= p["window"][1])
            for a, b in STAGES]


def doc(state, peers=(), **kw):
    peers = list(peers)
    base = dict(network="common-main-8b-v1", model="Qwen3-8B", peer="me0000000000000000000000000000aa",
                backend="mlx", gib=9.9, stages=STAGES, peers=peers, coverage=coverage(peers))
    base.update(kw)
    return blank_status(state, **base)


def normal(speed):
    me = "me0000000000000000000000000000aa"
    yield 2, doc("starting", error=None)
    mine = [9, 18]
    for pct in range(0, 101, 5):
        yield 0.6, doc("downloading", window=mine, download={"bytes": TOTAL * pct // 100, "total": TOTAL},
                       peers=[peer("a1" * 16, [0, 9])])
    yield 3, doc("loading", window=mine, peers=[peer("a1" * 16, [0, 9])])
    url = "https://example-demo.trycloudflare.com"
    peers = [peer(me, mine, me=True), peer("a1" * 16, [0, 9])]
    yield 3, doc("up", window=mine, url=url, peers=peers, api={"port": 11440, "ready": False})
    peers.append(peer("b2" * 16, [18, 27], backend="nf4"))
    yield 4, doc("up", window=mine, url=url, peers=peers, api={"port": 11440, "ready": True})
    peers.append(peer("c3" * 16, [27, 36]))
    yield 6, doc("up", window=mine, url=url, peers=peers, api={"port": 11440, "ready": True})
    peers = [p for p in peers if p["peer"] != "c3" * 16]  # the last stage loses its only peer
    yield 6, doc("up", window=mine, url=url, peers=peers, api={"port": 11440, "ready": True})
    peers.append(peer("d4" * 16, [27, 36], state="loading"))
    yield 4, doc("up", window=mine, url=url, peers=peers, api={"port": 11440, "ready": True})
    peers[-1] = peer("d4" * 16, [27, 36])
    yield 8, doc("up", window=mine, url=url, peers=peers, api={"port": 11440, "ready": True})
    # This peer is told to move to the thinnest stage.
    peers = [p for p in peers if not p["me"]]
    yield 4, doc("moving", window=[9, 18], url=url, peers=peers, api={"port": 11440, "ready": True},
                 download={"bytes": TOTAL // 3, "total": TOTAL})
    peers.append(peer(me, [9, 18], me=True))
    yield 8, doc("up", window=[9, 18], url=url, peers=peers, api={"port": 11440, "ready": True})


def run(path, scenario, speed, once):
    out = StatusFile(path)
    if scenario == "error":
        out.write(doc("starting"))
        time.sleep(2 / speed)
        out.write(doc("error", error="This machine needs a supported GPU (Apple silicon or NVIDIA)."))
        return
    steps = list(normal(speed))
    if scenario == "crash":
        steps = steps[:24]
    cycle = [steps] if once else itertools.repeat(steps)
    last = None
    try:
        for round_ in cycle:
            for seconds, status in round_:
                end = time.time() + seconds / speed
                while time.time() < end:  # rewritten every second, so ts moves like the real thing
                    status = dict(status, ts=time.time(), pid=os.getpid())
                    out.write(status)
                    last = status
                    time.sleep(min(1.0, max(0.05, end - time.time())))
        if scenario == "crash":
            out.write(dict(last, pid=2**22 + 1, ts=time.time() - 60))  # a dead pid and an old ts: it crashed
            return
    except KeyboardInterrupt:
        pass
    if last is not None:
        out.write(dict(last, state="stopped", url=None, ts=time.time()))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", type=Path, default=None, help="default: the node's own default path")
    p.add_argument("--scenario", choices=["normal", "error", "crash"], default="normal")
    p.add_argument("--speed", type=float, default=1.0, help="2 runs twice as fast")
    p.add_argument("--once", action="store_true", help="play the scenario once, then write `stopped`")
    args = p.parse_args(argv)
    run(args.path or default_status_path(), args.scenario, max(args.speed, 0.1), args.once)
    return 0


if __name__ == "__main__":
    sys.exit(main())
