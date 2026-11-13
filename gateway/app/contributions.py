"""The beta contribution log: how much work each machine has done for the network, kept across restarts.

Every accepted dashboard report becomes one JSON line holding what grew since that node's previous report:
tokens its layers processed, the compute time they took, route sessions it served to a clean close, chat
requests made through it, and the serving time the gateway itself saw between heartbeats. Totals are the sum
of the lines, so the file replays (and audits) on its own. Text-free, like the dashboard: identifiers and
numbers only.

Work counts are self-reported and signed by the node; uptime is measured here, from heartbeat arrival.
A machine is named by its device key (v4 reports), which stays the same across model switches and restarts.
Older nodes are named by their per-network peer key instead.

COMMON_CONTRIBUTIONS_LOG names the file (put it on a persistent volume). Without it the totals live only in
memory. Each line is also printed to stdout, prefixed `contribution `, as a backup in the host's logs.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

LOG_ENV = "COMMON_CONTRIBUTIONS_LOG"
# Report counter -> logged name. served_* come from v4 nodes; stage_count is route sessions closed cleanly.
WORK = {"served_tokens": "tokens", "served_compute_ms": "compute_ms", "stage_count": "requests_served",
        "completed": "chat_completed", "failed": "chat_failed", "output_tokens": "output_tokens"}
GAP_SECONDS = 60      # a heartbeat later than this means the node was away; the gap is not uptime
WINDOW_SECONDS = 310  # a node resends its last five minutes of buckets; remember them that long
MAX_DEVICES = 20000


class Log:
    def __init__(self, path=None, out=sys.stdout):
        self.path = Path(path) if path else None
        self.out = out
        self._lock = threading.Lock()
        self._seen = {}    # (network, peer, session) -> {bucket ts: {counter: value already logged}}
        self._last = {}    # (network, device) -> (time of last heartbeat, was serving)
        self.totals = {}   # device -> running totals
        self.since = None
        self._write_failed = False
        if self.path and self.path.exists():
            self._replay()

    def record(self, body: dict, now: float) -> None:
        """Log one validated dashboard report."""
        device = body.get("device") or body["peer"]
        with self._lock:
            if device not in self.totals and len(self.totals) >= MAX_DEVICES:
                return
            seen = self._seen.get((body["network"], body["peer"], body["session"]), {})
            buckets = {}
            for bucket in body["buckets"]:
                old = seen.get(bucket["ts"], {})
                grew = {name: bucket[name] - old.get(name, 0) for name in WORK
                        if bucket.get(name, 0) > old.get(name, 0) + 1e-6}  # not float dust from rounding
                if grew:  # _apply moves `seen` on, exactly as a replay of this line would
                    buckets[str(bucket["ts"])] = {name: _round(value) for name, value in grew.items()}
            serving = body["phase"] == "up" and body["layers"] is not None
            last = self._last.get((body["network"], device))
            uptime = round(now - last[0], 3) if last and last[1] and serving and 0 < now - last[0] <= GAP_SECONDS else 0
            line = {"t": round(now, 3), "device": device, "peer": body["peer"], "network": body["network"],
                    "session": body["session"], "backend": body["backend"], "layers": body["layers"],
                    "serving": serving, "uptime_s": uptime, "buckets": buckets}
            self._apply(line)
            self._prune(now)
            self._write(line)

    def snapshot(self) -> dict:
        with self._lock:
            devices = sorted((dict(t, networks=sorted(t["networks"]), backends=sorted(t["backends"]))
                              for t in self.totals.values()), key=lambda t: (-t["tokens"], t["device"]))
        return {"v": 1, "since": self.since, "persisted": self.path is not None,
                "measured": {"uptime_s": "gateway, from heartbeat arrival", "work": "self-reported by the node"},
                "devices": devices}

    def _apply(self, line: dict) -> None:
        """Fold one line into the totals and the replay state. Shared by record and replay."""
        self.since = line["t"] if self.since is None else min(self.since, line["t"])
        total = self.totals.get(line["device"])
        if total is None:
            total = self.totals[line["device"]] = {"device": line["device"], "networks": set(), "backends": set(),
                                                   **dict.fromkeys(WORK.values(), 0), "uptime_s": 0,
                                                   "first_seen": line["t"], "last_seen": line["t"]}
        total["networks"].add(line["network"])
        total["backends"].add(line["backend"])
        total["uptime_s"] = _round(total["uptime_s"] + line["uptime_s"])
        total["last_seen"] = max(total["last_seen"], line["t"])
        seen = self._seen.setdefault((line["network"], line["peer"], line["session"]), {})
        for stamp, grew in line["buckets"].items():
            old = seen.setdefault(int(stamp), {})
            for name, value in grew.items():
                old[name] = old.get(name, 0) + value
                total[WORK[name]] = _round(total[WORK[name]] + value)
        self._last[(line["network"], line["device"])] = (line["t"], line["serving"])

    def _prune(self, now: float) -> None:
        for key in list(self._seen):
            buckets = self._seen[key]
            for stamp in [s for s in buckets if s < now - WINDOW_SECONDS]:
                del buckets[stamp]
            if not buckets:
                del self._seen[key]
        for key in [k for k, (at, _) in self._last.items() if now - at > GAP_SECONDS]:
            del self._last[key]

    def _replay(self) -> None:
        latest = None
        with self.path.open() as f:
            for raw in f:
                try:
                    line = json.loads(raw)
                    self._apply(line)
                except (ValueError, KeyError, TypeError):
                    continue  # a line cut short by a crash; the rest still counts
                latest = line["t"]
                if len(self._seen) > 4 * MAX_DEVICES:
                    self._prune(latest)
        if latest is not None:
            self._prune(latest)

    def _write(self, line: dict) -> None:
        text = json.dumps(line, separators=(",", ":"))
        print("contribution " + text, file=self.out, flush=True)
        if self.path is None:
            return
        try:
            with self.path.open("a") as f:
                f.write(text + "\n")
        except OSError as e:
            if not self._write_failed:  # say so once; the stdout copy still has every line
                self._write_failed = True
                print(f"contribution log: cannot write {self.path}: {e}", file=self.out, flush=True)


def _round(value):
    return round(value, 3) if isinstance(value, float) else value


log = Log(os.environ.get(LOG_ENV))
