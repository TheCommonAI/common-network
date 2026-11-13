"""The beta contribution log: per-device totals, no double counting, gateway-measured uptime, replay."""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gateway"))
sys.path.insert(0, str(ROOT))
os.environ["COMMON_GATEWAY_TEST"] = "1"

from fastapi.testclient import TestClient
from app import contributions, dashboard, directory, main
from commonnet.metrics import COUNTERS, Metrics, Reporter
from commonnet.mesh.records import Identity

NET = "common-main-8b-v1"
NOW = 1700000000


def work(ts=NOW, tokens=10, compute_ms=40.0, sessions=1):
    return {"ts": ts, **dict.fromkeys(COUNTERS, 0), "served_tokens": tokens, "served_compute_ms": compute_ms,
            "stage_count": sessions, "stage_ms": compute_ms}


class Contributions(unittest.TestCase):
    def setUp(self):
        dashboard._peers.clear()
        dashboard._reports.clear()
        directory._records.clear()
        directory._hits.clear()
        self.now = NOW
        self.time = mock.patch.object(dashboard.time, "time", side_effect=lambda: self.now)
        self.time.start()
        self.addCleanup(self.time.stop)
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.file = Path(self.dir.name) / "contributions.jsonl"
        self.log = contributions.Log(self.file, out=io.StringIO())
        patch = mock.patch.object(contributions, "log", self.log)
        patch.start()
        self.addCleanup(patch.stop)
        self.client = TestClient(main.app)
        self.peer, self.device = Identity.generate(), Identity.generate()
        self.seq = 0

    def report(self, buckets, *, peer=None, network=NET, phase="up", layers=(0, 9), session="a" * 32, device=None):
        self.seq += 1
        peer = peer or self.peer
        snapshot = lambda: {"network": network, "phase": phase, "backend": "mlx", "gib": 8.0,
                            "layers": list(layers) if layers else None, "assigned_layers": None}
        reporter = Reporter(peer, "http://unused", snapshot, Metrics(), clock=lambda: self.now,
                            device=device or self.device)
        reporter.session, reporter.seq = session, self.seq - 1
        reporter.metrics.snapshot = lambda: buckets
        response = self.client.post("/api/network-report", content=json.dumps(reporter.payload()),
                                    headers={"x-forwarded-for": "1.1.1.1", "Content-Type": "application/json"})
        self.assertEqual(response.status_code, 200, response.text)

    def totals(self):
        devices = self.client.get("/api/contributions").json()["devices"]
        return {d["device"]: d for d in devices}

    def test_work_is_counted_once_per_device_although_buckets_are_resent(self):
        self.report([work(tokens=10)])
        self.now += 15
        self.report([work(tokens=10)])  # the same bucket, resent with the next heartbeat
        self.now += 15
        self.report([work(tokens=25, compute_ms=90.0, sessions=2), work(ts=NOW + 20, tokens=4)])
        mine = self.totals()[self.device.peer]
        self.assertEqual((mine["tokens"], mine["requests_served"]), (29, 3))
        self.assertAlmostEqual(mine["compute_ms"], 130.0)
        self.assertEqual(mine["networks"], [NET])

    def test_uptime_is_measured_between_serving_heartbeats_and_not_across_absences(self):
        self.report([])
        self.now += 15
        self.report([])
        self.now += 15
        self.report([], phase="loading", layers=None)  # not serving: this gap is not uptime
        self.now += 15
        self.report([])
        self.now += 15
        self.report([])
        self.now += 300  # away for five minutes
        self.report([])
        self.assertEqual(self.totals()[self.device.peer]["uptime_s"], 30)

    def test_one_machine_keeps_one_name_across_models(self):
        self.report([work(tokens=7)])
        self.report([work(tokens=5)], peer=Identity.generate(), network="common-qwen3-5-35b-a3b-v1")
        mine = self.totals()
        self.assertEqual(list(mine), [self.device.peer])
        self.assertEqual(mine[self.device.peer]["tokens"], 12)
        self.assertEqual(len(mine[self.device.peer]["networks"]), 2)

    def test_a_device_cannot_claim_a_peer_it_did_not_sign_for(self):
        snapshot = lambda: {"network": NET, "phase": "up", "backend": "mlx", "gib": 8.0, "layers": [0, 9],
                            "assigned_layers": None}
        reporter = Reporter(self.peer, "http://unused", snapshot, Metrics(), clock=lambda: self.now,
                            device=self.device)
        reporter._device_sig = Identity.generate().sign_device_binding(self.peer.pub)
        response = self.client.post("/api/network-report", content=json.dumps(reporter.payload()),
                                    headers={"x-forwarded-for": "1.1.1.1", "Content-Type": "application/json"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.totals(), {})

    def test_the_file_replays_to_the_same_totals_and_still_skips_resent_buckets(self):
        self.report([work(tokens=10)])
        self.now += 15
        self.report([work(tokens=10)])
        before = self.totals()
        restarted = contributions.Log(self.file, out=io.StringIO())
        self.assertEqual(restarted.snapshot()["devices"], self.log.snapshot()["devices"])
        with mock.patch.object(contributions, "log", restarted):
            self.now += 15
            self.report([work(tokens=12)])  # after a gateway restart: only the growth counts
            self.assertEqual(self.totals()[self.device.peer]["tokens"], 12)
            self.assertEqual(self.totals()[self.device.peer]["uptime_s"], before[self.device.peer]["uptime_s"] + 15)

    def test_older_reports_are_logged_under_their_peer(self):
        fields = dict(v=3, network=NET, session="b" * 32, seq=1, ts=self.now, phase="up", backend="mlx", gib=8,
                      layers=[0, 9], assigned_layers=None,
                      buckets=[{k: v for k, v in work().items() if k not in ("served_tokens", "served_compute_ms")}])
        response = self.client.post("/api/network-report", content=json.dumps(self.peer.sign_dashboard(fields)),
                                    headers={"x-forwarded-for": "1.1.1.1", "Content-Type": "application/json"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.totals()[self.peer.peer]["requests_served"], 1)

    def test_lines_are_text_free_and_written_to_stdout_too(self):
        self.report([work()])
        line = json.loads(self.file.read_text().splitlines()[0])
        self.assertEqual(set(line), {"t", "device", "peer", "network", "session", "backend", "layers", "serving",
                                     "uptime_s", "buckets"})
        self.assertIn("contribution {", self.log.out.getvalue())


if __name__ == "__main__":
    unittest.main()
