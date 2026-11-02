"""Public dashboard schema, CORS, redaction, signed reports and weighted metrics."""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gateway"))
sys.path.insert(0, str(ROOT))
os.environ["COMMON_GATEWAY_TEST"] = "1"

from fastapi.testclient import TestClient
from app import dashboard, directory, main
from commonnet.metrics import Metrics, Reporter
from commonnet.mesh.records import Identity

NET = "common-main-8b-v1"
NOW = 1700000000


def bucket(*, completed=1, output=3, decode_ms=200, ttft_ms=100):
    return {"ts": NOW, "requests": completed, "completed": completed, "failed": 0, "cancelled": 0,
            "output_tokens": output, "decode_tokens": output - completed, "decode_ms": decode_ms,
            "ttft_count": completed, "ttft_ms": ttft_ms, "total_ms": decode_ms + ttft_ms}


class Dashboard(unittest.TestCase):
    def setUp(self):
        dashboard._peers.clear()
        dashboard._reports.clear()
        directory._records.clear()
        directory._hits.clear()
        self.now = NOW
        self.time = mock.patch.object(dashboard.time, "time", side_effect=lambda: self.now)
        self.time.start()
        self.addCleanup(self.time.stop)
        self.path = mock.patch.object(main, "NETWORKS", ROOT / "commonnet" / "networks")
        self.path.start()
        self.addCleanup(self.path.stop)
        self.client = TestClient(main.app)
        self.headers = {"x-forwarded-for": "1.1.1.1"}

    def payload(self, identity=None, **overrides):
        fields = dict(v=1, network=NET, session="a" * 32, seq=1, ts=self.now, phase="up", backend="mlx",
                      gib=8, layers=[0, 36], assigned_layers=[0, 36], buckets=[])
        fields.update(overrides)
        return (identity or Identity.generate()).sign_dashboard(fields)

    def post(self, payload):
        return self.client.post("/api/network-report", content=json.dumps(payload),
                                headers={**self.headers, "Content-Type": "application/json"})

    def stats(self, network=NET):
        return self.client.get("/api/network-stats", params={"network": network}).json()

    def test_empty_stats_use_nulls_and_browser_cors_works(self):
        response = self.client.get("/api/network-stats", headers={"Origin": "https://my-site.example"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")
        self.assertEqual(response.json()["online_nodes"], 0)
        self.assertIsNone(response.json()["metrics"]["avg_tokens_per_second"])
        self.assertIsNone(response.json()["metrics"]["avg_ms_per_token"])
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(self.client.options("/api/network-stats", headers={"Origin": "https://my-site.example",
                         "Access-Control-Request-Method": "GET"}).status_code, 204)
        self.assertNotIn("Access-Control-Allow-Origin", self.client.get("/directory").headers)

    def test_real_reporter_payload_is_accepted_without_text_or_tokens(self):
        metrics = Metrics(clock=lambda: self.now)
        metrics.record(outcome="completed", count=3, first=.1, last=.3, started=0, ended=.3)
        reporter = Reporter(Identity.generate(), "https://unused", lambda: dict(network=NET, phase="up", backend="nf4",
                            gib=12, layers=[0, 36], assigned_layers=[0, 36]), metrics, clock=lambda: self.now)
        self.assertEqual(self.post(reporter.payload()).json(), {"ok": True})
        stats = self.stats()
        self.assertEqual(stats["online_nodes"], 1)
        self.assertEqual(stats["metrics"]["avg_tokens_per_second"], 10)
        self.assertEqual(stats["metrics"]["avg_ms_per_token"], 100)
        self.assertTrue(stats["models"][0]["route_ready"])
        node = stats["models"][0]["nodes"][0]
        self.assertEqual(node["display_state"], "ful")
        for private in ("token", "url", "pub", "sig", "source", "prompt", "answer"):
            self.assertNotIn(private, node)

    def test_directory_only_nodes_appear_with_redacted_records_and_expire(self):
        identity = Identity.generate()
        record = identity.sign_record(dict(network=NET, backend="mlx", state="up", gib=4,
                                      window=[0, 36], url="https://private-peer.example", token="hidden-bearer"), now=self.now)
        self.assertEqual(self.client.post("/announce", json=record, headers=self.headers).status_code, 200)
        stats = self.stats()
        self.assertEqual(stats["online_nodes"], 1)
        raw = json.dumps(stats)
        self.assertNotIn("hidden-bearer", raw)
        self.assertNotIn("private-peer.example", raw)
        self.now += 61
        stats = self.stats()
        self.assertEqual(stats["online_nodes"], 0)
        self.assertEqual(stats["models"][0]["nodes"][0]["state"], "offline")
        self.now += dashboard.HISTORY_SECONDS
        self.assertEqual(self.stats()["models"][0]["nodes"], [])

    def test_overlapping_coverage_is_not_a_complete_route(self):
        self.post(self.payload(layers=[0, 18]))
        self.post(self.payload(layers=[9, 36]))
        model = self.stats()["models"][0]
        self.assertTrue(all(stage["peers"] for stage in model["stages"]))
        self.assertFalse(model["route_ready"])
        self.post(self.payload(layers=[18, 36]))
        self.assertTrue(self.stats()["models"][0]["route_ready"])

    def test_weighted_metrics_retries_and_restarts_do_not_duplicate_samples(self):
        a, b = Identity.generate(), Identity.generate()
        first = self.payload(a, buckets=[bucket(completed=2, output=8, decode_ms=600, ttft_ms=100)])
        second = self.payload(b, buckets=[bucket(decode_ms=1000, ttft_ms=400)])
        self.post(first)
        self.post(second)
        self.assertTrue(self.post(first).json()["stale"])
        self.post(self.payload(a, seq=2, buckets=first["buckets"]))
        metrics = self.stats()["metrics"]
        self.assertEqual(metrics["sample_count"], 3)
        self.assertEqual(metrics["avg_tokens_per_second"], 5)
        self.assertEqual(metrics["avg_ms_per_token"], 200)
        self.assertEqual(metrics["avg_time_to_first_token_ms"], 166.667)
        self.post(self.payload(a, session="b" * 32, buckets=[bucket()]))
        self.assertEqual(self.stats()["metrics"]["sample_count"], 4)
        self.now += 301
        self.assertIsNone(self.stats()["metrics"]["avg_tokens_per_second"])

    def test_loading_and_unassigned_states_are_visible(self):
        self.post(self.payload(phase="downloading", layers=None, assigned_layers=[0, 9]))
        self.post(self.payload(layers=None, assigned_layers=None))
        self.assertEqual({n["state"] for n in self.stats()["models"][0]["nodes"]}, {"linking", "unassigned"})

    def test_peer_and_report_storage_are_bounded(self):
        with mock.patch.object(dashboard, "MAX_PEERS", 1):
            self.assertEqual(self.post(self.payload()).status_code, 200)
            self.assertEqual(self.post(self.payload()).status_code, 429)
        identity = Identity.generate()
        with mock.patch.object(dashboard, "MAX_SESSIONS", 1):
            self.assertEqual(self.post(self.payload(identity)).status_code, 429)

    def test_unknown_signed_text_fields_are_not_accepted(self):
        self.assertEqual(self.post(self.payload(prompt="not part of telemetry")).status_code, 400)
        self.assertEqual(self.stats()["online_nodes"], 0)

    def test_forgery_bad_counters_and_oversize_are_rejected(self):
        forged = self.payload()
        forged["gib"] = 50
        self.assertEqual(self.post(forged).status_code, 400)
        for overrides in (dict(ts=self.now - 61), dict(ts=self.now + 30), dict(gib=float("inf")),
                          dict(layers=[True, 36]), dict(buckets=[bucket(decode_ms=-1)]),
                          dict(buckets=[{**bucket(), "requests": 100}])):
            self.assertEqual(self.post(self.payload(**overrides)).status_code, 400)
        self.assertEqual(self.client.post("/api/network-report", content=b"x" * 40000, headers=self.headers).status_code, 413)

    def test_old_signed_report_cannot_resurrect_a_leaving_peer(self):
        identity = Identity.generate()
        old = self.payload(identity)
        self.post(old)
        self.now += 1
        self.post(self.payload(identity, phase="leaving", seq=2, layers=None))
        self.post(old)
        self.assertEqual(self.stats()["online_nodes"], 0)

    def test_network_filter_keeps_models_and_metrics_separate(self):
        self.post(self.payload(network=NET, buckets=[bucket()]))
        other = "common-qwen3-5-122b-a10b-v1"
        self.post(self.payload(network=other, layers=[0, 7], buckets=[bucket(output=6)]))
        stats = self.stats(other)
        self.assertEqual(len(stats["models"]), 1)
        self.assertEqual(stats["models"][0]["model"], "Qwen3.5-122B-A10B")
        self.assertEqual(stats["metrics"]["output_tokens"], 6)
        self.assertEqual(stats["online_nodes"], 1)


if __name__ == "__main__":
    unittest.main()
