"""`python -m commonnet.bench --live` against the synthetic cluster: real node processes, real HTTP, a stub model."""
import io
import json
import unittest
from contextlib import redirect_stdout

from commonnet.bench import __main__ as cli
from commonnet.bench import live
from commonnet.bench.synth import OracleDrafter, SynthClient


def synth_factory(cluster, drafter_accept=None):
    def make(net):
        client = SynthClient(None, layers=cluster.layers, hidden=64, seed=cluster.seed)
        vocab = client.config.vocab_size
        ids = lambda n: [(7 * i + 3) % vocab for i in range(n)]
        drafter = None
        if drafter_accept is not None:
            ep, tr = net.plan()
            client.transport = tr
            truth = client.generate(ep, ids(8), 12)
            tr.shutdown()
            drafter = OracleDrafter(8, truth, drafter_accept)
        return client, ids, None or (lambda fn: fn()), None, drafter
    return make


class LiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from commonnet.bench.harness import Cluster
        cls.cluster = Cluster(2, 1, layers=2, layer_ms=5.0, rtt_ms=20.0).start()
        cls.report = live.run_live(cls.cluster.seed_url, make_client=synth_factory(cls.cluster),
                                   cases=((4, 6), (12, 4)), repeats=2, log=lambda *_: None, settle_s=0.1)

    @classmethod
    def tearDownClass(cls):
        cls.cluster.stop()

    def test_route_and_breakdown_per_hop(self):
        r = self.report
        self.assertEqual(len(r["route"]), 2)
        self.assertEqual(r["warnings"], [])      # synthetic nodes all sit on 127.0.0.1 behind different proxy ports
        d = r["cases"][0]["decode"]
        self.assertEqual(d["steps"], 2 * 5)      # 6 tokens: 1 prefill + 5 decode, x2 repeats
        self.assertEqual([h["hop"] for h in d["per_hop"]], [0, 1])
        self.assertTrue(all(h["server_ms"] is not None and h["calls"] == 10 for h in d["per_hop"]))
        self.assertGreater(d["compute_ms"], 8.0)     # two hops x 5 ms of synthetic layer time
        self.assertGreater(d["network_ms"], 30.0)    # two hops x 20 ms emulated RTT
        total = d["compute_ms"] + d["network_ms"] + d["client_ms"]
        self.assertAlmostEqual(total, d["ms_mean"], delta=1.0)
        self.assertGreater(r["cases"][1]["prefill"]["tok_s"], 0)

    def test_hop_rtt_is_measured_and_shown_beside_network(self):
        for h in self.report["hop_rtt"]:
            self.assertIsNone(h["error"])
            self.assertGreater(h["warm_ms_p50"], 15.0)
        hop = self.report["cases"][0]["decode"]["per_hop"][0]
        self.assertIsNotNone(hop["rtt_ms"])
        text = live.format_report(self.report)
        self.assertIn("RTT ms", text)
        self.assertIn("compute", live.one_line(self.report))

    def test_json_has_no_token_and_no_query_strings(self):
        blob = json.dumps(self.report).lower()
        self.assertNotIn("token", blob)
        self.assertNotIn("?", blob)
        self.assertEqual(live.clean_url("https://a.example.com:8443/x/y?key=secret#f"), "https://a.example.com:8443")

    def test_cli_writes_json(self):
        import tempfile
        out = tempfile.mktemp(suffix=".json")
        args = cli.build_parser().parse_args(["--live", self.cluster.seed_url, "--prompt-lens", "4", "--new-tokens", "3",
                                              "--repeats", "1", "--out", out])
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.run_live_cli(args, make_client=synth_factory(self.cluster))
        self.assertEqual(rc, 0)
        self.assertIn("ms/token", buf.getvalue())
        self.assertNotIn("token", json.dumps(json.load(open(out))).lower())

    def test_speculative_reports_acceptance_and_same_output(self):
        make = synth_factory(self.cluster, drafter_accept=1.0)
        from commonnet.netclient import Network
        net = Network.from_spec(self.cluster.seed_url)
        net.refresh(force=True)
        client, ids, run, _, drafter = make(net)
        r = live.run_speculative(net, client, ids(8), 12, drafter=drafter, draft_k=4)
        self.assertTrue(r["same_output_as_plain"])
        self.assertGreater(r["acceptance"], 0.9)

    def test_speculative_skipped_without_drafter(self):
        rep = live.run_live(self.cluster.seed_url, make_client=synth_factory(self.cluster), cases=((4, 3),),
                            repeats=1, draft=True, log=lambda *_: None, settle_s=0.1)
        self.assertIn("skipped", rep["speculative"][0])
        self.assertIn("skipped", live.format_report(rep))


class SinglePeerTest(unittest.TestCase):
    def test_one_peer_holding_every_layer_is_called_out(self):
        from commonnet.bench.harness import Cluster
        logs = []
        with Cluster(1, 1, layers=2, layer_ms=0.0) as c:
            rep = live.run_live(c.seed_url, make_client=synth_factory(c), cases=((4, 3),), repeats=1,
                                log=logs.append, settle_s=0.1)
        self.assertEqual(len(rep["route"]), 1)
        self.assertTrue(any("ONE peer holding every layer" in w and "nothing to split" in w for w in rep["warnings"]))
        self.assertTrue(any(l.startswith("WARNING") for l in logs))
        self.assertNotIn("token", json.dumps(rep).lower())


if __name__ == "__main__":
    unittest.main()
