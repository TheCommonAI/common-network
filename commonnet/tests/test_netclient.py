"""Network: routes, replicas and waiting for a newcomer, with a fake peer table and real stub block servers."""
import json
import secrets
import tempfile
import unittest
from pathlib import Path

import torch

from commonnet.mesh.manifest import Manifest
from commonnet.mesh.placement import NoRoute
from commonnet.net.client import Unreachable
from commonnet.net.server import BlockServer, StubBlock
from commonnet.netclient import Network, NetworkUnreachable, NoLayersServed

MANIFEST = {"id": "t-net", "version": 1,
            "model": {"name": "Tiny", "num_layers": 4, "repos": {"mlx": "x/y"}, "client_repo": "x/y"},
            "boundaries": [0, 2, 4], "gib_per_layer": {"mlx": 0.1}, "fixed_gib": 0.0, "min_replicas": 2,
            "seeds": ["http://127.0.0.1:1"]}
NOW = 1_000_000.0


class FakeTable:
    def __init__(self):
        self.records = {}

    def merge_many(self, records):
        for r in records:
            self.records[r["peer"]] = r

    def export(self, limit=200):
        return list(self.records.values())

    def live(self, now=None):
        return [r for r in self.records.values() if r["state"] == "up"]


def record(peer, url, token, window, backend="mlx", state="up", ts=NOW):
    return {"v": 1, "network": "t-net", "peer": peer, "pub": "", "url": url, "token": token, "window": list(window),
            "backend": backend, "state": state, "gib": 8.0, "seq": 1, "ts": ts, "sig": ""}


class Peers:
    """Real stub servers plus the records that advertise them."""

    def __init__(self, test):
        self.test = test
        self.servers = {}

    def add(self, peer, window, backend="mlx"):
        token = secrets.token_urlsafe(32)
        block = StubBlock(*window)
        server = BlockServer(block, "127.0.0.1", 0, token=token).start()
        self.test.addCleanup(server.stop)
        self.servers[peer] = (server, block, record(peer, server.url, token, window, backend))
        return self.servers[peer][2]

    def rec(self, peer):
        return self.servers[peer][2]


def hidden(n=1, seed=0):
    return torch.randn(1, n, 32, generator=torch.Generator().manual_seed(seed))


class NetworkTests(unittest.TestCase):
    def network(self, advertised, later=None, **kw):
        """A Network whose exchange fills the table from `advertised`; `later(n)` may add records on call n."""
        table = FakeTable()
        calls = {"n": 0}

        def exchange(tbl, url, timeout):
            calls["n"] += 1
            tbl.merge_many(advertised)
            if later:
                tbl.merge_many(later(calls["n"]))

        net = Network(seeds=["http://seed"], manifest=MANIFEST, table=table, exchange=exchange,
                      now=lambda: NOW, refresh_interval=0, **kw)
        net.calls = calls
        return net

    def test_route_has_hops_for_each_window_and_replicas_for_the_doubled_one(self):
        peers = Peers(self)
        recs = [peers.add("a1", (0, 2)), peers.add("a2", (0, 2)), peers.add("b1", (2, 4))]
        net = self.network(recs)
        endpoints, transport = net.plan()
        self.assertEqual([(e.name, e.start, e.end) for e in endpoints], [("hop0", 0, 2), ("hop1", 2, 4)])
        out = hidden(3, 1)
        for ep in endpoints:
            self.assertTrue(torch.equal(transport.forward(ep, "s", out, 0), out))
        self.assertEqual(len(transport._candidates("hop0")), 2)
        self.assertEqual(len(transport._candidates("hop1")), 1)

    def test_a_dead_replica_is_failed_over_without_waiting(self):
        peers = Peers(self)
        recs = [peers.add("a1", (0, 2)), peers.add("a2", (0, 2)), peers.add("b1", (2, 4))]
        net = self.network(recs)
        endpoints, transport = net.plan(retries=0, sleep=lambda _: None)
        first = transport._candidates("hop0")[0]
        victim = next(p for p, (s, b, r) in peers.servers.items() if r["url"] == first[0] or r["url"] == first)
        peers.servers[victim][0].stop()
        for (name, i), client in transport._clients.items():
            client.close_connections()
        out = hidden(2, 2)
        self.assertTrue(torch.equal(transport.forward(endpoints[0], "s", out, 0), out))
        self.assertEqual(len(transport.failovers), 1)

    def test_no_route_names_the_layers_nobody_serves(self):
        peers = Peers(self)
        net = self.network([peers.add("a1", (0, 2))])
        with self.assertRaises(NoLayersServed) as caught:
            net.route()
        self.assertIsInstance(caught.exception, NoRoute)
        self.assertIn("layers 2-4", str(caught.exception))
        self.assertIn("Tiny", str(caught.exception))
        self.assertEqual(caught.exception.uncovered, [(2, 4)])

    def test_nobody_answering_is_a_network_error_not_a_route_error(self):
        net = Network(seeds=["http://seed"], manifest=MANIFEST, table=FakeTable(),
                      exchange=lambda t, u, timeout: (_ for _ in ()).throw(OSError("refused")), refresh_interval=0)
        with self.assertRaises(NetworkUnreachable) as caught:
            net.route()
        self.assertIn("refused", str(caught.exception))

    def test_an_emptied_hop_waits_for_a_newcomer_with_the_same_window(self):
        peers = Peers(self)
        recs = [peers.add("a1", (0, 2)), peers.add("b1", (2, 4))]
        newcomer = peers.add("b2", (2, 4))
        slept = []
        # b2 only appears in the mesh from the 12th exchange on
        net = self.network(recs, later=lambda n: [newcomer] if n >= 12 else [])
        clock = {"t": 0.0}
        endpoints, transport = net.plan(retries=0, sleep=lambda _: None, now=lambda: clock["t"],
                                        wait_sleep=lambda s: (slept.append(s), clock.__setitem__("t", clock["t"] + s)))
        first, second = hidden(4, 3), hidden(1, 4)
        for ep in endpoints:
            transport.forward(ep, "s", first, 0)
        peers.servers["b1"][0].stop()
        for client in transport._clients.values():
            client.close_connections()
        for ep in endpoints:
            self.assertTrue(torch.equal(transport.forward(ep, "s", second, 4), second))
        self.assertTrue(slept)  # it did wait
        newcomer_calls = [(op, pos) for op, _, pos in peers.servers["b2"][1].calls if op == "forward"]
        self.assertEqual(newcomer_calls, [(  "forward", 0), ("forward", 4)])  # the session was replayed to it

    def test_token_for_overrides_the_record_token(self):
        peers = Peers(self)
        recs = [peers.add("a1", (0, 2)), peers.add("b1", (2, 4))]
        net = self.network(recs)
        endpoints, transport = net.plan(token_for=lambda r: "wrong-token-wrong-token-wrong-token-1234")
        with self.assertRaises(Exception) as caught:
            transport.forward(endpoints[0], "s", hidden(), 0)
        self.assertIn("401", str(caught.exception))

    def test_cache_remembers_live_peer_urls(self):
        peers = Peers(self)
        recs = [peers.add("a1", (0, 2)), peers.add("b1", (2, 4))]
        with tempfile.TemporaryDirectory() as d:
            self.network(recs, cache_dir=d).route()
            data = json.loads((Path(d) / "peers.json").read_text())
            self.assertEqual(sorted(data["urls"]), sorted(r["url"] for r in recs))
            self.assertEqual(data["network"], "t-net")
            self.assertNotIn("token", json.dumps(data))  # urls only
            fresh = Network(seeds=[], manifest=MANIFEST, cache_dir=d, table=FakeTable(), exchange=lambda *a: None)
            self.assertEqual(sorted(fresh._cached), sorted(r["url"] for r in recs))

    def test_from_spec_reads_a_manifest_file_and_lists_its_seeds(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "net.json"
            path.write_text(json.dumps(MANIFEST))
            net = Network.from_spec(str(path))
            self.assertEqual(net.seeds, ["http://127.0.0.1:1"])
            self.assertEqual(net.manifest.id, "t-net")

    def test_from_spec_with_a_url_fetches_the_manifest_from_the_seed(self):
        asked = []
        net = Network.from_spec("http://peer.example/", fetch_manifest=lambda url, t: (asked.append(url), MANIFEST)[1])
        self.assertEqual(net.manifest.id, "t-net")
        self.assertEqual(asked, ["http://peer.example"])
        self.assertIn("http://127.0.0.1:1", net.seeds)  # the manifest's own seeds are added

    def test_status_shows_coverage_per_stage(self):
        peers = Peers(self)
        net = self.network([peers.add("a1", (0, 2))])
        net.refresh()
        status = net.status()
        self.assertEqual([s["peers"] for s in status["stages"]], [1, 0])


if __name__ == "__main__":
    unittest.main()
