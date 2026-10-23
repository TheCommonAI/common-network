import json
import random
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from commonnet.mesh import gossip
from commonnet.mesh.gossip import GossipError, Gossiper, PeerTable, exchange, fetch_manifest, fetch_peers, handle_gossip
from commonnet.mesh.manifest import Manifest
from commonnet.mesh.records import Identity

MANIFEST = Manifest.from_dict({
    "id": "g", "model": {"name": "T", "num_layers": 36, "repos": {}}, "boundaries": [0, 9, 18, 27, 36],
    "gib_per_layer": {"mlx": 0.125}, "fixed_gib": 0.3})


class Clock:
    def __init__(self, t=100_000.0):
        self.t = t

    def __call__(self):
        return self.t


def record(who, url, window=(0, 9), state="up", clock=None, network="g"):
    return who.sign_record({"network": network, "url": url, "token": "t", "window": list(window) if window else None,
                            "backend": "mlx", "state": state, "gib": 8.0}, now=clock() if clock else None)


class TableTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.table = PeerTable("g", now=self.clock)
        self.who = Identity.generate()

    def test_merge_keeps_the_newer_seq_only(self):
        a = record(self.who, "http://127.0.0.1:1", clock=self.clock)
        b = record(self.who, "http://127.0.0.1:2", clock=self.clock)
        self.assertTrue(self.table.merge(a))
        self.assertTrue(self.table.merge(b))
        self.assertFalse(self.table.merge(a))
        self.assertEqual(self.table.get(self.who.peer)["url"], "http://127.0.0.1:2")

    def test_tampered_and_foreign_network_records_are_dropped(self):
        r = record(self.who, "http://127.0.0.1:1", clock=self.clock)
        self.assertFalse(self.table.merge({**r, "window": [0, 36]}))
        self.assertFalse(self.table.merge(record(self.who, "http://127.0.0.1:1", clock=self.clock, network="other")))
        self.assertEqual(self.table.live(), [])

    def test_stale_future_and_ancient_records(self):
        self.assertFalse(self.table.merge(self.who.sign_record(
            {"network": "g", "url": "http://x", "token": "t", "window": [0, 9], "backend": "mlx", "state": "up", "gib": 1},
            now=self.clock() + 500)))
        self.assertFalse(self.table.merge(self.who.sign_record(
            {"network": "g", "url": "http://x", "token": "t", "window": [0, 9], "backend": "mlx", "state": "up", "gib": 1},
            now=self.clock() - 700)))

    def test_a_peer_that_stops_republishing_ages_out(self):
        self.table.merge(record(self.who, "http://127.0.0.1:1", clock=self.clock))
        self.assertEqual(len(self.table.live()), 1)
        self.clock.t += 61
        self.assertEqual(self.table.live(), [])

    def test_a_tombstone_removes_a_peer_at_once_and_is_passed_on(self):
        self.table.merge(record(self.who, "http://127.0.0.1:1", clock=self.clock))
        self.table.merge(record(self.who, "http://127.0.0.1:1", state="leaving", window=None, clock=self.clock))
        self.assertEqual(self.table.live(), [])
        self.assertEqual([r["state"] for r in self.table.export()], ["leaving"])
        self.clock.t += 301
        self.assertEqual(self.table.export(), [])

    def test_export_drops_stale_records_and_respects_limit(self):
        for i in range(5):
            self.table.merge(record(Identity.generate(), f"http://127.0.0.1:{i + 1}", clock=self.clock))
        self.assertEqual(len(self.table.export(limit=3)), 3)
        self.clock.t += 90
        self.assertEqual(self.table.export(), [])

    def test_sample_excludes_and_needs_a_url(self):
        mine = record(self.who, "http://127.0.0.1:1", clock=self.clock)
        loading = record(Identity.generate(), None, window=None, state="loading", clock=self.clock)
        self.table.merge(mine)
        self.table.merge(loading)
        self.assertEqual(self.table.sample(5), [mine])
        self.assertEqual(self.table.sample(5, exclude={mine["peer"]}), [])

    def test_handle_gossip_merges_and_answers_with_ours(self):
        other = Identity.generate()
        self.table.merge(record(self.who, "http://127.0.0.1:1", clock=self.clock))
        reply = handle_gossip(self.table, {"records": [record(other, "http://127.0.0.1:2", clock=self.clock)]})
        self.assertEqual(len(reply["records"]), 2)
        self.assertEqual(len(self.table.live()), 2)
        self.assertEqual(len(handle_gossip(self.table, "junk")["records"]), 2)


class Convergence(unittest.TestCase):
    def test_six_peers_that_each_know_only_a_seed_all_find_each_other(self):
        clock = Clock()
        n = 6
        whos = [Identity.generate() for _ in range(n)]
        urls = [f"http://peer{i}.test" for i in range(n)]
        tables = {urls[i]: PeerTable("g", now=clock) for i in range(n)}

        def fake_exchange(table, url, timeout):
            if url not in tables or tables[url] is None:
                raise GossipError("down")
            reply = handle_gossip(tables[url], {"records": table.export()})
            return table.merge_many(reply["records"])

        windows = [(0, 9), (0, 9), (9, 18), (18, 27), (27, 36), (27, 36)]
        gossipers = []
        for i in range(n):
            gossipers.append(Gossiper(
                tables[urls[i]], MANIFEST,
                (lambda i=i: record(whos[i], urls[i], windows[i], clock=clock)),
                seeds=[urls[0]], exchange_fn=fake_exchange, rng=random.Random(i)))
        for _ in range(8):
            for g in gossipers:
                g.run_round()
        for i in range(n):
            self.assertEqual(len(tables[urls[i]].live()), n, f"peer {i} sees {len(tables[urls[i]].live())}")

        # peer 5 dies silently: after a minute without a fresh record it is gone everywhere
        tables[urls[5]] = None
        for _ in range(6):
            clock.t += 15
            for g in gossipers[:5]:
                g.run_round()
        for i in range(5):
            self.assertEqual(len(tables[urls[i]].live()), 5)

    def test_a_peer_that_keeps_failing_is_skipped_but_the_seed_is_still_tried(self):
        clock = Clock()
        who, dead = Identity.generate(), Identity.generate()
        table = PeerTable("g", now=clock)
        table.merge(record(dead, "http://dead.test", clock=clock))
        calls = []

        def fake_exchange(t, url, timeout):
            calls.append(url)
            raise GossipError("down")

        g = Gossiper(table, MANIFEST, lambda: record(who, "http://me.test", clock=clock), seeds=["http://seed.test"],
                     exchange_fn=fake_exchange, rng=random.Random(1))
        for _ in range(6):
            g.run_round()
        self.assertEqual(calls.count("http://dead.test"), gossip.FAILURES_BEFORE_SKIP)
        self.assertGreater(calls.count("http://seed.test"), gossip.FAILURES_BEFORE_SKIP)


class Threaded(unittest.TestCase):
    def test_start_runs_rounds_until_stop(self):
        who = Identity.generate()
        table = PeerTable("g")
        calls = []
        seen = threading.Event()

        def fake_exchange(t, url, timeout):
            calls.append(url)
            if len(calls) >= 3:
                seen.set()
            return 0

        g = Gossiper(table, MANIFEST, lambda: record(who, "http://me.test"), seeds=["http://seed.test"],
                     exchange_fn=fake_exchange, interval=(0.01, 0.02)).start()
        self.assertTrue(seen.wait(5))
        g.stop()
        count = len(calls)
        threading.Event().wait(0.1)
        self.assertEqual(len(calls), count)  # stopped
        self.assertEqual(len(table.live()), 1)  # it published its own record


class Peer(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/peers":
            self._send(gossip.peers_body(self.server.table))
        elif self.path == "/manifest":
            self._send(gossip.manifest_body(MANIFEST))
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/gossip":
            self._send(handle_gossip(self.server.table, body))
        else:
            self.send_error(404)


def start_peer(table):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Peer)
    server.table = table
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


class RealSockets(unittest.TestCase):
    def test_two_peers_swap_records_over_http(self):
        a_who, b_who = Identity.generate(), Identity.generate()
        a, b = PeerTable("g"), PeerTable("g")
        sa, ua = start_peer(a)
        sb, ub = start_peer(b)
        self.addCleanup(lambda: (sa.shutdown(), sa.server_close(), sb.shutdown(), sb.server_close()))
        a.merge(record(a_who, ua, (0, 9)))
        b.merge(record(b_who, ub, (9, 18)))
        self.assertEqual(exchange(a, ub), 1)  # a learned b
        self.assertEqual({r["peer"] for r in a.live()}, {a_who.peer, b_who.peer})
        self.assertEqual({r["peer"] for r in b.live()}, {a_who.peer, b_who.peer})  # and b learned a from the push
        self.assertEqual(exchange(a, ub), 0)  # nothing new the second time

    def test_pull_only_and_manifest(self):
        who = Identity.generate()
        server_table, client_table = PeerTable("g"), PeerTable("g")
        s, url = start_peer(server_table)
        self.addCleanup(lambda: (s.shutdown(), s.server_close()))
        server_table.merge(record(who, url))
        self.assertEqual(fetch_peers(client_table, url), 1)
        self.assertEqual(fetch_manifest(url), MANIFEST)
        self.assertEqual(server_table.live()[0]["peer"], who.peer)

    def test_errors_are_gossip_errors(self):
        table = PeerTable("g")
        with self.assertRaises(GossipError):
            exchange(table, "http://127.0.0.1:9", timeout=1)  # nothing listens on the discard port
        with self.assertRaises(GossipError):
            exchange(table, "ftp://nope")


if __name__ == "__main__":
    unittest.main()
