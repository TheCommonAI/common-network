import json
import os
import random
import socket
import tempfile
import threading
import time
import unittest

from commonnet.dht import DhtNode, load_bootstrap
from commonnet.dht import ids
from commonnet.dht.bootstrap import ENV_VAR, parse_addr
from commonnet.dht.node import MAX_PACKET
from commonnet.dht.store import MAX_RECORD_BYTES, StoreError, ValueStore, check_record
from commonnet.dht.table import Contact, RoutingTable
from commonnet.mesh.records import LIVE_SECONDS, Identity

LO = ("127.0.0.1", 0)


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def fields(network, **kw):
    return {"network": network, "url": "https://x.trycloudflare.com", "token": "secret-token", "window": [0, 8],
            "backend": "mlx", "state": "up", "gib": 12.0, **kw}


def make_node(clock, network="net-a", boot=(), **kw):
    kw.setdefault("rpc_timeout", 0.3)
    kw.setdefault("retries", 2)
    node = DhtNode(Identity.generate(), bind=LO, bootstrap=boot, network_id=network, clock=clock,
                   now_rng=random.Random(1), **kw)
    node.start()
    return node


def dht_threads():
    return [t for t in threading.enumerate() if t.name.startswith("dht-")]


def cluster(n, clock, network="net-a", seeds=2, **kw):
    nodes = [make_node(clock, network, **kw)]
    for i in range(1, n):
        node = make_node(clock, network, **kw)
        node.bootstrap([x.address for x in nodes[:seeds]])
        nodes.append(node)
    return nodes


def announce_all(nodes, clock):
    recs = {}
    for node in nodes:
        rec = node.identity.sign_record(fields(node.network_id), now=clock())
        assert node.announce(rec) > 0
        recs[node.peer] = rec
    return recs


class Base(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.nodes = []

    def tearDown(self):
        for n in self.nodes:
            n.stop()
        self.assertEqual(dht_threads(), [])

    def spawn(self, *a, **kw):
        nodes = cluster(*a, **kw)
        self.nodes += nodes
        return nodes


class ConvergenceTests(Base):
    def test_24_nodes_find_each_other(self):
        t0 = time.time()
        nodes = self.spawn(24, self.clock)
        recs = announce_all(nodes, self.clock)
        boot_s = time.time() - t0
        t1 = time.time()
        before = sum(n.stats["lookups"] for n in nodes)
        for node in nodes:
            got = {r["peer"]: r for r in node.find_peers()}
            self.assertEqual(set(got), set(recs))
            self.assertEqual(got[nodes[3].peer], recs[nodes[3].peer])
        print(f"\n[dht] 24 nodes: bootstrap+announce {boot_s:.1f}s; 24 find_peers {time.time() - t1:.1f}s; "
              f"lookups {sum(n.stats['lookups'] for n in nodes) - before} (one per find_peers)")
        for n in nodes:
            n.table.check_invariants()

    def test_lookup_peer_by_id(self):
        nodes = self.spawn(12, self.clock)
        recs = announce_all(nodes, self.clock)
        far = nodes[-1]
        self.assertEqual(far.lookup_peer(nodes[1].peer), recs[nodes[1].peer])
        self.assertIsNone(far.lookup_peer("0" * 32))

    def test_newest_seq_wins(self):
        nodes = self.spawn(6, self.clock)
        a = nodes[2]
        old = a.identity.sign_record(fields("net-a", gib=1.0), now=self.clock())
        new = a.identity.sign_record(fields("net-a", gib=2.0), now=self.clock())
        a.announce(new)
        a.announce(old)  # arrives later but has the lower seq
        got = [r for r in nodes[5].find_peers() if r["peer"] == a.peer]
        self.assertEqual(got, [new])

    def test_first_node_alone(self):
        solo = self.spawn(1, self.clock)[0]
        rec = solo.identity.sign_record(fields("net-a"), now=self.clock())
        self.assertEqual(solo.announce(rec), 2)  # held locally under both keys
        self.assertEqual(solo.find_peers(), [rec])

    def test_loss_20_percent_still_converges(self):
        nodes = self.spawn(12, self.clock, retries=6, rpc_timeout=0.06)
        rng = random.Random(7)
        for n in nodes:
            real = n._sendto

            def lossy(data, addr, real=real):
                if rng.random() >= 0.2:
                    real(data, addr)
            n._sendto = lossy
        recs = announce_all(nodes, self.clock)
        t0 = time.time()
        for node in nodes[::3]:
            self.assertEqual({r["peer"] for r in node.find_peers()}, set(recs))
        print(f"\n[dht] 12 nodes, 20% packet loss: {len(nodes[::3])} find_peers all complete in {time.time() - t0:.1f}s")

    def test_bootstrap_via_constructor_in_background(self):
        a = self.spawn(1, self.clock)[0]
        b = make_node(self.clock, boot=[a.address])
        self.nodes.append(b)
        deadline = time.time() + 5
        while time.time() < deadline and not len(b.table):
            time.sleep(0.05)
        self.assertEqual(len(b.table), 1)


class ExpiryTests(Base):
    def test_dead_node_stops_being_returned(self):
        nodes = self.spawn(8, self.clock, rpc_timeout=0.1)
        announce_all(nodes, self.clock)
        dead = nodes.pop()
        dead.stop()
        self.clock.t += LIVE_SECONDS * 0.5
        for n in nodes:  # the survivors re-announce, the dead one cannot
            n.announce(n.identity.sign_record(fields("net-a"), now=self.clock()))
        self.assertIn(dead.peer, {r["peer"] for r in nodes[0].find_peers()})
        self.clock.t += LIVE_SECONDS * 0.8  # dead record is now older than LIVE_SECONDS
        got = {r["peer"] for r in nodes[0].find_peers()}
        self.assertNotIn(dead.peer, got)
        self.assertEqual(got, {n.peer for n in nodes})

    def test_everything_expires_without_refresh(self):
        nodes = self.spawn(5, self.clock)
        announce_all(nodes, self.clock)
        self.clock.t += LIVE_SECONDS + 1
        self.assertEqual(nodes[0].find_peers(), [])
        for n in nodes:
            n.run_maintenance()
            self.assertEqual(n.store.count(), 0)

    def test_leaving_record_is_not_listed(self):
        nodes = self.spawn(4, self.clock)
        rec = nodes[1].identity.sign_record(fields("net-a", state="leaving"), now=self.clock())
        nodes[1].announce(rec)
        self.assertNotIn(nodes[1].peer, {r["peer"] for r in nodes[0].find_peers()})


class RejectionTests(Base):
    def setUp(self):
        super().setUp()
        self.nodes = self.spawn(3, self.clock)
        self.target = self.nodes[0]
        self.who = Identity.generate()
        self.key = ids.network_key("net-a")

    def store(self, record, key=None):
        return self.target.store.put(key or self.key, record, "9.9.9.9")

    def test_good_record_accepted(self):
        self.assertTrue(self.store(self.who.sign_record(fields("net-a"), now=self.clock())))

    def test_bad_signature(self):
        rec = self.who.sign_record(fields("net-a"), now=self.clock())
        with self.assertRaisesRegex(StoreError, "signature"):
            self.store({**rec, "gib": 99.0})

    def test_stale_and_future(self):
        with self.assertRaisesRegex(StoreError, "stale"):
            self.store(self.who.sign_record(fields("net-a"), now=self.clock() - LIVE_SECONDS - 5, ) | {})
        old = self.who.sign_record(fields("net-a", ts=self.clock() - LIVE_SECONDS - 5), now=self.clock())
        with self.assertRaisesRegex(StoreError, "stale"):
            self.store(old)
        future = self.who.sign_record(fields("net-a", ts=self.clock() + 3600), now=self.clock())
        with self.assertRaisesRegex(StoreError, "future"):
            self.store(future)

    def test_oversized(self):
        big = self.who.sign_record(fields("net-a", token="t" * 256, url="https://" + "a" * 450), now=self.clock())
        self.assertGreater(len(json.dumps(big, separators=(",", ":"))), MAX_RECORD_BYTES)
        with self.assertRaisesRegex(StoreError, "large"):
            self.store(big)

    def test_wrong_network_key(self):
        rec = self.who.sign_record(fields("net-b"), now=self.clock())
        with self.assertRaisesRegex(StoreError, "key"):
            self.store(rec)  # net-b record under net-a's key
        with self.assertRaisesRegex(StoreError, "key"):
            self.store(rec, key=ids.peer_key("f" * 32))

    def test_announce_refuses_foreign_network_and_others_records(self):
        with self.assertRaises(ValueError):
            self.target.announce(self.target.identity.sign_record(fields("net-b"), now=self.clock()))
        with self.assertRaises(ValueError):
            self.target.announce(self.who.sign_record(fields("net-a"), now=self.clock()))

    def test_typical_record_fits_a_packet(self):
        rec = self.who.sign_record(fields("net-a"), now=self.clock())
        self.assertLess(len(json.dumps(rec, separators=(",", ":"))), 700)

    def test_caps(self):
        store = ValueStore(self.clock, max_per_key=3, max_per_source=5)
        for _ in range(3):
            store.put(self.key, Identity.generate().sign_record(fields("net-a"), now=self.clock()), "1.1.1.1")
        with self.assertRaisesRegex(StoreError, "full"):
            store.put(self.key, Identity.generate().sign_record(fields("net-a"), now=self.clock()), "1.1.1.1")
        store = ValueStore(self.clock, max_per_key=100, max_per_source=2)
        for _ in range(2):
            store.put(self.key, Identity.generate().sign_record(fields("net-a"), now=self.clock()), "2.2.2.2")
        with self.assertRaisesRegex(StoreError, "source"):
            store.put(self.key, Identity.generate().sign_record(fields("net-a"), now=self.clock()), "2.2.2.2")
        store.put(self.key, Identity.generate().sign_record(fields("net-a"), now=self.clock()), "3.3.3.3")

    def test_store_over_the_wire_needs_a_token(self):
        rec = self.who.sign_record(fields("net-a"), now=self.clock())
        client = self.nodes[1]
        client.bootstrap([self.target.address])
        resp = client._rpc_many([(self.target.address, None,
                                  {"t": "st", "k": self.key.hex(), "tok": "forged", "r": rec})])[0]
        self.assertEqual((resp["t"], resp["why"]), ("err", "token"))
        self.assertEqual(self.target.store.get(self.key), [])
        resp = client._rpc_many([(self.target.address, None,
                                  {"t": "st", "k": self.key.hex(), "tok": client._fresh_token(self.target.address),
                                   "r": {**rec, "gib": 5.0}})])[0]
        self.assertEqual(resp["t"], "err")

    def test_two_networks_do_not_mix(self):
        a = self.spawn(8, self.clock, network="net-a", seeds=1)
        b = [make_node(self.clock, "net-b") for _ in range(8)]
        self.nodes += b
        for n in b:
            n.bootstrap([a[0].address])  # one DHT, two networks
        ra, rb = announce_all(a, self.clock), announce_all(b, self.clock)
        self.assertEqual({r["peer"] for r in a[3].find_peers()}, set(ra))
        self.assertEqual({r["peer"] for r in b[3].find_peers()}, set(rb))
        self.assertTrue(set(ra).isdisjoint(rb))


class MalformedTests(Base):
    def test_garbage_is_ignored(self):
        node = self.spawn(2, self.clock)[0]
        node.bootstrap([self.nodes[1].address])
        raw = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        good_id = "a" * 32
        junk = [b"", b"\x00\xff\xfe", b"{", b"[]", b"null", b'{"t":"fn"}', b'{"t":"zzz","n":"0000000000000000","id":"%s"}' % good_id.encode(),
                json.dumps({"t": "fn", "n": "0" * 16, "id": good_id, "k": "zz"}).encode(),
                json.dumps({"t": "fv", "n": "0" * 16, "id": good_id, "k": "0" * 64, "a": 5}).encode(),
                json.dumps({"t": "st", "n": "0" * 16, "id": good_id, "k": "0" * 64, "tok": 1, "r": 5}).encode(),
                json.dumps({"t": "pong", "n": "0" * 16, "id": good_id}).encode(), os.urandom(1400), os.urandom(5000),
                b'{"t":"fn","n":"' + b"1" * 5000 + b'"}']
        for j in junk:
            raw.sendto(j, node.address)
        raw.close()
        time.sleep(0.3)
        self.assertGreater(node.stats["bad"], 5)
        self.assertEqual(len(node.find_peers()), 0)  # still alive and answering
        self.assertEqual(len(dht_threads()), 4)

    def test_replies_never_exceed_a_packet_or_the_request(self):
        nodes = self.spawn(25, self.clock)
        announce_all(nodes, self.clock)
        node = nodes[0]
        sizes = []
        real = node._sendto
        node._sendto = lambda data, addr: (sizes.append(len(data)), real(data, addr))
        raw = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        raw.settimeout(1)
        for req in ({"t": "ping"}, {"t": "fn", "k": "0" * 64}, {"t": "fv", "k": ids.network_key("net-a").hex(), "a": ""}):
            msg = json.dumps({**req, "n": "1" * 16, "id": "b" * 32}).encode()
            raw.sendto(msg, node.address)
            reply = raw.recv(65535)
            self.assertLessEqual(len(reply), max(300, len(msg)))
        raw.close()
        self.assertTrue(all(s <= MAX_PACKET for s in sizes))

    def test_unverified_sender_does_not_enter_table(self):
        node = self.spawn(1, self.clock)[0]
        liar = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        liar.settimeout(0.3)
        liar.sendto(json.dumps({"t": "ping", "n": "2" * 16, "id": "c" * 32}).encode(), node.address)
        liar.recv(65535)  # gets a pong, but never signs a reply to our verification ping
        time.sleep(0.6)
        self.assertEqual(len(node.table), 0)
        liar.close()

    def test_forged_response_is_dropped(self):
        node = self.spawn(1, self.clock)[0]
        fake = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        fake.bind(LO)
        fake.settimeout(2)
        result = {}

        def ask():
            result["r"] = node._rpc_many([(fake.getsockname(), None, {"t": "ping"})])[0]
        t = threading.Thread(target=ask)
        t.start()
        data, addr = fake.recvfrom(2000)
        req = json.loads(data)
        mallory = Identity.generate()
        forged = {"t": "pong", "n": req["n"], "id": mallory.peer, "pub": "AAAA", "sig": "AAAA"}
        fake.sendto(json.dumps(forged).encode(), addr)
        t.join(10)
        fake.close()
        self.assertIsNone(result["r"])
        self.assertEqual(len(node.table), 0)


class TableTests(unittest.TestCase):
    def test_invariants_and_eviction(self):
        rng = random.Random(3)
        own = rng.getrandbits(256)
        table = RoutingTable(own, k=4)
        evict_seen = 0
        for i in range(500):
            cid = rng.getrandbits(256)
            if cid == own:
                continue
            ev = table.add(Contact(f"{i:032x}", cid, ("127.0.0.1", 1000 + i)), float(i))
            if ev is not None:
                evict_seen += 1
                if i % 2:
                    table.remove(ev.id)
        table.check_invariants()
        self.assertGreater(evict_seen, 0)
        target = rng.getrandbits(256)
        got = table.closest(target, 4)
        self.assertEqual([c.id ^ target for c in got], sorted(c.id ^ target for c in table.contacts())[:4])

    def test_own_id_never_added_and_refresh_moves_to_tail(self):
        table = RoutingTable(5, k=3)
        self.assertIsNone(table.add(Contact("a" * 32, 5, ("1.1.1.1", 1)), 0.0))
        self.assertEqual(len(table), 0)
        ids_ = [1 << 200 | i for i in range(3)]
        for i in ids_:
            table.add(Contact(f"{i:032x}", i, ("1.1.1.1", i % 65535 + 1)), 1.0)
        table.add(Contact("x" * 32, ids_[0], ("2.2.2.2", 2)), 2.0)
        self.assertEqual(table.get(ids_[0]).addr, ("2.2.2.2", 2))
        self.assertEqual(table.contacts()[-1].id, ids_[0])

    def test_dead_contacts_are_replaced_from_cache(self):
        table = RoutingTable(0, k=2)
        a, b, c = (Contact(f"{i:032x}", (1 << 100) + i, ("1.1.1.1", i + 1)) for i in (1, 2, 3))
        table.add(a, 0.0)
        table.add(b, 0.0)
        self.assertIs(table.add(c, 0.0), a)  # full: ping the oldest
        table.remove(a.id)
        self.assertEqual({x.id for x in table.contacts()}, {b.id, c.id})


class BootstrapTests(unittest.TestCase):
    def test_shipped_file_is_an_empty_list(self):
        from commonnet.dht.bootstrap import DEFAULT_FILE
        self.assertEqual(json.loads(DEFAULT_FILE.read_text()), [])
        self.assertEqual(load_bootstrap(env={}), [])

    def test_file_and_env(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(["1.2.3.4:5000", "bad", "1.2.3.4:5000", ["5.6.7.8", 6000]], f)
        try:
            got = load_bootstrap(f.name, env={ENV_VAR: "9.9.9.9:1, nonsense, 1.2.3.4:5000"})
        finally:
            os.unlink(f.name)
        self.assertEqual(got, [("1.2.3.4", 5000), ("5.6.7.8", 6000), ("9.9.9.9", 1)])
        self.assertEqual(load_bootstrap("/nonexistent.json", env={}), [])
        with self.assertRaises(ValueError):
            parse_addr("nohost")


class LifecycleTests(unittest.TestCase):
    def test_stop_leaves_no_threads_and_is_repeatable(self):
        before = len(threading.enumerate())
        nodes = [make_node(Clock()) for _ in range(5)]
        self.assertEqual(len(dht_threads()), 10)
        for n in nodes:
            n.stop()
            n.stop()
        self.assertEqual(dht_threads(), [])
        self.assertEqual(len(threading.enumerate()), before)

    def test_cli_list_prints_no_token(self):
        import contextlib
        import io
        from commonnet.dht.__main__ import main
        clock = Clock(time.time())
        seed = make_node(clock, "cli-net")
        try:
            rec = seed.identity.sign_record(fields("cli-net", ts=time.time()), now=time.time())
            seed.announce(rec)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["--bootstrap", f"127.0.0.1:{seed.address[1]}", "--network", "cli-net", "--list",
                             "--bind", "127.0.0.1:0"])
        finally:
            seed.stop()
        self.assertEqual(code, 0)
        self.assertIn(seed.peer, out.getvalue())
        self.assertNotIn("secret-token", out.getvalue())


if __name__ == "__main__":
    unittest.main()
