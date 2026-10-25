"""Local front: layers [0, F) run in the client process, the chain starts at layer F with hidden states."""
import tempfile
import unittest
from unittest import mock

import torch

from commonnet import localfront, speculative as spec
from commonnet.bench.synth import OracleDrafter, SynthBlock, SynthClient
from commonnet.client import BlockEndpoint
from commonnet.localfront import FRONT_NAME, LocalFront
from commonnet.mesh.manifest import Manifest
from commonnet.mesh.placement import NoRoute, best_route
from commonnet.net.server import BlockServer
from commonnet.net.transport import HttpTransport

LAYERS, F, HIDDEN, VOCAB = 6, 2, 64, 256
PROMPT = [3, 10, 17, 24, 31, 38, 45]
N = 24


def synth(a, b):
    return SynthBlock(a, b, io_layers=LAYERS, hidden=HIDDEN, vocab=VOCAB, wire_dtype="float32")


class Rig:
    """Peers for [0,2) (only the no-front baseline uses it), [2,4), [4,6), plus a twin of [2,4)."""

    def __init__(self):
        self.base = BlockServer(synth(0, 2), allow_chain_private=True).start()
        self.a = BlockServer(synth(2, 4), allow_chain_private=True).start()
        self.a2 = BlockServer(synth(2, 4), allow_chain_private=True).start()
        self.b = BlockServer(synth(4, 6), allow_chain_private=True).start()
        self.front = LocalFront(synth(0, F), F)

    def stop(self):
        for s in (self.base, self.a, self.a2, self.b):
            s.stop()

    def run(self, front: bool, codec="int8_group", drafter=None, k=4, hook=None, replicas=False, bodies=None):
        if front:
            eps = [self.front.endpoint(), BlockEndpoint("hop1", 2, 4), BlockEndpoint("hop2", 4, 6)]
            urls = {"hop1": self.a.url, "hop2": self.b.url}
        else:
            eps = [BlockEndpoint("hop0", 0, 2), BlockEndpoint("hop1", 2, 4), BlockEndpoint("hop2", 4, 6)]
            urls = {"hop0": self.base.url, "hop1": self.a.url, "hop2": self.b.url}
        tr = HttpTransport(urls, activation_codec=codec, retries=0, sleep=lambda _: None,
                           replicas={"hop1": [self.a2.url]} if replicas else None)
        if front:
            tr.local_front = self.front
        if bodies is not None:
            send = tr._chain_send
            tr._chain_send = lambda state, body: (bodies.append(body), send(state, body))[1]
        client = SynthClient(tr, layers=LAYERS, hidden=HIDDEN, vocab=VOCAB, dtype=torch.float32)
        try:
            if drafter is not None:
                return list(spec.generate_stream(client, eps, PROMPT, N, drafter=drafter, k=k, adaptive=False)), tr
            return list(client.generate_stream(eps, PROMPT, N, before_step=hook)), tr
        finally:
            pass


class LocalFrontTests(unittest.TestCase):
    def setUp(self):
        self.rig = Rig()
        self.addCleanup(self.rig.stop)

    def test_greedy_output_is_identical_with_and_without_the_front(self):
        for codec in ("float", "int8_group"):
            with self.subTest(codec=codec):
                off, _ = self.rig.run(False, codec)
                on, _ = self.rig.run(True, codec)
                self.assertEqual(on, off)
                self.assertEqual(len(on), N)

    def test_speculative_output_is_identical_and_rolls_back_the_front(self):
        truth, _ = self.rig.run(False, "float")
        cuts = []
        real = self.rig.front.truncate
        self.rig.front.truncate = lambda s, n: (cuts.append(n), real(s, n))[1]
        for codec in ("float", "int8_group"):
            with self.subTest(codec=codec):
                off, _ = self.rig.run(False, codec)
                on, _ = self.rig.run(True, codec, drafter=OracleDrafter(len(PROMPT), truth, 0.6, vocab=VOCAB, seed=5))
                self.assertEqual(on, off)
        self.assertTrue(cuts, "no rollback reached the local front")

    def test_session_close_frees_the_local_kv(self):
        self.rig.run(True)
        self.assertEqual(self.rig.front.block._sessions, {})

    def test_front_kv_follows_the_truncate_semantics(self):
        f = self.rig.front
        f.forward("s", [1, 2, 3, 4, 5], 0)
        f.truncate("s", 3)
        self.assertEqual(f.block._sessions["s"]["pos"], 3)
        a = f.forward("s", [9], 3)
        g = LocalFront(synth(0, F), F)
        g.forward("t", [1, 2, 3], 0)
        self.assertTrue(torch.equal(a, g.forward("t", [9], 3)))

    def test_peers_get_hidden_states_and_the_layers_before_f_are_never_contacted(self):
        bodies = []
        calls = []
        real = self.rig.base.block.forward
        self.rig.base.block.forward = lambda *a: (calls.append(a), real(*a))[1]
        on, tr = self.rig.run(True, bodies=bodies)
        self.assertEqual(calls, [])                      # the [0,2) peer was never asked
        self.assertTrue(bodies and all("hidden" in b and "ids" not in b and b["layers"] == [F, 4] for b in bodies))
        self.assertEqual(self.rig.base.block._sessions, {})

    def test_failover_of_a_remote_peer_replays_hidden_states_exactly(self):
        off, _ = self.rig.run(False)
        killed = []

        def hook(step):
            if step == 6 and not killed:
                killed.append(1)
                self.rig.a.stop()
        on, tr = self.rig.run(True, hook=hook, replicas=True)
        self.assertEqual(killed, [1])
        self.assertEqual(on, off)

    def test_without_a_front_nothing_changes(self):
        tr = HttpTransport({"hop0": self.rig.base.url}, retries=0)
        self.assertIsNone(tr.local_front)
        self.assertEqual(tr._split_front([BlockEndpoint("hop0", 0, 2)])[0], None)

    def test_a_route_naming_a_front_the_transport_lacks_is_refused(self):
        tr = HttpTransport({"hop1": self.rig.a.url}, retries=0)
        self.assertFalse(tr.chain_route([BlockEndpoint(FRONT_NAME, 0, F), BlockEndpoint("hop1", 2, 4)]))


MANIFEST = Manifest.from_dict({
    "id": "t", "model": {"name": "T", "num_layers": 36, "repos": {"mlx": "a/b"}},
    "boundaries": [0, 9, 18, 27, 36], "gib_per_layer": {"mlx": 0.125}, "fixed_gib": 0.3})


class Settings(unittest.TestCase):
    def test_parse(self):
        self.assertIsNone(localfront.parse_setting(None))
        self.assertIsNone(localfront.parse_setting("off"))
        self.assertEqual(localfront.parse_setting("auto"), "auto")
        self.assertEqual(localfront.parse_setting("on"), "on")
        self.assertEqual(localfront.parse_setting("9"), 9)
        with self.assertRaises(localfront.FrontConfigError):
            localfront.parse_setting("nine")

    def test_flag_beats_env_and_default_is_off(self):
        self.assertIsNone(localfront.setting_from(None, {}))
        self.assertEqual(localfront.setting_from(None, {localfront.ENV: "9"}), 9)
        self.assertEqual(localfront.setting_from("auto", {localfront.ENV: "9"}), "auto")

    def test_n_must_be_a_stage_boundary(self):
        self.assertEqual(localfront.front_layers("on", MANIFEST), 9)
        self.assertEqual(localfront.front_layers("auto", MANIFEST), 9)
        self.assertEqual(localfront.front_layers(18, MANIFEST), 18)
        for bad in (10, 36, 0):
            with self.assertRaises(localfront.FrontConfigError):
                localfront.front_layers(bad, MANIFEST)

    def test_auto_decision(self):
        self.assertTrue(localfront.decide_auto(12.0, 4.0, 1.4).enabled)                 # fast, memory free
        slow = localfront.decide_auto(40.0, 4.0, 1.4)
        self.assertFalse(slow.enabled)
        self.assertIn("ms per decode token", slow.reason)
        tight = localfront.decide_auto(5.0, 2.0, 1.4)                                    # 2.0 < 1.5 * 1.4
        self.assertFalse(tight.enabled)
        self.assertIn("free", tight.reason)
        self.assertTrue(localfront.decide_auto(24.9, 2.1, 1.4).enabled)       # just under both limits is on
        self.assertFalse(localfront.decide_auto(25.0, 9.0, 1.4).enabled)                 # the limit itself is not under it

    def test_benchmark_runs_on_a_front(self):
        self.assertGreaterEqual(localfront.benchmark_ms(LocalFront(synth(0, F), F)), 0.0)


class NodeFront(unittest.TestCase):
    def node(self, setting, **kw):
        from commonnet import node
        from commonnet.mesh.records import Identity
        mf = Manifest.from_dict({"id": "t", "model": {"name": "T", "num_layers": 6, "repos": {"stub": "a/b"}},
                                 "boundaries": [0, 2, 4, 6], "gib_per_layer": {"stub": 0.1}, "fixed_gib": 0.0})
        d = tempfile.mkdtemp()
        n = node.Node(mf, [], Identity.generate(), node.Hardware("stub", 8.0, "stub"), stub=True, tunnel=False,
                      block_builder=lambda w, m, hw: synth(*w), local_front=setting, log=lambda *_: None)
        n._free_gib = lambda: kw.get("free", 16.0)
        n._bench_front = lambda front: kw.get("ms", 5.0)
        return n

    def test_off_by_default(self):
        n = self.node(None)
        net = mock.Mock(local_front=None)
        n._init_local_front(net)
        self.assertIsNone(n.local_front)
        self.assertIsNone(net.local_front)

    def test_on_loads_the_first_stage_once(self):
        n = self.node("on")
        net = mock.Mock()
        n._init_local_front(net)
        self.assertEqual(n.local_front.end, 2)
        self.assertIs(net.local_front, n.local_front)

    def test_auto_fast_slow_and_little_memory(self):
        for kw, enabled in (({"ms": 8.0}, True), ({"ms": 60.0}, False), ({"free": 0.1}, False)):
            with self.subTest(**kw):
                n = self.node("auto", **kw)
                n._init_local_front(mock.Mock())
                self.assertEqual(n.local_front is not None, enabled)

    def test_a_node_serving_from_layer_zero_reuses_its_block(self):
        n = self.node("on")
        n._block, n.window = synth(0, 2), (0, 2)
        n._init_local_front(mock.Mock())
        self.assertIs(n.local_front.block, n._block)
        self.assertFalse(n.local_front.owned)

    def test_a_node_that_serves_everything_has_no_use_for_it(self):
        n = self.node("on")
        n._block, n.window = synth(0, 6), (0, 6)
        n._init_local_front(mock.Mock())
        self.assertIsNone(n.local_front)


class Routing(unittest.TestCase):
    def test_route_may_start_at_f_and_falls_back_when_no_peer_starts_there(self):
        from commonnet.mesh.records import Identity

        def peer(window):
            return Identity.generate().sign_record({"network": "t", "url": "http://127.0.0.1:1", "token": "t",
                                                    "window": window, "backend": "mlx", "state": "up", "gib": 8.0}, now=10_000.0)
        live = [peer([9, 27]), peer([27, 36]), peer([0, 18])]
        route = best_route(MANIFEST, live, None, 10_000.0, start=9)
        self.assertEqual([h.window for h in route], [(9, 27), (27, 36)])
        with self.assertRaises(NoRoute):
            best_route(MANIFEST, [peer([0, 18]), peer([18, 36])], None, 10_000.0, start=9)


if __name__ == "__main__":
    unittest.main()
