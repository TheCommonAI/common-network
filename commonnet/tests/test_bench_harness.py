"""The benchmark harness measures what it claims to: injected delay shows up in the network column, injected
compute in the compute column, and the overhead with neither is small. Real node processes, real HTTP, tiny model."""
import unittest

from commonnet.bench import harness as h
from commonnet.bench.proxy import DelayProxy
from commonnet.bench.synth import SynthBlock, manifest_dict
from commonnet.net.protocol import Tensor

import numpy as np

KW = dict(hidden=64, prompt_tokens=4, new_tokens=8, repeats=1)


def tensor(arr):
    return Tensor(arr.shape, "float16", arr.astype(np.float16).tobytes())


class SynthBlockTest(unittest.TestCase):
    def test_weights_are_deterministic_from_the_seed(self):
        x = tensor(np.random.default_rng(0).standard_normal((1, 3, 32)))
        outs = []
        for seed in (1, 1, 2):
            b = SynthBlock(0, 2, seed=seed)
            b.open("s")
            outs.append(b.forward("s", x, 0).data)
        self.assertEqual(outs[0], outs[1])
        self.assertNotEqual(outs[0], outs[2])

    def test_output_depends_on_history_and_chunked_prefill_matches_one_shot(self):
        x = np.random.default_rng(1).standard_normal((1, 5, 32))
        whole = SynthBlock(0, 3)
        whole.open("s")
        full = np.frombuffer(whole.forward("s", tensor(x), 0).data, np.float16).reshape(1, 5, 32)
        parts = SynthBlock(0, 3)
        parts.open("s")
        a = parts.forward("s", tensor(x[:, :2]), 0)
        b = parts.forward("s", tensor(x[:, 2:]), 2)
        joined = np.concatenate([np.frombuffer(t.data, np.float16).reshape(t.shape) for t in (a, b)], axis=1)
        np.testing.assert_allclose(joined.astype(np.float32), full.astype(np.float32), atol=2e-2)
        fresh = SynthBlock(0, 3)
        fresh.open("s")
        last = fresh.forward("s", tensor(x[:, 4:]), 0)  # same last token, no history
        self.assertNotEqual(last.data, full[:, 4:].tobytes())

    def test_a_wrong_start_pos_is_refused(self):
        b = SynthBlock(0, 1)
        b.open("s")
        with self.assertRaises(ValueError):
            b.forward("s", tensor(np.zeros((1, 1, 8))), 3)

    def test_manifest_splits_evenly(self):
        self.assertEqual(manifest_dict(3, 12)["boundaries"], [0, 4, 8, 12])
        with self.assertRaises(ValueError):
            manifest_dict(5, 12)


class HarnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cluster = h.Cluster(2, 1, layers=2, layer_ms=0.0).start()

    @classmethod
    def tearDownClass(cls):
        cls.cluster.stop()

    def run_with(self, keepalive=True, **net):
        self.cluster.set_network(**{"rtt_ms": 0.0, "jitter_ms": 0.0, "loss": 0.0, **net})
        return h.scenario(self.cluster, keepalive=keepalive, **KW)

    def test_zero_delay_zero_compute_has_tiny_overhead(self):
        d = self.run_with()["decode"]
        self.assertLess(d["ms_mean"], 25.0)
        self.assertLess(d["compute_ms"], 2.0)
        self.assertLess(d["network_ms"], 20.0)
        self.assertEqual(d["hops"], 2)
        self.assertEqual(d["reuse_rate"], 1.0)
        self.assertGreater(d["bytes_per_step"], 0)

    def test_network_column_is_about_hops_times_rtt(self):
        d = self.run_with(rtt_ms=50.0)["decode"]
        self.assertGreater(d["network_ms"], 2 * 50.0 - 5)
        self.assertLess(d["network_ms"], 2 * 50.0 + 25.0)
        self.assertLess(d["compute_ms"], 2.0)

    def test_without_keepalive_every_call_pays_the_handshake(self):
        keep = self.run_with(rtt_ms=30.0)["decode"]
        fresh = self.run_with(rtt_ms=30.0, keepalive=False)["decode"]
        self.assertEqual(fresh["reuse_rate"], 0.0)
        self.assertGreater(fresh["network_ms"], keep["network_ms"] + 2 * 30.0 * 0.7)  # one extra RTT per hop

    def test_proxy_chunks_keep_order_and_loss_adds_a_timeout(self):
        import socket, threading
        srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
        def echo():
            c, _ = srv.accept()
            while (d := c.recv(65536)):
                c.sendall(d)
            c.close()
        threading.Thread(target=echo, daemon=True).start()
        p = DelayProxy(srv.getsockname()[1], rtt_ms=0.0)
        try:
            s = socket.create_connection(("127.0.0.1", p.port))
            payload = bytes(range(256)) * 400
            s.sendall(payload)
            got = b""
            while len(got) < len(payload):
                got += s.recv(65536)
            self.assertEqual(got, payload)
            p.loss = 1.0  # every chunk waits one retransmission timeout, each way
            import time
            t = time.perf_counter()
            s.sendall(b"x")
            s.recv(1)
            self.assertGreater((time.perf_counter() - t) * 1000, 350)
            s.close()
        finally:
            p.close()
            srv.close()


class ComputeColumnTest(unittest.TestCase):
    def test_compute_column_is_about_layers_times_layer_ms(self):
        with h.Cluster(2, 1, layers=2, layer_ms=40.0) as c:
            d = h.scenario(c, **KW)["decode"]
        self.assertGreater(d["compute_ms"], 2 * 40.0 - 2)
        self.assertLess(d["compute_ms"], 2 * 40.0 + 15.0)  # sleep() overshoots a little
        self.assertLess(d["network_ms"], 20.0)


class FailoverTest(unittest.TestCase):
    def test_killing_the_primary_mid_run_gives_the_same_tokens_and_a_measured_pause(self):
        r = h.failover_scenario(stages=2, layer_ms=1.0, rtt_ms=0.0, hidden=64, prompt_tokens=4, new_tokens=12,
                                kill_at=6)
        self.assertTrue(r["tokens_match_undisturbed_run"])
        self.assertGreaterEqual(r["failovers"], 1)
        self.assertGreater(r["pause_ms_over_steady"], 50.0)  # at least the retry backoff


if __name__ == "__main__":
    unittest.main()
