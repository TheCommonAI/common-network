"""Speculative decoding on the synthetic model: output equals plain greedy, rollback is exact, failover during a
round is invisible, a mismatched drafter tokenizer is refused, and an old server turns speculation off."""
import unittest

import torch

from commonnet.bench.synth import OracleDrafter, SynthBlock, SynthClient
from commonnet.client import BlockEndpoint
from commonnet.net.client import BlockClient, RemoteError
from commonnet.net.server import BlockServer, StubBlock
from commonnet.net.transport import HttpTransport
from commonnet import speculative as spec

LAYERS, HIDDEN, VOCAB = 4, 32, 256
EPS = [BlockEndpoint("a", 0, 2), BlockEndpoint("b", 2, 4)]
PROMPT = [(7 * i + 3) % VOCAB for i in range(9)]


def make_client(transport):
    return SynthClient(transport, layers=LAYERS, hidden=HIDDEN, vocab=VOCAB, dtype=torch.float32)


class Rig:
    """Two stages, two replicas each, all SynthBlocks behind real BlockServers."""

    def __init__(self):
        self.servers = {n: [BlockServer(SynthBlock(s, e), "127.0.0.1", 0).start() for _ in range(2)]
                        for n, s, e in (("a", 0, 2), ("b", 2, 4))}

    def transport(self):
        return HttpTransport({n: v[0].url for n, v in self.servers.items()},
                             replicas={n: [v[1].url] for n, v in self.servers.items()}, retries=0,
                             sleep=lambda _: None)

    def stop(self):
        for v in self.servers.values():
            for s in v:
                s.stop()


class SpecTest(unittest.TestCase):
    def setUp(self):
        self.rig = Rig()
        self.addCleanup(self.rig.stop)

    def plain(self, n=40):
        tr = self.rig.transport()
        return list(make_client(tr).generate_stream(EPS, PROMPT, n))

    def run_spec(self, accept, k, n=40, tr=None, adaptive=True, before=None):
        truth = self.plain(n + k + 2)
        tr = tr or self.rig.transport()
        stats = spec.SpecStats()
        drafter = OracleDrafter(len(PROMPT), truth, accept, vocab=VOCAB, seed=5)
        out = list(spec.generate_stream(make_client(tr), EPS, PROMPT, n, drafter=drafter, k=k, adaptive=adaptive,
                                        stats=stats))
        return out, truth[:n], stats, tr

    def test_equals_plain_across_acceptance_rates_and_k(self):
        for accept in (0.0, 0.5, 0.9, 1.0):
            for k in (1, 2, 4, 8):
                out, truth, stats, _ = self.run_spec(accept, k)
                self.assertEqual(out, truth, f"accept={accept} k={k}")
                self.assertEqual(stats.mode, "speculative")

    def test_perfect_drafter_takes_fewer_rounds_and_rolls_nothing_back(self):
        out, truth, stats, tr = self.run_spec(1.0, 8, adaptive=False)
        self.assertEqual(out, truth)
        self.assertLessEqual(stats.rounds, 5)
        self.assertEqual(stats.accepted, stats.drafted)

    def test_rollback_then_continue_is_exact(self):
        # A drafter that is always wrong: every round rolls back to 1 accepted position.
        out, truth, stats, _ = self.run_spec(0.0, 4)
        self.assertEqual(out, truth)
        self.assertGreater(stats.rounds, 20)

    def test_max_tokens_is_respected_exactly(self):
        for n in (1, 2, 7, 8, 9):
            out, truth, _, _ = self.run_spec(1.0, 8, n=n)
            self.assertEqual(len(out), n)
            self.assertEqual(out, truth)

    def test_failover_during_speculative_rounds_gives_identical_output(self):
        truth = self.plain(30)
        tr = self.rig.transport()
        client = make_client(tr)
        drafter = OracleDrafter(len(PROMPT), self.plain(40), 0.6, vocab=VOCAB, seed=9)
        out = []
        for i, t in enumerate(spec.generate_stream(client, EPS, PROMPT, 30, drafter=drafter, k=4)):
            out.append(t)
            if i == 10:   # kill the in-use replica of both stages mid-run, after rollbacks have happened
                for name in ("a", "b"):
                    self.rig.servers[name][0].stop()
                    for (n, idx), c in tr._clients.items():
                        if n == name and idx == 0:
                            c.close_connections()
        self.assertEqual(out, truth)
        self.assertEqual(len(tr.failovers), 2)

    def test_truncate_is_idempotent_and_history_replays_truncated(self):
        tr = self.rig.transport()
        ep = EPS[0]
        x = torch.randn(1, 6, HIDDEN, generator=torch.Generator().manual_seed(1))
        tr.forward(ep, "s", x, 0)
        tr.truncate(ep, "s", 4)
        tr.truncate(ep, "s", 4)
        y = torch.randn(1, 1, HIDDEN, generator=torch.Generator().manual_seed(2))
        a = tr.forward(ep, "s", y, 4)
        # a fresh session that never saw the cut tokens gives the same answer
        tr2 = self.rig.transport()
        tr2.forward(ep, "t", x[:, :4], 0)
        b = tr2.forward(ep, "t", y, 4)
        self.assertTrue(torch.allclose(a, b, atol=1e-5))
        # kill the server, fail over: the replay is the truncated history, so the next step still agrees
        self.rig.servers["a"][0].stop()
        for (n, idx), c in tr._clients.items():
            if idx == 0:
                c.close_connections()
        z = torch.randn(1, 1, HIDDEN, generator=torch.Generator().manual_seed(3))
        a2 = tr.forward(ep, "s", z, 5)
        b2 = tr2.forward(ep, "t", z, 5)
        self.assertTrue(torch.allclose(a2, b2, atol=1e-5))

    def test_plain_decoding_is_untouched_without_a_drafter_and_for_sampling(self):
        stats = spec.SpecStats()
        out = list(spec.generate_stream(make_client(self.rig.transport()), EPS, PROMPT, 12, stats=stats))
        self.assertEqual(out, self.plain(12))
        self.assertTrue(stats.mode.startswith("plain"))
        stats = spec.SpecStats()
        d = OracleDrafter(len(PROMPT), self.plain(20), 1.0, vocab=VOCAB)
        list(spec.generate_stream(make_client(self.rig.transport()), EPS, PROMPT, 5, drafter=d, temperature=0.8,
                                  stats=stats))
        self.assertIn("temperature", stats.mode)
        self.assertEqual(d.calls, 0)


class OldServer(unittest.TestCase):
    def test_an_old_server_is_detected_and_speculation_turns_off(self):
        srv = BlockServer(SynthBlock(0, 4), "127.0.0.1", 0).start()
        self.addCleanup(srv.stop)
        # Make it an old server: it has no /truncate path.
        import commonnet.net.server as server_mod
        orig = server_mod.POST_PATHS
        server_mod.POST_PATHS = frozenset(("/session", "/forward", "/close"))
        self.addCleanup(lambda: setattr(server_mod, "POST_PATHS", orig))
        ep = [BlockEndpoint("a", 0, 4)]
        tr = HttpTransport({"a": srv.url}, retries=0, sleep=lambda _: None)
        self.assertFalse(tr.supports_truncate(ep))
        stats = spec.SpecStats()
        out = list(spec.generate_stream(make_client(tr), ep, PROMPT, 6, drafter=OracleDrafter(len(PROMPT), [0] * 20, 1.0),
                                        stats=stats))
        self.assertEqual(len(out), 6)
        self.assertIn("old version", stats.mode)

    def test_new_server_is_detected_as_supporting_rollback(self):
        srv = BlockServer(SynthBlock(0, 4), "127.0.0.1", 0).start()
        self.addCleanup(srv.stop)
        tr = HttpTransport({"a": srv.url}, retries=0, sleep=lambda _: None)
        self.assertTrue(tr.supports_truncate([BlockEndpoint("a", 0, 4)]))

    def test_a_block_that_cannot_truncate_answers_501(self):
        class NoTruncate(StubBlock):
            truncate = None
        srv = BlockServer(NoTruncate(0, 4), "127.0.0.1", 0).start()
        self.addCleanup(srv.stop)
        c = BlockClient(token=None, retries=0)
        c.open(srv.url, "s")
        with self.assertRaises(RemoteError) as cm:
            c.truncate(srv.url, "s", 0)
        self.assertEqual(cm.exception.status, 501)


class Tokenizers(unittest.TestCase):
    class Tok:
        def __init__(self, vocab):
            self.v = vocab
            self.added_tokens_decoder = {}

        def get_vocab(self):
            return self.v

    def test_identical_vocab_is_accepted(self):
        spec.check_tokenizers_match(self.Tok({"a": 0, "b": 1}), self.Tok({"b": 1, "a": 0}))

    def test_different_ids_or_size_are_refused(self):
        with self.assertRaises(spec.SpeculationRefused):
            spec.check_tokenizers_match(self.Tok({"a": 0, "b": 1}), self.Tok({"a": 1, "b": 0}))
        with self.assertRaises(spec.SpeculationRefused):
            spec.check_tokenizers_match(self.Tok({"a": 0, "b": 1}), self.Tok({"a": 0}))


if __name__ == "__main__":
    unittest.main()
