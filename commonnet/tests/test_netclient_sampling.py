"""Sampling and streaming in Client.generate, on the tiny random Qwen3 checkpoint (no downloads)."""
import os
import shutil
import tempfile
import unittest

import torch

from commonnet.block import BlockRange
from commonnet.client import Client, LocalTransport, sample_token
from commonnet.tests.test_split import NEW_TOKENS, PROMPT, make_checkpoint, route_for


class Sampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="commonnet-sample-")
        cls.ckpt = os.path.join(cls.tmp, "tiny")
        make_checkpoint(cls.ckpt)
        cls.route = route_for([0, 2, 4])
        blocks = {ep.name: BlockRange(cls.ckpt, ep.start, ep.end) for ep in cls.route}
        cls.client = Client(cls.ckpt, LocalTransport(blocks), device="cpu", dtype=torch.float32)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_default_is_greedy_and_stream_matches_list(self):
        listed = self.client.generate(self.route, PROMPT, NEW_TOKENS)
        streamed = list(self.client.generate_stream(self.route, PROMPT, NEW_TOKENS))
        self.assertEqual(listed, streamed)
        self.assertEqual(len(listed), NEW_TOKENS)
        self.assertEqual(self.client.generate(self.route, PROMPT, NEW_TOKENS, temperature=0.0), listed)

    def test_same_seed_same_tokens_and_different_seed_differs(self):
        a = self.client.generate(self.route, PROMPT, 24, temperature=1.0, seed=1)
        b = self.client.generate(self.route, PROMPT, 24, temperature=1.0, seed=1)
        c = self.client.generate(self.route, PROMPT, 24, temperature=1.0, seed=2)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_stream_yields_as_it_goes_and_closes_the_session_when_dropped(self):
        closed = []
        real = self.client.transport.close
        self.client.transport.close = lambda ep, sid: (closed.append(ep.name), real(ep, sid))[1]
        try:
            stream = self.client.generate_stream(self.route, PROMPT, 10)
            first = next(stream)
            self.assertIsInstance(first, int)
            self.assertEqual(closed, [])
            stream.close()
        finally:
            self.client.transport.close = real
        self.assertEqual(sorted(closed), ["block0", "block1"])

    def test_stop_id_ends_the_stream_and_is_not_yielded(self):
        full = self.client.generate(self.route, PROMPT, NEW_TOKENS)
        stop = full[3]
        cut = self.client.generate(self.route, PROMPT, NEW_TOKENS, stop_ids=[stop])
        self.assertEqual(cut, full[:full.index(stop)])

    def test_zero_new_tokens_is_empty(self):
        self.assertEqual(self.client.generate(self.route, PROMPT, 0), [])


class SampleToken(unittest.TestCase):
    def test_greedy_is_argmax(self):
        self.assertEqual(sample_token(torch.tensor([0.1, 3.0, 2.0])), 1)

    def test_top_p_keeps_only_the_nucleus(self):
        logits = torch.tensor([10.0, 9.0, -20.0, -20.0])
        rng = torch.Generator().manual_seed(0)
        seen = {sample_token(logits, 1.0, 0.5, rng) for _ in range(200)}
        self.assertEqual(seen, {0})  # token 0 alone already holds more than half of the mass
        rng = torch.Generator().manual_seed(0)
        seen = {sample_token(logits, 1.0, 0.99, rng) for _ in range(300)}
        self.assertEqual(seen, {0, 1})

    def test_temperature_flattens_or_sharpens(self):
        logits = torch.tensor([2.0, 1.0, 0.0])
        rng = torch.Generator().manual_seed(3)
        cold = [sample_token(logits, 0.05, 1.0, rng) for _ in range(100)]
        self.assertEqual(set(cold), {0})
        rng = torch.Generator().manual_seed(3)
        hot = {sample_token(logits, 5.0, 1.0, rng) for _ in range(200)}
        self.assertEqual(hot, {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
