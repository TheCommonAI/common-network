"""Split vs full parity for a tiny random qwen3_5 (hybrid linear attention) checkpoint. Offline."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

from commonnet.block import BlockRange  # noqa: E402
from commonnet.client import BlockEndpoint, Client, LocalTransport  # noqa: E402
from commonnet.tests.qwen35_tiny import make_qwen35_checkpoint  # noqa: E402

PROMPT = [5, 17, 301, 42, 9, 133, 77, 8, 200]
NEW = 10


class Qwen35Parity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.ref = make_qwen35_checkpoint(cls.dir, layers=8)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def route(self, bounds):
        eps = [BlockEndpoint(f"b{i}", a, b) for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))]
        blocks = {e.name: BlockRange(self.dir, e.start, e.end) for e in eps}
        return eps, LocalTransport(blocks), blocks

    def ref_logits(self, ids):
        with torch.no_grad():
            return self.ref(torch.tensor([ids])).logits.float()

    def ref_greedy(self, ids, n):
        ids = list(ids)
        for _ in range(n):
            ids.append(int(self.ref_logits(ids)[0, -1].argmax()))
        return ids[-n:]

    def test_split_logits_match_full(self):
        for bounds in ([0, 8], [0, 3, 8], [0, 1, 4, 5, 8]):
            eps, tr, _ = self.route(bounds)
            client = Client(self.dir, tr)
            got = client.logits(eps, torch.tensor([PROMPT]))
            diff = (got - self.ref_logits(PROMPT)).abs().max().item()
            self.assertLess(diff, 1e-4, f"{bounds}: max diff {diff}")

    def test_greedy_tokens_equal(self):
        eps, tr, _ = self.route([0, 3, 8])
        client = Client(self.dir, tr)
        self.assertEqual(client.generate(eps, PROMPT, NEW), self.ref_greedy(PROMPT, NEW))

    def test_cached_incremental_equals_uncached(self):
        eps, tr, blocks = self.route([0, 5, 8])
        client = Client(self.dir, tr)
        full = self.ref_logits(PROMPT + [11, 12, 13])
        sid = "s"
        h = client.embed(torch.tensor([PROMPT]))
        pos = 0
        for e in eps:
            h = blocks[e.name].forward(sid, h, 0)
        pos = len(PROMPT)
        outs = [client._logits(h)]
        for tok in (11, 12, 13):
            h = client.embed(torch.tensor([[tok]]))
            for e in eps:
                h = blocks[e.name].forward(sid, h, pos)
            pos += 1
            outs.append(client._logits(h))
        inc = torch.cat([outs[0][:, -1:]] + outs[1:], dim=1)
        want = full[:, len(PROMPT) - 1:]
        self.assertLess((inc - want).abs().max().item(), 1e-4)
        # linear-attention state really is carried: layer 0 holds recurrent state, no KV
        st = blocks["b0"]._sessions[sid].cache
        self.assertIsNotNone(st.layers[0].recurrent_states)
        self.assertIsNone(st.keys[0])
        self.assertIsNotNone(st.keys[3])

    def test_chunked_continuation_equals_full(self):
        eps, tr, blocks = self.route([0, 8])
        b = blocks["b0"]
        client = Client(self.dir, tr)
        h1 = b.forward("c", client.embed(torch.tensor([PROMPT[:5]])), 0)
        h2 = b.forward("c", client.embed(torch.tensor([PROMPT[5:]])), 5)
        got = client._logits(torch.cat([h1, h2], 1))
        self.assertLess((got - self.ref_logits(PROMPT)).abs().max().item(), 1e-4)

    def test_negative_control_wrong_state_detected(self):
        """Dropping the linear-attention state between steps must break parity."""
        eps, tr, blocks = self.route([0, 8])
        b = blocks["b0"]
        client = Client(self.dir, tr)
        b.forward("n", client.embed(torch.tensor([PROMPT])), 0)
        for lyr in b._sessions["n"].cache.layers:
            lyr.recurrent_states = None
            lyr.conv_states = None
        h = b.forward("n", client.embed(torch.tensor([[11]])), len(PROMPT))
        got = client._logits(h)[:, -1]
        want = self.ref_logits(PROMPT + [11])[:, -1]
        self.assertGreater((got - want).abs().max().item(), 1e-3)

    def test_block_loads_only_its_tensors(self):
        b = BlockRange(self.dir, 2, 4)
        self.assertTrue(b.loaded_names)
        for n in b.loaded_names:
            self.assertTrue(n.startswith(("model.language_model.layers.2.", "model.language_model.layers.3.")), n)


if __name__ == "__main__":
    unittest.main()
