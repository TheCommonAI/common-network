"""MLX block for qwen3_5 on the tiny random hybrid checkpoint. Needs Apple Metal."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import torch  # noqa: E402

from commonnet.backends.mlx_block import MlxBlockRange  # noqa: E402
from commonnet.client import BlockEndpoint, Client, LocalTransport  # noqa: E402
from commonnet.tests.qwen35_tiny import make_qwen35_checkpoint  # noqa: E402

PROMPT = [5, 17, 301, 42, 9, 133, 77, 8, 200]


@unittest.skipUnless(mx.metal.is_available(), "needs Metal")
class MlxQwen35(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.ref = make_qwen35_checkpoint(cls.dir, layers=8)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def ref_logits(self, ids):
        with torch.no_grad():
            return self.ref(torch.tensor([ids])).logits.float()

    def chain(self, bounds, bits=None):
        eps = [BlockEndpoint(f"b{i}", a, b) for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))]
        blocks = {e.name: MlxBlockRange(self.dir, e.start, e.end, bits=bits, group_size=32) for e in eps}
        return eps, blocks

    def test_unquantized_split_matches_torch_reference(self):
        eps, blocks = self.chain([0, 3, 8])
        client = Client(self.dir, LocalTransport(blocks), dtype=torch.float16)
        # LocalTransport checks block.start/end; MlxBlockRange has them.
        got = client.logits(eps, torch.tensor([PROMPT]))
        diff = (got - self.ref_logits(PROMPT)).abs().max().item()
        self.assertLess(diff, 8e-2, diff)  # fp16 compute, logit std ~0.8
        self.assertTrue((got.argmax(-1) == self.ref_logits(PROMPT).argmax(-1)).all())

    def test_incremental_decode_matches_full(self):
        eps, blocks = self.chain([0, 5, 8])
        client = Client(self.dir, LocalTransport(blocks), dtype=torch.float16)
        h = client.embed(torch.tensor([PROMPT]))
        for e in eps:
            h = blocks[e.name].forward("s", h, 0)
        pos, outs = len(PROMPT), []
        for tok in (11, 12, 13):
            h = client.embed(torch.tensor([[tok]]))
            for e in eps:
                h = blocks[e.name].forward("s", h, pos)
            pos += 1
            outs.append(client._logits(h))
        want = self.ref_logits(PROMPT + [11, 12, 13])[:, len(PROMPT):]
        self.assertLess((torch.cat(outs, 1) - want).abs().max().item(), 8e-2)

    def test_4bit_runs(self):
        eps, blocks = self.chain([0, 8], bits=4)
        client = Client(self.dir, LocalTransport(blocks), dtype=torch.float16)
        got = client.logits(eps, torch.tensor([PROMPT]))
        self.assertTrue(torch.isfinite(got).all())


if __name__ == "__main__":
    unittest.main()
