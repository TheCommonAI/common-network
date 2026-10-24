"""CPU block on a tiny random Qwen3 checkpoint: output matches the Hugging Face reference, factory wiring.

Run from the repository root:
    python3 -m unittest commonnet.tests.test_cpu_block -v
"""
from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from transformers import Qwen3ForCausalLM  # noqa: E402

from commonnet.backends.cpu_block import CpuBlockRange  # noqa: E402
from commonnet.net import factory  # noqa: E402
from commonnet.tests.test_split import make_checkpoint  # noqa: E402

PROMPT = [5, 17, 301, 42, 9, 133, 77]


class CpuBlock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ckpt = os.path.join(cls.tmp.name, "tiny")
        make_checkpoint(cls.ckpt)
        cls.ref = Qwen3ForCausalLM.from_pretrained(cls.ckpt, torch_dtype=torch.float32).eval()
        with torch.no_grad():
            out = cls.ref(torch.tensor([PROMPT]), output_hidden_states=True)
        cls.hs = out.hidden_states  # [embeddings, after layer 1, ...]; the last entry is post-norm

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_default_dtype_is_bfloat16_on_cpu(self):
        block = CpuBlockRange(self.ckpt, 0, 2)
        self.assertEqual((block.dtype, block.device.type), (torch.bfloat16, "cpu"))
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in block.layers.parameters()))

    def test_refuses_a_non_cpu_device(self):
        with self.assertRaises(RuntimeError):
            CpuBlockRange(self.ckpt, 0, 2, device="cuda")

    def test_float32_matches_the_reference_tightly(self):
        block = CpuBlockRange(self.ckpt, 0, 2, dtype=torch.float32)
        out = block.forward("s", self.hs[0], 0)
        self.assertTrue(torch.allclose(out, self.hs[2], atol=1e-4), (out - self.hs[2]).abs().max())

    def test_bfloat16_matches_the_float32_reference_within_tolerance(self):
        block = CpuBlockRange(self.ckpt, 0, 2)
        out = block.forward("s", self.hs[0], 0)
        self.assertEqual(out.dtype, torch.bfloat16)
        err = (out.float() - self.hs[2]).abs().max().item()
        scale = self.hs[2].abs().max().item()
        self.assertLess(err, 0.05 * scale, (err, scale))

    def test_two_cpu_blocks_in_a_chain_decode_like_the_reference(self):
        a = CpuBlockRange(self.ckpt, 0, 2)
        b = CpuBlockRange(self.ckpt, 2, 4)
        h = a.forward("s", self.hs[0], 0).float()          # float32 on the wire between peers
        h = b.forward("s", h, 0)
        norm, head = self.ref.model.norm, self.ref.lm_head
        logits = head(norm(h.float()))[:, -1]
        want = self.ref(torch.tensor([PROMPT])).logits[:, -1]
        self.assertEqual(logits.argmax(-1).item(), want.argmax(-1).item())
        # one decoded token through the KV cache equals the reference on the longer prompt
        nxt = want.argmax(-1).item()
        emb = self.ref.model.embed_tokens(torch.tensor([[nxt]]))
        h2 = b.forward("s", a.forward("s", emb, len(PROMPT)).float(), len(PROMPT))
        got = head(norm(h2.float()))[:, -1]
        full = self.ref(torch.tensor([PROMPT + [nxt]])).logits[:, -1]
        self.assertTrue(torch.allclose(got, full, atol=0.15 * full.abs().max().item()))
        self.assertEqual(a.session_length("s"), len(PROMPT) + 1)
        a.close("s")
        self.assertEqual(a.open_sessions(), 0)

    def test_factory_builds_a_cpu_block_with_a_float32_wire(self):
        from commonnet.net.transport import from_wire, to_wire
        served = factory.build_block(0, 2, backend="cpu", checkpoint=self.ckpt, environ={})
        self.assertEqual(served.device, "cpu")
        self.assertEqual(served.inner.dtype, torch.bfloat16)
        wire = to_wire(self.hs[0])
        out = served.forward("w", wire, 0)
        self.assertEqual(from_wire(out).dtype, torch.float32)  # same wire dtype as the request
        self.assertIn("cpu", factory.BACKENDS)

    def test_factory_honours_an_explicit_dtype(self):
        served = factory.build_block(0, 1, backend="cpu", checkpoint=self.ckpt, environ={"COMMONNET_DTYPE": "float32"})
        self.assertEqual(served.inner.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
