"""MLX block tests on the tiny random Qwen3 checkpoint from the core tests. Needs Apple Metal.

Run on the Mac from the repository root:
    ~/pedal-venv/bin/python -m unittest pedal.tests.test_mlx_block -v

Stated tolerances (logits, absolute). Logit std on the tiny model is about 0.8.
  TOL_BOUNDARY: a split MLX chain vs a single MLX block. Only the float16 hidden
                crossing differs, so this is tight.
  TOL_Q4:       MLX 4-bit single block vs the fp32 reference. This is quantization error.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import torch  # noqa: E402
from transformers import Qwen3ForCausalLM  # noqa: E402

from pedal.backends.mlx_block import MlxBlockRange  # noqa: E402
from pedal.client import BlockEndpoint, Client, LocalTransport  # noqa: E402
from pedal.tests.test_split import NEW_TOKENS, PROMPT, make_checkpoint, route_for  # noqa: E402

TOL_BOUNDARY = 2e-2  # split chain vs single block (float16 hidden crossing only)
TOL_FLOAT = 2e-2  # unquantized MLX block vs fp32 reference (float16 compute)
TOL_Q8 = 0.2  # 8-bit MLX block vs fp32 reference (measured 0.051)
TOL_Q4 = 1.5  # 4-bit sanity bound: about 2x the logit std (measured 1.086); not a quality claim


def _metal_ok() -> bool:
    return mx.metal.is_available()


@unittest.skipUnless(_metal_ok(), "Metal GPU not available")
class MlxTiny(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pedal-mlx-")
        cls.ckpt = os.path.join(cls.tmp, "tiny-qwen3")
        make_checkpoint(cls.ckpt)
        cls.reference = Qwen3ForCausalLM.from_pretrained(cls.ckpt, dtype=torch.float32).eval()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def build(self, boundaries, bits=4):
        route = route_for(boundaries)
        blocks = {
            ep.name: MlxBlockRange(self.ckpt, ep.start, ep.end, bits=bits) for ep in route
        }
        client = Client(self.ckpt, LocalTransport(blocks), device="cpu", dtype=torch.float32)
        return client, route, blocks

    def _vs_reference(self, bits):
        ids = torch.tensor([PROMPT])
        with torch.inference_mode():
            full = self.reference(ids).logits
        client, route, _ = self.build([0, 4], bits=bits)
        return float((client.logits(route, ids) - full).abs().max()), float(full.std())

    def test_unquantized_block_matches_reference(self):
        diff, std = self._vs_reference(None)
        print(f"\n[mlx float16 single] max|mlx - fp32 ref| = {diff:.4e} (logit std {std:.3e}, tol {TOL_FLOAT})")
        self.assertLess(diff, TOL_FLOAT)

    def test_8bit_block_vs_reference(self):
        diff, std = self._vs_reference(8)
        print(f"\n[mlx 8-bit single] max|mlx - fp32 ref| = {diff:.4e} (logit std {std:.3e}, tol {TOL_Q8})")
        self.assertLess(diff, TOL_Q8)

    def test_4bit_block_vs_reference_sanity(self):
        diff, std = self._vs_reference(4)
        print(f"\n[mlx 4-bit single] max|mlx - fp32 ref| = {diff:.4e} (logit std {std:.3e}, tol {TOL_Q4})")
        self.assertLess(diff, TOL_Q4)

    def test_split_chain_matches_single_block(self):
        ids = torch.tensor([PROMPT])
        client1, route1, _ = self.build([0, 4])
        single = client1.logits(route1, ids)
        for boundaries in ([0, 2, 4], [0, 1, 3, 4]):
            with self.subTest(boundaries=boundaries):
                client, route, _ = self.build(boundaries)
                split = client.logits(route, ids)
                diff = float((split - single).abs().max())
                print(f"\n[mlx split {boundaries}] max|split - single| = {diff:.4e} (tol {TOL_BOUNDARY})")
                self.assertLess(diff, TOL_BOUNDARY)

    def test_generate_matches_single_block(self):
        client1, route1, _ = self.build([0, 4])
        single = client1.generate(route1, PROMPT, NEW_TOKENS)
        client2, route2, _ = self.build([0, 2, 4])
        split = client2.generate(route2, PROMPT, NEW_TOKENS)
        print(f"\n[mlx greedy] single={single} split={split} equal={single == split}")
        self.assertEqual(split, single)

    def test_sessions_and_errors(self):
        block = MlxBlockRange(self.ckpt, 0, 2)
        with torch.inference_mode():
            hidden = torch.randn(1, 3, 64)
            block.forward("s", hidden, 0)
            self.assertEqual(block.session_length("s"), 3)
            with self.assertRaises(ValueError):
                block.forward("s", hidden[:, :1], 5)
            with self.assertRaises(KeyError):
                block.forward("nope", hidden[:, :1], 3)
        self.assertEqual(block.open_sessions(), 1)
        block.close("s")
        block.close("s")
        self.assertEqual(block.open_sessions(), 0)

    def test_block_loads_only_its_layers(self):
        block = MlxBlockRange(self.ckpt, 2, 4)
        layer_ids = {n.split(".")[2] for n in block.loaded_names}
        self.assertEqual(layer_ids, {"2", "3"})
        self.assertTrue(all(n.startswith(("model.layers.2.", "model.layers.3.")) for n in block.loaded_names))

    def test_speed_and_memory(self):
        client, route, blocks = self.build([0, 2, 4])
        client.generate(route, PROMPT, 2)  # warm-up
        steps = 32
        t0 = time.perf_counter()
        client.generate(route, PROMPT, steps)
        dt = time.perf_counter() - t0
        mem = blocks[route[0].name].memory_bytes()
        print(
            f"\n[mlx speed] {steps} tokens after a {len(PROMPT)}-token prompt, 2 blocks: "
            f"{steps / dt:.1f} tok/s ({dt / steps * 1e3:.2f} ms/token incl. prompt amortised)"
        )
        print(f"[mlx memory] process-wide MLX active={mem['active']} B peak={mem['peak']} B")


if __name__ == "__main__":
    unittest.main()
