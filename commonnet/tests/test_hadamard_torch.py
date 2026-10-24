"""Hadamard packs (Bonsai 2) in torch: dequantisation, the activation transform, and a CPU window vs a float reference.

The tiny pack and its effective-float reference come from test_hadamard_pack (building them needs MLX/Metal, as there).
The full-size check against the MLX block (layers 0-8 of the real pack) was run by hand, see docs/bonsai-networks.md.
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from commonnet.backends.cpu_block import CpuBlockRange  # noqa: E402
from commonnet.backends.hadamard_torch import HadamardLinear, dequantize_affine2, plain_tensor  # noqa: E402
from commonnet.tests.test_nf4_quantise_on_load import nf4_roundtrip  # noqa: E402


def pack_codes(q: np.ndarray) -> np.ndarray:
    """MLX affine 2-bit layout: 16 codes per uint32, element j at bits 2j."""
    q = q.reshape(q.shape[0], -1, 16).astype(np.uint64)
    return (q << (2 * np.arange(16, dtype=np.uint64))).sum(-1).astype(np.uint32)


class Dequantise(unittest.TestCase):
    def test_matches_affine_formula(self):
        rng = np.random.default_rng(0)
        q = rng.integers(0, 4, (8, 256))
        sc = rng.random((8, 2)).astype(np.float16)
        bi = (rng.random((8, 2)) - 0.5).astype(np.float16)
        want = q.reshape(8, 2, 128) * sc[..., None].astype(np.float32) + bi[..., None].astype(np.float32)
        got = dequantize_affine2(pack_codes(q), sc, bi)
        np.testing.assert_allclose(got.numpy(), want.reshape(8, 256), rtol=0, atol=1e-6)

    def test_ternary_levels_survive_nf4_exactly(self):
        # bias = -scale, codes 0/1/2 -> levels -s, 0, +s (what the Bonsai 2 pack stores in the rotated basis)
        rng = np.random.default_rng(1)
        q = rng.integers(0, 3, (16, 512))
        sc = (rng.random((16, 4)) * 0.02 + 0.005).astype(np.float16)
        w = dequantize_affine2(pack_codes(q), sc, -sc)
        self.assertEqual(max(torch.unique(r).numel() for r in w.reshape(-1, 128)), 3)
        self.assertTrue(torch.equal(nf4_roundtrip(w.half()), w.half().float()))


class Transform(unittest.TestCase):
    def test_orthonormal_and_inverse(self):
        block = 512
        signs = torch.from_numpy(np.random.default_rng(2).choice([-1.0, 1.0], size=block * 2).astype(np.float32))
        eye = nn.Linear(block * 2, block * 2, bias=False)
        eye.weight.data = torch.eye(block * 2)
        lin = HadamardLinear(eye, signs, block)
        x = torch.randn(3, block * 2)
        y = lin(x)
        self.assertAlmostEqual(y.norm().item() / x.norm().item(), 1.0, places=5)
        # inverse: undo the Walsh-Hadamard (self-inverse), then the signs
        h = torch.ones(1, 1)
        while h.shape[0] < block:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        back = (y.reshape(-1, block) @ (h / math.sqrt(block))).reshape(x.shape) * signs
        torch.testing.assert_close(back, x, atol=1e-4, rtol=1e-4)

    def test_matches_mlx_hadamard_when_available(self):
        try:
            import mlx.core as mx
            from commonnet.backends import hadamard
            signs = np.random.default_rng(3).choice([-1.0, 1.0], size=1024).astype(np.float32)
            x = np.random.default_rng(4).standard_normal((2, 1024)).astype(np.float32)
            want = np.array(hadamard.fwht(mx.array(x), 512, mx.array(signs)))
        except Exception as e:  # no MLX / no Metal
            self.skipTest(f"MLX unavailable: {e}")
        eye = nn.Linear(1024, 1024, bias=False)
        eye.weight.data = torch.eye(1024)
        got = HadamardLinear(eye, torch.from_numpy(signs), 512)(torch.from_numpy(x)).detach().numpy()
        np.testing.assert_allclose(got, want, atol=1e-4)


class Layout(unittest.TestCase):
    def test_conv1d_and_plus_one_norms(self):
        c = torch.arange(24.0).reshape(6, 4, 1)
        self.assertEqual(tuple(plain_tensor("linear_attn.conv1d.weight", c).shape), (6, 1, 4))
        self.assertTrue(torch.equal(plain_tensor("input_layernorm.weight", torch.full((4,), 2.0)), torch.ones(4)))
        self.assertTrue(torch.equal(plain_tensor("linear_attn.norm.weight", torch.full((4,), 2.0)), torch.full((4,), 2.0)))


class TinyPack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import mlx.core as mx
            if not mx.metal.is_available():
                raise unittest.SkipTest("needs Metal to build the tiny pack")
        except ImportError:
            raise unittest.SkipTest("needs MLX to build the tiny pack")
        from commonnet.tests.test_hadamard_pack import build
        cls.tmp = tempfile.mkdtemp()
        cls.pack, cls.ref = build(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    def run_window(self, ckpt, a, b):
        blk = CpuBlockRange(ckpt, a, b, dtype=torch.float32)
        h = torch.randn(1, 5, 512, generator=torch.Generator().manual_seed(3)) * 0.5
        return blk.forward("s", h, 0), blk

    def test_cpu_window_matches_effective_float_reference(self):
        got, blk = self.run_window(self.pack, 0, 4)   # three linear-attention layers and one full-attention layer
        want, _ = self.run_window(self.ref, 0, 4)
        self.assertLess((got - want).abs().max().item(), 5e-3 * max(1.0, want.abs().max().item()))
        self.assertTrue(all(isinstance(m, HadamardLinear) for m in
                            (blk.layers[0].mlp.up_proj, blk.layers[3].self_attn.q_proj)))

    def test_window_reads_only_its_layers(self):
        _, blk = self.run_window(self.pack, 1, 3)
        self.assertTrue(all(".layers.1." in n or ".layers.2." in n for n in blk.loaded_names))


if __name__ == "__main__":
    unittest.main()
