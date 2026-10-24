"""NF4 quantise-on-load: which path each Linear takes, and what NF4 does to ternary weights.

CPU only. bitsandbytes is replaced by a recording fake; the NF4 quantiser here is a plain-torch SIMULATION of
bitsandbytes' blockwise NF4 (blocksize 64, fp32 absmax, nearest of the 16 NF4 levels), not bitsandbytes itself.
"""

from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from commonnet.backends import nf4_block  # noqa: E402
from commonnet.tests.test_split import make_checkpoint  # noqa: E402
from commonnet.weights import CheckpointReader  # noqa: E402

NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                    0.7229568362236023, 1.0])


def nf4_roundtrip(w: torch.Tensor, blocksize: int = 64) -> torch.Tensor:
    flat = w.float().reshape(-1, blocksize)
    absmax = flat.abs().amax(1, keepdim=True)
    safe = torch.where(absmax == 0, torch.ones_like(absmax), absmax)
    codes = (flat / safe).unsqueeze(-1).sub(NF4).abs().argmin(-1)
    return (NF4[codes] * absmax).reshape(w.shape)


def ternary(rows: int, cols: int, group: int = 128, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    scale = (torch.rand(rows, cols // group, generator=g) * 0.05 + 0.01).half().float()
    sign = torch.randint(-1, 2, (rows, cols), generator=g).float()
    return (sign * scale.repeat_interleave(group, 1)).half()


class FakeParams4bit:
    def __init__(self, data, requires_grad=False, quant_type=None, compress_statistics=None, module=None):
        self.data, self.quant_type, self.compress_statistics, self.kind = data, quant_type, compress_statistics, "quantised-on-load"

    @classmethod
    def from_prequantized(cls, data, quantized_stats, requires_grad, device, module):
        p = cls(data)
        p.kind, p.stats = "prequantized", quantized_stats
        return p

    def to(self, device):
        self.device = device
        return self


class FakeLinear4bit:
    def __init__(self, i, o, bias, compute_dtype, quant_type, device):
        self.in_features, self.out_features, self.compute_dtype, self.quant_type, self.device = i, o, compute_dtype, quant_type, device


class FakeBnb:
    class nn:
        Linear4bit = FakeLinear4bit


class Ternary(unittest.TestCase):
    def test_ternary_survives_nf4_exactly(self):
        w = ternary(256, 1024)
        self.assertTrue(torch.equal(nf4_roundtrip(w), w.float()))

    def test_in_bfloat16_compute_only_the_scale_rounds(self):
        w = ternary(256, 1024, seed=1)
        err = (nf4_roundtrip(w).bfloat16().float() - w.float()).abs()
        rel = err.max() / w.float().abs().max()
        self.assertLessEqual(rel.item(), 2 ** -8)  # bf16 keeps 8 significant bits

    def test_dense_weights_for_contrast(self):
        w = torch.randn(64, 256)
        rel = ((nf4_roundtrip(w) - w).norm() / w.norm()).item()
        self.assertGreater(rel, 0.05)  # NF4 is lossy on ordinary weights; only ternary is exact


class Selection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        make_checkpoint(self.tmp.name)
        self.reader = CheckpointReader(self.tmp.name)

    def test_plain_checkpoint_is_quantised_on_load(self):
        name = "model.layers.0.self_attn.q_proj.weight"
        self.assertFalse(nf4_block.is_prequantized(self.reader.weight_map, name))
        shape = self.reader.load([name])[name].shape
        mod = nn.Linear(shape[1], shape[0], bias=False, device="meta")
        new = nf4_block._load_linear(FakeBnb, FakeParams4bit, self.reader, name, mod, "cpu", torch.bfloat16)
        self.assertEqual(new.weight.kind, "quantised-on-load")
        self.assertEqual((new.weight.quant_type, new.weight.compress_statistics), ("nf4", False))
        self.assertIn(new.weight.data.dtype, (torch.float16, torch.bfloat16))
        self.assertEqual(new.compute_dtype, torch.bfloat16)

    def test_prequantized_tensor_takes_the_old_path(self):
        wm = {"a.weight": "f", "a.weight.absmax": "f"}
        self.assertTrue(nf4_block.is_prequantized(wm, "a.weight"))
        self.assertFalse(nf4_block.is_prequantized(wm, "b.weight"))
        # Only the NF4 stat names are read; a fake reader proves the pre-quantized branch is the one taken.
        reader = self.reader
        reader.weight_map = {**reader.weight_map, "x.weight.absmax": "f"}
        reader.weight_map.update({f"x.weight.{s}": "f" for s in nf4_block.NF4_SUFFIXES})
        reader.weight_map["x.weight"] = "f"
        seen = []
        orig = nf4_block._raw_tensor
        nf4_block._raw_tensor = lambda r, n, d: seen.append(n) or torch.zeros(1)
        try:
            mod = nn.Linear(8, 8, bias=False, device="meta")
            new = nf4_block._load_linear(FakeBnb, FakeParams4bit, reader, "x.weight", mod, "cpu", torch.bfloat16)
        finally:
            nf4_block._raw_tensor = orig
        self.assertEqual(new.weight.kind, "prequantized")
        self.assertIn("x.weight.absmax", seen)

    def test_wrong_shape_is_refused(self):
        name = "model.layers.0.self_attn.q_proj.weight"
        mod = nn.Linear(3, 5, bias=False, device="meta")
        with self.assertRaises(ValueError):
            nf4_block._load_linear(FakeBnb, FakeParams4bit, self.reader, name, mod, "cpu", torch.bfloat16)

class PlainKeys(unittest.TestCase):
    def test_quantised_linears_and_their_statistics_are_not_read_as_plain_tensors(self):
        quantized = {"self_attn.q_proj.weight", "mlp.down_proj.weight"}
        state = ["self_attn.q_proj.weight", "self_attn.q_proj.weight.absmax", "self_attn.q_proj.weight.quant_map",
                 "self_attn.q_proj.weight.nested_absmax", "self_attn.q_proj.weight.quant_state.bitsandbytes__nf4",
                 "mlp.down_proj.weight", "mlp.down_proj.weight.absmax", "input_layernorm.weight",
                 "self_attn.q_norm.weight", "post_attention_layernorm.weight"]
        self.assertEqual(nf4_block._plain_keys(state, quantized),
                         ["input_layernorm.weight", "self_attn.q_norm.weight", "post_attention_layernorm.weight"])

    def test_a_norm_whose_name_extends_a_quantised_one_is_kept(self):
        # q_proj.weight must not swallow q_proj_norm.weight: only keys under "q_proj.weight." are statistics
        self.assertEqual(nf4_block._plain_keys(["self_attn.q_proj_norm.weight"], {"self_attn.q_proj.weight"}),
                         ["self_attn.q_proj_norm.weight"])


if __name__ == "__main__":
    unittest.main()
