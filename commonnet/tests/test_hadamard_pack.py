"""Hadamard 2-bit packs (Ternary Bonsai 2 9B): MlxBlockRange loads a window and matches an explicit-float reference.

A tiny random qwen3_5 checkpoint is folded into the pack format (rotate each projection's input basis with a
sign vector and a Walsh-Hadamard transform, quantize to affine 2-bit, group 128, store as
language_model.model.* with the pack's config.json). The reference is a float checkpoint whose projections
are the effective weights (dequantized, then rotated back), run through the stock float path.
Needs Apple Metal.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import Qwen3_5TextConfig  # noqa: E402

from commonnet.backends import hadamard  # noqa: E402
from commonnet.backends.mlx_block import MlxBlockRange, _sanitize_qwen35  # noqa: E402
from commonnet.tests.qwen35_tiny import make_qwen35_checkpoint  # noqa: E402

BLOCK = 512
EFFECTIVE: dict = {}  # name -> the float matrix a packed embedding / head computes with
PROJ = ("in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj")


def wide_config(layers=4):
    return Qwen3_5TextConfig(
        vocab_size=512, hidden_size=BLOCK, intermediate_size=BLOCK, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=128,
        layer_types=["linear_attention" if (i + 1) % 4 else "full_attention" for i in range(layers)],
        linear_conv_kernel_dim=4, linear_key_head_dim=64, linear_value_head_dim=128,
        linear_num_key_heads=2, linear_num_value_heads=4, max_position_embeddings=256,
        tie_word_embeddings=False, initializer_range=0.1,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 0], "mrope_interleaved": True})


def rotate(w: mx.array, signs: mx.array) -> mx.array:
    """W' = W D H: rows signed, then Walsh-Hadamard along the input axis (H is its own inverse)."""
    return hadamard.fwht(w.astype(mx.float32), BLOCK, signs, inverse=True)  # x*signs after H == D H applied to rows


def build(tmp, layers=4):
    """Returns (pack_dir, reference_float_dir)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(wide_config(layers)).float().eval()
    rng = np.random.default_rng(0)
    pack, ref, modules = {}, {}, []
    sd = {("model.language_model." + k[6:] if k.startswith("model.") else k): v.detach().clone().contiguous()
          for k, v in model.state_dict().items()}
    for k, v in sd.items():
        ref[k] = v
        stem = k.split(".")[-2] if k.endswith(".weight") else None
        if stem in PROJ and ".layers." in k:
            w = mx.array(v.numpy())
            signs = mx.array(rng.choice([-1.0, 1.0], size=w.shape[1]).astype(np.float32))
            q, s, b = mx.quantize(rotate(w, signs).astype(mx.float16), group_size=128, bits=2)
            base = "language_model.model." + k[len("model.language_model."):-len(".weight")]
            pack[base + ".weight"], pack[base + ".scales"], pack[base + ".biases"], pack[base + ".signs"] = q, s, b, signs
            modules.append({"path": base[len("language_model."):], "block": BLOCK, "embedding": False, "dtype": "float16"})
            # effective weight the pack computes: dequantize, then undo the rotation
            deq = mx.dequantize(q, s, b, group_size=128, bits=2).astype(mx.float32)
            ref[k] = torch.from_numpy(np.array(_eff(deq, signs)))
        elif ".layers." in k:
            name = "language_model.model." + k[len("model.language_model."):]
            (name2, val), = _sanitize_qwen35([(name, mx.array(v.numpy()))], True)
            pack[name2] = val
    for name, rec_name, emb in (("model.language_model.embed_tokens.weight", "model.embed_tokens", True),
                                ("lm_head.weight", "lm_head", False)):
        w = mx.array(sd[name].numpy())
        signs = mx.array(rng.choice([-1.0, 1.0], size=w.shape[1]).astype(np.float32))
        stored = hadamard.fwht(w, BLOCK, signs) if emb else rotate(w, signs)  # embedding rows rotate the other way
        q, s_, b = mx.quantize(stored.astype(mx.float16), group_size=128, bits=2)
        base = "language_model." + rec_name
        pack[base + ".weight"], pack[base + ".scales"], pack[base + ".biases"], pack[base + ".signs"] = q, s_, b, signs
        modules.append({"path": rec_name, "block": BLOCK, "embedding": emb, "dtype": "float16"})
        deq = mx.dequantize(q, s_, b, group_size=128, bits=2).astype(mx.float32)
        EFFECTIVE[rec_name] = np.array(hadamard.fwht(deq, BLOCK, signs, inverse=True) if emb else _eff(deq, signs))
    pack["language_model.model.norm.weight"] = mx.array(sd["model.language_model.norm.weight"].numpy()) + 1.0
    pdir, rdir = os.path.join(tmp, "pack"), os.path.join(tmp, "ref")
    os.makedirs(pdir)
    mx.save_safetensors(os.path.join(pdir, "model.safetensors"), pack)
    tc = wide_config(layers).to_dict()
    json.dump({"model_type": "prism_hadamard_qwen35", "text_config": tc, "modules": modules,
               "quantization": {"bits": 2, "group_size": 128, "mode": "affine"}},
              open(os.path.join(pdir, "config.json"), "w"))
    os.makedirs(rdir)
    from transformers import Qwen3_5Config
    cfg = Qwen3_5Config(text_config=tc, vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 32,
                        "num_heads": 2, "out_hidden_size": BLOCK})
    cfg.tie_word_embeddings = False
    cfg.save_pretrained(rdir)
    save_file(ref, os.path.join(rdir, "model.safetensors"))
    return pdir, rdir


def _eff(deq, signs):
    """Effective weight: the pack computes deq @ (H D x), so W_eff = deq @ H @ D (row-wise on the input axis)."""
    h = mx.hadamard_transform(deq.reshape(-1, BLOCK), scale=1 / math.sqrt(BLOCK)).reshape(deq.shape)
    return h * signs


@unittest.skipUnless(mx.metal.is_available(), "needs Metal")
class HadamardPack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.pack, cls.ref = build(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_blocks(self, ckpt, bounds, **kw):
        blocks = [MlxBlockRange(ckpt, a, b, dtype=mx.float32, **kw) for a, b in zip(bounds, bounds[1:])]
        h = torch.randn(1, 5, BLOCK, generator=torch.Generator().manual_seed(3)) * 0.5
        for blk in blocks:
            h = blk.forward("s", h, 0)
        return h, blocks

    def test_pack_window_matches_effective_float_reference(self):
        got, blocks = self.run_blocks(self.pack, [0, 2, 4])
        self.assertTrue(all(b.hadamard for b in blocks))
        want, _ = self.run_blocks(self.ref, [0, 2, 4], bits=None)
        self.assertLess((got - want).abs().max().item(), 5e-3 * max(1.0, want.abs().max().item()))

    def test_window_reads_only_its_layers(self):
        blk = MlxBlockRange(self.pack, 1, 3, dtype=mx.float32)
        self.assertTrue(all(".layers.1." in n or ".layers.2." in n for n in blk.loaded_names))
        self.assertTrue(any(n.endswith("mlp.up_proj.signs") for n in blk.loaded_names))

    def test_packed_embedding_and_head_roundtrip(self):
        rng = np.random.default_rng(1)
        w = mx.array(rng.normal(size=(16, BLOCK)).astype(np.float32))
        signs = mx.array(rng.choice([-1.0, 1.0], size=BLOCK).astype(np.float32))
        q, s, b = mx.quantize(rotate(w, signs).astype(mx.float16), group_size=128, bits=2)
        deq = mx.dequantize(q, s, b, group_size=128, bits=2).astype(mx.float32)
        emb = hadamard.Packed(q, s, b, BLOCK, signs, embedding=True)
        got = emb(mx.array([[3, 7]]))
        want = hadamard.fwht(deq[mx.array([3, 7])], BLOCK, signs, inverse=True)
        self.assertLess(mx.abs(got.astype(mx.float32) - want[None]).max().item(), 5e-3)
        head = hadamard.Packed(q, s, b, BLOCK, signs)
        x = mx.array(rng.normal(size=(1, 2, BLOCK)).astype(np.float32)).astype(mx.float16)
        want = hadamard.fwht(x, BLOCK, signs).astype(mx.float32) @ deq.T
        self.assertLess(mx.abs(head(x).astype(mx.float32) - want).max().item(), 0.1)

    def test_client_embed_and_head_from_pack(self):
        from commonnet.client4 import Client4
        client = Client4(self.pack, transport=None, fmt="mlx4")
        ids = torch.tensor([[3, 7, 11]])
        emb = EFFECTIVE["model.embed_tokens"]
        self.assertLess(np.abs(client.embed(ids).numpy() - emb[[3, 7, 11]][None]).max(), 5e-3)
        h = torch.randn(1, 2, BLOCK, generator=torch.Generator().manual_seed(5)) * 0.5
        want = h.numpy() @ EFFECTIVE["lm_head"].T
        self.assertLess(np.abs(client.head(h).numpy() - want).max(), 0.1 * max(1.0, np.abs(want).max()))

    def test_linear_attention_state_cannot_roll_back(self):
        blk = MlxBlockRange(self.pack, 0, 4, dtype=mx.float32)
        blk.forward("s", torch.randn(1, 4, BLOCK), 0)
        with self.assertRaises(NotImplementedError):
            blk.truncate("s", 2)

    def test_a_pack_without_quantized_layers_is_refused(self):
        d = tempfile.mkdtemp()
        try:
            shutil.copy(os.path.join(self.pack, "config.json"), d)
            mx.save_safetensors(os.path.join(d, "model.safetensors"), {"x": mx.zeros((1,))})
            with self.assertRaises((ValueError, KeyError)):
                MlxBlockRange(d, 0, 1)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
