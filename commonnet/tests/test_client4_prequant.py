"""Regression: Client4(fmt="mlx4") on a checkpoint whose embed/head are already MLX-quantized.

Bug: Client4 read model.embed_tokens.weight as float32 even when the checkpoint was
mlx-community/Qwen3-8B-4bit style (packed uint32 + .scales/.biases), which fails or
gives garbage. It must use the stored 4-bit tables and match quantizing the float
checkpoint itself.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import torch  # noqa: E402

from commonnet.client4 import Client4  # noqa: E402
from commonnet.tests.test_split import make_checkpoint  # noqa: E402
from safetensors import safe_open  # noqa: E402


@unittest.skipUnless(mx.metal.is_available(), "Metal GPU not available")
class PrequantizedClient(unittest.TestCase):
    def test_prequantized_matches_quantize_at_load(self):
        tmp = tempfile.mkdtemp(prefix="commonnet-pq-")
        try:
            flt = os.path.join(tmp, "float")
            make_checkpoint(flt)
            pq = os.path.join(tmp, "pq")
            os.makedirs(pq)
            tensors = {}
            with safe_open(os.path.join(flt, "model.safetensors"), framework="pt") as f:
                for n in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"):
                    tensors[n] = mx.array(f.get_tensor(n).float().numpy()).astype(mx.float16)
            out = {"model.norm.weight": tensors["model.norm.weight"]}
            for n in ("model.embed_tokens", "lm_head"):
                w, s, b = mx.quantize(tensors[n + ".weight"], group_size=64, bits=4)
                out[n + ".weight"], out[n + ".scales"], out[n + ".biases"] = w, s, b
            mx.save_safetensors(os.path.join(pq, "model.safetensors"), out)
            cfg = json.load(open(os.path.join(flt, "config.json")))
            cfg["quantization"] = {"group_size": 64, "bits": 4}
            json.dump(cfg, open(os.path.join(pq, "config.json"), "w"))

            a = Client4(flt, transport=None, fmt="mlx4")
            b = Client4(pq, transport=None, fmt="mlx4")
            ids = torch.tensor([[5, 17, 301, 42]])
            self.assertTrue(torch.allclose(a.embed(ids), b.embed(ids), atol=1e-3))
            h = torch.randn(1, 3, 64)
            self.assertTrue(torch.allclose(a.head(h), b.head(h), atol=1e-2))
            self.assertEqual(a.client_bytes, b.client_bytes)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
