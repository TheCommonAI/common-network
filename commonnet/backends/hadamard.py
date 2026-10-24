"""Packed 2-bit linears and embeddings stored in a Hadamard-rotated basis (Prism ML "Hadamard" packs).

Ported from the pack's own runtime (runtime/runtime.py in prism-ml/Ternary-Bonsai-2-9B-mlx-2bit, MIT licence):
a folded weight W' = W H D is stored as MLX affine 2-bit (group 128), so the activation must be transformed
first (x -> H D x, a sign flip then a normalised Walsh-Hadamard transform per `block` values). The embedding
is stored rotated the other way, so its rows get the inverse transform after the lookup.
Stock mlx-lm cannot load such a checkpoint ("Unsupported model type: prism_hadamard_qwen35").
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

BLOCKS = (512, 1024, 2048, 4096)
GROUP_SIZE, BITS = 128, 2


def fwht(x, block: int, signs, inverse: bool = False):
    shape, dtype = x.shape, x.dtype
    if shape[-1] % block:
        raise ValueError("Hadamard block does not divide activation width")
    x = x.astype(mx.float32)
    if not inverse:
        x = x * signs
    x = mx.hadamard_transform(x.reshape(-1, block), scale=1 / math.sqrt(block)).reshape(shape)
    if inverse:
        x = x * signs
    return x.astype(dtype)


class Packed(nn.Module):
    """A 2-bit affine linear (or embedding) whose activations are Hadamard-transformed around it."""

    def __init__(self, weight, scales, biases, block: int = 0, signs=None, embedding: bool = False):
        super().__init__()
        if block and block not in BLOCKS:
            raise ValueError(f"unvalidated Hadamard block size {block}")
        if block and signs is None:
            raise ValueError("a Hadamard block needs its sign vector")
        self.weight, self.scales, self.biases = weight, scales, biases
        if block:
            self.signs = signs
        self.block, self.embedding = block, embedding

    def _signs(self):
        return self.signs if self.block else None

    def __call__(self, x):
        if self.embedding:
            shape, idx = x.shape, x.reshape(-1)
            out = mx.dequantize(self.weight[idx], self.scales[idx], self.biases[idx],
                                group_size=GROUP_SIZE, bits=BITS).reshape(*shape, -1).astype(mx.float16)
            return fwht(out, self.block, self._signs(), inverse=True) if self.block else out
        dtype = x.dtype
        if self.block:
            x = fwht(x, self.block, self._signs())
        y = mx.quantized_matmul(x, self.weight, self.scales, self.biases, transpose=True,
                                group_size=GROUP_SIZE, bits=BITS)
        return y.astype(dtype) if y.dtype != dtype else y


def install(root, records, prefix: str, get) -> list[str]:
    """Replace the modules named by `records` (config.json "modules"; each {"path", "block", "embedding"})
    that start with `prefix` by Packed modules. `get(path, suffix)` returns the stored tensor for a record path
    (weight, scales, biases, signs). `root` is the decoder block; record paths are relative to the model
    (model.layers.N.mlp.up_proj), so `prefix` is "model.layers.N.". Returns the paths replaced."""
    done = []
    for rec in records:
        path = rec["path"]
        if not path.startswith(prefix):
            continue
        parts = path[len(prefix):].split(".")
        parent = root
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        block = int(rec.get("block") or 0)
        signs = get(path, "signs") if block else None
        setattr(parent, parts[-1], Packed(get(path, "weight"), get(path, "scales"), get(path, "biases"),
                                          block, signs, bool(rec.get("embedding"))))
        done.append(path)
    return done
