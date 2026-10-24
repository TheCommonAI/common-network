"""Prism "Hadamard" packs (Bonsai 2) in PyTorch, for the CPU and NF4 (NVIDIA) blocks.

The MLX version is hadamard.py. A pack stores each projection as MLX affine 2-bit (group 128) in a rotated basis:
y = W' (H D x), with D a sign vector and H a normalised Sylvester-Walsh-Hadamard per `block` values. Here W' is
dequantised once to an ordinary weight (its three levels per group are -s, 0, +s, so a later NF4 pass keeps them
exactly) and `HadamardLinear` applies the same transform to the activation in float32 before the matmul.

Layout notes (pack = MLX/mlx-vlm tensor layout; torch layers are transformers' Qwen3_5 blocks):
conv1d is (C, K, 1) in the pack and (C, 1, K) in torch; the four RMSNorms that torch computes as (1 + w) are
stored already +1 in the pack.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
from safetensors import safe_open
from torch import nn

GROUP_SIZE, BITS = 128, 2
PLUS_ONE_NORMS = ("input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight", "k_norm.weight")


def dequantize_affine2(weight: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> torch.Tensor:
    """MLX affine 2-bit (uint32 words, element j at bits 2j, group 128) -> float32 (out, in)."""
    codes = np.ascontiguousarray(weight).view(np.uint8)  # little endian: 4 codes per byte, low bits first
    codes = (codes[..., None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3
    q = torch.from_numpy(codes.reshape(weight.shape[0], -1)).float()
    out, cols = q.shape
    sc = torch.from_numpy(scales.astype(np.float32))
    bi = torch.from_numpy(biases.astype(np.float32))
    return (q.reshape(out, cols // GROUP_SIZE, GROUP_SIZE) * sc.unsqueeze(-1) + bi.unsqueeze(-1)).reshape(out, cols)


@lru_cache(maxsize=8)
def _hadamard(block: int, device: str) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < block:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(block)).to(device)


class HadamardLinear(nn.Module):
    """inner(H D x): `inner` holds the dequantised W' (an nn.Linear, or a bitsandbytes Linear4bit)."""

    def __init__(self, inner: nn.Module, signs: torch.Tensor, block: int):
        super().__init__()
        self.inner, self.block = inner, block
        self.register_buffer("signs", signs.float(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape, dtype = x.shape, x.dtype
        h = _hadamard(self.block, str(x.device))
        t = (x.float() * self.signs).reshape(-1, self.block)
        t = (t @ h).reshape(shape).to(dtype)
        return self.inner(t)


def pack_tensors(reader, path: str) -> dict:
    """Raw weight/scales/biases/signs of one packed module (`path` like model.layers.3.mlp.up_proj)."""
    base = reader.prefix + path[len("model."):]
    out = {}
    for suffix in ("weight", "scales", "biases", "signs"):
        name = f"{base}.{suffix}"
        with safe_open(reader.weight_map[name], framework="np") as f:
            out[suffix] = f.get_tensor(name)
        reader.read_names.append(name)
    return out


def plain_tensor(name: str, t: torch.Tensor) -> torch.Tensor:
    """A non-packed layer tensor, converted from the pack's MLX layout to transformers'."""
    if name.endswith("conv1d.weight") and t.shape[-1] == 1:
        t = t.moveaxis(1, 2).contiguous()  # (C, K, 1) -> (C, 1, K)
    if name.endswith(PLUS_ONE_NORMS):
        t = t - 1.0
    return t


def load_pack_layer(reader, glob: int, factory, records, device, dtype, make_linear):
    """One decoder layer of a Hadamard pack. `make_linear(w, in, out)` turns the dequantised float32 W'
    (out x in, on the CPU) into the inner linear: an nn.Linear in `dtype`, or a Linear4bit."""
    with torch.device("meta"):
        layer = factory()
    mine = f"model.layers.{glob}."
    packed = {r["path"]: r for r in records if r["path"].startswith(mine)}
    prefix = f"{reader.prefix}layers.{glob}."
    state = {}
    for key in layer.state_dict().keys():
        if key.endswith(".weight") and mine + key[: -len(".weight")] in packed:
            continue
        t = reader.load([prefix + key], device="cpu", dtype=torch.float32)[prefix + key]
        state[key] = plain_tensor(key, t).to(device=device, dtype=dtype)
    layer.load_state_dict(state, strict=False, assign=True)
    for path, rec in packed.items():
        if rec.get("embedding"):
            continue
        parts = path[len(mine):].split(".")
        parent = layer
        for part in parts[:-1]:
            parent = getattr(parent, part)
        old = getattr(parent, parts[-1])
        raw = pack_tensors(reader, path)
        w = dequantize_affine2(raw["weight"], raw["scales"], raw["biases"])
        inner = make_linear(w, old.in_features, old.out_features)
        del w
        signs = torch.from_numpy(raw["signs"].astype(np.float32)).to(device)
        setattr(parent, parts[-1], HadamardLinear(inner, signs, int(rec["block"])))
    return layer
