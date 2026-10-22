"""Client with a choice of format for the embedding, final norm and lm_head.

fmt:
  fp32  same as commonnet.client.Client (default there)
  fp16  tensors loaded as float16 (torch)
  mlx4  embedding and lm_head quantized to 4-bit with MLX (Mac only); the norm stays fp16
The 4-bit tables are built from the checkpoint's bf16/fp tensors at load time,
reading only the three tensors the client needs.
"""
from __future__ import annotations

from functools import partial

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from commonnet.client import Client, make_norm
from commonnet.weights import CheckpointReader, build_module

FORMATS = ("fp32", "fp16", "mlx4")


class _MlxQuantized:
    """torch -> torch callables around an MLX quantized embedding or linear."""

    def __init__(self, weight: torch.Tensor, kind: str, bits: int, group_size: int):
        import mlx.core as mx
        import mlx.nn as mnn
        self.mx = mx
        w = mx.array(weight.to(torch.float32).numpy()).astype(mx.float16)
        if kind == "embed":
            layer = mnn.Embedding(w.shape[0], w.shape[1])
            layer.weight = w
            self.layer = mnn.QuantizedEmbedding.from_embedding(layer, group_size, bits)
        else:
            layer = mnn.Linear(w.shape[1], w.shape[0], bias=False)
            layer.weight = w
            self.layer = mnn.QuantizedLinear.from_linear(layer, group_size, bits)
        mx.eval(self.layer.parameters())
        self.nbytes = sum(v.nbytes for _, v in __import__("mlx.utils", fromlist=["x"]).tree_flatten(self.layer.parameters()))

    @classmethod
    def from_prequantized(cls, weight_map: dict, name: str, kind: str, bits: int, group_size: int):
        """Wrap an MLX-quantized embed/head already stored as name.weight/.scales/.biases (uint32 packed)."""
        import mlx.core as mx
        import mlx.nn as mnn
        from mlx.utils import tree_flatten
        files: dict = {}

        def get(suffix):
            path = weight_map[f"{name}.{suffix}"]
            if path not in files:
                files[path] = mx.load(path)
            return files[path][f"{name}.{suffix}"]

        w, scales, biases = get("weight"), get("scales"), get("biases")
        self = cls.__new__(cls)
        self.mx = mx
        out_dim, packed = w.shape
        in_dim = packed * 32 // bits
        if kind == "embed":
            layer = mnn.QuantizedEmbedding(out_dim, in_dim, group_size, bits)
        else:
            layer = mnn.QuantizedLinear(in_dim, out_dim, bias=False, group_size=group_size, bits=bits)
        layer.weight, layer.scales, layer.biases = w, scales, biases
        mx.eval(layer.parameters())
        self.layer = layer
        self.nbytes = sum(v.nbytes for _, v in tree_flatten(layer.parameters()))
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mx = self.mx
        if x.dtype in (torch.long, torch.int32, torch.int64):
            arr = mx.array(x.cpu().numpy().astype("int32"))
        else:
            arr = mx.array(x.detach().to(torch.float32).cpu().numpy()).astype(mx.float16)
        out = self.layer(arr)
        mx.eval(out)
        import numpy as np
        return torch.from_numpy(np.array(out.astype(mx.float32)))


class Client4(Client):
    def __init__(self, checkpoint_dir: str, transport, fmt: str = "fp16", device="cpu",
                 bits: int = 4, group_size: int = 64, hidden_dtype=torch.float16):
        if fmt not in FORMATS:
            raise ValueError(f"fmt must be one of {FORMATS}, got {fmt!r}")
        if fmt != "mlx4":
            super().__init__(checkpoint_dir, transport, device=device,
                             dtype=torch.float32 if fmt == "fp32" else torch.float16)
            self.fmt = fmt
            return
        try:
            import mlx.core  # noqa: F401
        except ImportError:
            raise RuntimeError("fmt mlx4 needs MLX (Apple silicon); use fp16 elsewhere") from None
        reader = CheckpointReader(checkpoint_dir)
        self.config = reader.config
        self.transport = transport
        self.device = torch.device("cpu")
        self.dtype = hidden_dtype  # what crosses the wire: float32 avoids float16 overflow on Qwen
        self.fmt = fmt
        self.arch = reader.arch
        embed_name = reader.prefix + "embed_tokens.weight"
        norm_name = reader.prefix + "norm.weight"
        head_name = reader.head_name
        hidden = self.config.hidden_size
        if reader.has(embed_name.replace(".weight", ".scales")):
            # Pre-quantized MLX checkpoint (e.g. mlx-community/Qwen3-8B-4bit): the stored weight is
            # packed uint32. Casting it to float32 would give garbage, so use the stored 4-bit tables.
            q = getattr(self.config, "quantization", None) or getattr(self.config, "quantization_config", None) or {}
            bits, group_size = q.get("bits", bits), q.get("group_size", group_size)
            t = reader.load([norm_name], dtype=torch.float32)
            base = lambda n: n[: -len(".weight")]
            self.embed = _MlxQuantized.from_prequantized(reader.weight_map, base(embed_name), "embed", bits, group_size)
            self.head = _MlxQuantized.from_prequantized(reader.weight_map, base(head_name), "linear", bits, group_size)
            reader.read_names += [embed_name, head_name]
        else:
            t = reader.load([embed_name, norm_name, head_name], dtype=torch.float32)
            self.embed = _MlxQuantized(t[embed_name], "embed", bits, group_size)
            self.head = _MlxQuantized(t[head_name], "linear", bits, group_size)
        self.norm = build_module(partial(make_norm, reader.arch, hidden, self.config.rms_norm_eps, reader.prefix),
                                 {"weight": t[norm_name].to(torch.float16)})
        self.loaded_names = list(reader.read_names)
        self.client_bytes = self.embed.nbytes + self.head.nbytes + self.norm.weight.numel() * 2
