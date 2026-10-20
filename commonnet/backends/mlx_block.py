"""MLX block server: decoder layers [start, end) of a Qwen3 checkpoint on Apple Silicon.

Layers are mlx-lm's Qwen3 TransformerBlock. This file loads only this block's
tensors, quantizes the linears to 4 bits (or uses an already-quantized MLX
checkpoint), keeps a KV cache per session, and implements the BlockRange
interface: forward(session_id, hidden, start_pos), close(session_id).

Hidden states cross the boundary as float16 bytes (commonnet.wire). Inside, the
block computes in float16.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx.utils import tree_flatten
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3 import ModelArgs, TransformerBlock

from commonnet.wire import decode_fp16, encode_fp16

INDEX_NAME = "model.safetensors.index.json"
SINGLE_NAME = "model.safetensors"


def _weight_files(checkpoint_dir: str) -> dict[str, str]:
    """Map each tensor name to the safetensors file that holds it."""
    index_path = os.path.join(checkpoint_dir, INDEX_NAME)
    if os.path.exists(index_path):
        with open(index_path) as f:
            return {n: os.path.join(checkpoint_dir, s) for n, s in json.load(f)["weight_map"].items()}
    single = os.path.join(checkpoint_dir, SINGLE_NAME)
    if os.path.exists(single):
        return {n: single for n in mx.load(single).keys()}
    raise FileNotFoundError(f"no safetensors checkpoint in {checkpoint_dir}")


@dataclass
class _Session:
    caches: list[KVCache]
    length: int = 0


@dataclass
class MlxBlockRange:
    checkpoint_dir: str
    start: int
    end: int
    group_size: int = 64
    bits: int | None = 4  # None keeps the linears unquantized (a correctness check)
    dtype: mx.Dtype = mx.float16
    require_metal: bool = True
    layer_types_ok: bool = field(init=False, default=True)

    def __post_init__(self):
        if self.require_metal and not mx.metal.is_available():
            raise RuntimeError("MLX block needs the Metal GPU; none is available")
        with open(os.path.join(self.checkpoint_dir, "config.json")) as f:
            cfg = json.load(f)
        n = cfg["num_hidden_layers"]
        if not (0 <= self.start < self.end <= n):
            raise ValueError(f"bad layer range [{self.start}, {self.end}) for {n} layers")
        layer_types = cfg.get("layer_types") or []
        if any(t == "sliding_attention" for t in layer_types[self.start : self.end]):
            raise NotImplementedError("sliding-window attention layers are not supported")
        self.config = cfg
        self.hidden_size = cfg["hidden_size"]
        args = ModelArgs.from_dict(cfg)

        files = _weight_files(self.checkpoint_dir)
        # A checkpoint is pre-quantized if it stores .scales next to the layer weights.
        prequantized = f"model.layers.{self.start}.self_attn.q_proj.scales" in files
        qcfg = cfg.get("quantization") or {}
        if prequantized:
            group_size, bits = qcfg["group_size"], qcfg["bits"]
        else:
            group_size, bits = self.group_size, self.bits
        if bits is None and prequantized:
            raise ValueError("bits=None is only for float checkpoints")

        cache_files: dict[str, dict] = {}

        def tensor(name: str) -> mx.array:
            path = files[name]
            if path not in cache_files:
                cache_files[path] = mx.load(path)
            return cache_files[path][name]

        self.layers: list[TransformerBlock] = []
        self.loaded_names: list[str] = []
        for glob in range(self.start, self.end):
            blk = TransformerBlock(args)
            prefix = f"model.layers.{glob}."
            if prequantized:
                nn.quantize(blk, group_size=group_size, bits=bits)
                keys = [k for k, _ in tree_flatten(blk.parameters())]
                pairs = [(k, tensor(prefix + k)) for k in keys]
                blk.load_weights(pairs, strict=True)
            else:
                keys = [k for k, _ in tree_flatten(blk.parameters())]
                pairs = [(k, tensor(prefix + k)) for k in keys]
                blk.load_weights(pairs, strict=True)
                if bits is not None:
                    nn.quantize(blk, group_size=group_size, bits=bits)
            blk.set_dtype(self.dtype)
            mx.eval(blk.parameters())
            self.loaded_names += [prefix + k for k in keys]
            self.layers.append(blk)
        self.num_local = self.end - self.start
        self._sessions: dict[str, _Session] = {}

    def forward(self, session_id: str, hidden_states: torch.Tensor, start_pos: int) -> torch.Tensor:
        if hidden_states.dim() != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"expected (batch, seq, {self.hidden_size}), got {tuple(hidden_states.shape)}")
        data, shape = encode_fp16(hidden_states)
        x = mx.array(np.frombuffer(data, dtype=np.float16).reshape(shape))
        q_len = shape[1]

        if start_pos == 0:
            sess = _Session(caches=[KVCache() for _ in range(self.num_local)])
            self._sessions[session_id] = sess
        else:
            sess = self._sessions.get(session_id)
            if sess is None:
                raise KeyError(f"unknown session {session_id!r} (start_pos={start_pos})")
            if sess.length != start_pos:
                raise ValueError(f"session {session_id!r} holds {sess.length} tokens, got start_pos={start_pos}")

        if q_len == 1:
            mask = None
        elif start_pos == 0:
            mask = "causal"
        else:
            mask = create_causal_mask(q_len, start_pos)

        for blk, cache in zip(self.layers, sess.caches):
            x = blk(x, mask, cache)
        sess.length += q_len
        out = np.array(x.astype(mx.float16))
        out_bytes, out_shape = out.tobytes(), out.shape
        return decode_fp16(out_bytes, out_shape)

    def session_length(self, session_id: str) -> int:
        sess = self._sessions.get(session_id)
        return 0 if sess is None else sess.length

    def close(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def open_sessions(self) -> int:
        return len(self._sessions)

    def memory_bytes(self) -> dict[str, int]:
        """MLX's own accounting: active (live) and peak memory in bytes."""
        return {"active": mx.get_active_memory(), "peak": mx.get_peak_memory()}
