"""MLX block server: decoder layers [start, end) of a Qwen3 checkpoint on Apple Silicon.

Layers are mlx-lm's Qwen3 TransformerBlock. This file loads only this block's
tensors, quantizes the linears to 4 bits (or uses an already-quantized MLX
checkpoint), keeps a KV cache per session, and implements the BlockRange
interface: forward(session_id, hidden, start_pos), close(session_id).

Hidden states cross the boundary as float16 bytes (commonnet.wire) or, when the
caller sends float32, as float32. Inside, the block computes in float16 unless
dtype says otherwise. Qwen checkpoints are trained in bfloat16 and carry
activations beyond float16's range on some prompts, which come out as NaN; for
those, compute in bfloat16 (COMMONNET_DTYPE=bfloat16) and send float32.
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
from mlx_lm.models.base import create_attention_mask, create_causal_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
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


def _sanitize_qwen35(pairs, unsanitized):
    """HF qwen3_5 floats -> mlx-lm layout: conv1d (C,1,K) -> (C,K,1); RMSNorm weights get +1.
    A checkpoint already converted for MLX (conv1d last dim 1) is left alone."""
    out = []
    for k, v in pairs:
        if "conv1d.weight" in k and v.shape[-1] != 1:
            v = v.moveaxis(2, 1)
        if unsanitized and v.ndim == 1 and k.endswith(("input_layernorm.weight", "post_attention_layernorm.weight",
                                                       "q_norm.weight", "k_norm.weight")):
            v = v + 1.0
        out.append((k, v))
    return out


@dataclass
class _Session:
    caches: list
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
        # MLX keeps freed buffers for reuse. Long prompts make large attention temporaries, and left
        # alone the cache grew to ~11 GB per block at 16k tokens, swapping a 16 GB Mac. Cap it.
        cache_mb = int(os.environ.get("COMMONNET_MLX_CACHE_MB", "1024"))
        (getattr(mx, "set_cache_limit", None) or mx.metal.set_cache_limit)(cache_mb * 1024 * 1024)
        with open(os.path.join(self.checkpoint_dir, "config.json")) as f:
            cfg = json.load(f)
        wrapper = cfg
        cfg = cfg.get("text_config", cfg)  # qwen3_5 nests the decoder config
        n = cfg["num_hidden_layers"]
        if not (0 <= self.start < self.end <= n):
            raise ValueError(f"bad layer range [{self.start}, {self.end}) for {n} layers")
        layer_types = cfg.get("layer_types") or []
        if any(t == "sliding_attention" for t in layer_types[self.start : self.end]):
            raise NotImplementedError("sliding-window attention layers are not supported")
        self.config = cfg
        self.hidden_size = cfg["hidden_size"]
        files = _weight_files(self.checkpoint_dir)
        model_type = str(wrapper.get("model_type", ""))
        # A Hadamard pack (Prism ML Bonsai 2): qwen3_5 layers whose 2-bit linears sit in a rotated basis.
        self.hadamard = model_type == "prism_hadamard_qwen35"
        self.qwen35 = self.hadamard or model_type.startswith("qwen3_5")
        if self.qwen35:
            from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs
            args = TextModelArgs.from_dict(cfg)
            self.layer_is_linear = [(i + 1) % args.full_attention_interval != 0 for i in range(n)]
            lp = next((p for p in ("model.language_model.", "language_model.model.")
                       if any(k.startswith(p) for k in files)), "model.")
            make_block = lambda glob: DecoderLayer(args, glob)
        else:
            args = ModelArgs.from_dict(cfg)
            lp = "model."
            make_block = lambda glob: TransformerBlock(args)
        # A checkpoint is pre-quantized if it stores .scales next to the layer weights.
        prequantized = f"{lp}layers.{self.start}.mlp.up_proj.scales" in files
        qcfg = wrapper.get("quantization") or cfg.get("quantization") or {}
        if self.hadamard and not prequantized:
            raise ValueError("a Hadamard pack stores 2-bit weights; this checkpoint has none for these layers")
        if prequantized:
            group_size, bits = qcfg["group_size"], qcfg["bits"]
        else:
            group_size, bits = self.group_size, self.bits
        if bits is None and prequantized:
            raise ValueError("bits=None is only for float checkpoints")

        cache_files: dict[str, dict] = {}
        hf_layout = False
        if self.qwen35 and not prequantized:
            # HF layout has conv1d (C, 1, K) and (1 + w) norms; MLX conversions have conv1d (C, K, 1).
            first_lin = self.layer_is_linear.index(True)
            hf_layout = mx.load(files[f"{lp}layers.{first_lin}.linear_attn.conv1d.weight"])[
                f"{lp}layers.{first_lin}.linear_attn.conv1d.weight"].shape[-1] != 1

        def tensor(name: str) -> mx.array:
            path = files[name]
            if path not in cache_files:
                cache_files[path] = mx.load(path)
            return cache_files[path][name]

        self.layers: list = []
        self.loaded_names: list[str] = []
        for glob in range(self.start, self.end):
            blk = make_block(glob)
            prefix = f"{lp}layers.{glob}."
            if prequantized:
                if self.hadamard:
                    from commonnet.backends import hadamard
                    if (group_size, bits) != (hadamard.GROUP_SIZE, hadamard.BITS):
                        raise ValueError(f"unsupported Hadamard quantization: {bits}-bit, group {group_size}")
                    hadamard.install(blk, wrapper["modules"], f"model.layers.{glob}.",
                                     lambda path, suffix: tensor(f"{lp}{path[len('model.'):]}.{suffix}"))
                else:
                    nn.quantize(blk, group_size=group_size, bits=bits)
                keys = [k for k, _ in tree_flatten(blk.parameters())]
                pairs = [(k, tensor(prefix + k)) for k in keys]
                blk.load_weights(pairs, strict=True)
            else:
                keys = [k for k, _ in tree_flatten(blk.parameters())]
                pairs = [(k, tensor(prefix + k)) for k in keys]
                if self.qwen35:
                    pairs = _sanitize_qwen35(pairs, hf_layout)
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
        wide = hidden_states.dtype == torch.float32
        if wide:
            x = mx.array(hidden_states.detach().cpu().numpy()).astype(self.dtype)
        else:
            data, shape = encode_fp16(hidden_states)
            x = mx.array(np.frombuffer(data, dtype=np.float16).reshape(shape)).astype(self.dtype)
        q_len = hidden_states.shape[1]

        if start_pos == 0:
            if self.qwen35:
                caches = [ArraysCache(size=2) if self.layer_is_linear[self.start + i] else KVCache()
                          for i in range(self.num_local)]
            else:
                caches = [KVCache() for _ in range(self.num_local)]
            sess = _Session(caches=caches)
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

        if self.qwen35:
            # one mask per kind: full attention uses a causal mask, linear attention a (usually None) ssm mask
            lin = [i for i in range(self.num_local) if self.layer_is_linear[self.start + i]]
            full = [i for i in range(self.num_local) if not self.layer_is_linear[self.start + i]]
            fa_mask = create_attention_mask(x, sess.caches[full[0]]) if full else None
            ssm_mask = create_ssm_mask(x, sess.caches[lin[0]]) if lin else None
            for i, (blk, cache) in enumerate(zip(self.layers, sess.caches)):
                x = blk(x, ssm_mask if blk.is_linear else fa_mask, cache)
        else:
            for blk, cache in zip(self.layers, sess.caches):
                x = blk(x, mask, cache)
        sess.length += q_len
        if wide:
            return torch.from_numpy(np.array(x.astype(mx.float32)))
        out = np.array(x.astype(mx.float16))
        out_bytes, out_shape = out.tobytes(), out.shape
        return decode_fp16(out_bytes, out_shape)

    def session_length(self, session_id: str) -> int:
        sess = self._sessions.get(session_id)
        return 0 if sess is None else sess.length

    def truncate(self, session_id: str, length: int) -> None:
        """Cut the session's KV cache back to `length` positions (speculative rollback). Idempotent."""
        sess = self._sessions.get(session_id)
        if sess is None:
            raise KeyError(f"unknown session {session_id!r}")
        if length >= sess.length:
            return
        if not all(c.is_trimmable() for c in sess.caches):
            raise NotImplementedError("this block has non-attention layers; its cache cannot be truncated")
        drop = sess.length - length
        for c in sess.caches:
            c.trim(drop)
        sess.length = length

    def close(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def open_sessions(self) -> int:
        return len(self._sessions)

    def memory_bytes(self) -> dict[str, int]:
        """MLX's own accounting: active (live) and peak memory in bytes."""
        return {"active": mx.get_active_memory(), "peak": mx.get_peak_memory()}


class OnWorkerThread:
    """Runs one MlxBlockRange, built and called, on a single dedicated thread.

    MLX streams are per-thread: weights, KV cache and linear-attention state made
    on one thread cannot be used from another. The HTTP server handles each request
    on a new thread, so every call is handed to this one worker.
    """

    def __init__(self, factory, *args, **kwargs):
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-worker")
        try:
            self._block = self._pool.submit(factory, *args, **kwargs).result()
        except BaseException:
            self._pool.shutdown(wait=True)
            raise

    def _run(self, name, *args):
        return self._pool.submit(getattr(self._block, name), *args).result()

    @property
    def layers(self):
        return getattr(self._block, "layers", (self._block.start, self._block.end))

    @property
    def start(self):
        return self._block.start

    @property
    def end(self):
        return self._block.end

    @property
    def device(self):
        return getattr(self._block, "device", "metal")

    def open(self, session_id):
        return self._run("open", session_id) if hasattr(self._block, "open") else None

    def close(self, session_id):
        return self._run("close", session_id)

    def forward(self, session_id, hidden_states, start_pos):
        return self._run("forward", session_id, hidden_states, start_pos)

    def session_length(self, session_id):
        return self._run("session_length", session_id)

    def truncate(self, session_id, length):
        return self._run("truncate", session_id, length)

    def open_sessions(self):
        return self._run("open_sessions")

    def memory_bytes(self):
        return self._run("memory_bytes")

    def shutdown(self):
        self._pool.shutdown(wait=True)
