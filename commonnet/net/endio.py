"""Embedding on the first-layer device and final norm/head on the last-layer device.

Only end windows construct EndIO. Tables are loaded lazily, on the same worker and
with the same torch device/dtype as their decoder layers. Packed MLX tables retain
their stored quantization on Metal; torch peers expand affine packs in bounded chunks.
"""
from __future__ import annotations

from functools import partial
import os
import threading

import torch
from torch import nn
from safetensors import safe_open

from commonnet.weights import CheckpointReader, build_module


class MissingTensors(RuntimeError):
    """The checkpoint has no usable embedding / norm / head for this window."""


def _base(name):
    return name.removesuffix(".weight")


def _head_name(reader):
    # CheckpointReader's fallback handles a tied head. Also detect a separately
    # packed head even if its index describes only scales beside the packed weight.
    return next((n for n in ("lm_head.weight", "language_model.lm_head.weight")
                 if reader.has(n) or reader.has(_base(n) + ".scales")), reader.head_name)


def _usable(reader, name):
    if not reader.has(name) or not os.path.isfile(reader.weight_map[name]):
        return False
    base = _base(name)
    if reader.has(name + ".absmax"):
        from commonnet.backends.nf4_block import NF4_SUFFIXES
        return all(reader.has(name + "." + suffix) and os.path.isfile(reader.weight_map[name + "." + suffix])
                   for suffix in NF4_SUFFIXES)
    if reader.has(base + ".scales"):
        return (reader.has(base + ".biases")
                and all(os.path.isfile(reader.weight_map[base + suffix]) for suffix in (".scales", ".biases")))
    return True


def _table_shape_ok(reader, name):
    """Inspect safetensors metadata before exposing a table to a device kernel."""
    if not _usable(reader, name):
        return False
    if reader.has(name + ".absmax"):
        return True  # bitsandbytes validates its stored shape when restoring QuantState
    base = _base(name)
    try:
        with safe_open(reader.weight_map[name], framework="pt", device="cpu") as f:
            tensor = f.get_slice(name)
            shape, dtype = tuple(tensor.get_shape()), tensor.get_dtype()
        vocab, hidden = reader.config.vocab_size, reader.config.hidden_size
        if reader.has(base + ".scales"):
            q = getattr(reader.config, "quantization", None) or getattr(reader.config, "quantization_config", None) or {}
            bits, group = int(q.get("bits", 4)), int(q.get("group_size", 64))
            if bits not in (2, 4, 8) or group < 1 or hidden % group or dtype != "U32":
                return False
            if shape != (vocab, hidden // (32 // bits)):
                return False
            for suffix in (".scales", ".biases"):
                n = base + suffix
                with safe_open(reader.weight_map[n], framework="pt", device="cpu") as f:
                    if tuple(f.get_slice(n).get_shape()) != (vocab, hidden // group):
                        return False
            if reader.has(base + ".signs"):
                n = base + ".signs"
                if not os.path.isfile(reader.weight_map[n]):
                    return False
                with safe_open(reader.weight_map[n], framework="pt", device="cpu") as f:
                    if tuple(f.get_slice(n).get_shape()) != (hidden,):
                        return False
            return True
        return shape == (vocab, hidden) and dtype in ("F16", "BF16", "F32", "F64")
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _norm_shape_ok(reader, name):
    if not _usable(reader, name):
        return False
    try:
        with safe_open(reader.weight_map[name], framework="pt", device="cpu") as f:
            return tuple(f.get_slice(name).get_shape()) == (reader.config.hidden_size,)
    except (OSError, ValueError, KeyError):
        return False


def caps_for(checkpoint: str, start: int, end: int, backend: str = "torch") -> tuple:
    """Advertise only complete end tables; no tensor allocation or weight download."""
    try:
        reader = CheckpointReader(checkpoint)
    except (OSError, ValueError, KeyError, TypeError):
        return ()
    embed, norm, head = reader.prefix + "embed_tokens.weight", reader.prefix + "norm.weight", _head_name(reader)
    caps = []
    if start == 0 and _table_shape_ok(reader, embed) and not reader.has(embed + ".absmax"):
        caps.append("ids_in")
    if end == reader.config.num_hidden_layers and _norm_shape_ok(reader, norm) and _table_shape_ok(reader, head):
        if backend == "nf4" or not reader.has(head + ".absmax"):
            caps.append("head_out")
    return tuple(caps)


def sample_at(logits: torch.Tensor, sampling: dict | None, position: int) -> int:
    """Position-seeded sampling makes an identical retry return the identical ID."""
    from commonnet.client import sample_token
    if not sampling:
        return int(logits.argmax())
    rng = torch.Generator().manual_seed((sampling["seed"] * 1000003 + position) % (2 ** 63))
    # A CPU generator and CPU logits are deliberate: CUDA generators cannot be
    # used by torch.multinomial on a different device.
    return sample_token(logits.float().cpu(), sampling["temperature"], sampling["top_p"], rng)


def _packed_weight(reader, name, device, dtype):
    """Expand an MLX affine table without allocating a full fp32 vocabulary matrix."""
    import numpy as np
    base = _base(name)
    q = getattr(reader.config, "quantization", None) or getattr(reader.config, "quantization_config", None) or {}
    bits, group = int(q.get("bits", 4)), int(q.get("group_size", 64))
    if bits not in (2, 4, 8):
        raise MissingTensors(f"unsupported packed end-table bit width {bits}")
    names = [name, base + ".scales", base + ".biases"]
    raw = {}
    for n in names:
        with safe_open(reader.weight_map[n], framework="np") as f:
            raw[n] = f.get_tensor(n)
        reader.read_names.append(n)
    words, scales, biases = (raw[n] for n in names)
    rows, cols = words.shape[0], words.shape[1] * (32 // bits)
    out = torch.empty((rows, cols), device=device, dtype=dtype)
    shifts = np.arange(0, 32, bits, dtype=np.uint32)
    mask = (1 << bits) - 1
    for a in range(0, rows, 512):
        b = min(rows, a + 512)
        codes = ((words[a:b, :, None].astype(np.uint32) >> shifts) & mask).reshape(b - a, -1)
        weight = codes.astype(np.float32).reshape(b - a, cols // group, group)
        weight = weight * scales[a:b, :, None].astype(np.float32) + biases[a:b, :, None].astype(np.float32)
        out[a:b].copy_(torch.from_numpy(weight.reshape(b - a, cols)).to(device=device, dtype=dtype))
    return out


class _RotatedEmbedding(nn.Module):
    """A Hadamard pack's embedding rows need the inverse activation transform."""
    def __init__(self, inner, signs, block):
        super().__init__()
        self.inner, self.block = inner, block
        self.register_buffer("signs", signs.float())

    def forward(self, ids):
        from commonnet.backends.hadamard_torch import _hadamard
        out = self.inner(ids)
        h = _hadamard(self.block, str(out.device))
        rotated = (out.float().reshape(-1, self.block) @ h).reshape(out.shape)
        return (rotated * self.signs).to(out.dtype)


class EndIO:
    def __init__(self, checkpoint: str, backend: str, *, want_embed: bool, want_head: bool,
                 device=None, dtype=None):
        self.checkpoint, self.backend = checkpoint, backend
        self.want_embed, self.want_head = want_embed, want_head
        self.device = torch.device(device or ("cuda" if backend == "nf4" else "cpu"))
        self.dtype = dtype or (torch.bfloat16 if backend in ("cpu", "nf4") else torch.float32)
        self._ready = False
        self._load_lock = threading.Lock()
        self.embed_mod = self.head_mod = self.norm = None
        self.nbytes = 0
        self.loaded_names = []
        config = CheckpointReader(checkpoint).config
        self.vocab_size, self.hidden_size = int(config.vocab_size), int(config.hidden_size)

    def _table(self, reader, name, kind):
        if not _table_shape_ok(reader, name):
            raise MissingTensors("end table is missing, incomplete, or has dimensions incompatible with the model")
        base = _base(name)
        packed = reader.has(base + ".scales")
        if self.backend == "mlx":
            from commonnet import client4
            q = getattr(reader.config, "quantization", None) or getattr(reader.config, "quantization_config", None) or {}
            bits, group = q.get("bits", 4), q.get("group_size", 64)
            if packed:
                if reader.has(base + ".signs"):
                    return client4._MlxHadamard(reader.weight_map, base, kind, reader.config.hidden_size)
                return client4._MlxQuantized.from_prequantized(reader.weight_map, base, kind, bits, group)
            return client4._MlxQuantized(reader.load([name])[name], kind, bits, group)
        if reader.has(name + ".absmax"):
            if kind == "embed" or self.backend != "nf4":
                raise MissingTensors("this backend cannot load an NF4-packed end table")
            import bitsandbytes as bnb
            from bitsandbytes.nn import Params4bit
            from commonnet.backends.nf4_block import _load_linear
            return _load_linear(bnb, Params4bit, reader, name,
                                nn.Linear(reader.config.hidden_size, reader.config.vocab_size, bias=False, device="meta"),
                                self.device, self.dtype)
        weight = (_packed_weight(reader, name, self.device, self.dtype) if packed else
                  reader.load([name], device=self.device, dtype=self.dtype)[name])
        if tuple(weight.shape) != (self.vocab_size, self.hidden_size):
            raise MissingTensors("end table dimensions do not match the model vocabulary and hidden width")
        layer = build_module(partial(nn.Embedding, weight.shape[0], weight.shape[1]) if kind == "embed" else
                             partial(nn.Linear, weight.shape[1], weight.shape[0], bias=False), {"weight": weight})
        if reader.has(base + ".signs"):
            from commonnet.backends.hadamard_torch import HadamardLinear
            signs = reader.load([base + ".signs"], device=self.device)[base + ".signs"]
            block = reader.config.hidden_size
            return (_RotatedEmbedding(layer, signs, block) if kind == "embed" else HadamardLinear(layer, signs, block))
        return layer

    def _load(self) -> None:
        with self._load_lock:
            if self._ready:
                return
            from commonnet.client import make_norm
            reader = CheckpointReader(self.checkpoint)
            embed, norm, head = reader.prefix + "embed_tokens.weight", reader.prefix + "norm.weight", _head_name(reader)
            need = ([embed] if self.want_embed else []) + ([norm, head] if self.want_head else [])
            if any(not _usable(reader, n) for n in need):
                raise MissingTensors("checkpoint lacks complete end-stage embedding, norm or head tensors")
            if self.want_embed:
                self.embed_mod = self._table(reader, embed, "embed")
            if self.want_head:
                self.head_mod = self._table(reader, head, "linear")
                norm_dtype = torch.float16 if self.backend == "mlx" else self.dtype
                norm_device = "cpu" if self.backend == "mlx" else self.device
                w = reader.load([norm], device=norm_device, dtype=norm_dtype)[norm]
                if tuple(w.shape) != (self.hidden_size,):
                    raise MissingTensors("final norm dimensions do not match the model hidden width")
                self.norm = build_module(partial(make_norm, reader.arch, reader.config.hidden_size,
                                                reader.config.rms_norm_eps, reader.prefix), {"weight": w})
                if self.want_embed and head == embed and self.backend != "mlx":
                    embed_layer = getattr(self.embed_mod, "inner", self.embed_mod)
                    head_layer = getattr(self.head_mod, "inner", self.head_mod)
                    if hasattr(embed_layer, "weight") and hasattr(head_layer, "weight"):
                        head_layer.weight = embed_layer.weight
            self.loaded_names = list(reader.read_names)
            seen = set()
            self.nbytes = 0
            for module in (self.embed_mod, self.norm, self.head_mod):
                if module is None:
                    continue
                if hasattr(module, "nbytes"):
                    self.nbytes += module.nbytes
                else:
                    for t in list(module.parameters()) + list(module.buffers()):
                        key = (str(t.device), t.data_ptr())
                        if key not in seen:
                            seen.add(key)
                            self.nbytes += t.numel() * t.element_size()
            self._ready = True

    @torch.inference_mode()
    def embed(self, ids: list[int], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if (not isinstance(ids, (list, tuple)) or not ids
                or any(type(t) is not int or not 0 <= t < self.vocab_size for t in ids)):
            raise ValueError("token ids must be a non-empty sequence of integers within the model vocabulary")
        self._load()
        if self.embed_mod is None:
            raise MissingTensors("this stage does not hold the embedding")
        t = torch.tensor([list(ids)], dtype=torch.long, device="cpu" if self.backend == "mlx" else self.device)
        return self.embed_mod(t).to(device="cpu", dtype=dtype)

    @torch.inference_mode()
    def head_ids(self, hidden: torch.Tensor, which: str, sampling: dict | None, start_pos: int) -> list[int]:
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1
                or hidden.shape[1] < 1 or hidden.shape[2] != self.hidden_size):
            raise ValueError("head input must have shape [1, positions, model hidden width]")
        self._load()
        if self.head_mod is None:
            raise MissingTensors("this stage does not hold the final norm and lm_head")
        h = hidden if which == "all" else hidden[:, -1:]
        first = start_pos if which == "all" else start_pos + hidden.shape[1] - 1
        if self.backend != "mlx":
            h = h.to(device=self.norm.weight.device, dtype=self.norm.weight.dtype)
        logits = self.head_mod(self.norm(h)).float()
        return [sample_at(logits[0, i], sampling, first + i) for i in range(logits.shape[1])]
