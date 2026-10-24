"""NF4 block server on PyTorch + bitsandbytes: decoder layers [start, end) on CUDA.

UNTESTED ON CUDA. The Mac that wrote this has no CUDA device, so the CUDA path has
never run. Only the refusal path and the CPU-side selection logic have been exercised. It must be
validated on the Windows RTX 4060 before anyone relies on it.

Two checkpoint kinds, chosen per Linear: a pre-quantized bitsandbytes tensor (it has a `.absmax` next to
it) loads as is; a floating-point weight (fp16/bf16/fp32) is quantised to NF4 on the GPU as it loads,
one Linear at a time, so peak VRAM stays near the quantised size plus one fp16 matrix.
Statistics are not double-quantised: the fp32 absmax keeps ternary weights (-s, 0, +s) exact.

It is a BlockRange subclass, so forward, close, session_length and open_sessions come
from commonnet.block. Only __post_init__ differs: each Linear is a bitsandbytes
Linear4bit loaded from a pre-quantized NF4 checkpoint (unsloth/Qwen3-32B-bnb-4bit
layout (also the community Qwen3.6-27B bnb repos, tensors under model.language_model.layers.N with
linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj} and self_attn.{q,k,v,o}_proj all NF4;
conv1d, A_log, dt_bias, norms stay bf16): weight, weight.absmax, weight.quant_map, weight.nested_absmax,
weight.nested_quant_map, weight.quant_state.bitsandbytes__nf4).
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from functools import partial

import torch
from safetensors import safe_open
from torch import nn

from commonnet.block import BlockRange, build_rotary, layer_classes
from commonnet.weights import CheckpointReader, build_module

NF4_SUFFIXES = (
    "absmax",
    "quant_map",
    "nested_absmax",
    "nested_quant_map",
    "quant_state.bitsandbytes__nf4",
)


def _raw_tensor(reader: CheckpointReader, name: str, device) -> torch.Tensor:
    """Read one tensor in its stored dtype. CheckpointReader would cast it, which breaks packed uint8."""
    path = reader.weight_map[name]
    with safe_open(path, framework="pt", device="cpu") as f:
        t = f.get_tensor(name)
    reader.read_names.append(name)
    return t.to(device)


def is_prequantized(weight_map, wname: str) -> bool:
    """True when `wname` is a bitsandbytes NF4 tensor (its absmax is stored beside it)."""
    return f"{wname}.absmax" in weight_map


def _load_linear(bnb, Params4bit, reader: CheckpointReader, wname: str, mod: nn.Linear, device, dtype):
    """A Linear4bit for `mod`: from its pre-quantized NF4 tensors, or quantised on the fly from float weights."""
    if is_prequantized(reader.weight_map, wname):
        stats = {s: _raw_tensor(reader, f"{wname}.{s}", device) for s in NF4_SUFFIXES}
        packed = _raw_tensor(reader, wname, device)
        new = bnb.nn.Linear4bit(mod.in_features, mod.out_features, bias=False, compute_dtype=dtype,
                                quant_type="nf4", device=device)
        new.weight = Params4bit.from_prequantized(
            data=packed, quantized_stats=stats, requires_grad=False, device=device, module=new
        )
        return new
    w = _raw_tensor(reader, wname, "cpu")
    if not w.dtype.is_floating_point or tuple(w.shape) != (mod.out_features, mod.in_features):
        raise ValueError(f"{wname}: expected a {mod.out_features}x{mod.in_features} float weight or an NF4 tensor, "
                         f"got {tuple(w.shape)} {w.dtype}")
    return _quantise(bnb, Params4bit, w, mod.in_features, mod.out_features, device, dtype)


def _quantise(bnb, Params4bit, w: torch.Tensor, n_in: int, n_out: int, device, dtype):
    """A Linear4bit from a float (out x in) weight on the CPU: quantised to NF4 as it moves to the GPU."""
    if w.dtype not in (torch.float16, torch.bfloat16):
        w = w.to(torch.float16)
    new = bnb.nn.Linear4bit(n_in, n_out, bias=False, compute_dtype=dtype, quant_type="nf4", device="meta")
    # Params4bit quantises when it is moved to the GPU. The fp16 copy is freed as soon as `w` goes out of scope.
    new.weight = Params4bit(w, requires_grad=False, quant_type="nf4", compress_statistics=False, module=new).to(device)
    return new


@dataclass
class Nf4BlockRange(BlockRange):
    device: torch.device | str = "cuda"
    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        # Refuse before anything else. bitsandbytes is imported only after this check.
        if not torch.cuda.is_available():
            raise RuntimeError("NF4 block refuses to start: no CUDA device was found")
        device = torch.device(self.device)
        if device.type != "cuda":
            raise RuntimeError(f"NF4 block runs on CUDA only, got device {device}")

        import bitsandbytes as bnb  # noqa: WPS433  (imported after the CUDA check on purpose)
        from bitsandbytes.nn import Params4bit

        reader = CheckpointReader(self.checkpoint_dir)
        config = copy.deepcopy(reader.config)
        config._attn_implementation = "sdpa"
        self.config = config
        n = config.num_hidden_layers
        if not (0 <= self.start < self.end <= n):
            raise ValueError(f"bad layer range [{self.start}, {self.end}) for {n} layers")
        layer_types = getattr(config, "layer_types", None) or []
        if any(t == "sliding_attention" for t in layer_types[self.start : self.end]):
            raise NotImplementedError("sliding-window attention layers are not supported")

        self.device = device
        self.num_local = self.end - self.start
        self.arch = reader.arch
        layer_cls, rotary_cls = layer_classes(self.arch)
        self.rotary_emb = build_rotary(rotary_cls, config, device)
        self.layer_types = list(layer_types) if layer_types else ["full_attention"] * n

        layers = []
        for local in range(self.num_local):
            glob = self.start + local
            if getattr(config, "hadamard_modules", None):
                # Bonsai 2 style Hadamard pack: dequantise the 2-bit projections, then NF4 them (hadamard_torch.py)
                from commonnet.backends.hadamard_torch import load_pack_layer
                layer = load_pack_layer(
                    reader, glob, partial(layer_cls, config, layer_idx=glob), config.hadamard_modules, device, self.dtype,
                    lambda w, n_in, n_out: _quantise(bnb, Params4bit, w, n_in, n_out, device, self.dtype))
                for mod in (getattr(layer, "self_attn", None), getattr(layer, "linear_attn", None)):
                    if mod is not None:
                        mod.layer_idx = local
                layer.to(device)
                layers.append(layer)
                torch.cuda.empty_cache()
                continue
            with torch.device("meta"):
                layer = layer_cls(config, layer_idx=glob)
            prefix = f"{reader.prefix}layers.{glob}."
            quantized = set()

            # Replace every Linear with a Linear4bit (pre-quantized NF4 tensors, or quantised from floats on load).
            for name, mod in list(layer.named_modules()):
                if not isinstance(mod, nn.Linear):
                    continue
                wname = prefix + name + ".weight"
                quantized.add(name + ".weight")
                new = _load_linear(bnb, Params4bit, reader, wname, mod, device, self.dtype)
                parent_name, _, child = name.rpartition(".")
                parent = layer.get_submodule(parent_name) if parent_name else layer
                setattr(parent, child, new)

            # Norms are not quantized. Load them in the compute dtype.
            # Everything that is not a quantized Linear (norms, conv1d, A_log, dt_bias) is a plain tensor.
            plain = {}
            for key in layer.state_dict().keys():
                if key in quantized:
                    continue
                plain[key] = _raw_tensor(reader, prefix + key, device).to(self.dtype)
            layer.load_state_dict(plain, strict=False, assign=True)
            for mod in (getattr(layer, "self_attn", None), getattr(layer, "linear_attn", None)):
                if mod is not None:
                    mod.layer_idx = local
            layer.to(device)
            layers.append(layer)
            torch.cuda.empty_cache()  # return the transient fp16 copies of this layer to the driver

        self.layers = nn.ModuleList(layers)
        self.loaded_names = list(reader.read_names)
        self._sessions = {}
        self.layers.eval()
