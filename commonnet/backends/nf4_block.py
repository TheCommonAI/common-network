"""NF4 block server on PyTorch + bitsandbytes: decoder layers [start, end) on CUDA.

UNTESTED ON CUDA. The Mac that wrote this has no CUDA device, so the CUDA path has
never run. Only the refusal path has been exercised. It must be validated on the
Windows RTX 4060 before anyone relies on it.

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

from commonnet.block import BlockRange, layer_classes
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
        self.rotary_emb = rotary_cls(config, device=device)
        self.layer_types = list(layer_types) if layer_types else ["full_attention"] * n

        layers = []
        for local in range(self.num_local):
            glob = self.start + local
            with torch.device("meta"):
                layer = layer_cls(config, layer_idx=glob)
            prefix = f"{reader.prefix}layers.{glob}."
            quantized = set()

            # Replace every Linear with a Linear4bit loaded from its NF4 tensors.
            for name, mod in list(layer.named_modules()):
                if not isinstance(mod, nn.Linear):
                    continue
                wname = prefix + name + ".weight"
                quantized.add(name + ".weight")
                stats = {s: _raw_tensor(reader, f"{wname}.{s}", device) for s in NF4_SUFFIXES}
                packed = _raw_tensor(reader, wname, device)
                new = bnb.nn.Linear4bit(
                    mod.in_features,
                    mod.out_features,
                    bias=False,
                    compute_dtype=self.dtype,
                    quant_type="nf4",
                    device=device,
                )
                new.weight = Params4bit.from_prequantized(
                    data=packed, quantized_stats=stats, requires_grad=False, device=device, module=new
                )
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

        self.layers = nn.ModuleList(layers)
        self.loaded_names = list(reader.read_names)
        self._sessions = {}
        self.layers.eval()
