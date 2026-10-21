"""Read only the named tensors from a Hugging Face safetensors checkpoint.

Works for a single `model.safetensors` and for a sharded checkpoint with a
`model.safetensors.index.json`. Only the shards that hold a requested tensor
are opened, and only the requested tensors are read from them.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Iterable

import torch
from safetensors import safe_open
from transformers import AutoConfig, Qwen3Config

INDEX_NAME = "model.safetensors.index.json"
SINGLE_NAME = "model.safetensors"


def load_text_config(checkpoint_dir: str):
    """Decoder config. For qwen3_5 the text_config of the (vision + text) wrapper config."""
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        mt = json.load(f).get("model_type", "")
    if mt.startswith("qwen3_5"):
        cfg = AutoConfig.from_pretrained(checkpoint_dir)
        return getattr(cfg, "text_config", cfg)
    return Qwen3Config.from_pretrained(checkpoint_dir)


class CheckpointReader:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.config = load_text_config(checkpoint_dir)
        index_path = os.path.join(checkpoint_dir, INDEX_NAME)
        single_path = os.path.join(checkpoint_dir, SINGLE_NAME)
        if os.path.exists(index_path):
            with open(index_path) as f:
                self.weight_map = {
                    name: os.path.join(checkpoint_dir, shard)
                    for name, shard in json.load(f)["weight_map"].items()
                }
        elif os.path.exists(single_path):
            with safe_open(single_path, framework="pt", device="cpu") as f:
                self.weight_map = {name: single_path for name in f.keys()}
        else:
            raise FileNotFoundError(f"no safetensors checkpoint in {checkpoint_dir}")
        # "model." for plain Qwen3, "model.language_model." for qwen3_5 (multimodal layout).
        # MLX conversions of the same model use "language_model.model.".
        self.prefix = next((p for p in ("model.language_model.", "language_model.model.")
                            if any(n.startswith(p) for n in self.weight_map)), "model.")
        self.head_name = next((n for n in ("lm_head.weight", "language_model.lm_head.weight")
                               if n in self.weight_map), self.prefix + "embed_tokens.weight")  # else tied
        self.arch = "qwen3_5" if getattr(self.config, "model_type", "").startswith("qwen3_5") else "qwen3"
        # Names actually read by load(), in order. Tests use it to prove that a
        # block never touches tensors outside its layer range.
        self.read_names: list[str] = []

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def load(
        self,
        names: Iterable[str],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        by_file: dict[str, list[str]] = {}
        for name in names:
            if name not in self.weight_map:
                raise KeyError(f"tensor {name!r} not in checkpoint")
            by_file.setdefault(self.weight_map[name], []).append(name)

        out: dict[str, torch.Tensor] = {}
        for path, file_names in by_file.items():
            with safe_open(path, framework="pt", device="cpu") as f:
                for name in file_names:
                    out[name] = f.get_tensor(name).to(device=device, dtype=dtype)
                    self.read_names.append(name)
        return out


def build_module(
    factory: Callable[[], torch.nn.Module],
    state: dict[str, torch.Tensor],
) -> torch.nn.Module:
    """Construct a module on the meta device and attach the loaded tensors.

    Avoids allocating and random-initialising full-size weights just to
    overwrite them. `assign=True` makes the loaded tensors the parameters.
    """
    with torch.device("meta"):
        module = factory()
    module.load_state_dict(state, assign=True, strict=True)
    return module
