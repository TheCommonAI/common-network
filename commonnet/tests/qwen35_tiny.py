"""Tiny random qwen3_5 checkpoint with the real tensor layout (model.language_model.*, lm_head)."""
from __future__ import annotations

import json
import os

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM


def tiny_text_config(layers: int = 8) -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        layer_types=["linear_attention" if (i + 1) % 4 else "full_attention" for i in range(layers)],
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        max_position_embeddings=256, tie_word_embeddings=False, initializer_range=0.1,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 0], "mrope_interleaved": True},
    )


def make_qwen35_checkpoint(directory: str, layers: int = 8, seed: int = 0, shards: int = 1):
    """Returns the reference Qwen3_5ForCausalLM (fp32). Tensors are saved under the real names."""
    torch.manual_seed(seed)
    tc = tiny_text_config(layers)
    model = Qwen3_5ForCausalLM(tc).float().eval()
    sd = {}
    for k, v in model.state_dict().items():
        if k.startswith("model."):
            k = "model.language_model." + k[len("model."):]
        sd[k] = v.detach().clone().contiguous()
    os.makedirs(directory, exist_ok=True)
    cfg = Qwen3_5Config(text_config=tc.to_dict(), vision_config={"depth": 1, "hidden_size": 32,
                        "intermediate_size": 32, "num_heads": 2, "out_hidden_size": tc.hidden_size})
    cfg.tie_word_embeddings = False
    cfg.save_pretrained(directory)
    names = sorted(sd)
    if shards == 1:
        save_file(sd, os.path.join(directory, "model.safetensors"))
    else:
        wm = {}
        for i in range(shards):
            part = {n: sd[n] for n in names[i::shards]}
            fn = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
            save_file(part, os.path.join(directory, fn))
            wm.update({n: fn for n in part})
        with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": wm}, f)
    return model
