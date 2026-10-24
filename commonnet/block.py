"""A block server's core: decoder layers [start, end) of a Qwen3 checkpoint.

The decoder layers are transformers' own Qwen3DecoderLayer modules. This file
supplies the parts a block needs around them: loading only its tensors, the
rotary embedding, a causal mask, and a KV cache per session.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import partial

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RotaryEmbedding,
)

from commonnet.weights import CheckpointReader, build_module


def layer_classes(arch: str):
    """(decoder layer class, rotary class) for an architecture name."""
    if arch == "qwen3_5":
        import transformers
        if not transformers.__version__.startswith("5.8."):
            raise RuntimeError(
                f"Qwen3.6 support is tested with transformers 5.8.x, found {transformers.__version__}. "
                'Run: pip install "transformers==5.8.0"')
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5TextRotaryEmbedding,
        )
        return Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding
    return Qwen3DecoderLayer, Qwen3RotaryEmbedding


def build_rotary(rotary_cls, config, device):
    """Build a rotary embedding. Newer transformers releases (5.20) dropped the `device` argument."""
    try:
        return rotary_cls(config, device=device)
    except TypeError:
        return rotary_cls(config).to(device)


class _LinState:
    conv_states = None
    recurrent_states = None


class KVCache:
    """Per-session keys and values, one entry per local layer.

    Matches the `update(key, value, layer_idx)` call that Qwen3Attention makes,
    so the transformers attention code can use it directly.
    """

    def __init__(self, num_layers: int):
        self.keys: list[torch.Tensor | None] = [None] * num_layers
        self.values: list[torch.Tensor | None] = [None] * num_layers
        # Gated-DeltaNet layers (qwen3_5): fixed-size conv and recurrent state per layer.
        self.layers = [_LinState() for _ in range(num_layers)]

    def has_previous_state(self, layer_idx: int = 0) -> bool:
        return self.layers[layer_idx].recurrent_states is not None

    def update_conv_state(self, state, layer_idx: int):
        self.layers[layer_idx].conv_states = state
        return state

    def update_recurrent_state(self, state, layer_idx: int):
        self.layers[layer_idx].recurrent_states = state
        return state

    def update(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int):
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = key
            self.values[layer_idx] = value
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], key], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], value], dim=2)
        return self.keys[layer_idx], self.values[layer_idx]


@dataclass
class _Session:
    cache: KVCache
    length: int = 0  # tokens already in the cache


@dataclass
class BlockRange:
    """Serves decoder layers [start, end) of the checkpoint in checkpoint_dir.

    forward(session_id, hidden_states, start_pos) runs the hidden states through
    the local layers. start_pos is the absolute position of the first new token.
    It must be 0 for a new session (this resets any old cache with that id) or
    equal to the number of tokens the session already holds.
    """

    checkpoint_dir: str
    start: int
    end: int
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32
    config: Qwen3Config = field(init=False)
    loaded_names: list[str] = field(init=False, default_factory=list)

    def __post_init__(self):
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

        self.device = torch.device(self.device)
        self.num_local = self.end - self.start
        self.arch = reader.arch
        layer_cls, rotary_cls = layer_classes(self.arch)
        self.rotary_emb = build_rotary(rotary_cls, config, self.device)
        self.layer_types = list(layer_types) if layer_types else ["full_attention"] * n

        # Each block loads only its own layers. The tensor names come from the
        # module's own state_dict keys, not a hard-coded list.
        layers = []
        for local in range(self.num_local):
            glob = self.start + local
            factory = partial(layer_cls, config, layer_idx=glob)
            with torch.device("meta"):
                keys = list(factory().state_dict().keys())
            prefix = f"{reader.prefix}layers.{glob}."
            tensors = reader.load([prefix + k for k in keys], device=self.device, dtype=self.dtype)
            state = {k[len(prefix):]: v for k, v in tensors.items()}
            layer = build_module(factory, state)
            # The attention code indexes the session cache by layer_idx, so the
            # local index is set here. The global index was used above only to
            # pick the right layer type.
            for mod in (getattr(layer, "self_attn", None), getattr(layer, "linear_attn", None)):
                if mod is not None:
                    mod.layer_idx = local
            layers.append(layer)
        self.layers = torch.nn.ModuleList(layers)
        self.loaded_names = list(reader.read_names)

        self._sessions: dict[str, _Session] = {}
        self.layers.eval()

    @torch.inference_mode()
    def forward(self, session_id: str, hidden_states: torch.Tensor, start_pos: int) -> torch.Tensor:
        if hidden_states.dim() != 3 or hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(f"expected (batch, seq, {self.config.hidden_size}), got {tuple(hidden_states.shape)}")
        hidden = hidden_states.to(device=self.device, dtype=self.dtype)
        q_len = hidden.shape[1]

        if start_pos == 0:
            session = _Session(cache=KVCache(self.num_local))
            self._sessions[session_id] = session
        else:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"unknown session {session_id!r} (start_pos={start_pos})")
            if session.length != start_pos:
                raise ValueError(
                    f"session {session_id!r} holds {session.length} tokens, got start_pos={start_pos}"
                )

        position_ids = torch.arange(start_pos, start_pos + q_len, device=self.device).unsqueeze(0)
        # qwen3_5 uses 3-axis (t, h, w) rope; for text all three axes are the same position.
        rope_ids = position_ids[None].expand(3, -1, -1) if self.arch == "qwen3_5" else position_ids
        position_embeddings = self.rotary_emb(hidden, rope_ids)
        mask = self._causal_mask(start_pos, q_len, hidden.dtype)

        for layer, ltype in zip(self.layers, self.layer_types[self.start : self.end]):
            hidden = layer(
                hidden,
                attention_mask=None if ltype == "linear_attention" else mask,
                position_ids=position_ids,
                past_key_values=session.cache,
                use_cache=True,
                position_embeddings=position_embeddings,
            )
        session.length += q_len
        return hidden

    def truncate(self, session_id: str, length: int) -> None:
        """Cut the session's KV cache back to `length` positions (speculative rollback). Idempotent."""
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown session {session_id!r}")
        if length >= session.length:
            return
        if self.arch == "qwen3_5":
            raise NotImplementedError("this block has non-attention layers; its cache cannot be truncated")
        cache = session.cache
        for i in range(len(cache.keys)):
            if cache.keys[i] is not None:
                cache.keys[i] = cache.keys[i][:, :, :length].contiguous()
                cache.values[i] = cache.values[i][:, :, :length].contiguous()
        session.length = length

    def _causal_mask(self, start_pos: int, q_len: int, dtype: torch.dtype) -> torch.Tensor:
        kv_len = start_pos + q_len
        q_pos = torch.arange(start_pos, kv_len, device=self.device)
        k_pos = torch.arange(kv_len, device=self.device)
        allowed = k_pos[None, :] <= q_pos[:, None]
        mask = torch.zeros(q_len, kv_len, device=self.device, dtype=dtype)
        mask = mask.masked_fill(~allowed, torch.finfo(dtype).min)
        return mask[None, None]

    def session_length(self, session_id: str) -> int:
        session = self._sessions.get(session_id)
        return 0 if session is None else session.length

    def close(self, session_id: str) -> None:
        """Drop a session's KV cache. Closing an unknown id is a no-op."""
        self._sessions.pop(session_id, None)

    def open_sessions(self) -> int:
        return len(self._sessions)
