"""Client: embedding, final norm and head, plus a chain of block endpoints.

The client holds no decoder layers. For each generated token it sends the
hidden states through every block in route order. Each block keeps its own KV
cache under a session id, so after the prompt only the new token is sent.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from functools import partial
from typing import Protocol, Sequence

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm


def make_norm(arch, hidden, eps):
    """Final norm module. qwen3_5 uses (1 + weight) scaling, plain Qwen3 uses weight."""
    if arch == "qwen3_5":
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
        return Qwen3_5RMSNorm(hidden, eps=eps)
    return Qwen3RMSNorm(hidden, eps=eps)

from commonnet.block import BlockRange
from commonnet.weights import CheckpointReader, build_module


@dataclass(frozen=True)
class BlockEndpoint:
    """Where a block lives and which decoder layers [start, end) it serves."""

    name: str
    start: int
    end: int


class Transport(Protocol):
    """How the client reaches a block. Remote transports implement the same two calls."""

    def forward(self, endpoint: BlockEndpoint, session_id: str, hidden: torch.Tensor, start_pos: int) -> torch.Tensor:
        ...

    def close(self, endpoint: BlockEndpoint, session_id: str) -> None:
        ...


class LocalTransport:
    """Calls BlockRange objects in this process, keyed by endpoint name."""

    def __init__(self, blocks: dict[str, BlockRange]):
        self.blocks = blocks

    def _block(self, endpoint: BlockEndpoint) -> BlockRange:
        block = self.blocks[endpoint.name]
        if (block.start, block.end) != (endpoint.start, endpoint.end):
            raise ValueError(
                f"endpoint {endpoint.name!r} claims [{endpoint.start}, {endpoint.end}) "
                f"but block serves [{block.start}, {block.end})"
            )
        return block

    def forward(self, endpoint: BlockEndpoint, session_id: str, hidden: torch.Tensor, start_pos: int) -> torch.Tensor:
        return self._block(endpoint).forward(session_id, hidden, start_pos)

    def close(self, endpoint: BlockEndpoint, session_id: str) -> None:
        self._block(endpoint).close(session_id)


class Client:
    def __init__(
        self,
        checkpoint_dir: str,
        transport: Transport,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        reader = CheckpointReader(checkpoint_dir)
        self.config = reader.config
        self.transport = transport
        self.device = torch.device(device)
        self.dtype = dtype

        self.arch = reader.arch
        embed_name = reader.prefix + "embed_tokens.weight"
        norm_name = reader.prefix + "norm.weight"
        head_name = "lm_head.weight" if reader.has("lm_head.weight") else embed_name  # tied embeddings
        tensors = reader.load(
            [embed_name, norm_name, head_name], device=self.device, dtype=dtype
        )
        vocab, hidden = self.config.vocab_size, self.config.hidden_size
        self.embed = build_module(
            partial(nn.Embedding, vocab, hidden), {"weight": tensors[embed_name]}
        )
        self.norm = build_module(
            partial(make_norm, reader.arch, hidden, self.config.rms_norm_eps),
            {"weight": tensors[norm_name]},
        )
        self.head = build_module(
            partial(nn.Linear, hidden, vocab, bias=False), {"weight": tensors[head_name]}
        )
        self.loaded_names = list(reader.read_names)

    @staticmethod
    def check_route(endpoints: Sequence[BlockEndpoint], num_layers: int) -> None:
        """Endpoints must be ordered and cover [0, num_layers) with no gaps or overlaps."""
        pos = 0
        for ep in endpoints:
            if ep.start != pos or ep.end <= ep.start:
                raise ValueError(f"route is not contiguous at {ep}: expected start {pos}")
            pos = ep.end
        if pos != num_layers:
            raise ValueError(f"route covers [0, {pos}) but the model has {num_layers} layers")

    def _run_blocks(self, endpoints, session_id: str, hidden: torch.Tensor, start_pos: int) -> torch.Tensor:
        for ep in endpoints:
            hidden = self.transport.forward(ep, session_id, hidden, start_pos)
        return hidden

    @torch.inference_mode()
    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(hidden)).float()

    @torch.inference_mode()
    def logits(self, endpoints: Sequence[BlockEndpoint], input_ids: torch.Tensor) -> torch.Tensor:
        """Logits for every position of input_ids (batch, seq), with no reuse of earlier sessions."""
        self.check_route(endpoints, self.config.num_hidden_layers)
        session_id = uuid.uuid4().hex
        try:
            hidden = self.embed(input_ids.to(self.device))
            hidden = self._run_blocks(endpoints, session_id, hidden.to(self.dtype), 0)
            return self._logits(hidden)
        finally:
            for ep in endpoints:
                self.transport.close(ep, session_id)

    @torch.inference_mode()
    def generate(
        self,
        endpoints: Sequence[BlockEndpoint],
        prompt_ids: Sequence[int],
        max_new_tokens: int,
    ) -> list[int]:
        """Greedy decoding. The prompt goes through the blocks once. Each later step sends one token."""
        self.check_route(endpoints, self.config.num_hidden_layers)
        if not prompt_ids:
            raise ValueError("prompt is empty")
        session_id = uuid.uuid4().hex
        new_tokens: list[int] = []
        try:
            ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=self.device)
            pos = 0
            while True:
                hidden = self.embed(ids).to(self.dtype)
                hidden = self._run_blocks(endpoints, session_id, hidden, pos)
                pos += ids.shape[1]
                # The last position's logits pick the next token.
                logits = self._logits(hidden[:, -1:])
                next_id = int(logits[0, -1].argmax())
                new_tokens.append(next_id)
                if len(new_tokens) == max_new_tokens:
                    return new_tokens
                ids = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
        finally:
            for ep in endpoints:
                self.transport.close(ep, session_id)
