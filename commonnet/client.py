"""Generation client with negotiated token-only or legacy hidden-state transport.

On a capable route, the first peer embeds input IDs, peers forward activations,
and the final peer returns output IDs. Legacy routes load local embedding/norm/
head tables lazily. Each block holds its own KV cache under the session ID.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from functools import partial
from typing import Iterator, Protocol, Sequence

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm


def make_norm(arch, hidden, eps, prefix="model."):
    """Final norm module. qwen3_5 HF checkpoints use (1 + weight) scaling, plain Qwen3 uses weight.
    MLX conversions (prefix "language_model.model.") store the norm already as 1 + weight."""
    if arch == "qwen3_5" and prefix != "language_model.model.":
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


def sample_token(logits: torch.Tensor, temperature: float = 0.0, top_p: float = 1.0,
                 rng: torch.Generator | None = None) -> int:
    """Pick one token id from a 1-D logits vector. temperature <= 0 is greedy (argmax)."""
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, order = torch.sort(probs, descending=True)
        before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs  # probability mass ahead of each token
        sorted_probs = torch.where(before < top_p, sorted_probs, torch.zeros_like(sorted_probs))
        choice = torch.multinomial(sorted_probs / sorted_probs.sum(), 1, generator=rng)
        return int(order[choice])
    return int(torch.multinomial(probs, 1, generator=rng))


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
        self._reader = reader
        self._local_lock = threading.RLock()
        self._local_ready = False
        self.loaded_names = []
        self.client_bytes = 0

    def _ensure_local_weights(self) -> None:
        # Target tables are unnecessary for an ID route. Attribute access keeps the
        # original embed/head/norm API and loads them only for a legacy route or logits.
        if getattr(self, "_local_ready", True):
            return
        with self._local_lock:
            if not self._local_ready:
                loader = getattr(self, "_endpoint_checkpoint_loader", None)
                if loader is not None:
                    # NetworkBackend starts with tokenizer/config/index only. An
                    # old route triggers the target table download exactly here.
                    self._reader = CheckpointReader(loader())
                    self._endpoint_checkpoint_loader = None
                self._load_local_weights()
                self._local_ready = True

    @property
    def embed(self):
        self._ensure_local_weights()
        return self._embed

    @embed.setter
    def embed(self, value):
        self._embed = value

    @property
    def norm(self):
        self._ensure_local_weights()
        return self._norm

    @norm.setter
    def norm(self, value):
        self._norm = value

    @property
    def head(self):
        self._ensure_local_weights()
        return self._head

    @head.setter
    def head(self, value):
        self._head = value

    def _load_local_weights(self) -> None:
        embed_name = self._reader.prefix + "embed_tokens.weight"
        norm_name = self._reader.prefix + "norm.weight"
        head_name = self._reader.head_name  # tied embeddings fall back to the embedding
        tensors = self._reader.load(
            [embed_name, norm_name, head_name], device=self.device, dtype=self.dtype
        )
        vocab, hidden = self.config.vocab_size, self.config.hidden_size
        self.embed = build_module(
            partial(nn.Embedding, vocab, hidden), {"weight": tensors[embed_name]}
        )
        self.norm = build_module(
            partial(make_norm, self._reader.arch, hidden, self.config.rms_norm_eps, self._reader.prefix),
            {"weight": tensors[norm_name]},
        )
        self.head = build_module(
            partial(nn.Linear, hidden, vocab, bias=False), {"weight": tensors[head_name]}
        )
        self.loaded_names = list(self._reader.read_names)

        self.client_bytes = sum(t.numel() * t.element_size() for t in tensors.values())

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

    def _use_ids(self, endpoints) -> bool:
        """True when the route can run on token ids: the first peer embeds (ids_in) and the last returns ids
        (head_out). Only a transport that can ask (HttpTransport) ever says yes; anything else is today's path."""
        probe = getattr(self.transport, "ids_route", None)
        if probe is None or getattr(self, "ids_disabled", False):
            return False
        try:
            return bool(probe(list(endpoints)))
        except Exception:
            return False

    def _use_chain(self, endpoints) -> bool:
        """Negotiate before opening a session; failures here choose the legacy route."""
        probe = getattr(self.transport, "chain_route", None)
        if probe is None or getattr(self, "ids_disabled", False):
            return False
        try:
            return bool(probe(list(endpoints)))
        except Exception:
            return False

    def _step_ids(self, endpoints, session_id: str, token_ids, start_pos: int, which: str = "last",
                  sampling=None, *, chain: bool = False) -> list[int]:
        """One trip through the route on token ids: ids go to the first stage, hidden states are relayed between
        the middle stages, the last stage returns token ids (the last position, or every one for which="all")."""
        if chain:
            # Once a session uses /chain, errors propagate. Switching to /forward
            # mid-session could replay a partly executed step with different caches.
            reply = self.transport.chain_step(endpoints, session_id, list(token_ids), start_pos,
                                              which=which, sampling=sampling)
            return reply["ids"]
        hidden, n = None, len(endpoints)
        for i, ep in enumerate(endpoints):
            last = i == n - 1
            hidden = self.transport.forward(ep, session_id, hidden, start_pos,
                                            ids=list(token_ids) if i == 0 else None,
                                            output="ids" if last else None, which=which, sampling=sampling)
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
    def generate_stream(
        self,
        endpoints: Sequence[BlockEndpoint],
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        stop_ids: Sequence[int] | None = None,
    ) -> Iterator[int]:
        """Yield new token ids one at a time. The prompt goes through the blocks once; each later step sends one token.

        temperature 0 (the default) is greedy. Above 0, tokens are sampled from softmax(logits / temperature),
        cut to the smallest set whose probability reaches top_p; seed makes that reproducible. A token in
        stop_ids ends the stream and is not yielded. The session is closed when the stream ends or is closed.
        """
        self.check_route(endpoints, self.config.num_hidden_layers)
        if not prompt_ids:
            raise ValueError("prompt is empty")
        if max_new_tokens < 1:
            return
        stops = frozenset(stop_ids or ())
        rng = None
        if temperature > 0:
            rng = torch.Generator()
            if seed is not None:
                rng.manual_seed(seed)
            else:
                rng.seed()
        session_id = uuid.uuid4().hex
        produced = 0
        try:
            use_chain = self._use_chain(endpoints)
            if use_chain or self._use_ids(endpoints):
                sampling = None
                if temperature > 0:
                    sampling = {"temperature": float(temperature), "top_p": float(top_p),
                                "seed": int(seed) if seed is not None else int(torch.randint(0, 2 ** 31, (1,)))}
                tokens, pos = list(prompt_ids), 0
                while True:
                    next_id = int(self._step_ids(endpoints, session_id, tokens, pos, "last", sampling,
                                                        chain=use_chain)[-1])
                    pos += len(tokens)
                    if next_id in stops:
                        return
                    produced += 1
                    yield next_id
                    if produced == max_new_tokens:
                        return
                    tokens = [next_id]
            ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=self.device)
            pos = 0
            while True:
                hidden = self.embed(ids).to(self.dtype)
                hidden = self._run_blocks(endpoints, session_id, hidden, pos)
                pos += ids.shape[1]
                # The last position's logits pick the next token.
                logits = self._logits(hidden[:, -1:])[0, -1]
                next_id = sample_token(logits, temperature, top_p, rng)
                if next_id in stops:
                    return
                produced += 1
                yield next_id
                if produced == max_new_tokens:
                    return
                ids = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
        finally:
            for ep in endpoints:
                self.transport.close(ep, session_id)

    def generate(
        self,
        endpoints: Sequence[BlockEndpoint],
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        **sampling,
    ) -> list[int]:
        """Decoding to a list (greedy unless temperature/top_p/seed/stop_ids are given): see generate_stream."""
        return list(self.generate_stream(endpoints, prompt_ids, max_new_tokens, **sampling))
