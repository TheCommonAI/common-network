"""A synthetic transformer stand-in: deterministic weights from a seed, a dial for block compute time.

SynthBlock is a block the real BlockServer can serve. Each layer is a residual low-rank MLP over a causal
running mean of the session's earlier positions, so the output depends on the whole history (a KV cache
stand-in). A failover that replays the history therefore reproduces exactly the tokens an undisturbed run gives,
and a replay that went wrong shows up as different tokens. layer_ms sleeps per layer, so block compute is a knob.

SynthClient is the client side (embedding, final norm, head, sampling), also synthetic: no download.
"""
from __future__ import annotations

import threading
import time
import uuid
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from commonnet.client import BlockEndpoint, Client, sample_token
from commonnet.net import protocol
from commonnet.net.protocol import Tensor

RANK = 16
DEFAULT_SEED = 1234


def synth_tables(seed: int, vocab: int, hidden: int):
    """The synthetic embedding and head weights (shared by SynthClient and a SynthBlock serving the ends)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(vocab, hidden, generator=g), torch.randn(vocab, hidden, generator=g) / hidden ** 0.5


def _rng(seed: int, *key) -> np.random.Generator:
    return np.random.default_rng([seed, *key])


class SynthBlock:
    """Layers [start, end) of the synthetic model. Weights are made on first use for the hidden size seen."""

    device = "synth"

    def __init__(self, start: int, end: int, *, layer_ms: float = 0.0, seed: int = DEFAULT_SEED,
                 io_layers: int | None = None, hidden: int = 256, vocab: int = 2048, wire_dtype: str = "float16"):
        """io_layers (the model's layer count) turns on token-id I/O: the window starting at 0 takes ids (ids_in),
        the one ending at io_layers returns ids (head_out), with the same synthetic tables as SynthClient."""
        self.layers = protocol.check_layers((start, end))
        self.caps = () if io_layers is None else tuple(
            c for c, ok in (("ids_in", start == 0), ("head_out", end == io_layers)) if ok)
        self.hidden, self.vocab, self.wire_dtype = hidden, vocab, wire_dtype
        self._tables = None
        self.layer_ms = float(layer_ms)
        self.seed = seed
        self._weights: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._sessions: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _w(self, layer: int, hidden: int):
        key = (layer, hidden)
        with self._lock:
            if key not in self._weights:
                g = _rng(self.seed, layer, hidden)
                a = g.standard_normal((hidden, RANK), dtype=np.float32) / np.sqrt(hidden)
                b = g.standard_normal((RANK, hidden), dtype=np.float32) / np.sqrt(RANK)
                self._weights[key] = (a, b)
            return self._weights[key]

    def open(self, session_id: str) -> None:
        self._sessions[session_id] = {"pos": 0, "sum": None, "cs": None}

    def close(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def truncate(self, session_id: str, length: int) -> None:
        state = self._sessions[session_id]
        if length >= state["pos"]:
            return
        state["pos"] = length
        if length == 0:
            state["sum"], state["cs"] = None, None
        else:
            state["cs"] = state["cs"][:, :length]
            state["sum"] = state["cs"][:, -1:, :].copy()

    def forward(self, session_id: str, hidden: Tensor, start_pos: int) -> Tensor:
        state = self._sessions[session_id]
        if start_pos != state["pos"]:
            raise ValueError(f"start_pos {start_pos} but the session is at {state['pos']}")
        dtype = np.float16 if hidden.dtype == "float16" else np.float32
        x = np.frombuffer(hidden.data, dtype=dtype).reshape(hidden.shape).astype(np.float32)
        batch, seq, width = x.shape
        csum = np.cumsum(x, axis=1)
        if state["sum"] is not None:
            csum = csum + state["sum"]
        state["sum"] = csum[:, -1:, :].copy()
        # per-position running sums: what a truncate (rollback) restores
        state["cs"] = csum.copy() if state["cs"] is None else np.concatenate([state["cs"], csum], axis=1)
        counts = np.arange(state["pos"] + 1, state["pos"] + seq + 1, dtype=np.float32).reshape(1, seq, 1)
        ctx = csum / counts
        state["pos"] += seq
        h = x
        for layer in range(*self.layers):
            a, b = self._w(layer, width)
            h = h + 0.1 * np.tanh(((h + ctx) @ a) @ b)
        if self.layer_ms:
            time.sleep(self.layer_ms * (self.layers[1] - self.layers[0]) / 1000.0)
        return Tensor(hidden.shape, hidden.dtype, np.ascontiguousarray(h.astype(dtype)).tobytes())

    def _io(self):
        if self._tables is None:
            e, h = synth_tables(self.seed, self.vocab, self.hidden)
            self._tables = (e, h, nn.RMSNorm(self.hidden))
        return self._tables

    def embed(self, ids) -> Tensor:
        t = self._io()[0][torch.tensor(list(ids))].unsqueeze(0)
        dt = np.float16 if self.wire_dtype == "float16" else np.float32
        arr = np.ascontiguousarray(t.numpy().astype(dt))
        return Tensor(tuple(arr.shape), self.wire_dtype, arr.tobytes())

    def head(self, hidden: Tensor, which: str, sampling, start_pos: int) -> list[int]:
        from commonnet.net.endio import sample_at
        _, head, norm = self._io()
        dt = np.float16 if hidden.dtype == "float16" else np.float32
        h = torch.from_numpy(np.frombuffer(hidden.data, dtype=dt).reshape(hidden.shape).copy())
        seq = h.shape[1]
        if which != "all":
            h = h[:, -1:]
        first = start_pos if which == "all" else start_pos + seq - 1
        with torch.inference_mode():
            logits = torch.nn.functional.linear(norm(h.float()), head).float()
        return [sample_at(logits[0, i], sampling, first + i) for i in range(logits.shape[1])]

    def shutdown(self) -> None:
        pass


def manifest_dict(stages: int, layers: int, *, network_id: str | None = None) -> dict:
    """A manifest of `layers` layers in `stages` equal stages (layers must divide evenly)."""
    if layers % stages:
        raise ValueError(f"{layers} layers do not split into {stages} equal stages")
    step = layers // stages
    return {"id": network_id or f"test-synth-{stages}x{layers}", "version": 1,
            "model": {"name": "Synth", "num_layers": layers, "repos": {"synth": "synthetic"},
                      "client_repo": "synthetic"},
            "boundaries": list(range(0, layers + 1, step)), "gib_per_layer": {"synth": 1.0},
            "fixed_gib": 0.0, "min_replicas": 1, "seeds": []}


class SynthClient(Client):
    """The Client loop with a synthetic embedding, norm and head, and per-phase timing in self.steps."""

    def __init__(self, transport, *, layers: int, hidden: int = 256, vocab: int = 2048,
                 seed: int = DEFAULT_SEED, dtype=torch.float16):
        self.transport = transport
        self.device = torch.device("cpu")
        self.dtype = dtype
        self.config = SimpleNamespace(num_hidden_layers=layers, hidden_size=hidden, vocab_size=vocab)
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.norm = nn.RMSNorm(hidden)
        with torch.no_grad():
            e, h = synth_tables(seed, vocab, hidden)
            self.embed.weight.copy_(e)
            self.head.weight.copy_(h)
        self.steps: list[dict] = []

    @torch.inference_mode()
    def _logits(self, hidden):
        return self.head(self.norm(hidden.float())).float()

    @torch.inference_mode()
    def generate_stream(self, endpoints, prompt_ids, max_new_tokens, *, before_step=None, **_ignored):
        """Greedy decode. Appends one dict per step to self.steps: kind, tokens_in and ms for each phase."""
        self.check_route(endpoints, self.config.num_hidden_layers)
        session_id = uuid.uuid4().hex
        ids = torch.tensor([list(prompt_ids)], dtype=torch.long)
        pos, produced = 0, 0
        use_chain = self._use_chain(endpoints)
        use_ids = use_chain or self._use_ids(endpoints)
        try:
            while produced < max_new_tokens:
                if before_step is not None:
                    before_step(produced)
                t0 = time.perf_counter()
                if use_ids:   # token ids in, token id out: no local embedding or head
                    next_id = int(self._step_ids(endpoints, session_id, ids[0].tolist(), pos, "last", chain=use_chain)[-1])
                    t4 = time.perf_counter()
                    self.steps.append({"kind": "prefill" if pos == 0 else "decode", "tokens_in": int(ids.shape[1]),
                                       "embed_ms": 0.0, "blocks_ms": (t4 - t0) * 1e3, "head_ms": 0.0,
                                       "sample_ms": 0.0, "wall_ms": (t4 - t0) * 1e3, "t_start": t0})
                    pos += ids.shape[1]
                    produced += 1
                    yield next_id
                    ids = torch.tensor([[next_id]], dtype=torch.long)
                    continue
                hidden = self.embed(ids).to(self.dtype)
                t1 = time.perf_counter()
                hidden = self._run_blocks(endpoints, session_id, hidden, pos)
                t2 = time.perf_counter()
                logits = self._logits(hidden[:, -1:])[0, -1]
                t3 = time.perf_counter()
                next_id = sample_token(logits)
                t4 = time.perf_counter()
                self.steps.append({"kind": "prefill" if pos == 0 else "decode", "tokens_in": int(ids.shape[1]),
                                   "embed_ms": (t1 - t0) * 1e3, "blocks_ms": (t2 - t1) * 1e3,
                                   "head_ms": (t3 - t2) * 1e3, "sample_ms": (t4 - t3) * 1e3,
                                   "wall_ms": (t4 - t0) * 1e3, "t_start": t0})
                pos += ids.shape[1]
                produced += 1
                yield next_id
                ids = torch.tensor([[next_id]], dtype=torch.long)
        finally:
            for ep in endpoints:
                self.transport.close(ep, session_id)


class OracleDrafter:
    """A drafter with a controlled acceptance rate: it knows the true greedy continuation and gets each draft
    token right with probability `accept` (a wrong token is a different random id)."""

    def __init__(self, prompt_len: int, truth: list[int], accept: float, vocab: int = 2048, seed: int = 0):
        self.prompt_len, self.truth, self.accept, self.vocab = prompt_len, list(truth), accept, vocab
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def reset(self) -> None:
        pass

    def propose(self, context, k):
        self.calls += 1
        idx = len(context) - self.prompt_len
        out = []
        for i in range(k):
            tok = self.truth[idx + i] if idx + i < len(self.truth) else 0
            if self.rng.random() >= self.accept:
                tok = (tok + 1 + int(self.rng.integers(self.vocab - 1))) % self.vocab
            out.append(tok)
        return out
