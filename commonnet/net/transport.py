"""Transport adapter: the core Client's Transport protocol over HTTP block servers.

Endpoint names map to base URLs (and optionally a per-endpoint token). Tensors are
converted to protocol.Tensor (float32 stays float32, anything else goes as float16).
A long prefill is sent in sequence chunks so each request stays under the server's
request limit; the block's KV cache makes chunked prefill equal to one-shot prefill.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from commonnet.net.client import BlockClient, BlockError, RemoteError, Unreachable
from commonnet.net.protocol import DTYPE_WIDTH, MAX_REQUEST_BYTES, Tensor

_NP = {"float16": np.float16, "float32": np.float32}
_TORCH = {"float16": torch.float16, "float32": torch.float32}
OVERHEAD_BYTES = 4096


def to_wire(t: torch.Tensor) -> Tensor:
    name = "float32" if t.dtype == torch.float32 else "float16"
    arr = t.detach().to(device="cpu", dtype=_TORCH[name]).contiguous().numpy()
    return Tensor(tuple(arr.shape), name, arr.tobytes())


def from_wire(t: Tensor) -> torch.Tensor:
    arr = np.frombuffer(t.data, dtype=_NP[t.dtype]).reshape(t.shape)
    return torch.from_numpy(arr.copy())


DEAD_SECONDS = 60.0


class HttpTransport:
    """The core Client's Transport over HTTP block servers.

    replicas maps an endpoint name to more servers for the same layers, each a URL or a
    (URL, token) pair. When the server in use stops answering (or has lost the session),
    the transport moves to the next one, opens the session there and replays every piece
    it already sent, exactly as sent, so the new server's cache is the one the old server
    held. The call that failed is then sent again. With no replicas nothing is kept and a
    dead server raises Unreachable as before.
    """

    def __init__(self, urls: dict[str, str], *, token=None, tokens: dict[str, str] | None = None,
                 replicas: dict[str, list] | None = None, max_request_bytes: int = MAX_REQUEST_BYTES,
                 now=time.monotonic, **client_kwargs):
        self.urls = dict(urls)
        self.max_request_bytes = max_request_bytes
        self._tokens = tokens or {}
        self._default_token = token
        self._kw = client_kwargs
        self._now = now
        self._replicas = {n: list(r) for n, r in (replicas or {}).items()}
        self._clients: dict[tuple[str, int], BlockClient] = {}
        self._open: set[tuple[str, str]] = set()
        self._active: dict[tuple[str, str], int] = {}
        self._history: dict[tuple[str, str], list] = {}
        self._dead: dict[tuple[str, int], float] = {}
        self.failovers: list[tuple[str, int, int]] = []  # (endpoint, from index, to index), for tests and logs

    def _candidates(self, name: str) -> list:
        try:
            primary = self.urls[name]
        except KeyError:
            raise KeyError(f"no URL for endpoint {name!r}") from None
        return [primary] + self._replicas.get(name, [])

    def _target(self, name: str, index: int) -> tuple[BlockClient, str]:
        entry = self._candidates(name)[index]
        url, token = entry if isinstance(entry, tuple) else (
            entry, self._tokens.get(name, self._default_token))
        if (name, index) not in self._clients:
            self._clients[(name, index)] = BlockClient(
                token=token, max_request_bytes=self.max_request_bytes, **self._kw)
        return self._clients[(name, index)], url

    def _alive(self, name: str, index: int) -> bool:
        died = self._dead.get((name, index))
        return died is None or self._now() - died > DEAD_SECONDS

    def _first_alive(self, name: str) -> int:
        for i in range(len(self._candidates(name))):
            if self._alive(name, i):
                return i
        return 0

    def _start(self, name: str, session_id: str, index: int) -> None:
        client, url = self._target(name, index)
        client.open(url, session_id)
        self._active[(name, session_id)] = index

    def _recover(self, name: str, session_id: str, failed: int, error: Exception) -> None:
        """Move the session to the next live server and replay what it held. Raises if none is left."""
        key = (name, session_id)
        self._dead[(name, failed)] = self._now()
        count = len(self._candidates(name))
        last = error
        for step in range(1, count + 1):
            index = (failed + step) % count
            if index != failed and not self._alive(name, index):
                continue
            try:
                self._start(name, session_id, index)
                client, url = self._target(name, index)
                for start_pos, piece in self._history.get(key, []):
                    client.forward(url, session_id, piece, start_pos)
            except BlockError as e:
                self._dead[(name, index)] = self._now()
                last = e
                continue
            self.failovers.append((name, failed, index))
            return
        raise last

    def _send(self, name: str, session_id: str, piece: Tensor, start_pos: int) -> Tensor:
        key = (name, session_id)
        while True:
            index = self._active[key]
            client, url = self._target(name, index)
            try:
                return client.forward(url, session_id, piece, start_pos)
            except (Unreachable, RemoteError) as e:
                if len(self._candidates(name)) < 2:
                    raise
                if isinstance(e, RemoteError) and e.status != 404:
                    raise  # an answer from a live server, not a lost one (404: it forgot the session)
                self._recover(name, session_id, index, e)

    def forward(self, endpoint, session_id: str, hidden: torch.Tensor, start_pos: int) -> torch.Tensor:
        name, key = endpoint.name, (endpoint.name, session_id)
        if key not in self._open:
            self._start(name, session_id, self._first_alive(name))
            self._open.add(key)
        keep = len(self._candidates(name)) > 1
        batch, seq, width = hidden.shape
        wire_name = "float32" if hidden.dtype == torch.float32 else "float16"
        per_token = batch * width * DTYPE_WIDTH[wire_name] * 4 // 3 + 4
        chunk = max(1, (self.max_request_bytes - OVERHEAD_BYTES) // per_token)
        outs = []
        for i in range(0, seq, chunk):
            piece = to_wire(hidden[:, i:i + chunk])
            out = self._send(name, session_id, piece, start_pos + i)
            if keep:
                self._history.setdefault(key, []).append((start_pos + i, piece))
            outs.append(from_wire(out))
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
        return out.to(hidden.dtype)

    def close(self, endpoint, session_id: str) -> None:
        key = (endpoint.name, session_id)
        if key not in self._open:
            return
        self._open.discard(key)
        self._history.pop(key, None)
        index = self._active.pop(key, 0)
        try:
            client, url = self._target(endpoint.name, index)
            client.close(url, session_id)
        except Exception:  # best effort: the server also drops sessions with the process
            pass

    def stats(self) -> dict:
        """Per endpoint: forward calls, mean total_ms, mean server_ms, and connections reused."""
        out = {}
        for (name, index), client in self._clients.items():
            name = name if index == 0 else f"{name}#{index}"
            rows = [t for t in client.timings if t["path"] == "/forward" and t["status"] == 200]
            if not rows:
                continue
            served = [t["server_ms"] for t in rows if t["server_ms"] is not None]
            out[name] = {
                "calls": len(rows),
                "total_ms": sum(t["total_ms"] for t in rows) / len(rows),
                "server_ms": sum(served) / len(served) if served else None,
                "reused": sum(1 for t in rows if t["reused"]),
            }
        return out

    def shutdown(self) -> None:
        for client in self._clients.values():
            client.close_connections()
