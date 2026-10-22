"""Transport adapter: the core Client's Transport protocol over HTTP block servers.

Endpoint names map to base URLs (and optionally a per-endpoint token). Tensors are
converted to protocol.Tensor (float32 stays float32, anything else goes as float16).
A long prefill is sent in sequence chunks so each request stays under the server's
request limit; the block's KV cache makes chunked prefill equal to one-shot prefill.
"""
from __future__ import annotations

import numpy as np
import torch

from commonnet.net.client import BlockClient
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


class HttpTransport:
    def __init__(self, urls: dict[str, str], *, token=None, tokens: dict[str, str] | None = None,
                 max_request_bytes: int = MAX_REQUEST_BYTES, **client_kwargs):
        self.urls = dict(urls)
        self.max_request_bytes = max_request_bytes
        self._tokens = tokens or {}
        self._default_token = token
        self._kw = client_kwargs
        self._clients: dict[str, BlockClient] = {}
        self._open: set[tuple[str, str]] = set()

    def _client(self, name: str) -> BlockClient:
        if name not in self._clients:
            self._clients[name] = BlockClient(token=self._tokens.get(name, self._default_token),
                                              max_request_bytes=self.max_request_bytes, **self._kw)
        return self._clients[name]

    def _url(self, endpoint) -> str:
        try:
            return self.urls[endpoint.name]
        except KeyError:
            raise KeyError(f"no URL for endpoint {endpoint.name!r}") from None

    def forward(self, endpoint, session_id: str, hidden: torch.Tensor, start_pos: int) -> torch.Tensor:
        client, url = self._client(endpoint.name), self._url(endpoint)
        if (endpoint.name, session_id) not in self._open:
            client.open(url, session_id)
            self._open.add((endpoint.name, session_id))
        batch, seq, width = hidden.shape
        wire_name = "float32" if hidden.dtype == torch.float32 else "float16"
        per_token = batch * width * DTYPE_WIDTH[wire_name] * 4 // 3 + 4
        chunk = max(1, (self.max_request_bytes - OVERHEAD_BYTES) // per_token)
        outs = []
        for i in range(0, seq, chunk):
            piece = hidden[:, i:i + chunk]
            out = client.forward(url, session_id, to_wire(piece), start_pos + i)
            outs.append(from_wire(out))
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
        return out.to(hidden.dtype)

    def close(self, endpoint, session_id: str) -> None:
        if (endpoint.name, session_id) not in self._open:
            return
        self._open.discard((endpoint.name, session_id))
        try:
            self._client(endpoint.name).close(self._url(endpoint), session_id)
        except Exception:  # best effort: the server also drops sessions with the process
            pass

    def stats(self) -> dict:
        """Per endpoint: forward calls, mean total_ms, mean server_ms, and connections reused."""
        out = {}
        for name, client in self._clients.items():
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
