"""Block factory for `python -m commonnet.net.server --block commonnet.net.factory:FACTORY`.

Environment:
  COMMONNET_CHECKPOINT  checkpoint directory (required)
  COMMONNET_BACKEND     torch (default) | mlx | nf4 | cpu
  COMMONNET_DEVICE      torch backend device (default cpu)
  COMMONNET_DTYPE       torch backend dtype: float32 (default) | float16 | bfloat16
                        mlx backend compute dtype: float16 (default) | bfloat16
"""
from __future__ import annotations

import os

import torch

from commonnet.net import protocol
from commonnet.net.transport import from_wire, to_wire

BACKENDS = ("torch", "mlx", "nf4", "cpu", "synth")


class ServedBlock:
    """Adapts a tensor-level block (forward(session, torch.Tensor, pos)) to the server's Tensor interface."""

    def __init__(self, inner, start: int, end: int, device: str, io=None, caps=()):
        self.inner = inner
        self.layers = protocol.check_layers((start, end))
        self.device = device
        self.io, self.caps = io, tuple(caps)  # token-id I/O for the end stages (commonnet/net/endio.py)
        self.wire_dtype = torch.float32

    def open(self, session_id: str) -> None:
        pass  # sessions are created by the block on first forward at start_pos 0

    def close(self, session_id: str) -> None:
        self.inner.close(session_id)

    def shutdown(self) -> None:
        stop = getattr(self.inner, "shutdown", None)
        if stop:
            stop()

    def truncate(self, session_id: str, length: int) -> None:
        cut = getattr(self.inner, "truncate", None)
        if cut is None:
            raise NotImplementedError("this block cannot truncate a session")
        cut(session_id, length)

    def _on_worker(self, fn, *args):
        run = getattr(self.inner, "run", None)  # MLX arrays belong to the worker thread
        return run(fn, *args) if run else fn(*args)

    def embed(self, ids):
        return to_wire(self._on_worker(self.io.embed, ids, self.wire_dtype))

    def head(self, hidden, which, sampling, start_pos):
        return self._on_worker(self.io.head_ids, from_wire(hidden), which, sampling, start_pos)

    def forward(self, session_id: str, hidden, start_pos: int):
        out = self.inner.forward(session_id, from_wire(hidden), start_pos)
        wire = to_wire(out)
        return wire if wire.dtype == hidden.dtype else to_wire(out.to(from_wire(hidden).dtype))


def _served(inner, start, end, device, checkpoint, backend, environ):
    """Wrap a tensor-level block, adding token-id I/O where the window is an end of the model and the checkpoint
    holds the tables. COMMONNET_IDS=0 turns it off."""
    from commonnet.net import endio
    caps = () if str((environ or {}).get("COMMONNET_IDS", "1")) == "0" else endio.caps_for(checkpoint, start, end, backend)
    dtype = getattr(inner, "dtype", None) if backend != "mlx" else None
    io = endio.EndIO(checkpoint, backend, want_embed="ids_in" in caps, want_head="head_out" in caps,
                    device=None if backend == "mlx" else device, dtype=dtype) if caps else None
    return ServedBlock(inner, start, end, device, io, caps)


def build_block(start: int, end: int, backend: str | None = None, checkpoint: str | None = None, environ=None):
    env = os.environ if environ is None else environ
    backend = (backend or env.get("COMMONNET_BACKEND") or "torch").lower()
    checkpoint = checkpoint or env.get("COMMONNET_CHECKPOINT")
    if backend not in BACKENDS:
        raise ValueError(f"unknown COMMONNET_BACKEND {backend!r}; choose one of {BACKENDS}")
    if backend == "synth":  # synthetic test model (commonnet/bench): no checkpoint, weights from a seed
        from commonnet.bench.synth import SynthBlock
        return SynthBlock(start, end, layer_ms=float(env.get("COMMONNET_SYNTH_LAYER_MS", "0")),
                          seed=int(env.get("COMMONNET_SYNTH_SEED", "1234")),
                          io_layers=int(env["COMMONNET_SYNTH_IO_LAYERS"]) if env.get("COMMONNET_SYNTH_IO_LAYERS") else None)
    if not checkpoint:
        raise ValueError("set COMMONNET_CHECKPOINT to the checkpoint directory")
    if backend == "torch":
        from commonnet.block import BlockRange
        device = env.get("COMMONNET_DEVICE", "cpu")
        dtype = getattr(torch, env.get("COMMONNET_DTYPE", "float32"))
        inner = BlockRange(checkpoint_dir=checkpoint, start=start, end=end, device=device, dtype=dtype)
        return _served(inner, start, end, device, checkpoint, backend, env)
    if backend == "cpu":
        from commonnet.backends.cpu_block import CpuBlockRange
        dtype = getattr(torch, env.get("COMMONNET_DTYPE") or "bfloat16")
        inner = CpuBlockRange(checkpoint_dir=checkpoint, start=start, end=end, dtype=dtype)
        return _served(inner, start, end, "cpu", checkpoint, backend, env)
    if backend == "mlx":
        from commonnet.backends.mlx_block import MlxBlockRange, OnWorkerThread
        # MLX streams are per-thread and the server runs each request on a new thread.
        extra = {}
        if env.get("COMMONNET_DTYPE"):
            import mlx.core as mx
            extra["dtype"] = getattr(mx, env["COMMONNET_DTYPE"])
        inner = OnWorkerThread(MlxBlockRange, checkpoint_dir=checkpoint, start=start, end=end, **extra)
        return _served(inner, start, end, "metal", checkpoint, backend, env)
    from commonnet.backends.nf4_block import Nf4BlockRange  # refuses without CUDA
    # bfloat16 compute: float16 activations overflow to NaN on some Qwen prompts (the Mac backend hit it too).
    # RTX 20-series and older have no bfloat16; set COMMONNET_DTYPE=float16 there.
    dtype = getattr(torch, env.get("COMMONNET_DTYPE") or "bfloat16")
    return _served(Nf4BlockRange(checkpoint_dir=checkpoint, start=start, end=end, dtype=dtype), start, end, "cuda",
                   checkpoint, backend, env)


def FACTORY(start: int, end: int):
    return build_block(start, end)
