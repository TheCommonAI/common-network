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

    def __init__(self, inner, start: int, end: int, device: str):
        self.inner = inner
        self.layers = protocol.check_layers((start, end))
        self.device = device

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

    def forward(self, session_id: str, hidden, start_pos: int):
        out = self.inner.forward(session_id, from_wire(hidden), start_pos)
        wire = to_wire(out)
        return wire if wire.dtype == hidden.dtype else to_wire(out.to(from_wire(hidden).dtype))


def build_block(start: int, end: int, backend: str | None = None, checkpoint: str | None = None, environ=None):
    env = os.environ if environ is None else environ
    backend = (backend or env.get("COMMONNET_BACKEND") or "torch").lower()
    checkpoint = checkpoint or env.get("COMMONNET_CHECKPOINT")
    if backend not in BACKENDS:
        raise ValueError(f"unknown COMMONNET_BACKEND {backend!r}; choose one of {BACKENDS}")
    if backend == "synth":  # synthetic test model (commonnet/bench): no checkpoint, weights from a seed
        from commonnet.bench.synth import SynthBlock
        return SynthBlock(start, end, layer_ms=float(env.get("COMMONNET_SYNTH_LAYER_MS", "0")),
                          seed=int(env.get("COMMONNET_SYNTH_SEED", "1234")))
    if not checkpoint:
        raise ValueError("set COMMONNET_CHECKPOINT to the checkpoint directory")
    if backend == "torch":
        from commonnet.block import BlockRange
        device = env.get("COMMONNET_DEVICE", "cpu")
        dtype = getattr(torch, env.get("COMMONNET_DTYPE", "float32"))
        inner = BlockRange(checkpoint_dir=checkpoint, start=start, end=end, device=device, dtype=dtype)
        return ServedBlock(inner, start, end, device)
    if backend == "cpu":
        from commonnet.backends.cpu_block import CpuBlockRange
        dtype = getattr(torch, env.get("COMMONNET_DTYPE") or "bfloat16")
        inner = CpuBlockRange(checkpoint_dir=checkpoint, start=start, end=end, dtype=dtype)
        return ServedBlock(inner, start, end, "cpu")
    if backend == "mlx":
        from commonnet.backends.mlx_block import MlxBlockRange, OnWorkerThread
        # MLX streams are per-thread and the server runs each request on a new thread.
        extra = {}
        if env.get("COMMONNET_DTYPE"):
            import mlx.core as mx
            extra["dtype"] = getattr(mx, env["COMMONNET_DTYPE"])
        inner = OnWorkerThread(MlxBlockRange, checkpoint_dir=checkpoint, start=start, end=end, **extra)
        return ServedBlock(inner, start, end, "metal")
    from commonnet.backends.nf4_block import Nf4BlockRange  # refuses without CUDA
    return ServedBlock(Nf4BlockRange(checkpoint_dir=checkpoint, start=start, end=end), start, end, "cuda")


def FACTORY(start: int, end: int):
    return build_block(start, end)
