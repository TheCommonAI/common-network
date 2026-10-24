"""CPU block server: decoder layers [start, end) in PyTorch on the CPU, for machines with no Apple silicon
and no NVIDIA GPU (for example a Windows PC with plenty of RAM).

It is a BlockRange subclass, so forward, close, session_length and open_sessions come from
commonnet.block and the wire format is the same as every other backend: float32 hidden states
between peers. Weights are the unquantised checkpoint loaded in bfloat16 (about 0.36 GiB per
Qwen3-8B layer), which keeps the numerics close to the reference; Qwen activations pass float16's
range on some prompts, so float16 is not used. It is slower than a GPU peer, so routing prefers GPU
peers when it costs nothing (commonnet.mesh.placement).

Thread count: torch's default (all physical cores). COMMONNET_CPU_THREADS overrides it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from commonnet.block import BlockRange


@dataclass
class CpuBlockRange(BlockRange):
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if torch.device(self.device).type != "cpu":
            raise RuntimeError(f"the CPU block runs on the CPU only, got device {self.device}")
        threads = os.environ.get("COMMONNET_CPU_THREADS")
        if threads:
            try:
                torch.set_num_threads(max(1, int(threads)))
            except ValueError:
                raise ValueError(f"COMMONNET_CPU_THREADS must be a number, got {threads!r}") from None
        super().__post_init__()
