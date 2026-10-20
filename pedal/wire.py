"""Hidden states on the wire: float16 bytes plus a shape.

Blocks convert at their own edge. Anything that crosses a machine boundary is
sent as these bytes, so the in-process path and a network path use the same format.
"""

from __future__ import annotations

import numpy as np
import torch


def encode_fp16(hidden: torch.Tensor) -> tuple[bytes, tuple[int, ...]]:
    arr = hidden.detach().to(device="cpu", dtype=torch.float16).contiguous().numpy()
    return arr.tobytes(), tuple(arr.shape)


def decode_fp16(data: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    arr = np.frombuffer(data, dtype=np.float16).reshape(shape)
    return torch.from_numpy(arr.copy())
