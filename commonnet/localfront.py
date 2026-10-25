"""Local front: this machine also holds the first layers [0, F) and runs them in the chat client's process.

The client then embeds and runs layers 0..F locally and sends the hidden state straight to the peer that holds layer F
onward. That removes the client-to-first-peer hop and keeps the prompt tokens off the network (peers only ever receive
layer-F hidden states). It costs the memory of those layers and only helps the machine that asks. Off by default.

The front is a block of the usual kind (factory.build_block) that is not served over HTTP. It appears in a route as the
endpoint FRONT_NAME [0, F); HttpTransport runs it locally and sends the rest of the route to peers.
"""
from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass

import torch

from commonnet.client import BlockEndpoint
from commonnet.net.transport import FRONT_NAME, from_wire, to_wire

ENV = "COMMONNET_LOCAL_FRONT"
AUTO_MAX_MS = 25.0      # auto: running the front layers for one decode token must take less than this (median)
AUTO_MEMORY_FACTOR = 1.5   # auto: free memory must be at least this times the front's weight size


class FrontConfigError(ValueError):
    pass


def parse_setting(value) -> str | int | None:
    """None (off), "auto", "on" (the first stage) or a layer count. Raises FrontConfigError on anything else."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "0", "off", "no", "false"):
        return None
    if text == "auto":
        return "auto"
    if text in ("on", "yes", "true"):
        return "on"
    if text.isdigit():
        return int(text)
    raise FrontConfigError(f"--local-front wants a layer count, auto or off, got {value!r}")


def setting_from(arg, environ=None):
    """The flag wins over COMMONNET_LOCAL_FRONT; both absent means off."""
    env = os.environ if environ is None else environ
    return parse_setting(arg if arg is not None else env.get(ENV))


def front_layers(setting, manifest) -> int:
    """F for this setting: must be an end of a manifest stage (auto and on use the first stage)."""
    stages = manifest.stages()
    if setting in ("auto", "on"):
        return stages[0][1]
    ends = [b for _, b in stages[:-1]]   # F = the last layer would leave nothing for the peers
    if setting not in ends:
        raise FrontConfigError(f"--local-front {setting} is not a stage boundary of this network "
                               f"(choose one of {', '.join(map(str, ends))}, or auto)")
    return int(setting)


@dataclass
class AutoDecision:
    enabled: bool
    reason: str
    ms: float | None = None


def decide_auto(ms: float | None, free_gib: float, weight_gib: float) -> AutoDecision:
    """auto: on only if one decode token through the front takes < AUTO_MAX_MS and memory is free for 1.5x the weights."""
    if ms is None:
        return AutoDecision(False, "the benchmark of the first stage did not run")
    if free_gib < AUTO_MEMORY_FACTOR * weight_gib:
        return AutoDecision(False, f"only {free_gib:.1f} GiB free, need {AUTO_MEMORY_FACTOR * weight_gib:.1f} GiB "
                                   f"(1.5x the {weight_gib:.1f} GiB of the first stage)", ms)
    if ms >= AUTO_MAX_MS:
        return AutoDecision(False, f"the first stage takes {ms:.1f} ms per decode token here (limit {AUTO_MAX_MS:.0f} ms)", ms)
    return AutoDecision(True, f"the first stage takes {ms:.1f} ms per decode token and memory is free", ms)


def benchmark_ms(front, tokens: int = 8, warmup: int = 2) -> float:
    """Median ms for one decode token through the front (a throwaway session), after a short warmup."""
    session = "bench-" + os.urandom(6).hex()
    times = []
    try:
        pos = 0
        front.forward(session, [1, 2, 3, 4], pos)
        pos = 4
        for i in range(warmup + tokens):
            t0 = time.perf_counter()
            front.forward(session, [5 + i], pos)
            dt = (time.perf_counter() - t0) * 1e3
            pos += 1
            if i >= warmup:
                times.append(dt)
    finally:
        front.close(session)
    return statistics.median(times)


def free_gib() -> float:
    """Memory this machine can still give, in GiB (available RAM; unified memory on a Mac)."""
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except Exception:
        try:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
        except (ValueError, OSError, AttributeError):
            return 0.0


class LocalFront:
    """Layers [0, end) of the model in this process. `block` is a ServedBlock (ids in, wire tensors) or a test double."""

    def __init__(self, block, end: int, owned: bool = True):
        self.block, self.end, self.owned = block, int(end), owned

    @property
    def has_ids(self) -> bool:
        return getattr(self.block, "io", None) is not None or "ids_in" in getattr(self.block, "caps", ())

    @property
    def can_truncate(self) -> bool:
        inner = getattr(self.block, "inner", self.block)
        return callable(getattr(inner, "truncate", None))

    def endpoint(self) -> BlockEndpoint:
        return BlockEndpoint(FRONT_NAME, 0, self.end)

    def forward(self, session_id: str, ids, start_pos: int) -> torch.Tensor:
        """Token ids through embedding and layers [0, F): the hidden state (1, T, hidden) that goes to the first peer."""
        if start_pos == 0:
            self.block.open(session_id)
        return from_wire(self.block.forward(session_id, self.block.embed(list(ids)), start_pos))

    def forward_hidden(self, session_id: str, hidden: torch.Tensor, start_pos: int) -> torch.Tensor:
        if start_pos == 0:
            self.block.open(session_id)
        return from_wire(self.block.forward(session_id, to_wire(hidden), start_pos)).to(hidden.dtype)

    def truncate(self, session_id: str, length: int) -> None:
        self.block.truncate(session_id, length)

    def close(self, session_id: str) -> None:
        self.block.close(session_id)

    def shutdown(self) -> None:
        if self.owned:
            stop = getattr(self.block, "shutdown", None)
            if stop:
                stop()


VRAM_HEADROOM_GIB = 0.5   # CUDA context, activations and KV for the front, on top of its weights


def free_vram_gib(pending_gib: float = 0.0):
    """(free GiB on the CUDA device after reserving `pending_gib` the node's own served window has yet to load, total GiB),
    or None when there is no CUDA device. torch.cuda.mem_get_info already counts what is loaded in any process."""
    try:
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
    except Exception:
        return None
    return max(0.0, free / 2 ** 30 - pending_gib), total / 2 ** 30


def vram_fit(setting, free_gib: float, weight_gib: float) -> tuple[bool, str]:
    """auto wants 1.5x the weights free, an explicit setting weights + headroom; either way refuse rather than OOM."""
    need = AUTO_MEMORY_FACTOR * weight_gib if setting == "auto" else weight_gib + VRAM_HEADROOM_GIB
    if free_gib < need:
        return False, f"only {free_gib:.1f} GiB of GPU memory free, the front needs {need:.1f} GiB ({weight_gib:.1f} GiB of weights)"
    return True, ""
