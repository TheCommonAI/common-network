"""python -m commonnet.bench.gpu_profile : where does one decode step of THIS machine's stage spend its time?

Runs a window of layers locally (no network, no peers) the way a peer would, and reports:
  - the weights' size in memory and the memory bandwidth this machine really achieves (a device copy),
    so the lowest time a memory-bound step could take (the 'floor');
  - wall time per decode step, and how much of it the GPU was actually busy (CUDA events);
  - the top operations by GPU time and by CPU time (torch.profiler), and kernel launches per step.
If the GPU is busy for much less than the wall time the step is limited by launches / Python, not by
reading weights, and a smaller weight format would not make it faster.

    python -m commonnet.bench.gpu_profile --layers 27-36            # NVIDIA: nf4 checkpoint from the default network
    python -m commonnet.bench.gpu_profile --layers 0-9 --backend cpu --checkpoint DIR   # dry run anywhere
Writes bench-results/gpu-profile-<time>.json. Nothing secret is read or written.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


def parse_layers(text: str) -> tuple[int, int]:
    a, _, b = text.partition("-")
    return int(a), int(b)


def device_bandwidth_gbs(device, megabytes: int = 512, repeats: int = 8) -> float | None:
    """GB/s of a device-to-device copy (reads and writes each byte once, so the figure counts both)."""
    if device.type != "cuda":
        return None
    src = torch.empty(megabytes * 2**20, dtype=torch.uint8, device=device)
    dst = torch.empty_like(src)
    dst.copy_(src)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(repeats):
        dst.copy_(src)
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - t) / repeats
    return 2 * src.numel() / seconds / 1e9


def weight_bytes(block) -> int:
    seen, total = set(), 0
    for tensor in list(block.layers.parameters()) + list(block.layers.buffers()):
        key = tensor.data_ptr()
        if key not in seen:
            seen.add(key)
            total += tensor.numel() * tensor.element_size()
    return total


def timed_steps(block, hidden_size: int, device, prefill: int, steps: int) -> dict:
    sid = "profile"
    cuda = device.type == "cuda"
    with torch.no_grad():
        block.forward(sid, torch.randn(1, prefill, hidden_size) * 0.1, 0)
        pos = prefill
        wall, cpu_side, gpu_busy = [], [], []
        for _ in range(steps):
            x = torch.randn(1, 1, hidden_size) * 0.1
            if cuda:
                torch.cuda.synchronize()
                start_ev, end_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_ev.record()
            t0 = time.perf_counter()
            out = block.forward(sid, x, pos)
            t1 = time.perf_counter()  # forward() returned: Python/launch side is done
            if cuda:
                end_ev.record()
                torch.cuda.synchronize()
                gpu_busy.append(start_ev.elapsed_time(end_ev))
            t2 = time.perf_counter()
            cpu_side.append((t1 - t0) * 1000)
            wall.append((t2 - t0) * 1000)
            pos += 1
        block.close(sid)
    del out
    result = {"steps": steps, "wall_ms_median": statistics.median(wall),
              "wall_ms_p95": sorted(wall)[int(0.95 * (len(wall) - 1))],
              "python_launch_ms_median": statistics.median(cpu_side)}
    if gpu_busy:
        result["gpu_event_ms_median"] = statistics.median(gpu_busy)
    return result


def profiled_steps(block, hidden_size: int, device, prefill: int, steps: int, top: int) -> dict:
    from torch.profiler import ProfilerActivity, profile
    sid = "profile2"
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with torch.no_grad():
        block.forward(sid, torch.randn(1, prefill, hidden_size) * 0.1, 0)
        pos = prefill
        for _ in range(3):  # warm up
            block.forward(sid, torch.randn(1, 1, hidden_size) * 0.1, pos)
            pos += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        with profile(activities=acts) as prof:
            for _ in range(steps):
                block.forward(sid, torch.randn(1, 1, hidden_size) * 0.1, pos)
                pos += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
        block.close(sid)
    events = prof.key_averages()
    def cuda_us(e):
        return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0) or 0
    def cpu_us(e):
        return getattr(e, "self_cpu_time_total", 0) or 0
    by_gpu = sorted(events, key=cuda_us, reverse=True)[:top]
    by_cpu = sorted(events, key=cpu_us, reverse=True)[:top]
    row = lambda e: {"op": e.key[:60], "count_per_step": round(e.count / steps, 1),
                     "gpu_ms_per_step": round(cuda_us(e) / steps / 1000, 3), "cpu_ms_per_step": round(cpu_us(e) / steps / 1000, 3)}
    total_gpu = sum(cuda_us(e) for e in events) / steps / 1000
    launches = sum(e.count for e in events if "cuda" in e.key.lower() and "launch" in e.key.lower()) / steps
    return {"steps": steps, "gpu_kernel_ms_per_step": round(total_gpu, 3), "launches_per_step": round(launches, 1),
            "top_by_gpu": [row(e) for e in by_gpu], "top_by_cpu": [row(e) for e in by_cpu]}


def verdict(timing: dict, nbytes: int, bw_gbs: float | None) -> list[str]:
    lines = []
    wall = timing["wall_ms_median"]
    if bw_gbs:
        floor = nbytes / (bw_gbs * 1e9 / 2) * 1000  # one read of the weights; copy bandwidth counts read+write
        lines.append(f"weights {nbytes / 2**30:.2f} GiB; achievable read bandwidth about {bw_gbs / 2:.0f} GB/s; "
                     f"floor for reading the weights once per token: {floor:.1f} ms")
        lines.append(f"measured step: {wall:.1f} ms median ({wall / floor:.1f}x the floor)")
    if "gpu_event_ms_median" in timing:
        busy = timing["gpu_event_ms_median"]
        lines.append(f"GPU busy (events): {busy:.1f} ms of {wall:.1f} ms wall; Python/launch side: {timing['python_launch_ms_median']:.1f} ms")
        if busy < 0.6 * wall:
            lines.append("VERDICT: launch/Python bound. The GPU waits for work; a smaller weight format would barely help.")
        elif bw_gbs and busy > 3 * floor:
            lines.append("VERDICT: GPU busy but far above the bandwidth floor: dequantise/compute bound, not reading bytes.")
        else:
            lines.append("VERDICT: close to memory-bound; fewer bytes per weight would help.")
    return lines


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", default="27-36")
    p.add_argument("--network", default=str(Path(__file__).resolve().parents[1] / "networks" / "main.json"))
    p.add_argument("--backend", choices=["nf4", "cpu", "mlx"], help="default: nf4 if CUDA is available, else cpu")
    p.add_argument("--checkpoint", help="checkpoint directory (default: download this window's tensors)")
    p.add_argument("--prefill", type=int, default=16)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--profile-steps", type=int, default=6)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--out")
    args = p.parse_args(argv)

    start, end = parse_layers(args.layers)
    backend = args.backend or ("nf4" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if backend == "nf4" else "cpu")
    info = {"torch": torch.__version__, "backend": backend, "layers": [start, end]}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=props.name, vram_gib=round(props.total_memory / 2**30, 2))
        try:
            import bitsandbytes
            info["bitsandbytes"] = bitsandbytes.__version__
        except Exception:
            info["bitsandbytes"] = None
    print(json.dumps(info), flush=True)

    checkpoint = args.checkpoint
    if checkpoint is None:
        from commonnet import hub
        from commonnet.mesh.manifest import Manifest
        manifest = Manifest.load(args.network)
        repo = manifest.repo(backend)
        if not repo:
            print(f"error: {manifest.id} has no {backend} checkpoint", file=sys.stderr)
            return 2
        print(f"fetching layers {start}-{end} of {repo} ...", flush=True)
        checkpoint = hub.Hub(repo).fetch(layers=(start, end))
    from commonnet.net import factory
    t = time.perf_counter()
    served = factory.build_block(start, end, backend=backend, checkpoint=checkpoint)
    block = served.inner
    if device.type == "cuda":
        torch.cuda.synchronize()
    info["load_s"] = round(time.perf_counter() - t, 1)
    nbytes = weight_bytes(block)
    info["weight_bytes"] = nbytes
    info["weight_gib"] = round(nbytes / 2**30, 3)  # parameters and buffers; NF4 scale tables live outside, see vram_allocated
    if device.type == "cuda":
        info["vram_allocated_gib"] = round(torch.cuda.memory_allocated() / 2**30, 3)
        info["vram_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    hidden = block.config.hidden_size
    bw = device_bandwidth_gbs(device)
    timing = timed_steps(block, hidden, device, args.prefill, args.steps)
    prof = profiled_steps(block, hidden, device, args.prefill, args.profile_steps, args.top)
    lines = verdict(timing, nbytes, bw)
    result = {"info": info, "copy_bandwidth_gbs": bw, "timing": timing, "profile": prof, "verdict": lines}

    print("\nstep timing:", json.dumps(timing))
    print(f"profiler (per step): GPU kernels {prof['gpu_kernel_ms_per_step']} ms, launches {prof['launches_per_step']}")
    for title, key in (("top by GPU time", "top_by_gpu"), ("top by CPU time", "top_by_cpu")):
        print(f"\n{title}:")
        for r in prof[key]:
            print(f"  {r['op']:<60} x{r['count_per_step']:<6} gpu {r['gpu_ms_per_step']:>7} ms  cpu {r['cpu_ms_per_step']:>7} ms")
    print()
    for line in lines:
        print(line)
    out = Path(args.out) if args.out else Path("bench-results") / time.strftime("gpu-profile-%Y%m%d-%H%M%S.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
