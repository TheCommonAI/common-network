"""Local front vs no local front on the real Qwen3-8B (MLX 4-bit), peers on loopback behind 30 ms delay proxies.

    python -m commonnet.bench.local_front_real --checkpoint DIR [--rtt-ms 30] [--front 9] [--new-tokens 48] [--out FILE]

One process holds three blocks: A = layers [0,F), B1 = [F,27), B2 = [27,36). Without the front the client sends token ids
to A (behind a proxy) which chains to B1 and B2. With the front the client runs A in its own process (no proxy, no HTTP)
and sends the layer-F hidden state to B1, which chains to B2. The same A object serves both, so weights load once.
Holds /tmp/commonnet-heavy.lock (exclusive) while the model is loaded. Loopback only: it shows the saved hop, not WAN
bandwidth, jitter or the peers' own load.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import pathlib
import statistics
import sys
import time

import torch

LOCK = "/tmp/commonnet-heavy.lock"
PROMPTS = ["Explain in a few sentences why the sky is blue, and why sunsets are red.",
           "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."]


def main(argv=None):
    os.environ.setdefault("COMMONNET_DTYPE", "bfloat16")
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--draft", default="mlx-community/Qwen3-0.6B-4bit")
    p.add_argument("--front", type=int, default=9)
    p.add_argument("--rtt-ms", type=float, default=30.0)
    p.add_argument("--new-tokens", type=int, default=48)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--out", default="bench-results/local-front-real.json")
    a = p.parse_args(argv)
    lock = open(LOCK, "a+")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        run(a)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def run(a):
    import mlx.core as mx
    from transformers import AutoTokenizer
    from commonnet import speculative as spec
    from commonnet.bench.proxy import DelayProxy
    from commonnet.client import BlockEndpoint
    from commonnet.client4 import Client4
    from commonnet.localfront import LocalFront, benchmark_ms
    from commonnet.net import factory
    from commonnet.net.server import BlockServer
    from commonnet.net.transport import HttpTransport

    nl = json.load(open(pathlib.Path(a.checkpoint) / "config.json"))["num_hidden_layers"]
    F, mid = a.front, 27
    mem0 = mx.get_active_memory()
    blockA = factory.build_block(0, F, backend="mlx", checkpoint=a.checkpoint)
    front = LocalFront(blockA, F, owned=False)
    mem_front = (mx.get_active_memory() - mem0) / 2 ** 30
    layer_ms = benchmark_ms(front, tokens=20, warmup=5)
    print(f"front layers 0-{F}: {mem_front:.2f} GiB resident (MLX active), {layer_ms:.1f} ms per decode token", flush=True)
    blockB1 = factory.build_block(F, mid, backend="mlx", checkpoint=a.checkpoint)
    blockB2 = factory.build_block(mid, nl, backend="mlx", checkpoint=a.checkpoint)
    servers = [BlockServer(b, "127.0.0.1", 0, allow_chain_private=True).start() for b in (blockA, blockB1, blockB2)]
    proxies = [DelayProxy(s.port, rtt_ms=0.0, seed=i) for i, s in enumerate(servers)]
    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    drafter = spec.MlxDrafter(a.draft, target_tokenizer=tok)
    stop = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    eps_off = [BlockEndpoint("hop0", 0, F), BlockEndpoint("hop1", F, mid), BlockEndpoint("hop2", mid, nl)]
    eps_on = [front.endpoint(), eps_off[1], eps_off[2]]
    urls = {ep.name: px.url for ep, px in zip(eps_off, proxies)}
    t_off = HttpTransport(urls, retries=1)
    t_on = HttpTransport({k: v for k, v in urls.items() if k != "hop0"}, retries=1)
    t_on.local_front = front
    client = Client4(a.checkpoint, t_off, fmt="mlx4", hidden_dtype=torch.float32)

    def prompt_ids(text):
        return tok(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                           tokenize=False, enable_thinking=False), add_special_tokens=False)["input_ids"]

    def gen(mode, ids, use_spec):
        client.transport = t_on if mode == "on" else t_off
        eps = eps_on if mode == "on" else eps_off
        t = time.perf_counter()
        if use_spec:
            st = spec.SpecStats()
            out = list(spec.generate_stream(client, eps, ids, a.new_tokens, drafter=drafter, k=a.k, stop_ids=stop, stats=st))
        else:
            out = list(client.generate_stream(eps, ids, a.new_tokens, stop_ids=stop))
        return out, time.perf_counter() - t

    rows = []
    try:
        w = prompt_ids("Hi")
        for mode in ("off", "on"):
            gen(mode, w, False)
            gen(mode, w, True)
        for px in proxies:
            px.rtt_ms = a.rtt_ms
        results = {}
        for use_spec in (False, True):
            for i, text in enumerate(PROMPTS):
                ids = prompt_ids(text)
                for rep in range(a.repeats):
                    for mode in ("off", "on"):
                        out, dt = gen(mode, ids, use_spec)
                        results.setdefault((use_spec, i, mode), []).append((out, dt))
                        rows.append({"spec": use_spec, "prompt": i, "mode": mode, "rep": rep, "tokens": len(out),
                                     "s": dt, "ms_per_token": dt / max(1, len(out)) * 1e3, "tok_s": len(out) / dt})
                        print(rows[-1], flush=True)
        same = all(results[(s, i, "off")][r][0] == results[(s, i, "on")][r][0]
                   for (s, i, m) in results if m == "off" for r in range(a.repeats))
        summary = {}
        for use_spec in (False, True):
            for mode in ("off", "on"):
                sel = [r for r in rows if r["spec"] == use_spec and r["mode"] == mode]
                summary[f"{'spec' if use_spec else 'plain'}_{mode}"] = {
                    "ms_per_token": statistics.mean(r["ms_per_token"] for r in sel),
                    "tok_s": statistics.mean(r["tok_s"] for r in sel)}
        doc = {"config": vars(a), "front_gib": mem_front, "front_decode_ms": layer_ms,
               "identical_output": same, "summary": summary, "rows": rows}
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(json.dumps(doc, indent=1))
        print(json.dumps({"front_gib": mem_front, "front_decode_ms": layer_ms, "identical": same, **summary}, indent=1))
    finally:
        for t in (t_on, t_off):
            t.shutdown()
        for px in proxies:
            px.close()
        for s in servers:
            s.stop()
            s.block.shutdown()


if __name__ == "__main__":
    sys.exit(main())
