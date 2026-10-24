"""Plain vs speculative decoding on the real Qwen3-8B (MLX 4-bit) split over local block servers behind delay proxies,
with a real Qwen3-0.6B drafter on the client.

    python -m commonnet.bench.spec_real --checkpoint DIR [--stages 3] [--rtt-ms 50] [--ks 4,8] [--new-tokens 64]

All servers run in this process (each block on its own MLX worker thread) so the whole run is one command; the wire path
(HTTP, base64 JSON, float32 hidden states, delay proxy) is the real one.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import os
import torch
from transformers import AutoTokenizer

from commonnet import speculative as spec
from commonnet.net import factory
from commonnet.bench.proxy import DelayProxy
from commonnet.client import BlockEndpoint
from commonnet.client4 import Client4
from commonnet.net.server import BlockServer
from commonnet.net.transport import HttpTransport

PROMPTS = {
    "chat": "Explain in a few sentences why the sky is blue, and why sunsets are red.",
    "code": "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and two example calls.",
    "copy": ("Repeat the following text exactly, word for word, and add nothing else:\n\n"
             "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
             "How vexingly quick daft zebras jump! Sphinx of black quartz, judge my vow."),
    "chat2": "Give me three tips for staying focused while working from home.",
}


def main(argv=None):
    os.environ.setdefault("COMMONNET_DTYPE", "bfloat16")   # as a real node does
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--draft", default="mlx-community/Qwen3-0.6B-4bit")
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--rtt-ms", type=float, default=50.0)
    p.add_argument("--ks", default="4,8")
    p.add_argument("--new-tokens", type=int, default=64)
    p.add_argument("--out", default="bench-results/spec-real.json")
    a = p.parse_args(argv)
    cfg = json.load(open(pathlib.Path(a.checkpoint) / "config.json"))
    layers = cfg["num_hidden_layers"]
    step = layers // a.stages
    bounds = [(i * step, layers if i == a.stages - 1 else (i + 1) * step) for i in range(a.stages)]
    servers, proxies, eps = [], [], []
    for i, (s, e) in enumerate(bounds):
        blk = factory.build_block(s, e, backend="mlx", checkpoint=a.checkpoint)
        srv = BlockServer(blk, "127.0.0.1", 0).start()
        servers.append(srv)
        proxies.append(DelayProxy(srv.port, rtt_ms=0.0, seed=i))
        eps.append(BlockEndpoint(f"hop{i}", s, e))
    urls = {ep.name: px.url for ep, px in zip(eps, proxies)}
    transport = HttpTransport(urls, retries=1)
    client = Client4(a.checkpoint, transport, fmt="mlx4", hidden_dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    t0 = time.time()
    drafter = spec.MlxDrafter(a.draft, target_tokenizer=tok)   # raises SpeculationRefused on a tokenizer mismatch
    print(f"drafter {a.draft} loaded in {time.time() - t0:.1f}s; tokenizer matches the target", flush=True)
    stop_ids = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    rows = []
    try:
        for px in proxies:
            px.rtt_ms = 0.0
        # warm-up (kernel compilation) with no delay
        ids0 = tok(tok.apply_chat_template([{"role": "user", "content": "Hi"}], add_generation_prompt=True, tokenize=False,
                                           enable_thinking=False), add_special_tokens=False)["input_ids"]
        list(client.generate_stream(eps, ids0, 6, stop_ids=stop_ids))
        list(spec.generate_stream(client, eps, ids0, 12, drafter=drafter, k=8, stop_ids=stop_ids))
        for px in proxies:
            px.rtt_ms = a.rtt_ms
        for name, text in PROMPTS.items():
            ids = tok(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                              tokenize=False, enable_thinking=False), add_special_tokens=False)["input_ids"]
            t = time.perf_counter()
            plain = list(client.generate_stream(eps, ids, a.new_tokens, stop_ids=stop_ids))
            tp = time.perf_counter() - t
            row = {"prompt": name, "prompt_tokens": len(ids), "tokens": len(plain), "plain_tok_s": len(plain) / tp, "spec": []}
            print(f"[{name}] plain {len(plain) / tp:.2f} tok/s ({len(plain)} tokens)", flush=True)
            for k in [int(x) for x in a.ks.split(",")]:
                st = spec.SpecStats()
                t = time.perf_counter()
                out = list(spec.generate_stream(client, eps, ids, a.new_tokens, drafter=drafter, k=k, stop_ids=stop_ids,
                                                stats=st))
                ts = time.perf_counter() - t
                same = out == plain
                first_diff = next((i for i, (x, y) in enumerate(zip(out, plain)) if x != y), None)
                r = {"k": k, "tok_s": len(out) / ts, "speedup": (len(out) / ts) / row["plain_tok_s"],
                     "acceptance": st.acceptance, "tokens_per_round": st.tokens_per_round, "rounds": st.rounds,
                     "identical_to_plain": same, "first_diff": first_diff, "mode": st.mode}
                row["spec"].append(r)
                print(f"   k={k}: {r['tok_s']:.2f} tok/s x{r['speedup']:.2f}  accept {st.acceptance:.2f}  "
                      f"{st.tokens_per_round:.1f} tok/round  identical={same} (first diff {first_diff})", flush=True)
            rows.append(row)
    finally:
        transport.shutdown()
        for px in proxies:
            px.close()
        for s in servers:
            s.stop()
            s.block.shutdown()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps({"config": vars(a), "rows": rows}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
