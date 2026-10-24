"""Speculative vs plain decoding on the synthetic cluster, across RTT, k and drafter acceptance.

    python -m commonnet.bench.spec_bench [--stages 3] [--layers 36] [--layer-ms 1.0] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from commonnet import speculative as spec
from commonnet.bench import harness as h
from commonnet.bench.synth import OracleDrafter, SynthClient


def one(cluster, net, hidden, prompt, n, *, drafter=None, k=8, adaptive=False):
    endpoints, transport = net.plan()
    client = SynthClient(transport, layers=cluster.layers, hidden=hidden, seed=cluster.seed, dtype=torch.float32)
    stats = spec.SpecStats()
    t0 = time.perf_counter()
    try:
        out = list(spec.generate_stream(client, endpoints, prompt, n, drafter=drafter, k=k, adaptive=adaptive,
                                        stats=stats))
    finally:
        transport.shutdown()
    return out, time.perf_counter() - t0, stats


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--layers", type=int, default=36)
    p.add_argument("--layer-ms", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--new-tokens", type=int, default=48)
    p.add_argument("--rtts", default="0,25,50,100")
    p.add_argument("--ks", default="4,8")
    p.add_argument("--accepts", default="0.5,0.7,0.9")
    p.add_argument("--out", default="bench-results/spec-synth.json")
    a = p.parse_args(argv)
    rows = []
    prompt = [(7 * i + 3) % 2048 for i in range(16)]
    with h.Cluster(a.stages, 1, layers=a.layers, layer_ms=a.layer_ms) as c:
        for rtt in [float(x) for x in a.rtts.split(",")]:
            c.set_network(rtt_ms=rtt)
            net = c.network()
            one(c, net, a.hidden, prompt, 4)                        # warm up
            truth, t_plain, _ = one(c, net, a.hidden, prompt, a.new_tokens)
            # the oracle needs the plain continuation; it must also cover the last round's drafts
            more, _, _ = one(c, net, a.hidden, prompt, a.new_tokens + 12)
            plain_tps = a.new_tokens / t_plain
            rows.append({"rtt_ms": rtt, "mode": "plain", "tok_s": plain_tps})
            print(f"rtt {rtt:g}: plain {plain_tps:.2f} tok/s", flush=True)
            for k in [int(x) for x in a.ks.split(",")]:
                for acc in [float(x) for x in a.accepts.split(",")]:
                    d = OracleDrafter(len(prompt), more, acc, seed=11)
                    out, t, st = one(c, net, a.hidden, prompt, a.new_tokens, drafter=d, k=k)
                    assert out == truth, "speculative output differs from plain"
                    row = {"rtt_ms": rtt, "mode": "spec", "k": k, "target_accept": acc,
                           "measured_accept": round(st.acceptance, 3), "tok_s": a.new_tokens / t,
                           "speedup": t_plain / t, "tokens_per_round": round(st.tokens_per_round, 2),
                           "rounds": st.rounds, "identical": True}
                    rows.append(row)
                    print(f"  k={k} accept~{acc}: {row['tok_s']:.2f} tok/s  x{row['speedup']:.2f}  "
                          f"({st.tokens_per_round:.1f} tok/round, measured accept {st.acceptance:.2f})", flush=True)
    import pathlib
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps({"config": vars(a), "rows": rows}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
