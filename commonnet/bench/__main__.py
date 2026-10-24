"""python -m commonnet.bench [--sweep|--failover|--stages N ...]  (see docs/network-bench.md)"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from commonnet.bench import harness as h


def build_parser():
    p = argparse.ArgumentParser(prog="commonnet.bench", description=__doc__)
    p.add_argument("--stages", type=int, default=2, help="stages in the route (default 2)")
    p.add_argument("--replicas", type=int, default=1)
    p.add_argument("--layers", type=int, help="model layers (default: one per stage)")
    p.add_argument("--layer-ms", type=float, default=0.0, help="synthetic compute per layer, ms")
    p.add_argument("--rtt-ms", type=float, default=0.0, help="emulated round trip per hop, ms")
    p.add_argument("--jitter-ms", type=float, default=0.0)
    p.add_argument("--loss", type=float, default=0.0, help="chance a chunk waits one 200 ms retransmit timeout")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--prompt-tokens", type=int, default=16)
    p.add_argument("--new-tokens", type=int, default=None, help="tokens to generate (default 20; with --live: every case)")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--bootstrap", action="append", default=[], metavar="HOST:PORT",
                   help="with --live: join the DHT through this node (repeatable); with no URL, peers are found by DHT only")
    p.add_argument("--live", metavar="URL_OR_MANIFEST", nargs="?", const="dht",
                   help="benchmark a REAL running network: a peer URL (as serve --network takes) or a manifest file")
    p.add_argument("--prompt-lens", help="with --live: comma list of prompt lengths in tokens (default 16,256,1024)")
    p.add_argument("--checkpoint", help="with --live: local client checkpoint (default: the manifest's client_repo)")
    p.add_argument("--client-format", choices=["mlx4", "fp16", "fp32"], default="mlx4", help="with --live")
    p.add_argument("--draft", nargs="?", const="mlx-community/Qwen3-0.6B-4bit", default=None, metavar="MODEL",
                   help="with --live: also run the Common Flash speculative path with this drafter model")
    p.add_argument("--draft-k", type=int, default=8, help="with --live --draft: tokens drafted per round")
    p.add_argument("--no-keepalive", action="store_true", help="close the HTTP connection after every request")
    p.add_argument("--sweep", action="store_true", help="the full sweep: stages, RTT, hidden size, keep-alive, failover")
    p.add_argument("--failover", action="store_true", help="only the kill-a-replica scenario")
    p.add_argument("--sweep-layers", type=int, default=12, help="model layers for --sweep (must divide by 1..4)")
    p.add_argument("--sweep-layer-ms", type=float, default=3.0)
    p.add_argument("--out", help="JSON output path (default bench-results/<timestamp>.json)")
    return p


def save(payload: dict, out: str | None) -> Path:
    path = Path(out) if out else Path("bench-results") / time.strftime("%Y%m%d-%H%M%S.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    return path


def run_sweep(args) -> dict:
    L, ms = args.sweep_layers, args.sweep_layer_ms
    kw = dict(prompt_tokens=args.prompt_tokens, new_tokens=args.new_tokens, repeats=args.repeats)
    results, groups = [], {}

    def log(r, group):
        results.append(r)
        groups.setdefault(group, []).append(r)
        print(h.format_table([r]).splitlines()[-1], flush=True)

    print(h.COLS, flush=True)
    for stages in (1, 2, 3, 4):
        with h.Cluster(stages, 1, layers=L, layer_ms=ms) as c:
            for rtt in (0, 25, 50, 100):
                c.set_network(rtt_ms=rtt)
                log(h.scenario(c, hidden=args.hidden, **kw), "stages x rtt")
    with h.Cluster(2, 1, layers=L, layer_ms=ms) as c:
        for hidden in (256, 1024, 4096, 8192):
            for rtt in (0, 50):
                c.set_network(rtt_ms=rtt)
                log(h.scenario(c, hidden=hidden, **kw), "hidden size (2 stages)")
        c.set_network(rtt_ms=50)
        for ka in (True, False):
            log(h.scenario(c, hidden=args.hidden, keepalive=ka, **kw), "keep-alive (2 stages, 50 ms)")
        c.set_network(rtt_ms=50, jitter_ms=20, loss=0.0)
        log(h.scenario(c, hidden=args.hidden, **kw), "jitter 20 ms (2 stages, 50 ms)")
        c.set_network(rtt_ms=50, jitter_ms=0, loss=0.02)
        log(h.scenario(c, hidden=args.hidden, **kw), "loss 2% (2 stages, 50 ms)")
    fo = h.failover_scenario(stages=2, layer_ms=ms, rtt_ms=50, new_tokens=24)
    print(json.dumps(fo), flush=True)
    return {"results": results, "groups": {k: [results.index(r) for r in v] for k, v in groups.items()},
            "failover": fo, "sweep": {"layers": L, "layer_ms": ms}}


def run_live_cli(args, make_client=None) -> int:
    from commonnet.bench import live
    if args.prompt_lens or args.new_tokens:
        lens = [int(x) for x in args.prompt_lens.split(",")] if args.prompt_lens else [16, 256, 1024]
        cases = [(n, args.new_tokens or 32) for n in lens]
    else:
        cases = list(live.DEFAULT_CASES)
    make_client = make_client or live.real_client_factory(
        args.checkpoint, client_format=args.client_format, draft=bool(args.draft), draft_model=args.draft,
        draft_k=args.draft_k)
    report = live.run_live(args.live, make_client=make_client, bootstrap=args.bootstrap, cases=cases, repeats=args.repeats,
                           draft=bool(args.draft), draft_k=args.draft_k)
    print(live.format_report(report))
    print()
    print(live.one_line(report))
    out = args.out or str(Path("bench-results") / time.strftime("live-%Y%m%d-%H%M%S.json", time.gmtime()))
    print(f"wrote {save(report, out)}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.live:
        return run_live_cli(args)
    args.new_tokens = args.new_tokens or 20
    if args.sweep:
        payload = run_sweep(args)
        path = save(payload, args.out)
        print()
        print(h.format_table(payload["results"]))
        pick = next((r for r in payload["results"] if r["config"]["stages"] == 2 and r["config"]["rtt_ms"] == 50), payload["results"][0])
        print(h.one_line(pick))
        print(f"wrote {path}")
        return 0
    if args.failover:
        fo = h.failover_scenario(stages=args.stages, layer_ms=args.layer_ms, rtt_ms=args.rtt_ms, hidden=args.hidden)
        print(json.dumps(fo, indent=1))
        print(f"wrote {save({'failover': fo}, args.out)}")
        return 0
    with h.Cluster(args.stages, args.replicas, layers=args.layers, layer_ms=args.layer_ms, rtt_ms=args.rtt_ms,
                   jitter_ms=args.jitter_ms, loss=args.loss) as c:
        c.set_network()
        r = h.scenario(c, hidden=args.hidden, prompt_tokens=args.prompt_tokens, new_tokens=args.new_tokens,
                       repeats=args.repeats, keepalive=not args.no_keepalive)
    print(h.format_table([r]))
    print(h.one_line(r))
    print(f"wrote {save({'results': [r]}, args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
