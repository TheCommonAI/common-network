"""Benchmark a real, running layer network over the real connection (python -m commonnet.bench --live SEED).

The route comes from gossip (Network.plan(), as `commonnet.serve` builds it), the generations are the real client's
(Client4 through NetworkBackend), and every number comes from BlockClient.timings: per hop, the call's total_ms and the
block's own compute (server_ms, from the Server-Timing header). Per token:

    compute  = sum of server_ms over the hops                        (the blocks' own time)
    network  = sum of (total_ms - server_ms) over the hops           (link RTT + serialisation + queueing + tunnel)
    client   = wall time of the token - sum of total_ms              (embedding, head, sampling, tensor conversion)

The bare round trip of each hop (GET /health on a warm connection) is shown beside its network time, so what is the
link and what is the stack's own cost can be told apart. Nothing secret is printed or saved: peer block tokens travel
in signed records and are only handed to the transport; the report holds scheme://host[:port] and short peer ids.
"""
from __future__ import annotations

import statistics
import time
import urllib.parse

from commonnet.bench.harness import pct
from commonnet.mesh.records import route_kind
from commonnet.net.client import BlockClient

DEFAULT_CASES = ((16, 16), (16, 64), (256, 32), (1024, 16))   # (prompt tokens, new tokens)
HEALTH_CALLS = 5
_PASSAGE = ("The quick brown fox jumps over the lazy dog while a distant train crosses the old iron bridge at dawn. "
            "Rivers carry stories from the mountains to the sea, and every harbour keeps a few of them. ")


def clean_url(url: str) -> str:
    """scheme://host[:port] only: no path, no query string, no credentials."""
    p = urllib.parse.urlsplit(url or "")
    if not p.hostname:
        return ""
    host = p.hostname + (f":{p.port}" if p.port else "")
    return f"{p.scheme}://{host}"


def prompt_ids(tokenizer, n: int) -> list[int]:
    """Exactly n ids of a fixed passage (repeated as needed), so every run sends the same prompt."""
    base = list(tokenizer(_PASSAGE, add_special_tokens=False)["input_ids"])
    return (base * (n // len(base) + 1))[:n]


# --- recording -------------------------------------------------------------------------------------

class _Tap:
    """A timings deque that also appends to a shared list, tagged with the client that made the call."""

    def __init__(self, sink: list, owner):
        self.sink, self.owner = sink, owner
        self.rows: list = []

    def append(self, row):
        row = dict(row, owner=id(self.owner), t=time.perf_counter())
        self.rows.append(row)
        self.sink.append(row)

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)


class Recording:
    """While active, every BlockClient created copies its timing rows into .sink."""

    def __init__(self):
        self.sink: list = []

    def __enter__(self):
        self._orig = BlockClient.__init__
        orig, sink = self._orig, self.sink

        def init(client, *a, **kw):
            orig(client, *a, **kw)
            client.timings = _Tap(sink, client)

        BlockClient.__init__ = init
        return self

    def __exit__(self, *exc):
        BlockClient.__init__ = self._orig


# --- the route -------------------------------------------------------------------------------------

def describe_route(net) -> list[dict]:
    """The hops net.plan() chose, with public facts only (never a token)."""
    out = []
    for i, hop in enumerate(net._hops):
        records = [hop.primary, *hop.alternates]
        out.append({"hop": i, "window": list(hop.window),
                    "peers": [{"peer": str(r.get("peer", ""))[:8], "host": clean_url(r.get("url", "")), "kind": route_kind(r.get("url", "")),
                               "backend": r.get("backend", "") or "?"} for r in records]})
    return out


def route_warnings(route: list[dict], num_layers: int | None) -> list[str]:
    warnings = []
    if len(route) == 1:
        warnings.append("this route is ONE peer holding every layer: there is nothing to split, so there is no "
                        "network hop to measure and the figures below are single-machine compute plus HTTP. "
                        "Start more peers (smaller --stage-gib) so the model is split across them.")
    elif len({h["peers"][0]["host"] for h in route}) == 1:
        warnings.append("every hop is on the same host: the 'network' column is loopback, not a real link.")
    return warnings


def hop_rtt(net, calls: int = HEALTH_CALLS) -> list[dict]:
    """Bare round trip of each hop's primary peer: GET /health, first call (new connection) then warm calls."""
    out = []
    for i, hop in enumerate(net._hops):
        url, token = hop.primary["url"], net._tokens(hop.primary)
        client = BlockClient(token=token, retries=0)
        times = []
        try:
            for _ in range(calls + 1):
                t0 = time.perf_counter()
                client.health(url)
                times.append((time.perf_counter() - t0) * 1000)
            error = None
        except Exception as e:   # the peer answering slowly or not at all is a result, not a crash
            error = f"{type(e).__name__}"
        finally:
            client.close_connections()
        warm = times[1:]
        out.append({"hop": i, "first_ms": times[0] if times else None,
                    "warm_ms_p50": statistics.median(warm) if warm else None,
                    "warm_ms_min": min(warm) if warm else None, "error": error})
    return out


# --- one generation ---------------------------------------------------------------------------------

def generate_once(net, client, ids, new_tokens: int, *, drafter=None, draft_k: int = 8, stop_ids=None) -> dict:
    """One generation through plan(); returns tokens, per-step rows (plain) or round stats (speculative).

    Must run on the thread that owns the client (NetworkBackend's worker) when the client is MLX.
    """
    from commonnet import speculative
    with Recording() as rec:
        endpoints, transport = net.plan()
        client.transport = transport
        stats = speculative.SpecStats()
        marks, stamps, tokens = [], [], []
        t0 = time.perf_counter()
        try:
            if drafter is None:
                stream = client.generate_stream(endpoints, ids, new_tokens, temperature=0.0, stop_ids=stop_ids)
            else:
                stream = speculative.generate_stream(client, endpoints, ids, new_tokens, drafter=drafter, k=draft_k,
                                                     temperature=0.0, stop_ids=stop_ids, stats=stats)
            for tok in stream:
                tokens.append(tok)
                stamps.append(time.perf_counter())
                marks.append(len(rec.sink))
        finally:
            transport.shutdown()
        rows = list(rec.sink)
        wall_s = (stamps[-1] if stamps else time.perf_counter()) - t0
        owners = {}
        for (name, index), blk in transport._clients.items():
            owners[id(blk)] = (int(name.removeprefix("hop")), index)
    out = {"tokens": tokens, "wall_s": wall_s, "mode": stats.mode or "plain", "stats": stats}
    if drafter is not None:
        return out
    steps, prev_t, prev_mark = [], t0, 0
    for k, (t, mark) in enumerate(zip(stamps, marks)):
        # Loopback benchmarks can record server-side forwarding clients in this
        # process too. Only the generation client's own calls belong to its wall time.
        window = [r for r in rows[prev_mark:mark] if r["owner"] in owners]
        wall = (t - prev_t) * 1000
        total = sum(r["total_ms"] for r in window)
        fwd = [r for r in window if r["path"] in ("/forward", "/chain") and r["status"] == 200]
        chained = any(r["path"] == "/chain" for r in fwd)
        # /chain's inclusive server duration includes downstream links and peers.
        # Calling it compute would invent a breakdown we cannot measure here.
        server = float("nan") if chained else sum(r["server_ms"] or 0.0 for r in fwd)
        steps.append({"kind": "prefill" if k == 0 else "decode", "wall_ms": wall, "compute_ms": server,
                      "network_ms": total - server, "client_ms": wall - total,
                      "timing_mode": "chain_inclusive" if chained else "per_stage",
                      "missing_server_ms": sum(1 for r in fwd if chained or r["server_ms"] is None),
                      "hops": [dict(r, hop=owners.get(r["owner"], (-1, 0))) for r in fwd]})
        prev_t, prev_mark = t, mark
    out["steps"] = steps
    return out


# --- aggregation ------------------------------------------------------------------------------------

def _mean(values):
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def summarize_steps(steps: list[dict], route: list[dict], rtts: list[dict]) -> dict:
    """The breakdown of a set of steps of one kind (all prefill or all decode steps)."""
    walls = [s["wall_ms"] for s in steps]
    ms = _mean(walls)
    out = {"steps": len(steps), "ms_mean": ms, "ms_p50": pct(walls, 0.5), "ms_p95": pct(walls, 0.95),
           "tok_s": 1000.0 / ms if steps else float("nan"),
           "compute_ms": _mean(s["compute_ms"] for s in steps), "network_ms": _mean(s["network_ms"] for s in steps),
           "client_ms": _mean(s["client_ms"] for s in steps),
           "missing_server_ms": sum(s["missing_server_ms"] for s in steps),
           "timing_mode": "chain_inclusive" if any(s.get("timing_mode") == "chain_inclusive" for s in steps) else "per_stage"}
    per_hop = {}
    for s in steps:
        for r in s["hops"]:
            per_hop.setdefault(r["hop"], []).append(r)
    rows = []
    for (hop, index), calls in sorted(per_hop.items()):
        peers = route[hop]["peers"] if 0 <= hop < len(route) else []
        peer = peers[index] if index < len(peers) else {"peer": "?", "host": "", "backend": "?"}
        served = [c["server_ms"] for c in calls if c["server_ms"] is not None]
        total = _mean(c["total_ms"] for c in calls)
        chained = any(c["path"] == "/chain" for c in calls)
        server = None if chained else (_mean(served) if served else None)
        rtt = next((r for r in rtts if r["hop"] == hop), {}) if index == 0 else {}
        rows.append({"hop": hop, "window": route[hop]["window"] if 0 <= hop < len(route) else None,
                     "peer": peer["peer"], "host": peer["host"], "kind": peer.get("kind", ""),
                     "backend": peer["backend"], "replica_index": index, "calls": len(calls), "total_ms": total, "server_ms": server,
                     "network_ms": None if chained else total - (server or 0.0),
                     "timing_mode": "chain_inclusive" if chained else "per_stage", "rtt_ms": rtt.get("warm_ms_p50"),
                     "reuse_rate": sum(1 for c in calls if c["reused"]) / len(calls),
                     "bytes_per_call": _mean(c["bytes_out"] + c["bytes_in"] for c in calls)})
    out["per_hop"] = rows
    return out


def run_case(net, client, route, rtts, prompt_len, new_tokens, *, ids, repeats, run=None, **gen) -> dict:
    """`repeats` plain generations of one prompt; `run` executes a callable on the client's thread (default: here)."""
    run = run or (lambda fn: fn())
    gens = [run(lambda: generate_once(net, client, ids, new_tokens, **gen)) for _ in range(repeats)]
    steps = [s for g in gens for s in g["steps"]]
    prefill = summarize_steps([s for s in steps if s["kind"] == "prefill"], route, rtts)
    decode = summarize_steps([s for s in steps if s["kind"] == "decode"], route, rtts)
    prefill["tok_s"] = 1000.0 * prompt_len / prefill["ms_mean"] if prefill["steps"] else float("nan")
    return {"prompt_len": prompt_len, "new": new_tokens, "repeats": repeats,
            "generated": [len(g["tokens"]) for g in gens], "prefill": prefill, "decode": decode,
            "first_run_ids": gens[0]["tokens"][:64]}


def run_speculative(net, client, ids, new_tokens, *, drafter, draft_k, run=None, stop_ids=None) -> dict:
    """Plain vs speculative on the same prompt, greedy. Returns tok/s of both, acceptance, and whether tokens match."""
    run = run or (lambda fn: fn())
    plain = run(lambda: generate_once(net, client, ids, new_tokens, stop_ids=stop_ids))
    spec = run(lambda: generate_once(net, client, ids, new_tokens, drafter=drafter, draft_k=draft_k, stop_ids=stop_ids))
    st = spec["stats"]
    if not st.mode.startswith("speculative"):
        return {"skipped": st.mode or "plain"}
    # tok/s over the whole generation (prefill included) for both, so the two are comparable
    return {"plain_tok_s": len(plain["tokens"]) / plain["wall_s"], "spec_tok_s": len(spec["tokens"]) / spec["wall_s"],
            "speedup": (len(spec["tokens"]) / spec["wall_s"]) / (len(plain["tokens"]) / plain["wall_s"]),
            "acceptance": st.acceptance, "rounds": st.rounds, "drafted": st.drafted, "accepted": st.accepted,
            "same_output_as_plain": spec["tokens"] == plain["tokens"]}


# --- the whole run ---------------------------------------------------------------------------------

def run_live(spec: str | None, *, make_client, bootstrap=(), cases=DEFAULT_CASES, repeats: int = 2, draft: bool = False, draft_k: int = 8,
             log=print, settle_s: float = 1.0) -> dict:
    """Learn the network from `spec`, show the route, measure every case. Returns the JSON-able report.

    make_client(net) -> (client, make_ids, run, stop_ids, drafter): the client, a function n -> prompt ids,
    a function that runs a callable on the client's thread, stop ids, and the drafter (or None).
    """
    from commonnet.netclient import Network
    net = Network.open(spec, bootstrap)   # a URL or manifest: gossip as before; none/"dht" or bootstrap: the DHT too
    manifest = net.manifest
    net.refresh(force=True)
    time.sleep(settle_s)
    net.refresh(force=True)       # a second round picks up peers the first one's peers had just learned
    client, make_ids, run, stop_ids, drafter = make_client(net)
    endpoints, transport = net.plan()
    transport.shutdown()
    route = describe_route(net)
    model = manifest.model.get("name", manifest.id)
    log(f"network {manifest.id}  model {model}  seed {clean_url(spec or '') or spec or 'dht (no seed URL)'}")
    log(f"route: {len(route)} stage(s), {len(net.table.live())} live peer(s) known")
    for h in route:
        extra = f"  (+{len(h['peers']) - 1} replica)" if len(h["peers"]) > 1 else ""
        p = h["peers"][0]
        log(f"  hop {h['hop']}: layers {h['window'][0]}-{h['window'][1]}  peer {p['peer']}  {p['backend']}  {p['kind']:<6} {p['host']}{extra}")
    warnings = route_warnings(route, manifest.model.get("num_layers"))
    for w in warnings:
        log(f"WARNING: {w}")
    rtts = hop_rtt(net)
    for r in rtts:
        if r["error"]:
            log(f"  hop {r['hop']} /health failed: {r['error']}")
    report = {"version": 1, "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "network": manifest.id,
              "model": model, "seed": clean_url(spec or ""), "route": route, "warnings": warnings, "hop_rtt": rtts,
              "config": {"repeats": repeats, "cases": [list(c) for c in cases], "draft": bool(draft), "draft_k": draft_k},
              "cases": [], "speculative": []}
    # warm-up: open the connections (TLS to every tunnel) before anything is timed
    run(lambda: generate_once(net, client, make_ids(16), 3, stop_ids=None))
    for prompt_len, new in cases:
        log(f"case: prompt {prompt_len} tokens, {new} new, x{repeats} ...")
        case = run_case(net, client, route, rtts, prompt_len, new, ids=make_ids(prompt_len), repeats=repeats,
                        run=run, stop_ids=None)
        report["cases"].append(case)
        if draft:
            if drafter is None:
                report["speculative"].append({"prompt_len": prompt_len, "new": new, "skipped": "no drafter loaded"})
            else:
                try:
                    r = run_speculative(net, client, make_ids(prompt_len), new, drafter=drafter, draft_k=draft_k, run=run)
                except Exception as e:
                    r = {"skipped": f"{type(e).__name__}: {e}"[:200]}
                report["speculative"].append({"prompt_len": prompt_len, "new": new, **r})
    report["summary"] = summary_of(report)
    return report


def summary_of(report: dict) -> dict:
    """All decode steps of all cases, weighted by step count."""
    cases = [c for c in report["cases"] if c["decode"]["steps"]]
    n = sum(c["decode"]["steps"] for c in cases)
    if not n:
        return {}

    def w(key):
        return sum(c["decode"][key] * c["decode"]["steps"] for c in cases) / n

    ms = w("ms_mean")
    return {"decode_ms_per_tok": ms, "decode_tok_s": 1000.0 / ms, "compute_ms": w("compute_ms"),
            "network_ms": w("network_ms"), "client_ms": w("client_ms"), "hops": len(report["route"]),
            "rtt_sum_ms": sum(r["warm_ms_p50"] or 0.0 for r in report["hop_rtt"]),
            "timing_mode": "chain_inclusive" if any(c["decode"].get("timing_mode") == "chain_inclusive" for c in cases) else "per_stage"}


# --- printing ---------------------------------------------------------------------------------------

def _f(x, nd=1):
    return "-" if x is None or x != x else f"{x:.{nd}f}"


def _table(header, rows) -> str:
    rows = [tuple(str(v) for v in header)] + [tuple(str(v) for v in r) for r in rows]
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    lines = ["  ".join(v.rjust(w) for v, w in zip(r, widths)) for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def format_report(report: dict) -> str:
    out = []
    head = ("prompt", "new", "ms/tok", "p50", "p95", "tok/s", "compute", "network", "client", "prefill ms", "prefill tok/s")
    rows = []
    for c in report["cases"]:
        d, p = c["decode"], c["prefill"]
        rows.append((c["prompt_len"], c["new"], _f(d["ms_mean"]), _f(d["ms_p50"]), _f(d["ms_p95"]), _f(d["tok_s"]),
                     _f(d["compute_ms"]), _f(d["network_ms"]), _f(d["client_ms"]), _f(p["ms_mean"]), _f(p["tok_s"])))
    chained = any(c["decode"].get("timing_mode") == "chain_inclusive" for c in report["cases"])
    label = ("Per token (decode), mean ms; chain compute/network breakdown unavailable"
             if chained else "Per token (decode), mean ms; compute + network + client = ms/tok")
    out += ["", label, _table(head, rows)]
    for c in report["cases"]:
        for kind in ("decode", "prefill"):
            hops = c[kind]["per_hop"]
            if not hops:
                continue
            head = ("hop", "layers", "peer", "backend", "host", "calls", "total ms", "server ms", "network ms",
                    "RTT ms", "reuse%", "KB/call")
            rows = [(h["hop"], f"{h['window'][0]}-{h['window'][1]}" if h["window"] else "?",
                     h["peer"] + (f"#{h['replica_index']}" if h["replica_index"] else ""), h["backend"], h["host"],
                     h["calls"], _f(h["total_ms"]), _f(h["server_ms"]), _f(h["network_ms"]), _f(h["rtt_ms"]),
                     f"{100 * h['reuse_rate']:.0f}", _f(h["bytes_per_call"] / 1024)) for h in hops]
            out += ["", f"Per hop, {kind}, prompt {c['prompt_len']} / {c['new']} new  "
                        f"(network = total - server; RTT = bare GET /health on a warm connection)", _table(head, rows)]
    for r in report["speculative"]:
        if r.get("skipped"):
            out += ["", f"speculative, prompt {r['prompt_len']}: skipped ({r['skipped']})"]
    spec = [r for r in report["speculative"] if not r.get("skipped")]
    if spec:
        head = ("prompt", "new", "plain tok/s", "spec tok/s", "speedup", "acceptance", "rounds", "same output")
        rows = [(r["prompt_len"], r["new"], _f(r["plain_tok_s"]), _f(r["spec_tok_s"]), _f(r["speedup"], 2),
                 f"{100 * r['acceptance']:.0f}%", r["rounds"], "yes" if r["same_output_as_plain"] else "NO") for r in spec]
        out += ["", "Plain vs speculative (greedy, whole generation incl. prefill)", _table(head, rows)]
    out += ["", "Bare round trip per hop (GET /health): " + ", ".join(
        f"hop {r['hop']} {_f(r['warm_ms_p50'])} ms (first call {_f(r['first_ms'])} ms)" for r in report["hop_rtt"])]
    return "\n".join(out)


def one_line(report: dict) -> str:
    s = report.get("summary") or {}
    if not s:
        return "no decode steps were measured"
    if s.get("timing_mode") == "chain_inclusive":
        return (f"decode {s['decode_ms_per_tok']:.1f} ms/token ({s['decode_tok_s']:.1f} tok/s); "
                "peer-forwarded chain: compute/network breakdown unavailable")
    return (f"decode {s['decode_ms_per_tok']:.1f} ms/token ({s['decode_tok_s']:.1f} tok/s) = compute "
            f"{s['compute_ms']:.1f} + network {s['network_ms']:.1f} ({s['hops']} hop(s), sum of bare RTTs "
            f"{s['rtt_sum_ms']:.1f} ms, the rest is serialisation and queueing) + client {s['client_ms']:.1f}")


# --- the real client --------------------------------------------------------------------------------

def real_client_factory(checkpoint: str | None, *, client_format: str = "mlx4", draft: bool = False,
                        draft_model: str | None = None, draft_k: int = 8):
    """make_client for run_live: NetworkBackend (the one `commonnet.serve` uses), run on its own worker thread."""
    def make(net):
        import torch
        from commonnet.serve import NetworkBackend
        backend = NetworkBackend(net, checkpoint, fmt=client_format, hidden_dtype=torch.float32,
                                 draft=(draft_model or "mlx-community/Qwen3-0.6B-4bit") if draft else None,
                                 draft_k=draft_k)
        if draft and backend.drafter is None:
            print(f"speculative path unavailable: {backend.draft_note}")
        return (backend.client, lambda n: prompt_ids(backend.tokenizer, n),
                lambda fn: backend._worker.submit(fn).result(), None, backend.drafter)
    return make
