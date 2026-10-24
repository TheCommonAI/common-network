"""Start synthetic nodes on loopback, put a delay proxy in front of each, drive the real client, break down the time.

The path under test is the real one: Node (gossip, placement, signed records) -> BlockServer (HTTP/1.1, JSON,
base64) <- BlockClient/HttpTransport (keep-alive, replica failover) <- Network (route from gossip). Only the
model (SynthBlock) and the wire delay (DelayProxy) are synthetic.
"""
from __future__ import annotations

import collections
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from commonnet.bench.proxy import DelayProxy
from commonnet.bench.synth import DEFAULT_SEED, SynthClient, manifest_dict
from commonnet.net.client import BlockClient
from commonnet.netclient import Network

_SINK: list = []


class _Rec(collections.deque):
    def append(self, item):
        _SINK.append(item)
        super().append(item)


@contextmanager
def recording(keepalive: bool = True):
    """Copy every BlockClient timing row into _SINK; with keepalive=False close each connection after use."""
    orig_init, orig_keep = BlockClient.__init__, BlockClient._keep

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        self.timings = _Rec(maxlen=4096)

    BlockClient.__init__ = init
    if not keepalive:
        BlockClient._keep = lambda self, key, conn: conn.close()
    try:
        yield
    finally:
        BlockClient.__init__, BlockClient._keep = orig_init, orig_keep


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Cluster:
    """stages x replicas nodes, each behind its own DelayProxy. Nodes run as separate processes."""

    def __init__(self, stages: int, replicas: int = 1, *, layers: int | None = None, layer_ms: float = 0.0,
                 rtt_ms: float = 0.0, jitter_ms: float = 0.0, loss: float = 0.0, seed: int = DEFAULT_SEED):
        self.stages, self.replicas = stages, replicas
        self.layers = layers or stages
        self.layer_ms, self.seed = layer_ms, seed
        self.manifest = manifest_dict(stages, self.layers)
        self.rtt_ms, self.jitter_ms, self.loss = rtt_ms, jitter_ms, loss
        self.nodes: list[dict] = []   # {proc, proxy, port, window}
        self._tmp = tempfile.TemporaryDirectory(prefix="commonnet-bench-")

    def set_network(self, rtt_ms=None, jitter_ms=None, loss=None) -> None:
        if rtt_ms is not None:
            self.rtt_ms = rtt_ms
        if jitter_ms is not None:
            self.jitter_ms = jitter_ms
        if loss is not None:
            self.loss = loss
        for n in self.nodes:
            n["proxy"].rtt_ms, n["proxy"].jitter_ms, n["proxy"].loss = self.rtt_ms, self.jitter_ms, self.loss

    def start(self) -> "Cluster":
        tmp = Path(self._tmp.name)
        manifest_path = tmp / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest))
        env = dict(os.environ, COMMON_NETWORK_HOME=str(tmp / "home"), PYTHONUNBUFFERED="1")
        seed_url = None
        order = [(s, r) for r in range(self.replicas) for s in range(self.stages)]
        for i, (_s, _r) in enumerate(order):
            port = _free_port()
            proxy = DelayProxy(port, rtt_ms=0.0, seed=i)
            cmd = [sys.executable, "-m", "commonnet.bench.nodeproc", "--manifest", str(manifest_path),
                   "--port", str(port), "--advertise", proxy.url, "--identity-dir", str(tmp / f"id{i}"),
                   "--layer-ms", str(self.layer_ms), "--weights-seed", str(self.seed),
                   "--stage-layers", str(self.layers // self.stages)]
            if seed_url:
                cmd += ["--seed", seed_url]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            line = proc.stdout.readline()
            if not line.startswith("READY"):
                err = proc.stderr.read() if proc.poll() is not None else ""
                proc.kill()
                raise RuntimeError(f"node {i} did not start: {line!r} {err[-500:]}")
            _, a, b = line.split()
            self.nodes.append({"proc": proc, "proxy": proxy, "port": port, "window": (int(a), int(b))})
            if seed_url is None:
                seed_url = f"http://127.0.0.1:{port}"   # gossip bootstrap goes direct; only block traffic is delayed
        self.seed_url = seed_url
        self.set_network()
        return self

    def network(self, **kw) -> Network:
        kw.setdefault("refresh_interval", 1.0)
        kw.setdefault("wait_s", 20.0)
        net = Network(seeds=[self.seed_url], manifest=self.manifest, **kw)
        deadline = time.time() + 30
        want = self.stages * self.replicas
        while True:  # gossip is asynchronous: wait until the client sees every node
            net.refresh(force=True)
            if len([r for r in net.table.live() if r.get("window")]) >= want:
                return net
            if time.time() > deadline:
                raise RuntimeError("client did not see every node by gossip in 30 s")
            time.sleep(0.3)

    def node_for_url(self, url: str) -> dict:
        for n in self.nodes:
            if n["proxy"].url == url:
                return n
        raise KeyError(url)

    def kill(self, node: dict) -> None:
        node["proc"].kill()
        node["proxy"].close()

    def stop(self) -> None:
        for n in self.nodes:
            if n["proc"].poll() is None:
                n["proc"].terminate()
        for n in self.nodes:
            try:
                n["proc"].wait(timeout=5)
            except subprocess.TimeoutExpired:
                n["proc"].kill()
            n["proxy"].close()
            for f in (n["proc"].stdout, n["proc"].stderr):
                if f:
                    f.close()
        self._tmp.cleanup()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def pct(values, q):
    if not values:
        return float("nan")
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def run_generation(cluster: Cluster, net: Network, *, hidden: int, prompt_tokens: int, new_tokens: int,
                   keepalive: bool = True, kill_at: int | None = None, kill_hop: int = 0) -> dict:
    """One prefill + decode through the real client. Returns per-step rows with the timing breakdown."""
    with recording(keepalive):
        endpoints, transport = net.plan()
        client = SynthClient(transport, layers=cluster.layers, hidden=hidden, seed=cluster.seed)
        prompt = [(7 * i + 3) % client.config.vocab_size for i in range(prompt_tokens)]
        marks, tokens, killed = [], [], {}

        def before(step):
            marks.append(len(_SINK))
            if kill_at is not None and step == kill_at:
                url = transport.urls[f"hop{kill_hop}"]
                killed["url"] = url
                cluster.kill(cluster.node_for_url(url))

        start_sink = len(_SINK)
        try:
            for t in client.generate_stream(endpoints, prompt, new_tokens, before_step=before):
                tokens.append(t)
        finally:
            transport.shutdown()
        marks.append(len(_SINK))
        rows = [r for r in _SINK[start_sink:]]
        failovers = list(transport.failovers)
    steps = []
    for i, s in enumerate(client.steps):
        # forward rows between this step's mark and the next one; the next mark is taken before the next step
        # but after this step's closing code, so the slice holds exactly this step's calls (and any replay).
        lo = marks[i]
        hi = marks[i + 1] if i + 1 < len(marks) else len(_SINK)
        fw = [r for r in _SINK[lo:hi] if r["path"] == "/forward" and r["status"] == 200]
        total = sum(r["total_ms"] for r in fw)
        server = sum(r["server_ms"] or 0.0 for r in fw)
        # a replayed history after a failover is extra forwards; count only the step's own hops for the split
        steps.append({**{k: s[k] for k in ("kind", "tokens_in", "embed_ms", "blocks_ms", "head_ms", "sample_ms",
                                            "wall_ms")},
                      "calls": len(fw), "compute_ms": server, "network_ms": total - server,
                      "codec_ms": s["blocks_ms"] - total,
                      "client_ms": s["wall_ms"] - total,
                      "reused": sum(1 for r in fw if r["reused"]),
                      "bytes": sum(r["bytes_out"] + r["bytes_in"] for r in fw)})
    return {"tokens": tokens, "steps": steps, "failovers": failovers, "killed": killed.get("url")}


def summarize(runs: list[dict]) -> dict:
    """Aggregate the steps of several generations into the per-token breakdown."""
    pre = [s for r in runs for s in r["steps"] if s["kind"] == "prefill"]
    dec = [s for r in runs for s in r["steps"] if s["kind"] == "decode"]

    def mean(rows, k):
        return statistics.fmean(r[k] for r in rows) if rows else float("nan")

    out = {}
    for name, rows in (("prefill", pre), ("decode", dec)):
        walls = [r["wall_ms"] for r in rows]
        calls = sum(r["calls"] for r in rows)
        out[name] = {
            "steps": len(rows), "ms_mean": mean(rows, "wall_ms"), "ms_p50": pct(walls, 0.5),
            "ms_p95": pct(walls, 0.95), "tok_s": 1000.0 / mean(rows, "wall_ms") if rows else float("nan"),
            "compute_ms": mean(rows, "compute_ms"), "network_ms": mean(rows, "network_ms"),
            "client_ms": mean(rows, "client_ms"),
            "client_embed_ms": mean(rows, "embed_ms"), "client_head_ms": mean(rows, "head_ms"),
            "client_sample_ms": mean(rows, "sample_ms"), "client_codec_ms": mean(rows, "codec_ms"),
            "hops": mean(rows, "calls"), "reuse_rate": (sum(r["reused"] for r in rows) / calls) if calls else float("nan"),
            "bytes_per_step": mean(rows, "bytes"),
        }
    return out


def scenario(cluster: Cluster, *, hidden: int = 256, prompt_tokens: int = 16, new_tokens: int = 20, repeats: int = 2,
             warmup: int = 1, keepalive: bool = True) -> dict:
    net = cluster.network()
    for _ in range(warmup):
        run_generation(cluster, net, hidden=hidden, prompt_tokens=prompt_tokens, new_tokens=min(new_tokens, 4),
                       keepalive=keepalive)
    runs = [run_generation(cluster, net, hidden=hidden, prompt_tokens=prompt_tokens, new_tokens=new_tokens,
                           keepalive=keepalive) for _ in range(repeats)]
    result = summarize(runs)
    result["config"] = {"stages": cluster.stages, "replicas": cluster.replicas, "layers": cluster.layers,
                        "layer_ms": cluster.layer_ms, "rtt_ms": cluster.rtt_ms, "jitter_ms": cluster.jitter_ms,
                        "loss": cluster.loss, "hidden": hidden, "prompt_tokens": prompt_tokens,
                        "new_tokens": new_tokens, "keepalive": keepalive}
    result["expected"] = {"compute_ms": cluster.layers * cluster.layer_ms, "network_ms_floor": cluster.stages * cluster.rtt_ms}
    return result


def failover_scenario(*, stages: int = 2, layer_ms: float = 2.0, rtt_ms: float = 0.0, hidden: int = 256,
                      prompt_tokens: int = 16, new_tokens: int = 24, kill_at: int = 10) -> dict:
    """Kill the in-use replica of hop 0 at decode step `kill_at`. Compare tokens and time with an undisturbed run."""
    with Cluster(stages, 2, layers=stages, layer_ms=layer_ms, rtt_ms=rtt_ms) as c:
        net = c.network()
        base = run_generation(c, net, hidden=hidden, prompt_tokens=prompt_tokens, new_tokens=new_tokens)
        broken = run_generation(c, net, hidden=hidden, prompt_tokens=prompt_tokens, new_tokens=new_tokens,
                                kill_at=kill_at)
    steady = [s["wall_ms"] for s in broken["steps"] if s["kind"] == "decode"][:kill_at - 1]
    hit = broken["steps"][kill_at]["wall_ms"]
    base_med = statistics.median(s["wall_ms"] for s in base["steps"] if s["kind"] == "decode")
    return {"config": {"stages": stages, "replicas": 2, "layer_ms": layer_ms, "rtt_ms": rtt_ms, "kill_at_step": kill_at,
                       "new_tokens": new_tokens, "hidden": hidden},
            "tokens_match_undisturbed_run": base["tokens"] == broken["tokens"],
            "failovers": len(broken["failovers"]), "steady_ms_p50": statistics.median(steady),
            "pause_step_ms": hit, "pause_ms_over_steady": hit - statistics.median(steady),
            "note": "the pause includes the transport's one retry backoff and replaying the whole history to the replica"}


def one_line(r: dict) -> str:
    d, e = r["decode"], r["expected"]
    cfg = r["config"]
    rtt_part = f"{cfg['stages']} hop x {cfg['rtt_ms']:g} ms RTT = {e['network_ms_floor']:g} ms floor"
    return (f"decode {d['ms_mean']:.1f} ms/token ({d['tok_s']:.1f} tok/s) = compute {d['compute_ms']:.1f} "
            f"(knobs say {e['compute_ms']:g}) + network {d['network_ms']:.1f} ({rtt_part} + serialisation) "
            f"+ client {d['client_ms']:.1f}")


COLS = ("stages", "rtt", "hid", "ka", "ms/tok", "p50", "p95", "tok/s", "compute", "network", "client", "reuse%",
        "KB/tok", "prefill ms")


def table_row(r: dict) -> tuple:
    c, d, p = r["config"], r["decode"], r["prefill"]
    return (c["stages"], f"{c['rtt_ms']:g}", c["hidden"], "on" if c["keepalive"] else "off", f"{d['ms_mean']:.1f}",
            f"{d['ms_p50']:.1f}", f"{d['ms_p95']:.1f}", f"{d['tok_s']:.1f}", f"{d['compute_ms']:.1f}",
            f"{d['network_ms']:.1f}", f"{d['client_ms']:.1f}", f"{100 * d['reuse_rate']:.0f}",
            f"{d['bytes_per_step'] / 1024:.1f}", f"{p['ms_mean']:.1f}")


def format_table(results: list[dict]) -> str:
    rows = [COLS] + [tuple(str(x) for x in table_row(r)) for r in results]
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(COLS))]
    lines = ["  ".join(str(v).rjust(w) for v, w in zip(r, widths)) for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)
