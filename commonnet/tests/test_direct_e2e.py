"""Direct mode end to end, on loopback, with real processes: a DHT bootstrap node and several synthetic peers
(`--direct-loopback`, TLS on 127.0.0.1, no tunnel, no seed URL). A client finds them through the DHT only, builds a
route and generates through certificate-pinned hops; a replica is killed and the run fails over; forged records are
refused; a plain-HTTP peer (what a tunnel peer looks like to the mesh) shares the mesh with the direct ones.

No router, no Internet, no model. Takes about 30 s.
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from commonnet.bench.synth import SynthClient, manifest_dict
from commonnet.mesh import records as rec
from commonnet.mesh.records import Identity
from commonnet.net import secure
from commonnet.netclient import Network

ROOT = Path(__file__).resolve().parents[2]
LAYERS, STAGES = 4, 2


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Mesh:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="commonnet-direct-")
        self.dir = Path(self.tmp.name)
        self.manifest = manifest_dict(STAGES, LAYERS, network_id="test-direct-e2e")
        self.manifest_path = self.dir / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.env = dict(os.environ, PYTHONPATH=str(ROOT), COMMON_NETWORK_HOME=str(self.dir / "home"),
                        PYTHONUNBUFFERED="1")
        self.procs: dict[str, subprocess.Popen] = {}
        self.ports: dict[str, int] = {}
        self.boot = None

    def start_bootstrap(self) -> str:
        self.boot = subprocess.Popen([sys.executable, "-m", "commonnet.dht", "--bind", "127.0.0.1:0"],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=self.env, cwd=ROOT)
        line = self.boot.stdout.readline()
        m = re.search(r"udp 127\.0\.0\.1:(\d+)", line)
        assert m, f"bootstrap did not start: {line!r} {self.boot.stderr.read() if self.boot.poll() is not None else ''}"
        self.bootstrap = f"127.0.0.1:{m.group(1)}"
        return self.bootstrap

    def start_node(self, name: str, window: str, *, direct: bool = True, seeds=()) -> int:
        port = free_port()
        cmd = [sys.executable, "-m", "commonnet.bench.nodeproc", "--manifest", str(self.manifest_path),
               "--port", str(port), "--identity-dir", str(self.dir / name), "--stage-layers", str(LAYERS // STAGES),
               "--window", window]
        if direct:
            cmd += ["--direct-loopback", "--bootstrap", self.bootstrap]
        else:
            cmd += ["--advertise", f"http://127.0.0.1:{port}"]
        for s in seeds:
            cmd += ["--seed", s]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=self.env, cwd=ROOT)
        line = proc.stdout.readline()
        if not line.startswith("READY"):
            err = proc.stderr.read() if proc.poll() is not None else ""
            proc.kill()
            raise RuntimeError(f"node {name} did not start: {line!r} {err[-800:]}")
        self.procs[name], self.ports[name] = proc, port
        return port

    def kill(self, name: str) -> None:
        self.procs[name].kill()
        self.procs[name].wait(timeout=5)

    def stop(self) -> None:
        for p in list(self.procs.values()) + ([self.boot] if self.boot else []):
            if p.poll() is None:
                p.terminate()
        for p in list(self.procs.values()) + ([self.boot] if self.boot else []):
            try:
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                p.kill()
            for f in (p.stdout, p.stderr):
                if f:
                    f.close()
        self.tmp.cleanup()


def generate(net: Network, mesh: Mesh, n: int = 8, kill=None):
    """Greedy-decode n tokens through the network's route with the synthetic client. Returns (tokens, transport)."""
    endpoints, transport = net.plan()
    client = SynthClient(transport, layers=LAYERS, hidden=64)
    prompt = [(7 * i + 3) % 2048 for i in range(6)]

    def before(step):
        if kill and step == kill[0]:
            kill[1](transport)

    try:
        tokens = list(client.generate_stream(endpoints, prompt, n, before_step=before))
    finally:
        transport.shutdown()
    return tokens, transport


class DirectE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh = m = Mesh()
        try:
            m.start_bootstrap()
            # a plain-HTTP peer, as a tunnel peer looks to the mesh (gossip only, no DHT)
            plain_port = m.start_node("plain", "0-2", direct=False)
            cls.plain_url = f"http://127.0.0.1:{plain_port}"
            # direct peers: found by the DHT; the plain peer is only a --seed (optional, for interop)
            m.start_node("d1", "0-2", seeds=[cls.plain_url])
            m.start_node("d2", "2-4", seeds=[cls.plain_url])
            m.start_node("d3", "2-4", seeds=[cls.plain_url])
        except BaseException:
            m.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.mesh.stop()

    def client(self, **kw) -> Network:
        kw.setdefault("refresh_interval", 0.5)
        kw.setdefault("wait_s", 15.0)
        net = Network.open(str(self.mesh.manifest_path), [self.mesh.bootstrap], **kw)
        self.addCleanup(net.close)
        deadline = time.time() + 30
        while True:   # announcements are asynchronous: wait until every peer is visible
            net.refresh(force=True)
            if len([r for r in net.table.live() if r.get("window")]) >= 4:
                return net
            self.assertLess(time.time(), deadline, f"client sees only {len(net.table.live())} peers")
            time.sleep(0.4)

    def test_1_found_through_the_dht_alone_and_generates_through_pinned_tls(self):
        net = self.client()
        self.assertEqual(net.seeds, [])                      # no seed URL anywhere
        live = net.table.live()
        direct = [r for r in live if rec.route_kind(r["url"]) == "direct"]
        self.assertEqual(len(direct), 3)
        for r in direct:                                     # every direct record carries a verified binding
            self.assertTrue(secure.verify_binding(r["pub"], r["fp"], r["fpsig"]))
            self.assertIn(urlsplit_netloc(r["url"]), net._pins)
        tokens, transport = generate(net, self.mesh)
        self.assertEqual(len(tokens), 8)
        # the route really is https with a pin for every direct hop
        kinds = {rec.route_kind(u) for u in (transport.urls.values())}
        self.assertTrue(kinds <= {"direct", "plain"})
        # same tokens as an in-process chain of the same synthetic weights
        self.assertEqual(tokens, reference_tokens(8))

    def test_2_default_manifest_and_dht_only(self):
        with mock.patch("commonnet.node.DEFAULT_MANIFEST", self.mesh.manifest_path):
            net = Network.open("dht", [self.mesh.bootstrap], refresh_interval=0.5)
        self.addCleanup(net.close)
        deadline = time.time() + 30
        while len(net.table.live()) < 3 and time.time() < deadline:
            net.refresh(force=True)
            time.sleep(0.4)
        self.assertGreaterEqual(len(net.table.live()), 3)
        self.assertEqual(net.seeds, [])

    def test_3_direct_and_plain_peers_share_one_mesh(self):
        net = self.client()
        by_kind = {}
        for r in net.table.live():
            by_kind.setdefault(rec.route_kind(r["url"]), []).append(r)
        self.assertEqual(len(by_kind.get("plain", [])), 1)   # learned from the direct peers' gossip, not from the DHT
        self.assertEqual(len(by_kind.get("direct", [])), 3)
        # the plain peer's own table lists the direct peers (they pushed their records to it)
        import urllib.request
        with urllib.request.urlopen(self.plain_url + "/peers", timeout=5) as resp:
            seen = json.loads(resp.read())["records"]
        self.assertGreaterEqual(len([r for r in seen if r.get("fp")]), 1)
        self.assertTrue(all(rec.verify_record(r) for r in seen))
        # a route that uses stage 0 on whichever of {plain, d1} the client prefers still generates
        tokens, _ = generate(net, self.mesh, 4)
        self.assertEqual(tokens, reference_tokens(4))

    def test_4_forged_records_are_refused(self):
        net = self.client()
        victim = next(r for r in net.table.live() if r["window"] == [0, 2] and rec.route_kind(r["url"]) == "direct")
        evil = Identity.generate()
        base = dict(network=self.mesh.manifest["id"], url=victim["url"], token="x", window=[0, 2], backend="synth",
                    state="up", gib=2.0)
        # 1. the victim's real fingerprint, bound by the wrong key
        forged = evil.sign_record(dict(base, fp=victim["fp"], fpsig=secure.bind_fingerprint(evil, "ab" * 32)))
        self.assertFalse(rec.verify_record(forged))
        self.assertFalse(net.table.merge(forged))
        # 2. a direct URL with no fingerprint at all
        self.assertFalse(rec.verify_record(evil.sign_record(base)))
        # 3. a wrong (malformed) fingerprint
        with self.assertRaises(ValueError):
            evil.sign_record(dict(base, fp="not-a-fingerprint"))
        # 4. a well formed record of the attacker pointing at the victim's address: genuine signature, but the
        #    victim's certificate is not the attacker's, so the pinned connection is refused
        own_fp = "cd" * 32
        impostor = evil.sign_record(dict(base, fp=own_fp))
        self.assertTrue(rec.verify_record(impostor))
        from commonnet.net.client import BlockClient, Unreachable
        client = BlockClient(token=victim["token"], retries=0, pin={urlsplit_netloc(victim["url"]): own_fp})
        with self.assertRaises(Unreachable):
            client.health(victim["url"])
        # 5. the DHT will not take a forged record either
        from commonnet.dht import DhtNode
        node = DhtNode(evil, bind=("127.0.0.1", 0), network_id=self.mesh.manifest["id"])
        node.start()
        self.addCleanup(node.stop)
        with self.assertRaises(ValueError):
            node.announce(forged)

    def test_5_killing_a_replica_fails_over(self):
        net = self.client()
        baseline, _ = generate(net, self.mesh, 10)

        def kill_primary(transport):
            url = transport.urls["hop1"]
            name = next(n for n, p in self.mesh.ports.items() if url.endswith(f":{p}"))
            self.assertIn(name, ("d2", "d3"))
            self.mesh.kill(name)

        net2 = self.client()
        tokens, transport = generate(net2, self.mesh, 10, kill=(4, kill_primary))
        self.assertEqual(tokens, baseline)
        self.assertTrue(transport.failovers, "expected a failover to the surviving replica")


def urlsplit_netloc(url: str) -> str:
    from urllib.parse import urlsplit
    return urlsplit(url).netloc


def reference_tokens(n: int):
    """The same generation in one process: two SynthBlocks wired directly."""
    from commonnet.bench.synth import SynthBlock
    from commonnet.net.server import BlockServer
    servers = [BlockServer(SynthBlock(a, b), port=0).start() for a, b in ((0, 2), (2, 4))]
    try:
        from commonnet.client import BlockEndpoint
        from commonnet.net.transport import HttpTransport as T
        transport = T({"hop0": servers[0].url, "hop1": servers[1].url})
        endpoints = [BlockEndpoint("hop0", 0, 2), BlockEndpoint("hop1", 2, 4)]
        client = SynthClient(transport, layers=LAYERS, hidden=64)
        prompt = [(7 * i + 3) % 2048 for i in range(6)]
        return list(client.generate_stream(endpoints, prompt, n))
    finally:
        for s in servers:
            s.stop()


if __name__ == "__main__":
    unittest.main()
