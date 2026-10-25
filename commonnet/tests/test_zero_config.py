"""Zero-config join: the gateway directory as rendezvous. Gateway down or old => behaviour unchanged; address
decision with fake gateway answers; and an end-to-end run on loopback against the REAL gateway app."""
from __future__ import annotations

import http.server
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from commonnet import direct
from commonnet.bench.synth import SynthClient, manifest_dict
from commonnet.mesh import directory as md
from commonnet.mesh import records as rec
from commonnet.mesh.manifest import Manifest
from commonnet.mesh.records import Identity
from commonnet.net import nat
from commonnet.netclient import Network

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gateway"))
os.environ["COMMON_GATEWAY_TEST"] = "1"
PUBLIC = "93.184.216.34"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class OldGateway:
    """An HTTP server that answers 404 (or 410) to everything: a gateway without the directory."""

    def __init__(self, code=404):
        class H(http.server.BaseHTTPRequestHandler):
            def _r(self):
                self.send_response(code)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")
            do_GET = do_POST = _r

            def log_message(self, *a):
                pass
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


class GatewayClient(unittest.TestCase):
    def test_gateway_url_rules(self):
        self.assertEqual(md.gateway_url({}), md.DEFAULT_GATEWAY)
        self.assertEqual(md.gateway_url({"COMMONNET_GATEWAY": "http://x:1/"}), "http://x:1")
        for off in ("off", "", "none", "ftp://x"):
            self.assertIsNone(md.gateway_url({"COMMONNET_GATEWAY": off}))
        self.assertIsNone(md.gateway_url({}, default=None))

    def test_down_or_old_gateway_fails_soft(self):
        dead = f"http://127.0.0.1:{free_port()}"
        r = Identity.generate().sign_record({"network": "n", "url": "http://127.0.0.1:1", "window": [0, 1]})
        for code in (404, 410):
            old = OldGateway(code)
            self.addCleanup(old.close)
            self.assertEqual(md.fetch(old.url, "n"), [])
            self.assertFalse(md.announce(old.url, r))
            self.assertIsNone(md.check(old.url, 9999))
        self.assertEqual(md.fetch(dead, "n", timeout=1), [])
        self.assertFalse(md.announce(dead, r, timeout=1))
        self.assertIsNone(md.check(dead, 9999, timeout=1))
        self.assertEqual(md.fetch(None, "n"), [])

    def test_forged_directory_records_are_dropped(self):
        good = Identity.generate().sign_record({"network": "n", "url": "http://127.0.0.1:1", "window": [0, 1]})
        forged = dict(good, url="http://evil.example")
        other = Identity.generate().sign_record({"network": "other", "url": "http://127.0.0.1:2", "window": [0, 1]})
        with mock.patch.object(md, "_call", return_value=[good, forged, other, "junk"]):
            self.assertEqual(md.fetch("http://gw", "n"), [good])

    def test_dht_contacts_only_from_direct_records_with_a_port(self):
        ident = Identity.generate()
        from commonnet.net import secure
        fp = "ab" * 32
        direct_rec = ident.sign_record({"network": "n", "url": "https://1.2.3.4:7471", "window": [0, 1], "fp": fp,
                                        "fpsig": secure.bind_fingerprint(ident, fp), "dht": 7472})
        tunnel = Identity.generate().sign_record({"network": "n", "url": "https://x.trycloudflare.com", "window": [0, 1], "dht": 7472})
        self.assertEqual(md.dht_contacts([direct_rec, tunnel]), [("1.2.3.4", 7472)])
        self.assertTrue(rec.verify_record(direct_rec))


class Decision(unittest.TestCase):
    """The address decision of an auto-direct node, with a fake router, STUN and gateway."""

    def node(self, tmp, check_answer):
        from commonnet.bench.synth import SynthBlock
        from commonnet.node import Hardware, Node

        def decider(cfg, port, dial, log=None):
            def no_map(_p):
                raise nat.NatError("no router mapping")
            return direct.decide(cfg, port, dial, mapper=no_map, stun=lambda: {"public_ip": PUBLIC}, log=lambda m: None)

        logs = []
        n = Node(Manifest.load(manifest_dict(1, 2, network_id="t-zero")), [], Identity.generate(),
                 Hardware("synth", 2.0, "synthetic"), tunnel=True, direct=direct.DirectConfig(auto=True), state_dir=tmp,
                 decider=decider, log=logs.append, gateway="http://gw.invalid")
        n.logs = logs
        return n, SynthBlock(0, 2)

    def run_case(self, answer):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(direct, "DEFAULT_TCP_PORT", free_port()), \
                mock.patch.object(md, "check", return_value=answer):
            n, block = self.node(tmp, answer)
            n.expected_port = direct.DEFAULT_TCP_PORT
            served = n._serve_direct(block)
            if n.server is not None:
                n.server.stop()
            return n, served

    def test_gateway_says_reachable_publishes_the_direct_address(self):
        n, served = self.run_case({"reachable": True, "ip": PUBLIC})
        self.assertTrue(served)
        self.assertEqual(n.url, f"https://{PUBLIC}:{n.expected_port}")
        self.assertTrue(n.fp and n.fpsig)
        self.assertTrue(any("mode: direct (verified by the gateway" in m for m in n.logs))

    def test_gateway_says_unreachable_falls_back_to_the_tunnel(self):
        n, served = self.run_case({"reachable": False, "ip": PUBLIC})
        self.assertFalse(served)
        self.assertIsNone(n.url)
        self.assertTrue(any("automatic direct connections" in m for m in n.logs))
        self.assertFalse(any("forward TCP" in m for m in n.logs))

    def test_old_gateway_means_no_answer_and_the_tunnel(self):
        n, served = self.run_case(None)
        self.assertFalse(served)

    def test_gateway_seeing_another_address_is_not_trusted(self):
        n, served = self.run_case({"reachable": True, "ip": "8.8.4.4"})
        self.assertFalse(served)
        self.assertFalse(any("not trusting its check" in m for m in n.logs))

    def test_never_publishes_a_private_address(self):
        d = direct.decide(direct.DirectConfig(auto=True), 7471, lambda ip, p: True,
                          mapper=lambda p: (_ for _ in ()).throw(nat.NatError("x")),
                          stun=lambda: {"public_ip": "100.64.1.2"})
        self.assertFalse(d.direct)


class Cli(unittest.TestCase):
    def parse(self, *argv):
        from commonnet import node
        return node.direct_config(node.build_parser().parse_args(list(argv)))

    def test_auto_direct_flag(self):
        cfg = self.parse("--auto-direct")
        self.assertTrue(cfg.auto and cfg.tunnel_fallback)
        self.assertIsNone(self.parse())
        self.assertIsNone(self.parse("--auto-direct", "--no-tunnel"))
        self.assertFalse(self.parse("--direct", "--auto-direct").auto)

    def test_join_passes_auto_by_default_and_not_with_no_direct(self):
        sys.path.insert(0, str(ROOT / "join"))
        import types
        import join
        ns = types.SimpleNamespace
        self.assertEqual(join._direct_flags(ns(direct=False, no_direct=False)), ["--auto-direct"])
        self.assertEqual(join._direct_flags(ns(direct=False, no_direct=True)), ["--no-rtc"])
        self.assertEqual(join._direct_flags(ns(direct=True, no_direct=False))[0], "--direct")


class NodeWithDeadGateway(unittest.TestCase):
    def test_node_starts_and_serves_when_the_gateway_is_down_or_old(self):
        from commonnet.bench.synth import SynthBlock
        from commonnet.node import Hardware, Node
        old = OldGateway(404)
        self.addCleanup(old.close)
        for gw in (f"http://127.0.0.1:{free_port()}", old.url):
            with tempfile.TemporaryDirectory() as tmp:
                n = Node(Manifest.load(manifest_dict(1, 2, network_id="t-down")), [], Identity.generate(),
                         Hardware("synth", 2.0, "synthetic"), tunnel=False, advertise="http://127.0.0.1:1", state_dir=tmp,
                         block_builder=lambda w, m, h: SynthBlock(w[0], w[1]), log=lambda m: None, gateway=gw,
                         fixed_window=(0, 2), resign_s=0.2)
                try:
                    n.start()
                    time.sleep(0.6)   # a couple of maintenance rounds hit the gateway
                    self.assertEqual(n.state, "up")
                finally:
                    n.stop()


def _gateway_deps():
    try:
        import fastapi, uvicorn  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_gateway_deps(), "needs fastapi and uvicorn (gateway/requirements.txt)")
class EndToEnd(unittest.TestCase):
    """The real gateway app on a socket; two plain synthetic peers with no seed, no bootstrap, only the gateway."""

    @classmethod
    def setUpClass(cls):
        import uvicorn
        from app import directory, main
        directory._records.clear()
        directory._hits.clear()
        directory.ANNOUNCE_PER_MIN = 10_000       # every process here shares 127.0.0.1
        cls.port = free_port()
        cls.server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=cls.port, log_level="error"))
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        while not cls.server.started:
            time.sleep(0.05)
        cls.gw = f"http://127.0.0.1:{cls.port}"
        cls.tmp = tempfile.TemporaryDirectory(prefix="commonnet-zc-")
        cls.dir = Path(cls.tmp.name)
        cls.manifest = manifest_dict(2, 4, network_id="test-zero-config")
        cls.manifest_path = cls.dir / "manifest.json"
        cls.manifest_path.write_text(json.dumps(cls.manifest))
        cls.env = dict(os.environ, PYTHONPATH=str(ROOT), COMMON_NETWORK_HOME=str(cls.dir / "home"),
                       PYTHONUNBUFFERED="1", COMMONNET_GATEWAY=cls.gw)
        cls.procs = []
        try:
            for name, window in (("a", "0-2"), ("b", "2-4")):
                cls.start_node(name, window)
        except BaseException:
            cls.tearDownClass()
            raise

    @classmethod
    def start_node(cls, name, window):
        cmd = [sys.executable, "-m", "commonnet.bench.nodeproc", "--manifest", str(cls.manifest_path), "--port",
               str(free_port()), "--identity-dir", str(cls.dir / name), "--stage-layers", "2", "--window", window,
               "--direct-loopback"]   # no --seed, no --bootstrap
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=cls.env, cwd=ROOT)
        line = p.stdout.readline()
        if not line.startswith("READY"):
            err = p.stderr.read() if p.poll() is not None else ""
            p.kill()
            raise RuntimeError(f"node {name}: {line!r} {err[-600:]}")
        cls.procs.append(p)

    @classmethod
    def tearDownClass(cls):
        for p in cls.procs:
            p.terminate()
        for p in cls.procs:
            try:
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                p.kill()
            for f in (p.stdout, p.stderr):
                f and f.close()
        if getattr(cls, "server", None):
            cls.server.should_exit = True
            cls.thread.join(5)
        cls.tmp.cleanup()

    def test_peers_find_each_other_and_a_client_with_no_url_generates(self):
        # the directory holds both peers, and only verified records
        deadline = time.time() + 30
        listed = []
        while time.time() < deadline:
            listed = md.fetch(self.gw, self.manifest["id"])
            if len(listed) >= 2:
                break
            time.sleep(0.3)
        self.assertEqual(len(listed), 2)
        self.assertTrue(all(rec.route_kind(r["url"]) == "direct" and r.get("dht") for r in listed))
        # a client with no URL and no bootstrap: only the gateway (COMMONNET_GATEWAY)
        with mock.patch("commonnet.node.DEFAULT_MANIFEST", self.manifest_path), mock.patch.dict(os.environ, {md.GATEWAY_ENV: self.gw}):
            net = Network.open(None, [], refresh_interval=0.3, wait_s=10.0)
        self.addCleanup(net.close)
        self.assertEqual(net.seeds, [])
        self.assertEqual(net.gateway, self.gw)
        deadline = time.time() + 30
        while len([r for r in net.table.live() if r.get("window")]) < 2 and time.time() < deadline:
            net.refresh(force=True)
            time.sleep(0.3)
        endpoints, transport = net.plan()
        client = SynthClient(transport, layers=4, hidden=64)
        try:
            tokens = list(client.generate_stream(endpoints, [3, 10, 17, 24, 31, 38], 6))
        finally:
            transport.shutdown()
        self.assertEqual(len(tokens), 6)

    def test_the_two_peers_know_each_other_without_a_seed(self):
        # each peer's own gossip table (GET /peers) lists the other
        listed = md.fetch(self.gw, self.manifest["id"])
        import ssl
        import urllib.request
        for r in listed:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            deadline, seen = time.time() + 30, []
            while time.time() < deadline:
                with urllib.request.urlopen(r["url"] + "/peers", timeout=5, context=ctx) as resp:
                    seen = json.loads(resp.read())["records"]
                if len({x["peer"] for x in seen}) >= 2:
                    break
                time.sleep(0.5)
            self.assertGreaterEqual(len({x["peer"] for x in seen}), 2)


if __name__ == "__main__":
    unittest.main()
