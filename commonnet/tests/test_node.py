"""Node tests: stub blocks on loopback, no model, no tunnel, no network. Ports 18900-18999 only."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
from unittest import mock
import urllib.request
from pathlib import Path

from commonnet import node
from commonnet.mesh.manifest import Manifest
from commonnet.mesh.records import Identity, verify_record
from commonnet.net.server import StubBlock

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = {
    "id": "test-net", "version": 1,
    "model": {"name": "Toy", "num_layers": 36, "repos": {}, "client_repo": None},
    "boundaries": [0, 9, 18, 27, 36], "gib_per_layer": {"stub": 1.0}, "fixed_gib": 0.0, "min_replicas": 2,
}
ONE_STAGE_GIB = 9.0  # one 9-layer stage per peer


def wait_for(condition, timeout=15.0, step=0.1):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(step)
    return condition()


def get(url, path, token=None):
    request = urllib.request.Request(url + path, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(request, timeout=5) as r:
        return r.status, json.loads(r.read())


def post(url, path, body):
    request = urllib.request.Request(url + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


class NodeCase(unittest.TestCase):
    def setUp(self):
        self.nodes = []
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self._stop_all)
        self.manifest = Manifest.from_dict(MANIFEST)

    def _stop_all(self):
        for n in reversed(self.nodes):
            n.stop()

    def make(self, port, seeds=(), gib=ONE_STAGE_GIB, **kw):
        identity = Identity.load_or_create(Path(self.tmp.name) / f"id{port}.key")
        n = node.Node(self.manifest, list(seeds), identity, node.Hardware("stub", gib, "stub"), port=port, stub=True,
                      tunnel=False, log=lambda *_: None, resign_s=kw.pop("resign_s", 0.3),
                      gossip_s=kw.pop("gossip_s", 0.3), **kw)
        self.nodes.append(n)
        return n


class PublicEndpoints(NodeCase):
    def test_manifest_and_peers_are_served_without_a_token_and_forward_still_needs_one(self):
        a = self.make(18901).start()
        status, body = get(a.url, "/manifest")
        self.assertEqual((status, body["id"], body["boundaries"]), (200, "test-net", [0, 9, 18, 27, 36]))
        status, body = get(a.url, "/peers")
        mine = [r for r in body["records"] if r["peer"] == a.identity.peer]
        self.assertEqual(len(mine), 1)
        self.assertTrue(verify_record(mine[0]))
        self.assertEqual((mine[0]["state"], mine[0]["window"], mine[0]["url"]), ("up", [0, 9], a.url))
        with self.assertRaises(urllib.error.HTTPError) as caught:
            get(a.url, "/health")  # the block's own endpoints are unchanged: no token, no answer
        self.assertEqual(caught.exception.code, 401)
        status, body = get(a.url, "/health", token=a.token)
        self.assertEqual(status, 200)
        self.assertEqual(post(a.url, "/forward", {})[0], 401)

    def test_bad_gossip_bodies_are_refused(self):
        a = self.make(18902).start()
        self.assertEqual(post(a.url, "/gossip", {"records": "nope"})[0], 400)
        self.assertEqual(post(a.url, "/gossip", [1, 2])[0], 400)
        status, body = post(a.url, "/gossip", {"records": [{"junk": 1}]})  # unverifiable records are ignored
        self.assertEqual(status, 200)
        self.assertEqual([r["peer"] for r in body["records"]], [a.identity.peer])

    def test_the_token_is_in_the_signed_record_and_works(self):
        a = self.make(18903).start()
        record = a.table.get(a.identity.peer)
        self.assertEqual(record["token"], a.token)
        self.assertEqual(get(record["url"], "/health", token=record["token"])[0], 200)


class Gossip(NodeCase):
    def test_two_nodes_see_each_other_and_a_third_takes_the_uncovered_window(self):
        a = self.make(18911).start()
        b = self.make(18912, seeds=[a.url]).start()
        self.assertEqual(a.window, (0, 9))
        self.assertEqual(b.window, (9, 18))  # a covers the first stage, so b takes the next
        self.assertTrue(wait_for(lambda: {r["peer"] for r in a.table.live()} == {a.identity.peer, b.identity.peer}))
        self.assertTrue(wait_for(lambda: {r["peer"] for r in b.table.live()} == {a.identity.peer, b.identity.peer}))
        c = self.make(18913, seeds=[b.url]).start()  # knows only b, learns a through b
        self.assertEqual(c.window, (18, 27))
        self.assertTrue(wait_for(lambda: len(a.table.live()) == 3 and len(c.table.live()) == 3))

    def test_a_node_that_leaves_publishes_leaving_to_its_peers(self):
        a = self.make(18921).start()
        b = self.make(18922, seeds=[a.url]).start()
        self.assertTrue(wait_for(lambda: len(a.table.live()) == 2))
        b.stop()
        self.assertTrue(wait_for(lambda: (a.table.get(b.identity.peer) or {}).get("state") == "leaving", timeout=5))
        self.assertEqual([r["peer"] for r in a.table.live()], [a.identity.peer])

    def test_a_third_node_with_more_memory_serves_more_stages(self):
        a = self.make(18931).start()
        big = self.make(18932, seeds=[a.url], gib=3 * ONE_STAGE_GIB).start()
        self.assertEqual(big.window, (9, 36))  # the three stages a does not cover, in one window


class Rebalance(NodeCase):
    def test_a_spare_peer_moves_to_a_stage_that_lost_every_peer(self):
        m = Manifest.from_dict({**MANIFEST, "boundaries": [0, 18, 36]})
        self.manifest = m
        a = self.make(18941, gib=18.0, rebalance_s=0.4, cooldown_s=0).start()
        b = self.make(18942, seeds=[a.url], gib=18.0, rebalance_s=0.4, cooldown_s=0).start()
        c = self.make(18943, seeds=[a.url], gib=18.0, rebalance_s=0.4, cooldown_s=0).start()
        self.assertEqual((a.window, b.window), ((0, 18), (18, 36)))
        self.assertIn(c.window, {(0, 18), (18, 36)})
        spare_window = c.window
        self.assertTrue(wait_for(lambda: len(a.table.live()) == 3))
        # whoever holds the window that c does not hold loses a peer; c then has a twin and is spare
        loser = b if spare_window == (0, 18) else a
        survivor_twin = a if spare_window == (0, 18) else b
        loser.stop()
        # the stage the loser held is now empty; its two surviving peers both hold the other stage.
        # Exactly one of them (the lowest id) moves; the other must stay put.
        twins = [c, survivor_twin]
        self.assertTrue(wait_for(lambda: any(p.window == loser.window and p.state == "up" for p in twins), timeout=20))
        time.sleep(2.5)  # a second mover would show up by now
        movers = [p for p in twins if p.window == loser.window]
        stayers = [p for p in twins if p.window == spare_window]
        self.assertEqual((len(movers), len(stayers)), (1, 1))
        self.assertEqual(movers[0].identity.peer, min(p.identity.peer for p in twins))
        self.assertEqual(get(movers[0].url, "/health", token=movers[0].token)[0], 200)
        self.assertEqual(movers[0].server.port, {c: 18943, survivor_twin: survivor_twin.port}[movers[0]])


class BlockStart(NodeCase):
    """What the node hands the MLX block: bfloat16 compute, with the cache limit left to the environment."""

    def _start_real(self, backend, environ):
        seen = {}
        manifest = Manifest.from_dict({**MANIFEST, "model": {**MANIFEST["model"], "repos": {backend: "org/repo"}}})
        identity = Identity.load_or_create(Path(self.tmp.name) / "idb.key")
        n = node.Node(manifest, [], identity, node.Hardware(backend, 30.0, "x"), port=18971, stub=False, tunnel=False,
                      log=lambda *_: None, fetcher=lambda repo, window: "/ckpt", resign_s=5, gossip_s=5)

        def fake_build(start, end, backend=None, checkpoint=None):
            seen.update(dtype=os.environ.get("COMMONNET_DTYPE"), cache=os.environ.get("COMMONNET_MLX_CACHE_MB"),
                        backend=backend, checkpoint=checkpoint, layers=(start, end))
            return StubBlock(start, end)

        with mock.patch.dict(os.environ, environ, clear=False), mock.patch("commonnet.net.factory.build_block", fake_build):
            os.environ.pop("COMMONNET_DTYPE", None) if "COMMONNET_DTYPE" not in environ else None
            n._make_block((0, 9))
        return seen

    def test_mlx_blocks_compute_in_bfloat16_and_the_cache_limit_passes_through(self):
        seen = self._start_real("mlx", {"COMMONNET_MLX_CACHE_MB": "256"})
        self.assertEqual((seen["dtype"], seen["cache"], seen["backend"], seen["layers"]), ("bfloat16", "256", "mlx", (0, 9)))

    def test_an_explicit_dtype_is_respected(self):
        self.assertEqual(self._start_real("mlx", {"COMMONNET_DTYPE": "float16"})["dtype"], "float16")

    def test_nf4_is_not_given_an_mlx_dtype(self):
        self.assertIsNone(self._start_real("nf4", {})["dtype"])


class Hardware(unittest.TestCase):
    def test_apple_silicon_offers_a_share_of_memory_on_mlx(self):
        hw = node.detect_hardware("Darwin", "arm64", memsize=lambda: 16 * 2**30)
        self.assertEqual((hw.backend, hw.gib), ("mlx", round(16 * node.MAC_MEMORY_FRACTION, 2)))

    def test_nvidia_offers_free_vram_minus_a_reserve_on_nf4(self):
        hw = node.detect_hardware("Windows", "AMD64", nvidia=lambda: (7168.0, "RTX 4060"))
        self.assertEqual((hw.backend, hw.gib, hw.name), ("nf4", 7.0 - node.VRAM_RESERVE_GIB, "RTX 4060"))

    def test_machines_without_a_gpu_offer_a_conservative_share_of_ram_on_cpu(self):
        gib = 2**30
        for system, machine in (("Linux", "x86_64"), ("Windows", "AMD64"), ("Darwin", "x86_64")):
            with self.subTest(system=system, machine=machine):
                hw = node.detect_hardware(system, machine, nvidia=lambda: None, meminfo=lambda s: (31 * gib, 20 * gib))
                self.assertEqual((hw.backend, hw.gib), ("cpu", 12.0))  # min(20*0.6, 31*0.5)
        low = node.detect_hardware("Windows", "AMD64", nvidia=lambda: None, meminfo=lambda s: (31 * gib, 4 * gib))
        self.assertEqual(low.gib, 2.4)  # busy machine: the free-RAM share wins
        small = node.detect_hardware("Linux", "x86_64", nvidia=lambda: None, meminfo=lambda s: (4 * gib, 4 * gib))
        self.assertEqual(small.gib, 2.0)  # too little for a stage; choose_window then says so

    def test_unreadable_memory_is_refused_with_a_reason(self):
        with self.assertRaises(node.NodeError) as caught:
            node.detect_hardware("Linux", "x86_64", nvidia=lambda: None, meminfo=lambda s: None)
        self.assertIn("memory", str(caught.exception))


class Network(NodeCase):
    def test_resolve_network_from_a_file_a_seed_and_nothing(self):
        path = Path(self.tmp.name) / "m.json"
        path.write_text(json.dumps({**MANIFEST, "seeds": ["http://127.0.0.1:1"]}))
        manifest, seeds = node.resolve_network(str(path), ["http://127.0.0.1:2"], environ={})
        self.assertEqual((manifest.id, seeds), ("test-net", ["http://127.0.0.1:2", "http://127.0.0.1:1"]))
        a = self.make(18951).start()
        manifest, seeds = node.resolve_network(a.url, [], environ={})
        self.assertEqual((manifest.id, seeds[0]), ("test-net", a.url))
        manifest, seeds = node.resolve_network(None, [], environ={node.NETWORK_ENV: str(path)})
        self.assertEqual(manifest.id, "test-net")
        with self.assertRaises(node.NodeError):
            node.resolve_network("http://127.0.0.1:9", [], environ={})  # nothing there


class CommandLine(unittest.TestCase):
    def test_cli_comes_up_and_leaves_cleanly_on_sigterm(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "m.json"
            manifest.write_text(json.dumps(MANIFEST))
            env = dict(os.environ, PYTHONPATH=str(ROOT), COMMON_NETWORK_HOME=tmp)
            port = 18961
            proc = subprocess.Popen([sys.executable, "-m", "commonnet.node", "--network", str(manifest), "--stub",
                                     "--no-tunnel", "--port", str(port), "--identity-dir", tmp],
                                    cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                url = f"http://127.0.0.1:{port}"

                def up():
                    try:
                        return any(r["state"] == "up" for r in get(url, "/peers")[1]["records"])
                    except (OSError, ValueError):
                        return False

                self.assertTrue(wait_for(up, timeout=30), "node never came up")
                self.assertTrue(os.path.exists(os.path.join(tmp, "identity.key")))
                self.assertEqual(oct(os.stat(os.path.join(tmp, "identity.key")).st_mode & 0o777), "0o600")
                proc.send_signal(signal.SIGTERM)
                self.assertEqual(proc.wait(timeout=20), 0)
                output = proc.stdout.read()
                self.assertNotIn("token", output.lower().replace("block token", ""))
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()


if __name__ == "__main__":
    unittest.main()
