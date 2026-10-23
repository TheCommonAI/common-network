"""The node's status file (read by the desktop app) and its local chat API (--api). Stub blocks, no model, no tunnel.

Ports 18980-18999 only.
"""
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import torch

from commonnet import node
from commonnet.mesh.manifest import Manifest
from commonnet.mesh.records import Identity
from commonnet.tests.test_node import MANIFEST, ONE_STAGE_GIB, wait_for
from commonnet.tests.test_serve import FakeStream, LetterTokenizer

ROOT = Path(__file__).resolve().parents[2]
FIELDS = {"v", "state", "network", "model", "peer", "window", "backend", "gib", "url", "download", "stages",
          "coverage", "peers", "error", "api", "pid", "ts"}
PEER_FIELDS = {"peer", "window", "backend", "state", "age_s", "me"}


class RecordingStatus(node.StatusFile):
    """A StatusFile that also remembers every document it wrote."""

    def __init__(self, path):
        super().__init__(path)
        self.history = []

    def write(self, status):
        self.history.append(status)
        super().write(status)


def read(path):
    return json.loads(Path(path).read_text())


class StatusCase(unittest.TestCase):
    def setUp(self):
        self.nodes = []
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(lambda: [n.stop() for n in reversed(self.nodes)])
        self.manifest = Manifest.from_dict(MANIFEST)

    def make(self, port, name, seeds=(), gib=ONE_STAGE_GIB, **kw):
        identity = Identity.load_or_create(Path(self.tmp.name) / f"id{port}.key")
        path = Path(self.tmp.name) / "net" / f"{name}.json"
        status = RecordingStatus(path)
        n = node.Node(self.manifest, list(seeds), identity, node.Hardware("stub", gib, "stub"), port=port, stub=True,
                      tunnel=False, log=lambda *_: None, resign_s=0.3, gossip_s=0.3, status=status, status_s=0.1, **kw)
        self.nodes.append(n)
        return n, path, status


class Lifecycle(StatusCase):
    def test_the_file_goes_starting_loading_up_then_stopped_with_every_contract_field(self):
        n, path, status = self.make(18981, "a")
        n.start()
        doc = read(path)
        self.assertEqual(set(doc), FIELDS)
        self.assertEqual((doc["v"], doc["state"], doc["network"], doc["model"], doc["backend"]),
                         (1, "up", "test-net", "Toy", "stub"))
        self.assertEqual((doc["window"], doc["peer"], doc["url"], doc["pid"]), ([0, 9], n.identity.peer, n.url, os.getpid()))
        self.assertEqual(doc["stages"], [[0, 9], [9, 18], [18, 27], [27, 36]])
        self.assertEqual(doc["coverage"], [1, 0, 0, 0])
        self.assertIsNone(doc["download"])
        self.assertIsNone(doc["error"])
        self.assertIsNone(doc["api"])
        self.assertAlmostEqual(doc["ts"], time.time(), delta=5)
        self.assertEqual([p["peer"] for p in doc["peers"]], [n.identity.peer])
        self.assertEqual(set(doc["peers"][0]), PEER_FIELDS)
        self.assertTrue(doc["peers"][0]["me"])
        n.stop()
        self.assertEqual(read(path)["state"], "stopped")
        states = [d["state"] for d in status.history]
        order = [s for i, s in enumerate(states) if i == 0 or s != states[i - 1]]
        self.assertEqual(order, ["starting", "loading", "up", "leaving", "stopped"])

    def test_it_is_rewritten_every_interval_not_only_on_changes(self):
        n, path, _ = self.make(18982, "a")
        n.start()
        first = read(path)["ts"]
        self.assertTrue(wait_for(lambda: read(path)["ts"] > first + 0.2, timeout=5))

    def test_two_nodes_each_see_the_other_and_the_coverage_adds_up(self):
        a, path_a, _ = self.make(18983, "a")
        a.start()
        b, path_b, _ = self.make(18984, "b", seeds=[a.url])
        b.start()
        self.assertEqual(b.window, (9, 18))

        def settled():
            da, db = read(path_a), read(path_b)
            return da["coverage"] == [1, 1, 0, 0] and db["coverage"] == [1, 1, 0, 0] and len(da["peers"]) == 2

        self.assertTrue(wait_for(settled, timeout=10))
        doc = read(path_b)
        mine = [p for p in doc["peers"] if p["me"]]
        theirs = [p for p in doc["peers"] if not p["me"]]
        self.assertEqual([p["peer"] for p in mine], [b.identity.peer])
        self.assertEqual((theirs[0]["peer"], theirs[0]["window"], theirs[0]["state"]), (a.identity.peer, [0, 9], "up"))
        self.assertEqual(doc["peers"][0]["me"], True)  # me first

    def test_a_peer_that_left_disappears_from_the_list(self):
        a, path_a, _ = self.make(18985, "a")
        a.start()
        b, _, _ = self.make(18986, "b", seeds=[a.url])
        b.start()
        self.assertTrue(wait_for(lambda: len(read(path_a)["peers"]) == 2, timeout=10))
        b.stop()
        self.assertTrue(wait_for(lambda: len(read(path_a)["peers"]) == 1 and read(path_a)["coverage"] == [1, 0, 0, 0],
                                 timeout=10))

    def test_the_file_never_holds_a_token_or_a_key_and_is_private(self):
        a, path_a, _ = self.make(18987, "a")
        a.start()
        b, path_b, _ = self.make(18988, "b", seeds=[a.url])
        b.start()
        self.assertTrue(wait_for(lambda: len(read(path_b)["peers"]) == 2, timeout=10))
        raw = path_b.read_bytes().decode()
        record = a.table.get(a.identity.peer)
        for secret in (a.token, b.token, record["pub"], record["sig"]):
            self.assertNotIn(secret, raw)
        self.assertNotIn("token", raw)
        self.assertEqual(stat.S_IMODE(path_b.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(path_b.parent.stat().st_mode), 0o700)

    def test_a_reader_never_sees_a_half_written_file(self):
        n, path, _ = self.make(18989, "a")
        n.status_s = 0.005
        n.start()
        bad, reads, done = [], [0], threading.Event()

        def reader():
            while not done.is_set():
                try:
                    json.loads(path.read_text())
                    reads[0] += 1
                except FileNotFoundError:
                    pass
                except ValueError as e:
                    bad.append(str(e))

        t = threading.Thread(target=reader)
        t.start()
        time.sleep(1.0)
        done.set()
        t.join()
        self.assertGreater(reads[0], 50)
        self.assertEqual(bad, [])

    def test_a_status_that_cannot_be_written_does_not_stop_the_peer(self):
        blocker = Path(self.tmp.name) / "file"
        blocker.write_text("x")
        n, _, _ = self.make(18990, "a")
        n._status = node.StatusFile(blocker / "inside" / "status.json")  # its parent is a file: every write fails
        n.start()
        self.assertEqual(n.phase, "up")


class Progress(unittest.TestCase):
    def test_it_adds_up_the_hub_lines_over_every_file_and_ignores_the_rest(self):
        p = node.Progress()
        self.assertIsNone(p.snapshot())
        p.feed("model.safetensors: 3 tensors in 1 range(s)")
        self.assertIsNone(p.snapshot())
        p.feed("model.safetensors: 100 MB of 400 MB")
        self.assertEqual(p.snapshot(), {"bytes": 100 << 20, "total": 400 << 20})
        p.feed("model.safetensors: 300 MB of 400 MB")
        p.feed("shard-2.safetensors: 50 MB of 200 MB")
        self.assertEqual(p.snapshot(), {"bytes": 350 << 20, "total": 600 << 20})
        p.feed("model.safetensors: 400 MB written (12 tensors)")
        self.assertEqual(p.snapshot(), {"bytes": 450 << 20, "total": 600 << 20})
        p.reset()
        self.assertIsNone(p.snapshot())


class Download(StatusCase):
    def test_the_download_is_reported_while_it_runs_and_a_failure_becomes_a_friendly_error(self):
        self.manifest = Manifest.from_dict({**MANIFEST, "model": {**MANIFEST["model"], "repos": {"stub": "x/y"}}})
        seen = []

        def fetcher(repo, window):
            n._hub_log("model.safetensors: 100 MB of 400 MB")
            seen.append(n.status_snapshot())
            raise OSError("connection reset")

        n, path, status = self.make(18991, "a", fetcher=fetcher)
        n.stub = False
        with self.assertRaises(node.NodeError) as caught:
            n.start()
        self.assertEqual(seen[0]["state"], "downloading")
        self.assertEqual(seen[0]["download"], {"bytes": 100 << 20, "total": 400 << 20})
        self.assertEqual(seen[0]["window"], [0, 9])
        n.fail(str(caught.exception))
        n.stop()
        doc = read(path)
        self.assertEqual(doc["state"], "error")  # an error stays an error after the process cleans up
        self.assertTrue(doc["error"].startswith("The download failed"))
        self.assertIsNone(doc["download"])


class CommandLine(unittest.TestCase):
    def run_cli(self, tmp, *args):
        env = dict(os.environ, PYTHONPATH=str(ROOT), COMMON_NETWORK_HOME=tmp)
        return subprocess.run([sys.executable, "-m", "commonnet.node", *args], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=60)

    def test_a_stage_that_does_not_fit_leaves_a_friendly_error_in_the_status_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "m.json"
            manifest.write_text(json.dumps(MANIFEST))
            path = Path(tmp) / "status.json"
            out = self.run_cli(tmp, "--network", str(manifest), "--stub", "--no-tunnel", "--stage-gib", "0.5",
                               "--port", "18992", "--status-file", str(path))
            self.assertEqual(out.returncode, 1)
            doc = read(path)
            self.assertEqual((doc["state"], doc["network"]), ("error", "test-net"))
            self.assertIn("This network needs at least 9.0 GiB", doc["error"])
            self.assertEqual(set(doc), FIELDS)

    def test_a_missing_manifest_is_an_error_state_too_and_the_env_var_picks_the_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "elsewhere" / "s.json"
            env = dict(os.environ, PYTHONPATH=str(ROOT), COMMON_NETWORK_HOME=tmp, COMMONNET_STATUS_FILE=str(path))
            out = subprocess.run([sys.executable, "-m", "commonnet.node", "--network", str(Path(tmp) / "nope.json")],
                                 cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(out.returncode, 2)
            doc = read(path)
            self.assertEqual(doc["state"], "error")
            self.assertTrue(doc["error"])

    def test_the_default_path_is_under_the_network_home(self):
        self.assertEqual(node.default_status_path({}), node.HOME / "net" / "status.json")
        self.assertEqual(node.default_status_path({node.STATUS_ENV: "/x/y.json"}), Path("/x/y.json"))


class RouteBackend:
    """A backend that sends one tiny activation through the node's real route and answers with letters."""

    model_name = "Toy"
    tokenizer = LetterTokenizer()

    def __init__(self, network):
        self.network = network
        self.windows = []

    def encode_chat(self, messages, template_kwargs):
        return [1, 2, 3]

    def stream(self, prompt_ids, max_tokens, temperature, top_p, seed):
        endpoints, transport = self.network.plan()
        self.windows = [(e.start, e.end) for e in endpoints]
        session = uuid.uuid4().hex
        hidden = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
        try:
            for ep in endpoints:
                hidden = transport.forward(ep, session, hidden, 0)
        finally:
            for ep in endpoints:
                transport.close(ep, session)
        return FakeStream([int(v) + 1 for v in hidden.flatten()[:3]])


def http(port, method, path, body=None):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


class LocalApi(StatusCase):
    def two_stage_manifest(self):
        self.manifest = Manifest.from_dict({**MANIFEST, "boundaries": [0, 18, 36]})

    def test_the_api_answers_at_once_says_loading_then_serves_a_chat_through_two_stub_blocks(self):
        self.two_stage_manifest()
        gate = threading.Event()
        backends = []

        def factory(network):
            gate.wait(10)
            backends.append(RouteBackend(network))
            return backends[0]

        a, path_a, _ = self.make(18993, "a", gib=18.0, api_port=18995, api_backend=factory)
        a.start()
        b, _, _ = self.make(18994, "b", seeds=[a.url], gib=18.0)
        b.start()
        self.assertEqual((a.window, b.window), ((0, 18), (18, 36)))
        # The server is up before the model part has loaded.
        status, models = http(18995, "GET", "/v1/models")
        self.assertEqual((status, models["data"][0]["id"]), (200, "Toy"))
        self.assertEqual(http(18995, "GET", "/health")[1]["ready"], False)
        status, body = http(18995, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual((status, body["error"]["code"]), (503, "loading"))
        self.assertEqual(read(path_a)["api"], {"port": 18995, "ready": False})
        gate.set()
        self.assertTrue(wait_for(lambda: read(path_a)["api"] == {"port": 18995, "ready": True}, timeout=10))
        self.assertTrue(wait_for(lambda: len(a.table.live()) == 2, timeout=10))
        status, body = http(18995, "POST", "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3})
        self.assertEqual(status, 200, body)
        self.assertEqual(body["choices"][0]["message"]["content"], "abc")  # the activation crossed both blocks intact
        self.assertEqual(backends[0].windows, [(0, 18), (18, 36)])  # the node's own route: itself, then its peer
        self.assertEqual(http(18995, "GET", "/health")[1]["ready"], True)
        a.stop()
        with self.assertRaises(urllib.error.URLError):
            urllib.request.urlopen("http://127.0.0.1:18995/v1/models", timeout=3)  # the port is released on leaving

    def test_a_model_that_fails_to_load_is_reported_not_hidden(self):
        def factory(network):
            raise RuntimeError("no memory")

        a, path_a, _ = self.make(18996, "a", api_port=18997, api_backend=factory)
        a.start()
        self.assertTrue(wait_for(lambda: read(path_a)["error"], timeout=10))
        doc = read(path_a)
        self.assertEqual(doc["state"], "up")  # the peer itself is fine
        self.assertEqual(doc["api"], {"port": 18997, "ready": False})
        self.assertIn("could not load its model part", doc["error"])
        status, body = http(18997, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(status, 503)
        self.assertIn("could not load", body["error"]["message"])

    def test_a_backend_that_exits_instead_of_raising_is_still_reported(self):
        def factory(network):
            raise SystemExit("the manifest names no client_repo; pass --checkpoint DIR")

        a, path_a, _ = self.make(18979, "a", api_port=18978, api_backend=factory)
        a.start()
        self.assertTrue(wait_for(lambda: read(path_a)["error"], timeout=10))
        self.assertIn("client_repo", read(path_a)["error"])
        self.assertEqual(read(path_a)["api"], {"port": 18978, "ready": False})

    def test_a_busy_port_is_an_error_in_the_file_and_the_peer_stays_up(self):
        import socket
        busy = socket.socket()
        busy.bind(("127.0.0.1", 18998))
        busy.listen(1)
        self.addCleanup(busy.close)
        a, path_a, _ = self.make(18999, "a", api_port=18998, api_backend=lambda network: RouteBackend(network))
        a.start()
        doc = read(path_a)
        self.assertEqual(doc["state"], "up")
        self.assertIn("could not start on port 18998", doc["error"])

    def test_requests_for_this_nodes_own_window_go_to_loopback_not_out_through_the_tunnel(self):
        a, _, _ = self.make(18980, "a")
        a.start()
        a.url = "https://example.trycloudflare.com"  # what a tunnelled node publishes

        class T:
            urls = {"hop0": "https://example.trycloudflare.com", "hop1": "https://other.example"}
            _replicas = {"hop0": [("https://other.example", "t")], "hop1": [("https://example.trycloudflare.com", "tok"),
                                                                           "https://example.trycloudflare.com"]}

        t = T()
        a.localize(t)
        self.assertEqual(t.urls, {"hop0": a.server.url, "hop1": "https://other.example"})
        self.assertEqual(t._replicas["hop1"], [(a.server.url, "tok"), a.server.url])
        self.assertEqual(t._replicas["hop0"], [("https://other.example", "t")])


if __name__ == "__main__":
    unittest.main()
