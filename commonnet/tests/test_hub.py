"""hub.py tests against a local http.server double with Range support. Fully offline."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from transformers import Qwen3ForCausalLM  # noqa: E402

from commonnet import hub  # noqa: E402
from commonnet.block import BlockRange  # noqa: E402
from commonnet.tests.test_split import tiny_config  # noqa: E402


def make_server(files: dict):
    log = []
    drop = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self, head=False):
            name = self.path.split("/resolve/main/", 1)[-1]
            log.append((self.command, name, self.headers.get("Range"), self.headers.get("Authorization")))
            if name not in files:
                self.send_error(404)
                return
            data = files[name]
            rng = self.headers.get("Range")
            status, lo = 200, 0
            if rng:
                m = re.fullmatch(r"bytes=(\d+)-(\d+)", rng)
                lo, hi = int(m[1]), min(int(m[2]) + 1, len(data))
                data, status = data[lo:hi], 206
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if not head:
                cut = drop.get(name)
                if cut and not rng:
                    data = data[:cut]
                    self.wfile.write(data)
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.wfile.write(data)

        def do_HEAD(self):
            self.do_GET(head=True)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, log, drop


def model_files(sharded: bool):
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(tiny_config(False)).eval()
    d = tempfile.mkdtemp()
    model.save_pretrained(d, max_shard_size="40KB" if sharded else "1GB")
    files = {n: open(os.path.join(d, n), "rb").read() for n in os.listdir(d)}
    shutil.rmtree(d)
    return model, files


class HubTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.env = {"COMMONNET_CACHE": self.tmp}
        self.servers = []

    def tearDown(self):
        for s in self.servers:
            s.shutdown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def hub(self, files, env=None):
        srv, log, drop = make_server(files)
        self.servers.append(srv)
        e = dict(self.env, **(env or {}))
        h = hub.Hub("org/tiny", "main", endpoint=f"http://127.0.0.1:{srv.server_port}", environ=e,
                    log=lambda m: self.logs.append(m), sleep=lambda s: None)
        return h, log, drop

    logs: list

    def run(self, result=None):
        self.logs = []
        return super().run(result)

    def test_single_file_only_needed_bytes_and_same_output(self):
        model, files = model_files(False)
        h, log, _ = self.hub(files)
        d = h.fetch(layers=(1, 3))
        blob = files["model.safetensors"]
        import struct
        n = struct.unpack("<Q", blob[:8])[0]
        header = json.loads(blob[8:8 + n])
        other = [(header[k]["data_offsets"]) for k in header if k != "__metadata__"
                 and not k.startswith(("model.layers.1.", "model.layers.2."))]
        base = 8 + n
        for _, name, rng, _ in log:
            if name == "model.safetensors" and rng:
                lo, hi = map(int, rng[6:].split("-"))
                for a, b in other:
                    self.assertTrue(hi < base + a or lo >= base + b, f"requested other bytes {rng}")
        self.assertFalse(any(name == "model.safetensors" and not rng and m == "GET" for m, name, rng, _ in log))
        ref = BlockRange(checkpoint_dir=self._full(files), start=1, end=3)
        got = BlockRange(checkpoint_dir=d, start=1, end=3)
        x = torch.randn(1, 5, 64)
        torch.testing.assert_close(got.forward("s", x, 0), ref.forward("s", x, 0))

    def _full(self, files):
        d = os.path.join(self.tmp, "full")
        os.makedirs(d, exist_ok=True)
        for n, b in files.items():
            open(os.path.join(d, n), "wb").write(b)
        return d

    def test_sharded_only_needed_shards(self):
        model, files = model_files(True)
        wm = json.loads(files[hub.INDEX_NAME])["weight_map"]
        shards = sorted(set(wm.values()))
        self.assertGreater(len(shards), 2)
        h, log, _ = self.hub(files)
        d = h.fetch(layers=(3, 4))
        want = {wm[k] for k in wm if k.startswith("model.layers.3.")}
        got = {n for _, n, _, _ in log if n.endswith(".safetensors")}
        self.assertEqual(got, want)
        self.assertLess(len(want), len(shards))
        ref = BlockRange(checkpoint_dir=self._full(files), start=3, end=4)
        out = BlockRange(checkpoint_dir=d, start=3, end=4)
        x = torch.randn(1, 4, 64)
        torch.testing.assert_close(out.forward("s", x, 0), ref.forward("s", x, 0))

    def test_one_file_with_an_index_downloads_only_needed_tensors(self):
        """A pack that is one model.safetensors plus an index (the Bonsai MLX packs): ranges, not the whole file."""
        model, files = model_files(False)
        import struct
        blob = files["model.safetensors"]
        n = struct.unpack("<Q", blob[:8])[0]
        names = [k for k in json.loads(blob[8:8 + n]) if k != "__metadata__"]
        files[hub.INDEX_NAME] = json.dumps({"metadata": {}, "weight_map": {k: "model.safetensors" for k in names}}).encode()
        h, log, _ = self.hub(files)
        d = h.fetch(layers=(1, 3))
        self.assertFalse(any(name == "model.safetensors" and not rng and m == "GET" for m, name, rng, _ in log))
        ref = BlockRange(checkpoint_dir=self._full(files), start=1, end=3)
        got = BlockRange(checkpoint_dir=d, start=1, end=3)
        x = torch.randn(1, 5, 64)
        torch.testing.assert_close(got.forward("s", x, 0), ref.forward("s", x, 0))
        self.assertLess(os.path.getsize(os.path.join(d, "model.safetensors")), len(blob))

    def test_client_tensors(self):
        model, files = model_files(False)
        h, log, _ = self.hub(files)
        d = h.fetch(client=True)
        from safetensors import safe_open
        with safe_open(os.path.join(d, "model.safetensors"), "pt") as f:
            names = set(f.keys())
        self.assertEqual(names, {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"})

    def test_resume_partial(self):
        model, files = model_files(False)
        h, log, _ = self.hub(files)
        d = h.fetch(layers=(0, 2))
        final = os.path.join(d, "model.safetensors")
        whole = open(final, "rb").read()
        os.rename(final, final + ".part")
        with open(final + ".part", "r+b") as f:
            f.truncate(len(whole) // 2)
        log.clear()
        h.dir = os.path.dirname(final)
        h.fetch_compact("model.safetensors", hub.needed_prefixes((0, 2)))
        self.assertEqual(open(final, "rb").read(), whole)
        total = sum(int(r[6:].split("-")[1]) - int(r[6:].split("-")[0]) + 1 for _, n, r, _ in log if r and n == "model.safetensors")
        self.assertLess(total, len(whole))

    def test_resume_whole_shard(self):
        model, files = model_files(True)
        name = sorted(set(json.loads(files[hub.INDEX_NAME])["weight_map"].values()))[0]
        h, log, _ = self.hub(files)
        h.dir = os.path.join(self.tmp, "x")
        os.makedirs(h.dir)
        open(os.path.join(h.dir, name + ".part"), "wb").write(files[name][:100])
        h.fetch_file(name)
        self.assertEqual(open(os.path.join(h.dir, name), "rb").read(), files[name])
        self.assertEqual([r for _, n, r, _ in log if n == name and r], ["bytes=100-%d" % (len(files[name]) - 1)])

    def test_wrong_size_rejected(self):
        model, files = model_files(False)
        h, log, _ = self.hub(files)
        h.dir = os.path.join(self.tmp, "y")
        os.makedirs(h.dir)
        open(os.path.join(h.dir, "model.safetensors"), "wb").write(b"short")
        with self.assertRaises(hub.HubError):
            h.fetch_file("model.safetensors")

    def test_truncated_server_rejected(self):
        files = {"model.safetensors": b"x" * 1000}
        h, log, _ = self.hub(files)
        orig = h.get_bytes
        # server returns fewer bytes than asked: ask beyond the end
        with self.assertRaises(hub.HubError):
            h.get_bytes("model.safetensors", (900, 1100))

    def test_token_sent_not_logged(self):
        model, files = model_files(False)
        h, log, _ = self.hub(files, {"HF_TOKEN": "hf_SECRET123"})
        h.fetch(layers=(0, 1))
        self.assertTrue(all(a == "Bearer hf_SECRET123" for *_, a in log))
        self.assertNotIn("SECRET123", "\n".join(self.logs))
        h2, log2, _ = self.hub(files)
        h2.fetch(layers=(0, 1))
        self.assertTrue(all(a is None for *_, a in log2))


def many_tensor_file(n=40, size=1000, gap_every=None, gap=0):
    """Single safetensors blob with n small tensors t.0..; optional big unneeded gap after every gap_every."""
    import struct
    header, blob = {}, b""
    for i in range(n):
        if gap_every and i and i % gap_every == 0:
            header[f"skip.{i}"] = {"dtype": "U8", "shape": [gap], "data_offsets": [len(blob), len(blob) + gap]}
            blob += b"\xee" * gap
        header[f"model.layers.0.t{i}"] = {"dtype": "U8", "shape": [size], "data_offsets": [len(blob), len(blob) + size]}
        blob += bytes([i % 251]) * size
    hb = json.dumps(header).encode()
    hb += b" " * (-len(hb) % 8)
    return struct.pack("<Q", len(hb)) + hb + blob


class CoalesceTest(HubTest):
    def _prefix(self):
        return ["model.layers.0."]

    def test_requests_drop_to_segments(self):
        files = {"model.safetensors": many_tensor_file()}
        h, log, _ = self.hub(files)
        h.fetch_compact("model.safetensors", self._prefix())
        data_reqs = [r for _, n, r, _ in log if r and n == "model.safetensors"]
        self.assertEqual(len(data_reqs), 2 + 1)  # 8-byte size, header, ONE merged range (was 40)
        out = open(os.path.join(h.dir, "model.safetensors"), "rb").read()
        self.assertEqual(out.count(bytes([7]) * 1000), 1)
        self.assertTrue(any("40 tensors in 1 range" in m for m in self.logs))

    def test_large_gap_not_fetched(self):
        gap = 300
        files = {"model.safetensors": many_tensor_file(20, 100, gap_every=10, gap=gap)}
        h, log, _ = self.hub(files)
        orig = hub.BRIDGE
        hub.BRIDGE = 200
        try:
            h.fetch_compact("model.safetensors", self._prefix())
        finally:
            hub.BRIDGE = orig
        import struct
        blob = files["model.safetensors"]
        base = 8 + struct.unpack("<Q", blob[:8])[0]
        data = [tuple(map(int, r[6:].split("-"))) for _, n, r, _ in log if r and n == "model.safetensors"][2:]
        self.assertEqual(len(data), 2)
        skip = blob.index(b"\xee" * gap)
        for lo, hi in data:
            self.assertFalse(lo <= skip < hi, "gap was fetched")
        out = open(os.path.join(h.dir, "model.safetensors"), "rb").read()
        self.assertNotIn(b"\xee", out)

    def test_resume_after_interrupted_segment(self):
        files = {"model.safetensors": many_tensor_file()}
        h, log, _ = self.hub(files)
        h.fetch_compact("model.safetensors", self._prefix())
        final = os.path.join(h.dir, "model.safetensors")
        whole = open(final, "rb").read()
        os.rename(final, final + ".part")
        with open(final + ".part", "r+b") as f:
            f.truncate(len(whole) // 2 + 123)  # mid-tensor
        log.clear()
        h.fetch_compact("model.safetensors", self._prefix())
        self.assertEqual(open(final, "rb").read(), whole)
        got = sum(int(r[6:].split("-")[1]) - int(r[6:].split("-")[0]) + 1 for _, n, r, _ in log if r and n == "model.safetensors")
        self.assertLess(got, len(whole) * 3 // 4)

    def test_mid_stream_drop_retries(self):
        files = {"model.safetensors": many_tensor_file()}
        h, log, _ = self.hub(files)
        calls = {"n": 0}
        orig = h._request

        class Cut:
            def __init__(s, r):
                s.r, s.n = r, 0
                s.status = r.status
            def __enter__(s): return s
            def __exit__(s, *a): s.r.close()
            def read(s, k):
                if s.n >= 1:
                    raise ConnectionResetError()
                s.n += 1
                return s.r.read(min(k, 5000))

        def req(name, rng=None, method="GET"):
            r = orig(name, rng, method)
            if rng and rng[0] > 100 and calls["n"] == 0:
                calls["n"] = 1
                return Cut(r)
            return r
        h._request = req
        h.fetch_compact("model.safetensors", self._prefix())
        ref = many_tensor_file()
        out = open(os.path.join(h.dir, "model.safetensors"), "rb").read()
        self.assertEqual(out[-40000:], ref[-40000:])



if __name__ == "__main__":
    unittest.main()


class RetryAndLocal(unittest.TestCase):
    def test_fetch_retries_and_resumes_after_incomplete_download(self):
        from commonnet import hub
        h = hub.Hub("a/b", environ={"COMMONNET_CACHE": "/nonexistent-cache"}, log=lambda m: None, sleep=lambda s: None)
        calls = []

        def once(*a, **k):
            calls.append(1)
            if len(calls) < 3:
                raise hub.HubError("x: incomplete download (1 of 2 bytes)")
            return "dir"
        h._fetch_once = once
        self.assertEqual(h.fetch(layers=(0, 9)), "dir")
        self.assertEqual(len(calls), 3)

    def test_no_space_is_not_retried_and_says_so(self):
        from commonnet import hub
        h = hub.Hub("a/b", environ={"COMMONNET_CACHE": "/nonexistent-cache"}, log=lambda m: None, sleep=lambda s: None)
        calls = []

        def once(*a, **k):
            calls.append(1)
            raise OSError(28, "No space left on device")
        h._fetch_once = once
        with self.assertRaises(hub.NoDiskSpace) as cm:
            h.fetch()
        self.assertEqual(len(calls), 1)
        self.assertIn("COMMONNET_CACHE", str(cm.exception))

    def test_own_endpoint_is_local_and_never_rtc(self):
        from commonnet.net import client as nc
        nc.register_local_endpoint("https://me-abc.trycloudflare.com")
        self.assertEqual(nc.BlockClient(token="t" * 24).transport_report("https://me-abc.trycloudflare.com/")["transport"], "local")
        self.assertFalse(nc.BlockClient(token="t" * 24, rtc=True)._rtc_endpoint_allowed("https://me-abc.trycloudflare.com"))
        self.assertIsNone(nc.BlockClient(token="t" * 24, rtc=True)._rtc_peer("https://me-abc.trycloudflare.com"))
