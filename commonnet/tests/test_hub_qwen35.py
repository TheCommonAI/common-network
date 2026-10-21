"""hub.py with the qwen3_5 layouts (HF model.language_model.*, MLX language_model.model.*), sharded and single. Offline."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

from commonnet import hub  # noqa: E402
from commonnet.block import BlockRange  # noqa: E402
from commonnet.tests.qwen35_tiny import make_qwen35_checkpoint  # noqa: E402
from commonnet.tests.test_hub import make_server  # noqa: E402


def files_for(shards: int):
    d = tempfile.mkdtemp()
    make_qwen35_checkpoint(d, layers=8, shards=shards)
    files = {n: open(os.path.join(d, n), "rb").read() for n in os.listdir(d)}
    return d, files


class HubQwen35(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.srv = []

    def tearDown(self):
        for s in self.srv:
            s.shutdown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def hub(self, files):
        srv, log, _ = make_server(files)
        self.srv.append(srv)
        return hub.Hub("org/tiny35", "main", endpoint=f"http://127.0.0.1:{srv.server_port}",
                       environ={"COMMONNET_CACHE": self.tmp}, log=lambda m: None, sleep=lambda s: None), log

    def test_detect_layout(self):
        self.assertEqual(hub.detect_layout(["model.language_model.layers.0.x", "model.visual.blocks.0.y"]), "model.language_model.")
        self.assertEqual(hub.detect_layout(["language_model.model.layers.0.x"]), "language_model.model.")
        self.assertEqual(hub.detect_layout(["model.layers.0.x"]), "model.")

    def test_sharded_downloads_only_needed_shards_and_runs(self):
        full, files = files_for(4)
        h, log = self.hub(files)
        d = h.fetch(layers=(2, 4))
        wm = json.loads(files["model.safetensors.index.json"])["weight_map"]
        want = {wm[n] for n in wm if n.startswith(("model.language_model.layers.2.", "model.language_model.layers.3."))}
        got = {n for n in os.listdir(d) if n.endswith(".safetensors")}
        self.assertEqual(got, want)
        kept = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.assertTrue(kept and all(n.startswith(("model.language_model.layers.2.", "model.language_model.layers.3.")) for n in kept))
        # no whole-shard GET: every shard request carried a Range
        self.assertFalse(any(n.endswith(".safetensors") and not r and m == "GET" for m, n, r, _ in log))
        b = BlockRange(d, 2, 4)
        ref = BlockRange(full, 2, 4)
        x = torch.randn(1, 5, 64)
        torch.testing.assert_close(b.forward("s", x, 0), ref.forward("s", x, 0))

    def test_single_file_ranges_only_for_needed_layers(self):
        full, files = files_for(1)
        h, log = self.hub(files)
        d = h.fetch(layers=(4, 5))  # layer 4 is linear attention
        b = BlockRange(d, 4, 5)
        self.assertTrue(all(n.startswith("model.language_model.layers.4.") for n in b.loaded_names))
        x = torch.randn(1, 5, 64)
        torch.testing.assert_close(b.forward("s", x, 0), BlockRange(full, 4, 5).forward("s", x, 0))
        self.assertFalse(any(n == "model.safetensors" and not r and m == "GET" for m, n, r, _ in log))

    def test_client_tensors_fetched_with_head(self):
        full, files = files_for(1)
        h, _ = self.hub(files)
        d = h.fetch(client=True)
        from safetensors import safe_open
        with safe_open(os.path.join(d, "model.safetensors"), "pt") as f:
            self.assertEqual(set(f.keys()), {"model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"})


if __name__ == "__main__":
    unittest.main()
