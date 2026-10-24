"""The Bonsai network manifests, the early refusal on unsupported backends, and drafter compatibility."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from commonnet import node
from commonnet.mesh.manifest import Manifest
from commonnet.speculative import SpeculationRefused, check_tokenizers_match

NETWORKS = Path(node.__file__).resolve().parent / "networks"


class FakeTokenizer:
    def __init__(self, vocab, added=None):
        self._vocab, self.added_tokens_decoder = vocab, added or {}

    def get_vocab(self):
        return dict(self._vocab)


def qwen3_vocab():
    v = {f"t{i}": i for i in range(151643)}
    v.update({f"<special{i}>": 151643 + i for i in range(26)})
    return v


class Manifests(unittest.TestCase):
    def load(self, name):
        return Manifest.load(NETWORKS / name)

    def test_bonsai_8b(self):
        m = self.load("bonsai-8b.json")
        self.assertEqual((m.num_layers, m.boundaries), (36, (0, 9, 18, 27, 36)))
        self.assertEqual(m.repo("mlx"), "prism-ml/Ternary-Bonsai-8B-mlx-2bit")
        self.assertEqual(m.repo("cpu"), "prism-ml/Ternary-Bonsai-8B-unpacked")
        self.assertEqual(m.repo("nf4"), "prism-ml/Ternary-Bonsai-8B-unpacked")  # float16, quantised to NF4 on load
        self.assertEqual(m.client_repo, m.repo("mlx"))
        self.assertLess(m.window_gib(0, 36, "mlx"), 3.0)   # the whole 2-bit model fits one small Mac
        # a 9-layer window fits the 4 GiB a 4060 offers; the whole model (about 3.7 GiB NF4 + overhead) does not
        self.assertLess(m.window_gib(0, 9, "nf4"), 1.5)
        self.assertGreater(m.window_gib(0, 36, "nf4"), 4.0)

    def test_bonsai_2_9b(self):
        m = self.load("bonsai-2-9b.json")
        self.assertEqual((m.num_layers, m.boundaries), (32, (0, 8, 16, 24, 32)))
        self.assertEqual(m.repo("mlx"), "prism-ml/Ternary-Bonsai-2-9B-mlx-2bit")
        self.assertEqual(m.repo("cpu"), m.repo("mlx"))   # the same 2-bit pack, dequantised in torch (hadamard_torch.py)
        self.assertEqual(m.repo("nf4"), m.repo("mlx"))
        self.assertLess(m.window_gib(0, 8, "nf4"), 1.5)
        self.assertIn("linear attention", m.model["no_speculation"])
        self.assertTrue(all(b % 4 == 0 for b in m.boundaries))  # stages hold whole 4-layer hybrid groups

    def test_default_network_is_untouched(self):
        m = self.load("main.json")
        self.assertEqual(m.id, "common-main-8b-v1")
        self.assertEqual(m.model["name"], "Qwen3-8B")


class Refusal(unittest.TestCase):
    def manifest_without(self, name, backend):
        d = json.loads((NETWORKS / name).read_text())
        d["model"]["repos"].pop(backend)
        return Manifest.from_dict(d) if hasattr(Manifest, "from_dict") else Manifest.parse(json.dumps(d))

    def test_bonsai_networks_accept_every_backend(self):
        for name in ("bonsai-8b.json", "bonsai-2-9b.json"):
            m = Manifest.load(NETWORKS / name)
            for backend in ("mlx", "nf4", "cpu", "stub"):
                node.check_backend_supported(m, node.Hardware(backend, 16.0, "x"))

    def test_a_network_without_an_nf4_repo_still_refuses_nvidia_and_says_what_runs(self):
        m = self.manifest_without("bonsai-8b.json", "nf4")
        with self.assertRaises(node.NodeError) as cm:
            node.check_backend_supported(m, node.Hardware("nf4", 4.0, "RTX 4060"))
        text = str(cm.exception)
        self.assertIn("no NVIDIA GPU (4-bit) checkpoint", text)
        self.assertIn("Apple silicon (MLX), CPU", text)

    def test_main_network_unaffected(self):
        m = Manifest.load(NETWORKS / "main.json")
        for backend in ("mlx", "nf4", "cpu"):
            node.check_backend_supported(m, node.Hardware(backend, 16.0, "x"))

    def test_main_refuses_before_downloading_anything(self):
        import io, contextlib, tempfile
        err = io.StringIO()
        d = json.loads((NETWORKS / "bonsai-2-9b.json").read_text())
        d["model"]["repos"].pop("cpu")
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stderr(err):
            path = os.path.join(t, "m.json")
            with open(path, "w") as f:
                json.dump(d, f)
            code = node.main(["--network", path, "--backend", "cpu",
                              "--identity-dir", t, "--status-file", os.path.join(t, "s.json"), "--no-tunnel"])
        self.assertEqual(code, 2)
        self.assertIn("has no CPU checkpoint", err.getvalue())


class DrafterCompat(unittest.TestCase):
    def test_bonsai_1_shares_qwen3_tokenizer_so_the_drafter_is_allowed(self):
        check_tokenizers_match(FakeTokenizer(qwen3_vocab()), FakeTokenizer(qwen3_vocab()))

    def test_bonsai_2_vocabulary_is_refused(self):
        big = {f"u{i}": i for i in range(248077)}
        with self.assertRaises(SpeculationRefused):
            check_tokenizers_match(FakeTokenizer(big), FakeTokenizer(qwen3_vocab()))

    def test_real_tokenizers_when_present(self):
        from transformers import AutoTokenizer
        b1 = next(iter(sorted((Path.home() / ".cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-8B-mlx-2bit/snapshots").glob("*"))), None) \
            if (Path.home() / ".cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-8B-mlx-2bit/snapshots").is_dir() else None
        q = Path.home() / ".common-network/commonnet/models/Qwen/Qwen3-8B/main"
        if not (b1 and (b1 / "tokenizer.json").exists() and (q / "tokenizer.json").exists()):
            self.skipTest("tokenizers not cached")
        check_tokenizers_match(AutoTokenizer.from_pretrained(str(b1)), AutoTokenizer.from_pretrained(str(q)))


class NoSpeculationFlag(unittest.TestCase):
    def test_backend_reports_the_manifest_reason_instead_of_loading_a_drafter(self):
        from commonnet import serve
        src = Path(serve.__file__).read_text()
        self.assertIn('manifest.model.get("no_speculation")', src)


if __name__ == "__main__":
    unittest.main()
