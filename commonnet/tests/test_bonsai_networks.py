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
        self.assertIsNone(m.repo("nf4"))
        self.assertEqual(m.client_repo, m.repo("mlx"))
        self.assertLess(m.window_gib(0, 36, "mlx"), 3.0)   # the whole 2-bit model fits one small Mac

    def test_bonsai_2_9b(self):
        m = self.load("bonsai-2-9b.json")
        self.assertEqual((m.num_layers, m.boundaries), (32, (0, 8, 16, 24, 32)))
        self.assertEqual(m.repo("mlx"), "prism-ml/Ternary-Bonsai-2-9B-mlx-2bit")
        self.assertIsNone(m.repo("cpu"))
        self.assertIsNone(m.repo("nf4"))
        self.assertIn("linear attention", m.model["no_speculation"])
        self.assertTrue(all(b % 4 == 0 for b in m.boundaries))  # stages hold whole 4-layer hybrid groups

    def test_default_network_is_untouched(self):
        m = self.load("main.json")
        self.assertEqual(m.id, "common-main-8b-v1")
        self.assertEqual(m.model["name"], "Qwen3-8B")


class Refusal(unittest.TestCase):
    def test_cpu_machine_refused_for_9b_and_names_supported_backends(self):
        m = Manifest.load(NETWORKS / "bonsai-2-9b.json")
        for backend in ("cpu", "nf4"):
            with self.assertRaises(node.NodeError) as cm:
                node.check_backend_supported(m, node.Hardware(backend, 16.0, "x"))
            text = str(cm.exception)
            self.assertIn("Apple silicon (MLX)", text)
            self.assertNotIn("NVIDIA", text.split("Supported here:")[1].split(".")[0])

    def test_nf4_refused_for_8b_but_cpu_and_mlx_allowed(self):
        m = Manifest.load(NETWORKS / "bonsai-8b.json")
        with self.assertRaises(node.NodeError) as cm:
            node.check_backend_supported(m, node.Hardware("nf4", 16.0, "x"))
        self.assertIn("Apple silicon (MLX), CPU", str(cm.exception))
        node.check_backend_supported(m, node.Hardware("mlx", 16.0, "x"))
        node.check_backend_supported(m, node.Hardware("cpu", 16.0, "x"))
        node.check_backend_supported(m, node.Hardware("stub", 16.0, "stub"))

    def test_main_network_unaffected(self):
        m = Manifest.load(NETWORKS / "main.json")
        for backend in ("mlx", "nf4", "cpu"):
            node.check_backend_supported(m, node.Hardware(backend, 16.0, "x"))

    def test_main_refuses_before_downloading_anything(self):
        import io, contextlib, tempfile
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(err):
            code = node.main(["--network", str(NETWORKS / "bonsai-2-9b.json"), "--backend", "cpu",
                              "--identity-dir", d, "--status-file", os.path.join(d, "s.json"), "--no-tunnel"])
        self.assertEqual(code, 2)
        self.assertIn("Supported here: Apple silicon (MLX)", err.getvalue())


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
