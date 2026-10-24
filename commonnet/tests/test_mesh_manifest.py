import json
import os
import tempfile
import unittest

from commonnet.mesh.manifest import Manifest, ManifestError

BASE = {"id": "t-1", "version": 1,
        "model": {"name": "T", "num_layers": 36, "repos": {"mlx": "a/b", "nf4": "c/d"}, "client_repo": "a/b"},
        "boundaries": [0, 9, 18, 27, 36], "gib_per_layer": {"mlx": 0.125, "nf4": 0.17}, "fixed_gib": 0.3,
        "min_replicas": 2, "seeds": ["http://127.0.0.1:1"]}


class ManifestTests(unittest.TestCase):
    def test_load_from_dict_json_and_file(self):
        a = Manifest.load(BASE)
        self.assertEqual(Manifest.load(json.dumps(BASE)), a)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            open(path, "w").write(json.dumps(BASE))
            self.assertEqual(Manifest.load(path), a)
            self.assertEqual(Manifest.load(a), a)

    def test_stages_and_roundtrip(self):
        m = Manifest.load(BASE)
        self.assertEqual(m.stages(), [(0, 9), (9, 18), (18, 27), (27, 36)])
        self.assertEqual(Manifest.from_dict(m.to_dict()), m)
        self.assertEqual(m.repo("mlx"), "a/b")
        self.assertIsNone(m.repo("torch"))
        self.assertEqual(m.client_repo, "a/b")

    def test_window_gib_and_fitting(self):
        m = Manifest.load(BASE)
        self.assertAlmostEqual(m.window_gib(0, 9, "mlx"), 0.3 + 9 * 0.125)
        self.assertAlmostEqual(m.window_gib(0, 36, "nf4"), 0.3 + 36 * 0.17)
        fits = m.windows_fitting(2.0, "mlx")  # 0.3 + 0.125 * n <= 2.0 -> n <= 13.6
        self.assertIn((0, 9), fits)
        self.assertNotIn((0, 18), fits)
        self.assertEqual(m.windows_fitting(0.1, "mlx"), [])
        self.assertEqual(len(m.windows_fitting(100, "mlx")), 10)  # all 4+3+2+1 aligned windows
        self.assertEqual(m.windows_fitting(100, "mlx")[0], (0, 36))

    def test_unknown_backend_is_costed_at_the_largest(self):
        m = Manifest.load(BASE)
        self.assertEqual(m.per_layer("stub"), 0.17)

    def test_is_window(self):
        m = Manifest.load(BASE)
        self.assertTrue(m.is_window(9, 27))
        self.assertFalse(m.is_window(9, 20))
        self.assertFalse(m.is_window(18, 9))

    def test_bad_manifests_are_refused(self):
        for patch in ({"boundaries": [0, 9, 9, 36]}, {"boundaries": [1, 36]}, {"boundaries": [0, 35]},
                      {"boundaries": [0]}, {"id": ""}, {"min_replicas": 0}):
            with self.subTest(patch=patch):
                with self.assertRaises(ManifestError):
                    Manifest.from_dict({**BASE, **patch})
        with self.assertRaises(ManifestError):
            Manifest.from_dict({k: v for k, v in BASE.items() if k != "model"})
        with self.assertRaises(ManifestError):
            Manifest.load("/no/such/file.json")


if __name__ == "__main__":
    unittest.main()


class CpuBackendManifest(unittest.TestCase):
    def test_main_manifest_has_a_cpu_checkpoint_and_cost(self):
        from pathlib import Path
        m = Manifest.load(Path(__file__).resolve().parents[1] / "networks" / "main.json")
        self.assertEqual(m.repo("cpu"), "Qwen/Qwen3-8B")
        self.assertEqual(m.model["dtype"]["cpu"], "bfloat16")
        # an unquantised bf16 Qwen3-8B layer is ~0.36 GiB; a 9-layer window must be costed at least that
        self.assertGreaterEqual(m.window_gib(0, 9, "cpu"), 0.3 + 9 * 0.36)
        self.assertEqual(m.windows_fitting(5.0, "cpu")[0], (0, 9))
        self.assertEqual(m.windows_fitting(3.0, "cpu"), [])

    def test_a_peer_that_does_not_know_cpu_is_unaffected(self):
        from pathlib import Path
        m = Manifest.load(Path(__file__).resolve().parents[1] / "networks" / "main.json")
        self.assertAlmostEqual(m.window_gib(0, 9, "mlx"), 0.3 + 9 * 0.125)
        self.assertAlmostEqual(m.window_gib(0, 9, "nf4"), 0.3 + 9 * 0.17)
