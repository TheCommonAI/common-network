"""The retired gateway: signpost endpoints work, everything the old network used says 410."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gateway"))

from fastapi.testclient import TestClient  # noqa: E402

from app import main  # noqa: E402


class Retired(unittest.TestCase):
    def setUp(self):
        main.NETWORKS = ROOT / "commonnet" / "networks"
        self.client = TestClient(main.app)

    def test_health_manifest_seeds_and_home(self):
        self.assertEqual(self.client.get("/health").json()["mode"], "retired")
        self.assertEqual(self.client.get("/manifest").json()["id"], "common-main-8b-v1")
        self.assertIn("seeds", self.client.get("/seeds").json())
        self.assertIn("common join", self.client.get("/").text)

    def test_seeds_from_the_environment_are_added_once(self):
        import os
        os.environ["COMMON_SEEDS"] = "https://a.example, https://a.example"
        try:
            self.assertEqual(self.client.get("/seeds").json()["seeds"].count("https://a.example"), 1)
        finally:
            del os.environ["COMMON_SEEDS"]

    def test_every_old_endpoint_says_it_moved(self):
        for method, path in (("get", "/nodes"), ("post", "/nodes"), ("post", "/v1/chat/completions"),
                             ("get", "/v1/models"), ("get", "/catalogue"), ("post", "/assign"),
                             ("get", "/admin/anything"), ("delete", "/nodes/x"), ("get", "/mobile/ws")):
            with self.subTest(path=path):
                r = getattr(self.client, method)(path)
                self.assertEqual(r.status_code, 410)
                self.assertIn("common join", r.json()["error"])


if __name__ == "__main__":
    unittest.main()
