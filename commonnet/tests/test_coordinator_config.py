"""The coordinator must count layers from the decoder config, not from a plain Qwen3Config.

A qwen3_5 checkpoint nests its decoder config under text_config; reading it as Qwen3Config
reported 32 layers for the 64-layer Qwen3.6-27B and the route check failed.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from unittest import mock

from commonnet import coordinator
from commonnet.tests.qwen35_tiny import make_qwen35_checkpoint


class CoordinatorLayerCount(unittest.TestCase):
    def test_qwen35_layer_count_comes_from_text_config(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        info = make_qwen35_checkpoint(d)
        seen = {}

        def stop(members, n):
            seen["n"] = n
            raise RuntimeError("stop after the layer count")

        with mock.patch.object(coordinator, "build_route", stop):
            with self.assertRaises(RuntimeError):
                coordinator.generate([], d, [1, 2, 3], 1)
        self.assertEqual(seen["n"], 8)


if __name__ == "__main__":
    unittest.main()
