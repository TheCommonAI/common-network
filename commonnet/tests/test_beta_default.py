"""Beta model selection must preserve explicit network and preset choices."""
import unittest
from unittest.mock import patch

from commonnet import node
from commonnet.presets import model_for_nodes


class BetaDefault(unittest.TestCase):
    def resolve(self, network=None, environ=None):
        with patch.object(node, "fetch_default_seeds", return_value=[]):
            return node.resolve_network(network, [], environ or {})[0]

    def test_plain_join_selects_beta_network(self):
        manifest = self.resolve()
        self.assertEqual(manifest.model["name"], "Qwen3.5-35B-A3B")
        self.assertEqual(manifest.num_layers, 40)
        self.assertEqual(manifest.model["repos"]["mlx"], "mlx-community/Qwen3.5-35B-A3B-4bit")
        self.assertIn("rollback", manifest.model["no_speculation"])

    def test_explicit_8b_remains_a_separate_network(self):
        fallback = self.resolve("qwen3-8b")
        self.assertEqual(fallback.model["name"], "Qwen3-8B")
        self.assertNotEqual(fallback.id, self.resolve().id)

    def test_environment_and_explicit_selection_override_default(self):
        env = {node.NETWORK_ENV: "qwen3.5-122b-a10b"}
        self.assertEqual(self.resolve(environ=env).model["name"], "Qwen3.5-122B-A10B")
        self.assertEqual(self.resolve("qwen3-8b", env).model["name"], "Qwen3-8B")

    def test_direct_discovery_uses_the_same_default(self):
        self.assertEqual(self.resolve("dht").id, self.resolve().id)

    def test_node_count_presets_remain_explicit(self):
        self.assertEqual(model_for_nodes(2), "qwen3-8b")
        self.assertEqual(model_for_nodes(3), "qwen3.5-35b-a3b")
        self.assertEqual(model_for_nodes(7), "qwen3.5-122b-a10b")


if __name__ == "__main__":
    unittest.main()
