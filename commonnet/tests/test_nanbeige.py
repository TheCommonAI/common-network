"""Default network placement and Llama split inference."""
import tempfile
import unittest

from commonnet.mesh import placement
from commonnet.node import resolve_network


class Nanbeige(unittest.TestCase):
    def test_default_routes_on_two_and_three_mlx_peers(self):
        manifest, _ = resolve_network("dht", [], {})
        self.assertEqual(manifest.model["name"], "Nanbeige4.3-33B")
        for count, budget in [(2, 10.5), (3, 7.7)]:
            with self.subTest(count=count):
                peers = []
                for i in range(count):
                    window = placement.choose_window(manifest, peers, budget, "mlx")
                    peers.append(dict(peer=str(i), backend="mlx", window=window, state="up"))
                self.assertTrue(all(placement.coverage(manifest, peers)))
                self.assertEqual(len(placement.best_route(manifest, peers)), count)

    def test_llama_split_matches_full_logits(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from commonnet.block import BlockRange
        from commonnet.client import Client, LocalTransport
        from commonnet.tests.test_split import route_for

        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=64, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        )).eval()
        ids = torch.tensor([[1, 5, 9]])
        with tempfile.TemporaryDirectory() as checkpoint, torch.inference_mode():
            model.save_pretrained(checkpoint)
            route = route_for([0, 1, 2])
            blocks = {ep.name: BlockRange(checkpoint, ep.start, ep.end, device="cpu", dtype=torch.float32)
                      for ep in route}
            client = Client(checkpoint, LocalTransport(blocks), device="cpu", dtype=torch.float32)
            reference = model(ids).logits
            torch.testing.assert_close(client.logits(route, ids), reference)
            try:
                import mlx.core as mx
            except ImportError:
                return
            if mx.metal.is_available():
                from commonnet.backends.mlx_block import MlxBlockRange
                blocks = {ep.name: MlxBlockRange(checkpoint, ep.start, ep.end, bits=None)
                          for ep in route}
                client = Client(checkpoint, LocalTransport(blocks), device="cpu", dtype=torch.float32)
                torch.testing.assert_close(client.logits(route, ids), reference, atol=0.002, rtol=0.002)
