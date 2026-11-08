"""Real tiny Qwen3.5 MoE layers: split parity, expert routing and packed MLX checkpoints."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

from commonnet.block import BlockRange
from commonnet.client import BlockEndpoint, Client, LocalTransport
from commonnet.tests.qwen35_tiny import tiny_text_config
from commonnet.weights import CheckpointReader


def checkpoint(directory):
    torch.manual_seed(42)
    values = tiny_text_config(8).to_dict()
    values.pop("model_type", None)
    values.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64, linear_num_value_heads=8,
                  shared_expert_intermediate_size=64)
    config = Qwen3_5MoeTextConfig(**values)
    model = Qwen3_5MoeForCausalLM(config).float().eval()
    wrapper = Qwen3_5MoeConfig(text_config=config.to_dict(), vision_config={
        "depth": 1, "hidden_size": 32, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 64})
    wrapper.save_pretrained(directory)
    weights = {"model.language_model." + name[len("model."):] if name.startswith("model.") else name:
               value.detach().contiguous() for name, value in model.state_dict().items()}
    save_file(weights, str(Path(directory) / "model.safetensors"))
    return model


class MoeParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = cls.directory.name
        cls.reference = checkpoint(cls.path)
        cls.prompt = [5, 17, 301, 42, 9]

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_split_logits_and_incremental_decode_match_reference(self):
        for bounds in ([0, 3, 8], [0, 1, 2, 3, 4, 5, 6, 8]):
            with self.subTest(bounds=bounds), torch.inference_mode():
                endpoints = [BlockEndpoint(str(i), a, b) for i, (a, b) in enumerate(zip(bounds, bounds[1:]))]
                blocks = {ep.name: BlockRange(self.path, ep.start, ep.end) for ep in endpoints}
                client = Client(self.path, LocalTransport(blocks))
                got = client.logits(endpoints, torch.tensor([self.prompt]))
                expected = self.reference(torch.tensor([self.prompt])).logits
                self.assertLess((got - expected).abs().max().item(), 1e-4)
                hidden = client.embed(torch.tensor([self.prompt]))
                for ep in endpoints:
                    hidden = blocks[ep.name].forward("incremental", hidden, 0)
                hidden = client.embed(torch.tensor([[7]]))
                for ep in endpoints:
                    hidden = blocks[ep.name].forward("incremental", hidden, len(self.prompt))
                expected = self.reference(torch.tensor([self.prompt + [7]])).logits[:, -1:]
                self.assertLess((client._logits(hidden) - expected).abs().max().item(), 1e-4)
                self.assertEqual(CheckpointReader(self.path).arch, "qwen3_5_moe")
                with self.assertRaises(ValueError):
                    next(iter(blocks.values())).truncate("incremental", 1)

    def test_expert_slice_loader_preserves_routing_without_loading_whole_fused_tensors(self):
        from commonnet.backends.nf4_block import load_experts
        reader = CheckpointReader(self.path)
        original = self.reference.model.layers[0].mlp.experts
        shapes = []
        def linear(weight):
            shapes.append(tuple(weight.shape))
            result = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
            result.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
            return result
        loaded = load_experts(reader, reader.prefix + "layers.0.mlp.experts", original, linear)
        hidden = torch.randn(7, 64)
        experts = torch.tensor([[0, 2], [1, 3], [2, 1], [3, 0], [0, 1], [2, 3], [3, 1]])
        weights = torch.softmax(torch.randn(7, 2), dim=-1)
        with torch.inference_mode():
            self.assertLess((loaded(hidden, experts, weights) - original(hidden, experts, weights)).abs().max().item(), 1e-5)
        self.assertEqual(len(shapes), 12)
        self.assertTrue(all(len(shape) == 2 for shape in shapes))

    def test_seven_http_peers_generate_reference_tokens(self):
        from commonnet.net.factory import build_block
        from commonnet.net.server import BlockServer
        from commonnet.net.transport import HttpTransport
        bounds = [0, 1, 2, 3, 4, 5, 6, 8]
        endpoints = [BlockEndpoint(str(i), a, b) for i, (a, b) in enumerate(zip(bounds, bounds[1:]))]
        servers = []
        transport = None
        token = "m" * 32
        try:
            for ep in endpoints:
                block = build_block(ep.start, ep.end, backend="torch", checkpoint=self.path,
                                    environ={"COMMONNET_DTYPE": "float32"})
                servers.append(BlockServer(block, token=token, rtc_enabled=False, allow_chain_private=True).start())
            transport = HttpTransport({ep.name: server.url for ep, server in zip(endpoints, servers)},
                                      token=token, activation_codec="float")
            got = Client(self.path, transport).generate(endpoints, self.prompt, 4)
            ids = list(self.prompt)
            with torch.inference_mode():
                for _ in range(4):
                    ids.append(int(self.reference(torch.tensor([ids])).logits[0, -1].argmax()))
            self.assertEqual(got, ids[-4:])
        finally:
            if transport is not None:
                transport.shutdown()
            for server in reversed(servers):
                server.stop()

    @unittest.skipUnless(torch.cuda.is_available(), "needs NVIDIA CUDA hardware")
    def test_cuda_nf4_split_tracks_unquantized_reference(self):
        from commonnet.backends.nf4_block import Nf4BlockRange
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        endpoints = [BlockEndpoint("first", 0, 3), BlockEndpoint("last", 3, 8)]
        blocks = {ep.name: Nf4BlockRange(self.path, ep.start, ep.end, dtype=dtype) for ep in endpoints}
        client = Client(self.path, LocalTransport(blocks))
        got = client.logits(endpoints, torch.tensor([self.prompt])).float()
        with torch.inference_mode():
            expected = self.reference(torch.tensor([self.prompt])).logits.float()
        self.assertTrue(torch.isfinite(got).all())
        self.assertLess(((got - expected).norm() / expected.norm()).item(), 0.15)

    def test_mlx_float_and_packed_splits(self):
        try:
            import mlx.core as mx
            import mlx.nn as nn
            from mlx.utils import tree_flatten
            from mlx_lm.models.qwen3_5_moe import Model, ModelArgs
        except ImportError:
            self.skipTest("MLX is unavailable")
        if not mx.metal.is_available():
            self.skipTest("Metal is unavailable")
        from commonnet.backends.mlx_block import MlxBlockRange
        from commonnet.client4 import Client4
        endpoints = [BlockEndpoint("first", 0, 3), BlockEndpoint("last", 3, 8)]
        blocks = {ep.name: MlxBlockRange(self.path, ep.start, ep.end, bits=None, dtype=mx.float32) for ep in endpoints}
        client = Client(self.path, LocalTransport(blocks))
        with torch.inference_mode():
            want = self.reference(torch.tensor([self.prompt])).logits
        self.assertLess((client.logits(endpoints, torch.tensor([self.prompt])) - want).abs().max().item(), 1e-3)

        config = json.loads((Path(self.path) / "config.json").read_text())
        full = Model(ModelArgs.from_dict(config))
        full.load_weights(list(full.sanitize(mx.load(str(Path(self.path) / "model.safetensors"))).items()), strict=True)
        nn.quantize(full, group_size=64, bits=4)
        mx.eval(full.parameters())
        with tempfile.TemporaryDirectory() as packed:
            config["quantization"] = {"bits": 4, "group_size": 64, "mode": "affine"}
            (Path(packed) / "config.json").write_text(json.dumps(config))
            mx.save_safetensors(str(Path(packed) / "model.safetensors"), dict(tree_flatten(full.parameters())))
            blocks = {ep.name: MlxBlockRange(packed, ep.start, ep.end, dtype=mx.float32) for ep in endpoints}
            client = Client4(packed, LocalTransport(blocks), fmt="mlx4", hidden_dtype=torch.float32)
            got = client.logits(endpoints, torch.tensor([self.prompt]))
            import numpy as np
            expected = torch.from_numpy(np.array(full(mx.array([self.prompt])).astype(mx.float32)))
            self.assertLess((got - expected).abs().max().item(), 5e-3)


if __name__ == "__main__":
    unittest.main()

