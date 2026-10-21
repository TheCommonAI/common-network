"""Split-model tests on a tiny random Qwen3 checkpoint. All offline.

Run from the repository root:
    python3 -m unittest commonnet.tests.test_split -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from commonnet.block import BlockRange  # noqa: E402
from commonnet.client import BlockEndpoint, Client, LocalTransport  # noqa: E402

TOL_FP32 = 1e-4
PROMPT = [5, 17, 301, 42, 9, 133, 77]
NEW_TOKENS = 12


def tiny_config(tie: bool) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        tie_word_embeddings=tie,
        # 0.1 rather than the default 0.02: with 0.02 the random model's greedy
        # output collapses to one repeated token, which tests very little.
        initializer_range=0.1,
    )


def make_checkpoint(directory: str, tie: bool = False, max_shard_size: str = "5GB", seed: int = 0) -> None:
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(tiny_config(tie)).float()
    # Keep greedy decoding free of an EOS stop so both sides run the same number of steps.
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = 0
    model.save_pretrained(directory, max_shard_size=max_shard_size)


def route_for(boundaries: list[int]) -> list[BlockEndpoint]:
    """Route [0, b1, b2, ..., L] into endpoints named block0, block1, ..."""
    return [
        BlockEndpoint(name=f"block{i}", start=a, end=b)
        for i, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:]))
    ]


class RecordingTransport(LocalTransport):
    """LocalTransport that logs every forward call, to check what crosses the wire."""

    def __init__(self, blocks):
        super().__init__(blocks)
        self.calls: list[tuple[str, int, int]] = []  # (endpoint name, start_pos, tokens sent)

    def forward(self, endpoint, session_id, hidden, start_pos):
        self.calls.append((endpoint.name, start_pos, hidden.shape[1]))
        return super().forward(endpoint, session_id, hidden, start_pos)


class SplitTestBase(unittest.TestCase):
    tie = False
    max_shard_size = "5GB"

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="commonnet-ckpt-")
        cls.ckpt = os.path.join(cls.tmp, "tiny-qwen3")
        make_checkpoint(cls.ckpt, tie=cls.tie, max_shard_size=cls.max_shard_size)
        cls.reference = Qwen3ForCausalLM.from_pretrained(cls.ckpt, dtype=torch.float32).eval()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def build(self, boundaries, device="cpu", dtype=torch.float32):
        route = route_for(boundaries)
        blocks = {
            ep.name: BlockRange(self.ckpt, ep.start, ep.end, device=device, dtype=dtype) for ep in route
        }
        transport = RecordingTransport(blocks)
        client = Client(self.ckpt, transport, device=device, dtype=dtype)
        return client, route, blocks, transport


class TestSplitFp32(SplitTestBase):
    def test_split_logits_match_full_model(self):
        ids = torch.tensor([PROMPT])
        with torch.inference_mode():
            full = self.reference(ids).logits
        spread = float(full.std())
        for boundaries in ([0, 2, 4], [0, 1, 3, 4]):
            with self.subTest(boundaries=boundaries):
                client, route, _, _ = self.build(boundaries)
                split = client.logits(route, ids)
                diff = float((split - full).abs().max())
                print(f"\n[fp32 cpu] ranges={boundaries} max|split-full| logits = {diff:.3e} (logit std {spread:.3e})")
                self.assertLess(diff, TOL_FP32)
                self.assertGreater(spread, 1e-2, "logits too flat to make the comparison meaningful")

    def test_comparison_detects_a_wrong_block(self):
        # Negative control: if one block's weights differ, the comparison must fail.
        ids = torch.tensor([PROMPT])
        with torch.inference_mode():
            full = self.reference(ids).logits
        client, route, blocks, _ = self.build([0, 2, 4])
        with torch.no_grad():
            blocks["block1"].layers[0].mlp.down_proj.weight.mul_(1.5)
        diff = float((client.logits(route, ids) - full).abs().max())
        print(f"\n[control] perturbed block1 -> max|split-full| = {diff:.3e}")
        self.assertGreater(diff, 1e-2)

    def test_split_logits_match_with_one_block_too(self):
        ids = torch.tensor([PROMPT])
        with torch.inference_mode():
            full = self.reference(ids).logits
        client, route, _, _ = self.build([0, 4])
        diff = float((client.logits(route, ids) - full).abs().max())
        self.assertLess(diff, TOL_FP32)

    def test_generate_matches_full_model_greedy(self):
        with torch.inference_mode():
            hf = self.reference.generate(
                torch.tensor([PROMPT]),
                max_new_tokens=NEW_TOKENS,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=0,
            )
        hf_new = hf[0, len(PROMPT):].tolist()
        client, route, _, _ = self.build([0, 2, 4])
        ours = client.generate(route, PROMPT, NEW_TOKENS)
        print(f"\n[fp32 cpu] greedy tokens HF={hf_new} split={ours}")
        self.assertEqual(ours, hf_new)

    def test_second_step_reuses_kv_cache(self):
        client, route, blocks, transport = self.build([0, 2, 4])
        steps = 4
        client.generate(route, PROMPT, steps)
        n = len(PROMPT)
        # Step 0 sends the whole prompt at position 0. Step k>0 sends one token
        # at position n + k - 1. Each step visits every block in route order.
        expected = []
        for step in range(steps):
            start, toks = (0, n) if step == 0 else (n + step - 1, 1)
            expected += [(ep.name, start, toks) for ep in route]
        self.assertEqual(transport.calls, expected)
        # Across the whole run each block receives the prompt once plus one token per later step.
        per_block = {ep.name: sum(t for name, _, t in transport.calls if name == ep.name) for ep in route}
        self.assertEqual(per_block, {ep.name: n + steps - 1 for ep in route})

    def test_sessions_are_closed_after_generate(self):
        client, route, blocks, _ = self.build([0, 2, 4])
        client.generate(route, PROMPT, 3)
        for block in blocks.values():
            self.assertEqual(block.open_sessions(), 0)

    def test_cache_holds_prompt_plus_generated_tokens(self):
        # Inspect a block mid-generation by running the steps by hand.
        client, route, blocks, _ = self.build([0, 2, 4])
        sid = "manual"
        ids = torch.tensor([PROMPT])
        with torch.inference_mode():
            hidden = client.embed(ids)
            for ep in route:
                hidden = blocks[ep.name].forward(sid, hidden, 0)
            self.assertEqual(blocks["block0"].session_length(sid), len(PROMPT))
            self.assertEqual(blocks["block1"].session_length(sid), len(PROMPT))
            hidden = client.embed(torch.tensor([[7]]))
            for ep in route:
                hidden = blocks[ep.name].forward(sid, hidden, len(PROMPT))
            self.assertEqual(blocks["block0"].session_length(sid), len(PROMPT) + 1)
        for block in blocks.values():
            block.close(sid)
            block.close(sid)  # closing twice is a no-op

    def test_wrong_start_pos_is_rejected(self):
        client, route, blocks, _ = self.build([0, 4])
        block = blocks["block0"]
        with torch.inference_mode():
            hidden = client.embed(torch.tensor([PROMPT]))
            block.forward("s", hidden, 0)
            with self.assertRaises(ValueError):
                block.forward("s", hidden[:, :1], 3)  # cache holds 7, not 3
            with self.assertRaises(KeyError):
                block.forward("unknown", hidden[:, :1], 5)

    def test_block_loads_only_its_layers(self):
        block = BlockRange(self.ckpt, 2, 4)
        expected = {
            f"model.layers.{i}.{suffix}"
            for i in (2, 3)
            for suffix in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            )
        }
        self.assertEqual(set(block.loaded_names), expected)
        self.assertEqual(len(block.loaded_names), len(set(block.loaded_names)))

    def test_route_must_be_contiguous_and_complete(self):
        with self.assertRaises(ValueError):
            Client.check_route(route_for([0, 2, 3]), 4)  # does not reach layer 4
        with self.assertRaises(ValueError):
            Client.check_route([BlockEndpoint("a", 0, 2), BlockEndpoint("b", 3, 4)], 4)  # gap
        with self.assertRaises(ValueError):
            Client.check_route([BlockEndpoint("a", 0, 2), BlockEndpoint("b", 1, 4)], 4)  # overlap

    def test_endpoint_range_must_match_block(self):
        client, _, blocks, _ = self.build([0, 2, 4])
        bad = [BlockEndpoint("block0", 0, 3), BlockEndpoint("block1", 3, 4)]
        with self.assertRaises(ValueError):
            client.generate(bad, PROMPT, 1)


class TestSplitFp32Sharded(TestSplitFp32):
    """Same tests, with the checkpoint sharded into several safetensors files and an index."""

    max_shard_size = "60KB"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        index = os.path.join(cls.ckpt, "model.safetensors.index.json")
        if not os.path.exists(index):
            raise unittest.SkipTest("checkpoint was not sharded; raise the shard limit")


class TestSplitFp32Tied(TestSplitFp32):
    """Tied embeddings: lm_head.weight is absent and the head reuses embed_tokens."""

    tie = True


class TestSplitMps(SplitTestBase):
    device = "mps"

    @classmethod
    def setUpClass(cls):
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            raise unittest.SkipTest("MPS is not available")
        super().setUpClass()

    def test_split_fp32_on_mps(self):
        ids = torch.tensor([PROMPT])
        with torch.inference_mode():
            full_cpu = self.reference(ids).logits
            full_mps = Qwen3ForCausalLM.from_pretrained(self.ckpt, dtype=torch.float32).eval().to("mps")(
                ids.to("mps")
            ).logits.cpu()
        client, route, _, _ = self.build([0, 2, 4], device="mps", dtype=torch.float32)
        split = client.logits(route, ids).cpu()
        d_full = float((split - full_mps).abs().max())
        d_cpu = float((split - full_cpu).abs().max())
        print(f"\n[fp32 mps] max|split_mps - full_mps| = {d_full:.3e}; max|split_mps - full_cpu| = {d_cpu:.3e}")
        self.assertLess(d_full, TOL_FP32)
        with torch.inference_mode():
            hf = self.reference.generate(
                torch.tensor([PROMPT]), max_new_tokens=NEW_TOKENS, do_sample=False, eos_token_id=None, pad_token_id=0
            )
        ours = client.generate(route, PROMPT, NEW_TOKENS)
        print(f"[fp32 mps] greedy tokens HF_cpu={hf[0, len(PROMPT):].tolist()} split_mps={ours}")
        self.assertEqual(ours, hf[0, len(PROMPT):].tolist())

    def test_split_fp16_on_mps(self):
        ids = torch.tensor([PROMPT])
        with torch.inference_mode():
            full_cpu = self.reference(ids).logits
            full_mps16 = Qwen3ForCausalLM.from_pretrained(self.ckpt, dtype=torch.float16).eval().to("mps")(
                ids.to("mps")
            ).logits.float().cpu()
        client, route, _, _ = self.build([0, 2, 4], device="mps", dtype=torch.float16)
        split = client.logits(route, ids).cpu()
        d_full = float((split - full_mps16).abs().max())
        d_cpu = float((split - full_cpu).abs().max())
        print(f"\n[fp16 mps] max|split_mps - full_mps (both fp16)| = {d_full:.3e}; max|split_mps_fp16 - full_cpu_fp32| = {d_cpu:.3e}")
        self.assertTrue(torch.isfinite(split).all())
        self.assertLess(d_full, 1e-2)


if __name__ == "__main__":
    unittest.main()
