"""Real tiny MoE verification and hybrid drafter state after rejected drafts. Offline."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from commonnet import speculative, serve
from commonnet.bench.synth import OracleDrafter
from commonnet.block import BlockRange
from commonnet.client import BlockEndpoint, Client, LocalTransport
from commonnet.hybrid_draft import HybridTorchDrafter
from commonnet.recurrent import CAPABILITY
from commonnet.tests.test_qwen35_moe import checkpoint
from commonnet.tests.qwen35_tiny import make_qwen35_checkpoint
from commonnet.tests.test_default_draft import backend, FakeTransport


class SpecTransport(LocalTransport):
    rollbacks = 0

    def capabilities(self, ep):
        return ("chain_v1", CAPABILITY)

    def supports_truncate(self, endpoints):
        return True

    def truncate(self, ep, session_id, length, defer=False):
        self.rollbacks += 1
        self._block(ep).truncate(session_id, length)


class RecurrentFlash(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = cls.directory.name
        cls.reference = checkpoint(cls.path)
        cls.eps = [BlockEndpoint("a", 0, 3), BlockEndpoint("b", 3, 8)]

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def rig(self):
        blocks = {ep.name: BlockRange(self.path, ep.start, ep.end) for ep in self.eps}
        transport = SpecTransport(blocks)
        return Client(self.path, transport), transport

    def test_real_moe_tokens_match_across_acceptance_and_draft_lengths(self):
        with torch.inference_mode():
            for prompt in ([5, 17, 301, 42, 9], [42, 6, 73, 88, 21, 19, 2]):
                ids = list(prompt)
                for _ in range(33):
                    ids.append(int(self.reference(torch.tensor([ids])).logits[0, -1].argmax()))
                truth = ids[len(prompt):]
                for acceptance in (0, 0.5, 1):
                    for k in (1, 4, 8):
                        with self.subTest(acceptance=acceptance, k=k, prompt=prompt):
                            client, transport = self.rig()
                            stats = speculative.SpecStats()
                            draft = OracleDrafter(len(prompt), truth, acceptance, vocab=512, seed=7)
                            output = list(speculative.generate_stream(client, self.eps, prompt, 24,
                                          drafter=draft, k=k, adaptive=False, stats=stats))
                            self.assertEqual(output, truth[:24])
                            self.assertEqual(stats.mode, "speculative")
                            if acceptance == 0:
                                self.assertGreater(transport.rollbacks, 0)
                            for block in transport.blocks.values():
                                self.assertEqual(block.open_sessions(), 0)
                                self.assertFalse(getattr(block, "_replay", {}))

    def test_rollback_replay_is_idempotent_and_matches_clean_state(self):
        client, transport = self.rig()
        block = transport.blocks["a"]
        with torch.inference_mode():
            hidden = client.embed(torch.tensor([[5, 17, 301, 42, 9]]))
            block.forward("cut", hidden, 0)
            block.forward("cut", hidden[:, :4], 5)
            block.truncate("cut", 7)
            block.truncate("cut", 7)
            block.truncate("cut", 6)
            block.forward("clean", hidden, 0)
            block.forward("clean", hidden[:, :1], 5)
            got = block.forward("cut", hidden[:, :1], 6)
            want = block.forward("clean", hidden[:, :1], 6)
            self.assertTrue(torch.allclose(got, want, atol=1e-5))

    def test_older_peer_refuses_before_any_draft(self):
        client, transport = self.rig()
        with patch.object(transport, "capabilities", return_value=("chain_v1",)):
            ok, reason = speculative.can_speculate(client, self.eps)
        self.assertFalse(ok)
        self.assertIn("recurrent", reason)

    def test_network_wire_adapter_advertises_and_repairs_rollback(self):
        from commonnet.net.factory import build_block
        from commonnet.net.transport import to_wire, from_wire
        block = build_block(0, 3, backend="torch", checkpoint=self.path,
                            environ={"COMMONNET_DTYPE": "float32", "COMMONNET_IDS": "0"})
        self.assertIn(CAPABILITY, block.caps)
        client, _ = self.rig()
        with torch.inference_mode():
            hidden = client.embed(torch.tensor([[5, 17, 301, 42, 9]]))
            block.forward("cut", to_wire(hidden), 0)
            block.forward("cut", to_wire(hidden[:, :4]), 5)
            block.truncate("cut", 7)
            block.forward("clean", to_wire(hidden), 0)
            block.forward("clean", to_wire(hidden[:, :2]), 5)
            got = from_wire(block.forward("cut", to_wire(hidden[:, :1]), 7))
            want = from_wire(block.forward("clean", to_wire(hidden[:, :1]), 7))
            self.assertTrue(torch.allclose(got, want, atol=1e-5))
        block.close("cut")
        block.close("clean")
        self.assertFalse(block.inner._replay)

    def test_lazy_drafter_waits_for_all_peers_to_upgrade(self):
        called = []
        b = backend(FakeTransport(caps=("chain_v1",)), factory=lambda: called.append(1))
        b.recurrent_rollback = True
        endpoints, transport = b.network.plan()
        self.assertIsNone(b._lazy_drafter(endpoints, transport, 0))
        self.assertFalse(called)
        transport.caps = ("chain_v1", CAPABILITY)
        b._lazy_drafter(endpoints, transport, 0)
        self.assertEqual(called, [1])

    def test_35b_manifest_enables_matching_family_drafter(self):
        model = json.loads((Path(serve.__file__).parent / "networks/qwen3.5-35b-a3b.json").read_text())["model"]
        self.assertNotIn("no_speculation", model)
        self.assertEqual(model["speculative_rollback"], CAPABILITY)
        self.assertEqual(model["draft_repos"]["torch"], "Qwen/Qwen3.5-0.8B")

    def test_hybrid_drafter_repairs_rejected_and_accepted_context(self):
        with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
            model = make_qwen35_checkpoint(directory)
            drafter = HybridTorchDrafter("tiny", model=model)
            prompt = [5, 17, 301, 42]
            previous = drafter.propose(prompt, 4)
            for keep in (0, 2, 4, 1):
                context = prompt + previous[:keep] + [(previous[keep % 4] + 1) % 512]
                expected = []
                for _ in range(4):
                    expected.append(int(model(torch.tensor([context + expected])).logits[0, -1].argmax()))
                self.assertEqual(drafter.propose(context, 4), expected)
                prompt, previous = context, expected


if __name__ == "__main__":
    unittest.main()
