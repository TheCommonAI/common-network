"""Planned layer assignment against the manual split: same prompt, same tokens. Tiny random Qwen3, offline.

Run from the repository root:
    python3 -m unittest commonnet.tests.test_assign_split -v
"""

from __future__ import annotations

import unittest

import torch  # noqa: E402

from commonnet.block import BlockRange  # noqa: E402
from commonnet.client import BlockEndpoint, Client, LocalTransport  # noqa: E402
from commonnet.mesh import assign  # noqa: E402
from commonnet.mesh.assign import changes  # noqa: E402
from commonnet.mesh.manifest import Manifest  # noqa: E402
from commonnet.mesh.records import Identity  # noqa: E402
from commonnet.tests.test_split import NEW_TOKENS, PROMPT, SplitTestBase, route_for  # noqa: E402

MANIFEST = Manifest.from_dict({
    "id": "tiny", "model": {"name": "Tiny", "num_layers": 4, "repos": {"cpu": "x/y"}},
    "boundaries": [0, 2, 4], "gib_per_layer": {"cpu": 1.0}, "fixed_gib": 0.0})


def record(who: Identity, gib: float, window=None) -> dict:
    return who.sign_record({"network": "tiny", "url": "http://127.0.0.1:1", "token": "t", "window": window,
                            "backend": "cpu", "state": "up", "gib": gib})


class PlannedSplit(SplitTestBase):
    def endpoints_for(self, result) -> list[BlockEndpoint]:
        return [BlockEndpoint(f"block{i}", a, b) for i, (a, b) in enumerate(result.chain)]

    def generate(self, endpoints) -> list[int]:
        blocks = {ep.name: BlockRange(self.ckpt, ep.start, ep.end, device="cpu", dtype=torch.float32)
                  for ep in endpoints}
        client = Client(self.ckpt, LocalTransport(blocks), device="cpu", dtype=torch.float32)
        return client.generate(endpoints, PROMPT, NEW_TOKENS)

    def reference_tokens(self) -> list[int]:
        with torch.inference_mode():
            out = self.reference.generate(torch.tensor([PROMPT]), max_new_tokens=NEW_TOKENS, do_sample=False,
                                          eos_token_id=None, pad_token_id=0)
        return out[0, len(PROMPT):].tolist()

    def test_planned_split_matches_the_manual_split_and_the_full_model(self):
        a, b = Identity.generate(), Identity.generate()
        result = assign.plan(MANIFEST, [record(a, 2.0), record(b, 2.0)])
        self.assertEqual(result.chain, ((0, 2), (2, 4)))
        planned = self.generate(self.endpoints_for(result))
        manual = self.generate(route_for([0, 2, 4]))
        print(f"\n[planned vs manual] planned={planned} manual={manual}")
        self.assertEqual(planned, manual)
        self.assertEqual(planned, self.reference_tokens())
        self.assertGreater(len(set(planned)), 1, "greedy output is one repeated token; the comparison means little")

    def test_a_joining_peer_changes_nothing_and_the_output_is_unchanged(self):
        a, b, joiner = Identity.generate(), Identity.generate(), Identity.generate()
        first = assign.plan(MANIFEST, [record(a, 2.0), record(b, 2.0)])
        live = [record(a, 2.0, list(first.windows[a.peer])), record(b, 2.0, list(first.windows[b.peer])),
                record(joiner, 2.0)]
        second = assign.plan(MANIFEST, live)
        self.assertEqual(changes(first.windows, second), {})
        self.assertEqual(self.generate(self.endpoints_for(second)), self.reference_tokens())

    def test_a_dropped_chain_peer_is_replaced_by_a_spare_and_the_output_is_unchanged(self):
        a, b, spare = Identity.generate(), Identity.generate(), Identity.generate()
        before = {a.peer: (0, 2), b.peer: (2, 4), spare.peer: (0, 2)}
        live = [record(a, 2.0, [0, 2]), record(spare, 2.0, [0, 2])]  # b dropped
        after = assign.plan(MANIFEST, live)
        self.assertEqual(after.chain, ((0, 2), (2, 4)))
        moved = changes({k: v for k, v in before.items() if k != b.peer}, after)
        self.assertEqual(len(moved), 1)
        self.assertEqual(self.generate(self.endpoints_for(after)), self.reference_tokens())


if __name__ == "__main__":
    unittest.main()
