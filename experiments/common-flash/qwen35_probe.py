"""Offline Common Flash correctness probe on a tiny Qwen3.5 MoE.

Run with a current Common source tree and its torch/transformers dependencies:
  PYTHONPATH=/path/to/current/common-network python qwen35_probe.py --out result.json

Uses random tiny weights and controlled oracle drafts, not the trained 35B model.
This tests cache repair and token parity, never model quality or real-world speed.
No production cache implementation or speculation guard is changed.
"""
import argparse
import copy
import json
import tempfile
import time
from pathlib import Path

import torch

from commonnet import speculative as flash
from commonnet.bench.synth import OracleDrafter
from commonnet.block import BlockRange
from commonnet.client import BlockEndpoint, Client, LocalTransport
from commonnet.tests.test_qwen35_moe import checkpoint


class CheckpointReplayTransport(LocalTransport):
    """Experimental in-process rollback: restore before verification, replay kept prefix.

    Attention KV, convolution and recurrent states are copied together. Rejected
    tokens never survive into the next round. Copies are deliberately unoptimised.
    """

    def __init__(self, blocks):
        super().__init__(blocks)
        self.saved = {}
        self.rollbacks = 0
        self.replayed_positions = 0

    def supports_truncate(self, endpoints):
        return True

    def forward(self, endpoint, session_id, hidden, start_pos):
        block = self._block(endpoint)
        session = block._sessions.get(session_id) if start_pos else None
        self.saved[endpoint.name, session_id] = (copy.deepcopy(session), hidden.clone(), start_pos)
        return block.forward(session_id, hidden, start_pos)

    def truncate(self, endpoint, session_id, length, defer=False):
        block = self._block(endpoint)
        before, hidden, start = self.saved[endpoint.name, session_id]
        if not start <= length <= start + hidden.shape[1]:
            raise ValueError("rollback exceeds the saved verification chunk")
        if before is None:
            block.close(session_id)
        else:
            block._sessions[session_id] = copy.deepcopy(before)
        keep = length - start
        if keep:
            block.forward(session_id, hidden[:, :keep], start)
        self.rollbacks += 1
        self.replayed_positions += keep

    def close(self, endpoint, session_id):
        self.saved.pop((endpoint.name, session_id), None)
        super().close(endpoint, session_id)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=24)
    args = parser.parse_args()
    if args.tokens < 2:
        parser.error("--tokens must be at least 2")
    torch.set_num_threads(1)
    rows = []
    with tempfile.TemporaryDirectory(prefix="common-flash-qwen35-") as directory, torch.inference_mode():
        reference = checkpoint(directory)
        endpoints = [BlockEndpoint("first", 0, 3), BlockEndpoint("last", 3, 8)]
        blocks = {ep.name: BlockRange(directory, ep.start, ep.end) for ep in endpoints}
        transport = CheckpointReplayTransport(blocks)
        client = Client(directory, transport)
        stock = next(iter(blocks.values()))
        stock.forward("guard-check", client.embed(torch.tensor([[5, 17, 301]])), 0)
        try:
            stock.truncate("guard-check", 1)
        except (NotImplementedError, ValueError):
            stock_refuses_rollback = True
        else:
            raise AssertionError("stock cache unexpectedly permits rollback")
        finally:
            stock.close("guard-check")

        for prompt in ([5, 17, 301, 42, 9], [42, 6, 73, 88, 21, 19, 2]):
            # Check the split target against the unsplit reference, too.
            ids = list(prompt)
            for _ in range(args.tokens + 9):
                ids.append(int(reference(torch.tensor([ids])).logits[0, -1].argmax()))
            truth = ids[len(prompt):]
            plain = client.generate(endpoints, prompt, args.tokens)
            assert plain == truth[:args.tokens], "split target differs from reference"
            for acceptance in (0.0, 0.5, 1.0):
                for k in (1, 4, 8):
                    before = transport.rollbacks
                    replayed = transport.replayed_positions
                    stats = flash.SpecStats()
                    draft = OracleDrafter(len(prompt), truth, acceptance,
                                          vocab=client.config.vocab_size, seed=7)
                    started = time.perf_counter()
                    output = list(flash.generate_stream(client, endpoints, prompt, args.tokens,
                                                       drafter=draft, k=k, adaptive=False, stats=stats))
                    assert output == plain, f"greedy parity failed: acceptance={acceptance}, k={k}"
                    assert stats.mode == "speculative", stats.mode
                    assert not transport.saved, "saved state leaked after generation"
                    if acceptance == 0:
                        assert transport.rollbacks > before, "negative control never rolled back"
                    if acceptance == 1:
                        assert transport.rollbacks == before, "perfect draft unexpectedly rejected"
                    row = {"prompt_tokens": len(prompt), "k": k, "oracle_accept_probability": acceptance,
                           "exact_token_match": True, "rounds": stats.rounds,
                           "drafted": stats.drafted, "accepted": stats.accepted,
                           "block_rollbacks": transport.rollbacks - before,
                           "replayed_block_positions": transport.replayed_positions - replayed,
                           "local_probe_seconds": round(time.perf_counter() - started, 4)}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    result = {"scope": "tiny random eight-layer Qwen3.5 MoE; two in-process blocks; float32 CPU",
              "method": "Common Flash verifier with experimental checkpoint and accepted-prefix replay",
              "stock_refuses_rollback": stock_refuses_rollback,
              "not_measured": ["trained 35B model", "real drafter acceptance", "4-bit decoding",
                               "network speed", "end-to-end speedup"],
              "passed": len(rows), "cases": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Passed {len(rows)} exact-match cases; saved {args.out}")


if __name__ == "__main__":
    run()
