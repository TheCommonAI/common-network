"""Token-only client links and final-peer speculative verification over real HTTP."""
from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import torch

from commonnet import speculative as spec
from commonnet.bench import live
from commonnet.bench.synth import OracleDrafter, SynthBlock, SynthClient
from commonnet.client import BlockEndpoint, Client
from commonnet.client4 import Client4
from commonnet.net.client import Unreachable
from commonnet.net.server import BlockServer
from commonnet.net.transport import HttpTransport

LAYERS, HIDDEN, VOCAB = 4, 64, 256
ENDPOINTS = [BlockEndpoint("hop0", 0, 2), BlockEndpoint("hop1", 2, 4)]
PROMPT = [3, 10, 17, 24, 31, 38, 45]


class CaptureTransport(HttpTransport):
    def __init__(self, urls, **kw):
        # These tests verify exact baseline parity and float payload accounting.
        # The default int8 path has separate protocol smoke coverage.
        kw.setdefault("activation_codec", "float")
        super().__init__(urls, retries=0, sleep=lambda _: None, **kw)
        self.answers = []

    def chain_step(self, *a, **kw):
        reply = super().chain_step(*a, **kw)
        self.answers.append((dict(kw), reply))
        return reply


class ChainRig:
    def __init__(self, hidden=HIDDEN, *, single=False):
        self.hidden = hidden
        self.endpoints = [BlockEndpoint("hop0", 0, 4)] if single else ENDPOINTS
        self.servers = [BlockServer(SynthBlock(ep.start, ep.end, io_layers=LAYERS, hidden=hidden,
                                                vocab=VOCAB, wire_dtype="float32"),
                                    allow_chain_private=True).start() for ep in self.endpoints]

    def client(self, *, legacy=False):
        transport = CaptureTransport({ep.name: server.url for ep, server in zip(self.endpoints, self.servers)})
        client = SynthClient(transport, layers=LAYERS, hidden=self.hidden, vocab=VOCAB, dtype=torch.float32)
        client.ids_disabled = legacy
        return client

    def close(self):
        for srv in self.servers:
            srv.stop()


def traffic_comparison(*, hidden=4096, new_tokens=24, k_values=(4, 8)) -> dict:
    """Reproducible payload-byte comparison, not a WAN latency or speed benchmark.

    Captures every HTTP request and reply in this process, including peer-to-peer
    forwarding. Only /forward and /chain payload bytes are included; health and
    session-control calls are excluded in both modes.
    """
    rig = ChainRig(hidden)
    try:
        truth_client = rig.client(legacy=True)
        try:
            truth = truth_client.generate(rig.endpoints, PROMPT, new_tokens + max(k_values) + 2)
        finally:
            truth_client.transport.shutdown()
        rows = []
        for k in k_values:
            row = {"k": k, "new_tokens": new_tokens}
            for mode in ("hidden_relay", "token_chain"):
                # Every measured generation must create its forwarding clients
                # inside this capture; a pooled client could still point at the
                # previous generation's timing sink.
                for server in rig.servers:
                    for peer_client in server._chain_clients.values():
                        peer_client.close_connections()
                    server._chain_clients.clear()
                with live.Recording() as rec:
                    client = rig.client(legacy=mode == "hidden_relay")
                    stats = spec.SpecStats()
                    try:
                        tokens = list(spec.generate_stream(
                            client, rig.endpoints, PROMPT, new_tokens,
                            drafter=OracleDrafter(len(PROMPT), truth, 0.7, vocab=VOCAB, seed=11),
                            k=k, adaptive=False, stats=stats))
                    finally:
                        client.transport.shutdown()
                calls = [r for r in rec.sink if r["path"] in ("/forward", "/chain") and r["status"] == 200]
                ingress_owners = {id(c) for c in client.transport._clients.values()}
                ingress = [r for r in calls if r["owner"] in ingress_owners]
                row[mode] = {"payload_bytes": sum(r["bytes_out"] + r["bytes_in"] for r in calls),
                             "client_payload_bytes": sum(r["bytes_out"] + r["bytes_in"] for r in ingress),
                             "data_calls": len(calls), "rounds": stats.rounds,
                             "accepted": stats.accepted, "matches_plain": tokens == truth[:new_tokens]}
            row["payload_reduction"] = 1 - row["token_chain"]["payload_bytes"] / row["hidden_relay"]["payload_bytes"]
            row["client_payload_reduction"] = 1 - row["token_chain"]["client_payload_bytes"] / row["hidden_relay"]["client_payload_bytes"]
            rows.append(row)
        return {"model": "synthetic seeded causal model", "hidden_width": hidden,
                "stages": len(rig.endpoints), "wire_dtype": "float32", "measurement": "HTTP JSON payload bytes",
                "scope": "loopback; no WAN speed inference", "cases": rows}
    finally:
        rig.close()


class SpeculativeChain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = ChainRig()
        client = cls.rig.client(legacy=True)
        try:
            cls.truth = client.generate(cls.rig.endpoints, PROMPT, 54)
        finally:
            client.transport.shutdown()

    @classmethod
    def tearDownClass(cls):
        cls.rig.close()

    def run_spec(self, *, accept=1.0, k=4, n=24, stops=()):
        client = self.rig.client()
        self.addCleanup(client.transport.shutdown)
        client.transport.forward = mock.Mock(side_effect=AssertionError("client relayed hidden states"))
        client._embed = mock.Mock(side_effect=AssertionError("client embedded tokens"))
        client._head = mock.Mock(side_effect=AssertionError("client ran target head"))
        stats = spec.SpecStats()
        out = list(spec.generate_stream(client, self.rig.endpoints, PROMPT, n,
                                        drafter=OracleDrafter(len(PROMPT), self.truth, accept, vocab=VOCAB, seed=5),
                                        k=k, adaptive=False, stop_ids=stops, stats=stats))
        self.assertFalse(client.transport.forward.called)
        self.assertEqual(client.transport._chain_sessions, {})
        return out, client, stats

    def test_exact_plain_output_across_k_and_acceptance(self):
        for k in (1, 4, 8):
            for accept in (0.0, 0.6, 1.0):
                with self.subTest(k=k, accept=accept):
                    out, client, stats = self.run_spec(k=k, accept=accept)
                    self.assertEqual(out, self.truth[:24])
                    answers = [(kw, a) for kw, a in client.transport.answers if kw.get("drafts") is not None]
                    self.assertTrue(answers)
                    for kw, answer in answers:
                        self.assertEqual(answer["ids"][:-1], kw["drafts"][:answer["accepted"]])
                        self.assertEqual(len(answer["ids"]), answer["accepted"] + 1)
                        self.assertNotIn("hidden", answer)
                    self.assertEqual(stats.mode, "speculative")

    def test_stop_and_max_output_limits(self):
        for n in (1, 2, 7, 8, 9):
            with self.subTest(n=n):
                out, _, _ = self.run_spec(k=8, n=n)
                self.assertEqual(out, self.truth[:n])
        stop = self.truth[5]
        out, _, _ = self.run_spec(k=8, stops=[stop])
        self.assertEqual(out, self.truth[:self.truth.index(stop)])

    def test_error_after_chain_start_does_not_fall_back(self):
        client = self.rig.client()
        self.addCleanup(client.transport.shutdown)
        original = client.transport.chain_step
        calls = 0

        def fail(*a, **kw):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise Unreachable("injected lost chain")
            return original(*a, **kw)

        client.transport.chain_step = fail
        client.transport.forward = mock.Mock(side_effect=AssertionError("unsafe fallback"))
        with self.assertRaises(Unreachable):
            list(spec.generate_stream(client, self.rig.endpoints, PROMPT, 8,
                                      drafter=OracleDrafter(len(PROMPT), self.truth, 1.0, vocab=VOCAB), k=4))
        self.assertFalse(client.transport.forward.called)
        self.assertEqual(client.transport._chain_sessions, {})

    def test_old_route_uses_existing_hidden_protocol(self):
        old = [server.block.caps for server in self.rig.servers]
        try:
            for srv in self.rig.servers:
                srv.block.caps = ()
            client = self.rig.client()
            self.addCleanup(client.transport.shutdown)
            self.assertFalse(client._use_chain(self.rig.endpoints))
            self.assertFalse(client._use_ids(self.rig.endpoints))
            out = list(spec.generate_stream(client, self.rig.endpoints, PROMPT, 12,
                                            drafter=OracleDrafter(len(PROMPT), self.truth, 0.6, vocab=VOCAB), k=8))
            self.assertEqual(out, self.truth[:12])
            self.assertFalse(client.transport.answers)
        finally:
            for srv, caps in zip(self.rig.servers, old):
                srv.block.caps = caps

    def test_malformed_compact_reply_is_rejected(self):
        for reply in ({"accepted": True, "next": 2, "ids": [2]},
                      {"accepted": 2, "next": 2, "ids": [1, 2]},
                      {"accepted": 1, "next": 2, "ids": [9, 2]},
                      {"accepted": 0, "next": -1, "ids": [-1]}):
            with self.subTest(reply=reply), self.assertRaises(ValueError):
                spec._verified_prefix(reply, [1, 3], VOCAB)

    def test_single_peer_holds_both_end_tables(self):
        rig = ChainRig(single=True)
        try:
            legacy, chained = rig.client(legacy=True), rig.client()
            try:
                self.assertEqual(chained.generate(rig.endpoints, PROMPT, 16), legacy.generate(rig.endpoints, PROMPT, 16))
                self.assertTrue(chained.transport.answers)
            finally:
                legacy.transport.shutdown()
                chained.transport.shutdown()
        finally:
            rig.close()

    def test_live_chain_timing_does_not_call_inclusive_time_compute(self):
        client = self.rig.client()
        from types import SimpleNamespace
        net = SimpleNamespace(plan=lambda: (self.rig.endpoints, client.transport))
        result = live.generate_once(net, client, PROMPT, 4)
        steps = result["steps"]
        self.assertTrue(all(s["timing_mode"] == "chain_inclusive" for s in steps))
        self.assertTrue(all(s["compute_ms"] != s["compute_ms"] for s in steps))
        self.assertTrue(all(s["network_ms"] != s["network_ms"] for s in steps))
        summary = live.summarize_steps(steps, [], [])
        self.assertEqual(summary["timing_mode"], "chain_inclusive")
        self.assertTrue(all(row["server_ms"] is None and row["network_ms"] is None
                            for row in summary["per_hop"]))
        self.assertIn("breakdown unavailable", live.one_line({"summary": {
            "decode_ms_per_tok": summary["ms_mean"], "decode_tok_s": summary["tok_s"],
            "timing_mode": "chain_inclusive"}}))

    def test_real_payload_bytes_fall_while_tokens_match(self):
        report = traffic_comparison(hidden=256, new_tokens=18)
        for row in report["cases"]:
            self.assertTrue(row["hidden_relay"]["matches_plain"])
            self.assertTrue(row["token_chain"]["matches_plain"])
            self.assertGreater(row["client_payload_reduction"], 0.8)
            self.assertGreater(row["payload_reduction"], 0.4)


class TinyQwenChain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from commonnet.tests.test_split import make_checkpoint
        cls.tmp = tempfile.TemporaryDirectory(prefix="commonnet-chain-qwen-")
        make_checkpoint(cls.tmp.name, seed=7)
        from commonnet.net.factory import build_block
        cls.servers = [BlockServer(build_block(ep.start, ep.end, checkpoint=cls.tmp.name,
                                                environ={"COMMONNET_DTYPE": "float32"}),
                                    allow_chain_private=True).start() for ep in ENDPOINTS]
        client = cls.client(legacy=True)
        try:
            cls.truth = client.generate(ENDPOINTS, PROMPT, 38)
        finally:
            client.transport.shutdown()

    @classmethod
    def tearDownClass(cls):
        for server in cls.servers:
            server.stop()
        cls.tmp.cleanup()

    @classmethod
    def client(cls, *, legacy=False, cls_type=Client):
        transport = CaptureTransport({ep.name: srv.url for ep, srv in zip(ENDPOINTS, cls.servers)})
        client = cls_type(cls.tmp.name, transport)
        client.ids_disabled = legacy
        return client

    def test_target_tables_remain_unloaded_for_chained_plain_and_speculative(self):
        for cls_type in (Client, Client4):
            for k in (4, 8):
                with self.subTest(client=cls_type.__name__, k=k):
                    client = self.client(cls_type=cls_type)
                    if cls_type is Client4:
                        # Peer fixture runs float32; client tables are unused regardless of local format.
                        self.assertEqual(client.fmt, "fp16")
                    try:
                        self.assertEqual(client.loaded_names, [])
                        out = list(spec.generate_stream(client, ENDPOINTS, PROMPT, 24,
                                                        drafter=OracleDrafter(len(PROMPT), self.truth, 0.7,
                                                                             vocab=512, seed=3), k=k, adaptive=False))
                        self.assertEqual(out, self.truth[:24])
                        self.assertEqual(client.loaded_names, [])
                        self.assertEqual(client.client_bytes, 0)
                        self.assertFalse(client._local_ready)
                    finally:
                        client.transport.shutdown()

    def test_sampling_is_repeatable_and_drafter_falls_back_to_plain(self):
        def generate(use_drafter):
            client = self.client()
            drafter = OracleDrafter(len(PROMPT), self.truth, 1.0, vocab=512)
            stats = spec.SpecStats()
            try:
                out = list(spec.generate_stream(client, ENDPOINTS, PROMPT, 10,
                                                drafter=drafter if use_drafter else None,
                                                temperature=0.8, top_p=0.9, seed=42, stats=stats))
                self.assertEqual(drafter.calls, 0)
                self.assertFalse(client._local_ready)
                return out
            finally:
                client.transport.shutdown()
        self.assertEqual(generate(True), generate(False))

    def test_metadata_only_client_downloads_tables_only_for_legacy_fallback(self):
        import json
        import shutil
        from pathlib import Path
        from commonnet.weights import CheckpointReader
        with tempfile.TemporaryDirectory(prefix="commonnet-client-metadata-") as folder:
            metadata = Path(folder)
            shutil.copyfile(Path(self.tmp.name, "config.json"), metadata / "config.json")
            names = CheckpointReader(self.tmp.name).weight_map
            (metadata / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {name: "not-downloaded.safetensors" for name in names}}))
            transport = CaptureTransport({ep.name: srv.url for ep, srv in zip(ENDPOINTS, self.servers)})
            client = Client(str(metadata), transport)
            loader = mock.Mock(return_value=self.tmp.name)
            client._endpoint_checkpoint_loader = loader
            try:
                self.assertEqual(client.generate(ENDPOINTS, PROMPT, 8), self.truth[:8])
                self.assertFalse(loader.called)
                self.assertFalse(client._local_ready)
                client.ids_disabled = True
                self.assertEqual(client.generate(ENDPOINTS, PROMPT, 8), self.truth[:8])
                loader.assert_called_once_with()
                self.assertTrue(client._local_ready)
                self.assertEqual(client.generate(ENDPOINTS, PROMPT, 4), self.truth[:4])
                loader.assert_called_once_with()
            finally:
                transport.shutdown()

    def test_local_api_loads_tables_only_when_requested(self):
        client = self.client()
        try:
            self.assertEqual(client.loaded_names, [])
            result = client.embed(torch.tensor([PROMPT]))
            self.assertEqual(tuple(result.shape), (1, len(PROMPT), 64))
            self.assertTrue(client.loaded_names)
            self.assertGreater(client.client_bytes, 0)
        finally:
            client.transport.shutdown()


if __name__ == "__main__":
    unittest.main()
