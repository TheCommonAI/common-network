"""End to end on localhost: tiny Qwen3, real block servers advertised by fake peer records, generation through Network.

A stage with two replicas loses its primary mid-generation; a stage with one server loses it and a newcomer appears.
Both must produce exactly the tokens of the full model.
"""
import os
import threading
import time
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch

from commonnet import coordinator
from commonnet.net import factory
from commonnet.net.server import BlockServer
from commonnet.netclient import Network
from commonnet.tests.test_netclient import FakeTable, record
from commonnet.tests.test_split import NEW_TOKENS, PROMPT, SplitTestBase

TOKEN = "t" * 32
MANIFEST = {"id": "tiny-e2e", "version": 1,
            "model": {"name": "Tiny", "num_layers": 4, "repos": {"torch": "x/y"}, "client_repo": "x/y"},
            "boundaries": [0, 2, 4], "gib_per_layer": {"torch": 0.01}, "min_replicas": 2, "seeds": ["http://seed"]}
NOW = 1_000_000.0


class NetworkE2E(SplitTestBase):
    def start(self, window, peer):
        blk = factory.build_block(window[0], window[1], environ={"COMMONNET_CHECKPOINT": self.ckpt})
        srv = BlockServer(blk, token=TOKEN).start()
        self.servers[peer] = srv
        rec = record(peer, srv.url, TOKEN, window, backend="torch")
        rec["network"] = "tiny-e2e"
        self.records[peer] = rec
        return rec

    def setUp(self):
        self.servers, self.records = {}, {}

    def tearDown(self):
        for s in self.servers.values():
            s.stop()

    def reference_tokens(self):
        with torch.inference_mode():
            out = self.reference.generate(torch.tensor([PROMPT]), max_new_tokens=NEW_TOKENS,
                                          do_sample=False, eos_token_id=None, pad_token_id=0)
        return out[0, len(PROMPT):].tolist()

    def network(self, visible):
        """`visible()` -> the peer ids the mesh currently lists."""
        def exchange(table, url, timeout):
            table.records = {p: self.records[p] for p in visible()}
        return Network(seeds=["http://seed"], manifest=MANIFEST, table=FakeTable(), exchange=exchange,
                       now=lambda: NOW, refresh_interval=0, wait_s=20.0)

    def kill_at(self, peer, hop, call_number, transport):
        """Stop a peer's server (and drop the pooled connections to it) just before the transport's Nth call to `hop`."""
        real, seen = transport.forward, {"n": 0}

        def forward(endpoint, session_id, hidden, start_pos):
            if endpoint.name == hop:
                seen["n"] += 1
                if seen["n"] == call_number:
                    self.servers[peer].stop()
                    for c in list(transport._clients.values()):
                        c.close_connections()
            return real(endpoint, session_id, hidden, start_pos)

        transport.forward = forward

    def test_generation_over_the_network_matches_the_full_model(self):
        for peer, window in (("a1", (0, 2)), ("b1", (2, 4))):
            self.start(window, peer)
        net = self.network(lambda: ["a1", "b1"])
        got = coordinator.generate_network(net, self.ckpt, PROMPT, NEW_TOKENS, fmt="fp32")
        self.assertEqual(got, self.reference_tokens())

    def test_a_replica_takes_over_when_the_primary_dies_mid_generation(self):
        for peer, window in (("a1", (0, 2)), ("b1", (2, 4)), ("b2", (2, 4))):
            self.start(window, peer)
        net = self.network(lambda: ["a1", "b1", "b2"])
        endpoints, transport = net.plan(retries=0, sleep=lambda _: None)
        primary = [p for p in ("b1", "b2") if self.records[p]["url"] == transport._candidates("hop1")[0]
                   or self.records[p]["url"] == transport._candidates("hop1")[0][0]][0]
        self.kill_at(primary, "hop1", 5, transport)  # the prompt, then four tokens in
        from commonnet.client4 import Client4
        client = Client4(self.ckpt, transport, fmt="fp32")
        got = client.generate(endpoints, PROMPT, NEW_TOKENS)
        self.assertEqual(got, self.reference_tokens())
        self.assertTrue(transport.failovers, "the primary should have been replaced")

    def test_a_lone_server_dies_and_a_newcomer_with_the_same_window_finishes_the_job(self):
        for peer, window in (("a1", (0, 2)), ("b1", (2, 4))):
            self.start(window, peer)
        visible = ["a1", "b1"]
        net = self.network(lambda: list(visible))
        endpoints, transport = net.plan(retries=0, sleep=lambda _: None, wait_poll_s=0.05)
        self.kill_at("b1", "hop1", 5, transport)

        def newcomer():
            time.sleep(0.6)           # the peer joins a moment after the other one died
            self.start((2, 4), "b2")
            visible[:] = ["a1", "b2"]  # and the dead one has aged out of the mesh
        threading.Thread(target=newcomer, daemon=True).start()
        from commonnet.client4 import Client4
        client = Client4(self.ckpt, transport, fmt="fp32")
        got = client.generate(endpoints, PROMPT, NEW_TOKENS)
        self.assertEqual(got, self.reference_tokens())
        self.assertEqual(transport.failovers[-1][0], "hop1")


if __name__ == "__main__":
    unittest.main()


class FakeTokenizer:
    """Just enough of a Hugging Face tokenizer for NetworkBackend: ids are the characters' code points mod 500."""
    eos_token_id = 511
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 510}.get(token, 0)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kw):
        return "".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 500 for c in text]}

    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids)


class BackendE2E(NetworkE2E):
    """serve.NetworkBackend: one worker thread, a fresh route per request, failover inside a request."""

    def backend(self, net):
        from unittest import mock
        from commonnet.serve import NetworkBackend
        with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()):
            return NetworkBackend(net, self.ckpt, fmt="fp32", hidden_dtype=torch.float32)

    def test_stream_matches_the_full_model_and_each_request_rebuilds_the_route(self):
        for peer, window in (("a1", (0, 2)), ("b1", (2, 4))):
            self.start(window, peer)
        net = self.network(lambda: ["a1", "b1"])
        backend = self.backend(net)
        self.assertEqual(backend.model_name, "Tiny")
        want = self.reference_tokens()
        self.assertEqual(list(backend.stream(PROMPT, NEW_TOKENS, 0.0, 1.0, None)), want)
        self.assertEqual(list(backend.stream(PROMPT, NEW_TOKENS, 0.0, 1.0, None)), want)
        self.assertEqual(backend.health()["peers"], 2)

    def test_a_request_survives_its_primary_dying(self):
        for peer, window in (("a1", (0, 2)), ("b1", (2, 4)), ("b2", (2, 4))):
            self.start(window, peer)
        net = self.network(lambda: ["a1", "b1", "b2"])
        backend = self.backend(net)
        real_plan = net.plan

        def plan_and_arm(**kw):
            endpoints, transport = real_plan(retries=0, sleep=lambda _: None)
            victim = [p for p in ("b1", "b2") if self.records[p]["url"] in (
                transport._candidates("hop1")[0], transport._candidates("hop1")[0][0])][0]
            self.kill_at(victim, "hop1", 5, transport)
            return endpoints, transport

        net.plan = plan_and_arm
        self.assertEqual(list(backend.stream(PROMPT, NEW_TOKENS, 0.0, 1.0, None)), self.reference_tokens())

    def test_no_route_reaches_the_reader_of_the_stream(self):
        self.start((0, 2), "a1")
        net = self.network(lambda: ["a1"])
        backend = self.backend(net)
        from commonnet.mesh.placement import NoRoute
        with self.assertRaises(NoRoute) as caught:
            next(iter(backend.stream(PROMPT, 3, 0.0, 1.0, None)))
        self.assertIn("layers 2-4", str(caught.exception))
