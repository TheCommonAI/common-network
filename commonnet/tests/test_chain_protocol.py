"""Token-only ingress, peer forwarding, rollback and replay over real HTTP sockets."""
import json
import socket
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from commonnet.client import BlockEndpoint
from commonnet.net import chain, protocol
from commonnet.net.client import BlockClient, RemoteError, Unreachable, _connect_resolved
from commonnet.net.server import BlockServer, StubBlock
from commonnet.net.transport import HttpTransport


class TokenBlock(StubBlock):
    """Causal position-checked block; the final head predicts input token plus one."""
    def __init__(self, start, end, total=6):
        super().__init__(start, end)
        self.caps = tuple(c for c, yes in (("ids_in", start == 0), ("head_out", end == total)) if yes)
        self.positions = {}
        self.head_calls = 0

    def open(self, sid):
        super().open(sid)
        self.positions[sid] = 0

    def close(self, sid):
        super().close(sid)
        self.positions.pop(sid, None)

    def truncate(self, sid, length):
        super().truncate(sid, length)
        self.positions[sid] = min(self.positions[sid], length)

    def forward(self, sid, hidden, pos):
        if self.positions[sid] != pos:
            raise ValueError("cache position mismatch")
        self.positions[sid] += hidden.shape[1]
        return super().forward(sid, hidden, pos)

    def embed(self, ids):
        return protocol.Tensor((1, len(ids), 1), "float32", struct.pack("<" + "f" * len(ids), *ids))

    def head(self, hidden, which, sampling, start_pos):
        self.head_calls += 1
        toks = [int(x) + 1 for x in struct.unpack("<" + "f" * hidden.shape[1], hidden.data)]
        return toks if which == "all" else toks[-1:]


class ChainHttp(unittest.TestCase):
    def setUp(self):
        self.blocks = [TokenBlock(i, i + 2) for i in (0, 2, 4)]
        self.servers = [BlockServer(b, "127.0.0.1", 0, allow_chain_private=True).start() for b in self.blocks]
        self.endpoints = [BlockEndpoint(str(i), *b.layers) for i, b in enumerate(self.blocks)]
        self.tr = HttpTransport({e.name: s.url for e, s in zip(self.endpoints, self.servers)},
                                token=None, retries=1, backoff=0, activation_codec="float")
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.tr.shutdown()
        for server in self.servers:
            server.stop()

    def step(self, ids, pos=0, **kwargs):
        return self.tr.chain_step(self.endpoints, "s", ids, pos, **kwargs)

    def forwards(self, block):
        return [c for c in block.calls if c[0] == "forward"]

    def test_only_first_peer_is_called_by_client_and_only_ids_return(self):
        self.assertTrue(self.tr.chain_route(self.endpoints))
        self.assertEqual(self.step([3, 4]), {"ids": [5]})
        for i, block in enumerate(self.blocks):
            self.assertEqual(len(self.forwards(block)), 1)
            client = self.tr._clients[(str(i), 0)]
            self.assertEqual(sum(t["path"] == "/chain" for t in client.timings), int(i == 0))
        first = self.servers[0]
        self.assertEqual(len(first._chain_clients), 1)
        self.assertEqual(self.blocks[0].head_calls, 0)
        self.assertEqual(self.blocks[1].head_calls, 0)
        self.assertEqual(self.blocks[2].head_calls, 1)

    def test_final_peer_returns_only_matching_prefix_and_correction(self):
        self.step([3, 4])
        out = self.step([5, 6, 7, 99], 2, which="all", drafts=[6, 7, 99])
        self.assertEqual(out, {"ids": [6, 7, 8], "accepted": 2, "next": 8})
        for ep in self.endpoints:
            self.tr.truncate(ep, "s", 5, defer=True)
        self.assertEqual(self.step([8], 5), {"ids": [9]})
        for block in self.blocks:
            self.assertEqual(list(block.positions.values()), [6])
            self.assertEqual([c[2] for c in block.calls if c[0] == "truncate"], [5])

    def test_all_accepted_returns_bonus_and_empty_drafts_is_valid(self):
        self.step([3, 4])
        self.assertEqual(self.step([5, 6, 7], 2, which="all", drafts=[6, 7]),
                         {"ids": [6, 7, 8], "accepted": 2, "next": 8})
        self.assertEqual(self.step([8], 5, which="all", drafts=[]),
                         {"ids": [9], "accepted": 0, "next": 9})

    def test_lost_final_reply_retries_without_appending_kv_twice(self):
        self.tr.chain_route(self.endpoints)
        client, _ = self.tr._target("0", 0)
        original = client._once
        dropped = [False]
        def drop(method, parts, path, data, headers, timeout):
            result = original(method, parts, path, data, headers, timeout)
            if path == "/chain" and not dropped[0]:
                dropped[0] = True
                raise OSError("injected dropped reply")
            return result
        with mock.patch.object(client, "_once", side_effect=drop):
            self.assertEqual(self.step([3, 4]), {"ids": [5]})
        self.assertTrue(dropped[0])
        self.assertTrue(all(len(self.forwards(b)) == 1 for b in self.blocks))
        self.assertEqual(self.blocks[-1].head_calls, 1)

    def test_downstream_failure_reuses_local_compute_but_forwards_again(self):
        self.step([3, 4])
        onward = next(iter(self.servers[0]._chain_clients.values()))
        original = onward.chain
        calls = [0]
        def fail_once(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 1:
                raise Unreachable("injected downstream failure")
            return original(*args, **kwargs)
        with mock.patch.object(onward, "chain", side_effect=fail_once):
            self.assertEqual(self.step([5], 2), {"ids": [6]})
        self.assertEqual(calls[0], 2)
        self.assertTrue(all(len(self.forwards(b)) == 2 for b in self.blocks))

    def test_lost_downstream_session_rebuilds_whole_route_from_ids(self):
        self.step([3, 4])
        old = self.tr._chain_sessions["s"]["wire_id"]
        self.servers[1].close(old)
        self.assertEqual(self.step([5], 2), {"ids": [6]})
        fresh = self.tr._chain_sessions["s"]["wire_id"]
        self.assertNotEqual(fresh, old)
        self.assertTrue(all(list(b.positions.values()) == [3] for b in self.blocks))
        self.assertEqual(self.tr._chain_sessions["s"]["history"],
                         [{"ids": [3, 4], "start_pos": 0}, {"ids": [5], "start_pos": 2}])

    def test_failed_final_peer_is_replaced_and_complete_chain_replayed(self):
        replica_block = TokenBlock(4, 6)
        replica = BlockServer(replica_block, "127.0.0.1", 0, allow_chain_private=True).start()
        self.addCleanup(replica.stop)
        self.tr._replicas["2"] = [replica.url]
        self.step([3, 4])
        for client in self.servers[1]._chain_clients.values():
            client.close_connections()
        self.tr._clients[("2", 0)].close_connections()
        self.servers[2].stop()
        self.assertEqual(self.step([5], 2), {"ids": [6]})
        state = self.tr._chain_sessions["s"]
        self.assertEqual(state["hops"][-1]["index"], 1)
        self.assertEqual(list(replica_block.positions.values()), [3])
        self.assertEqual([c[2] for c in self.forwards(replica_block)], [0, 2])

    def test_failover_to_peer_without_chain_capability_is_refused_safely(self):
        self.step([3, 4])
        old = self.tr._chain_sessions["s"]["wire_id"]
        self.servers[1].close(old)
        # Simulate an old process replacing a stage after initial negotiation.
        self.tr._chain_caps.clear()
        class OldServer(BlockServer):
            @property
            def caps(self):
                return ()
        self.servers[1].__class__ = OldServer
        with self.assertRaises(ValueError):
            self.step([5], 2)
        self.assertEqual(self.tr._chain_sessions["s"]["history"], [{"ids": [3, 4], "start_pos": 0}])

    def test_replay_after_rollback_retains_only_committed_ids(self):
        self.step([3, 4])
        self.step([5, 99, 99], 2, which="all", drafts=[99, 99])
        for ep in self.endpoints:
            self.tr.truncate(ep, "s", 3, defer=True)
        old = self.tr._chain_sessions["s"]["wire_id"]
        self.servers[1].close(old)
        self.assertEqual(self.step([6], 3), {"ids": [7]})
        self.assertTrue(all(list(b.positions.values()) == [4] for b in self.blocks))
        self.assertEqual(self.tr._chain_sessions["s"]["history"][1]["ids"], [5])

    def test_close_and_shutdown_release_all_stage_sessions(self):
        self.step([3, 4])
        for ep in self.endpoints:
            self.tr.close(ep, "s")
        self.assertEqual(self.tr._chain_sessions, {})
        self.assertTrue(all(not b.positions for b in self.blocks))
        self.step([10])
        self.tr.shutdown()
        self.assertTrue(all(not b.positions for b in self.blocks))

    def test_default_public_peer_policy_declines_private_route_before_session(self):
        self.servers[0].allow_chain_private = False
        self.assertFalse(self.tr.chain_route(self.endpoints))
        self.assertEqual(self.tr._chain_sessions, {})

    def test_old_peer_or_wrong_layer_window_is_not_negotiated(self):
        with mock.patch.object(type(self.servers[1]), "caps", new_callable=mock.PropertyMock,
                               return_value=()):
            self.assertFalse(self.tr.chain_route(self.endpoints))
        self.tr._chain_caps.clear()
        wrong = [self.endpoints[0], BlockEndpoint("1", 2, 5), BlockEndpoint("2", 5, 6)]
        self.assertFalse(self.tr.chain_route(wrong))

    def test_invalid_request_does_not_create_a_session(self):
        client = BlockClient(token=None, retries=0)
        self.addCleanup(client.close_connections)
        body = {"id": "bad", "ids": [3], "start_pos": 0, "layers": [0, 2],
                "which": "last", "route": [{"url": self.servers[1].url, "layers": [3, 4]}]}
        with self.assertRaises(RemoteError) as caught:
            client.chain(self.servers[0].url, body)
        self.assertEqual(caught.exception.status, 400)
        self.assertTrue(all(not b.positions for b in self.blocks))

    def test_default_int8_between_peers_has_token_only_ingress_and_reply(self):
        from commonnet.net.activation_codec import encode_int8_group
        self.tr._activation_codec = "int8_group_v1"
        with mock.patch("commonnet.net.activation_codec.encode_int8_group", wraps=encode_int8_group) as encode:
            reply = self.step([3, 4])
        self.assertEqual(self.tr._chain_sessions["s"]["codec"], "int8_group_v1")
        self.assertEqual(encode.call_count, 2)
        self.assertEqual(set(reply), {"ids"})
        self.assertTrue(all(type(t) is int for t in reply["ids"]))

    def test_a_peer_without_int8_negotiates_float_before_generation(self):
        class FloatOnlyServer(BlockServer):
            @property
            def caps(self):
                return tuple(c for c in super().caps if c != "activation_int8_group_v1")
        self.servers[1].__class__ = FloatOnlyServer
        self.tr._activation_codec = "int8_group_v1"
        self.assertTrue(self.tr.chain_route(self.endpoints))
        self.step([3, 4])
        self.assertEqual(self.tr._chain_sessions["s"]["codec"], "float")


class ChainValidation(unittest.TestCase):
    def test_bad_origins_credentials_and_pins_are_rejected_without_echoing_credentials(self):
        for url in ("http://8.8.8.8", "https://127.0.0.1:8080", "https://169.254.169.254",
                    "https://user:secret@example.com", "https://example.com/path", "https://example.com?q=x"):
            with self.subTest(url=url), self.assertRaises(protocol.ProtocolError):
                chain.check_hop({"url": url, "layers": [2, 4]})
        with self.assertRaises(protocol.ProtocolError) as caught:
            chain.check_hop({"url": "https://example.com", "layers": [2, 4], "token": "do-not-echo"})
        self.assertNotIn("do-not-echo", str(caught.exception))

    def test_connect_time_dns_check_refuses_mixed_public_private_answers(self):
        conn = SimpleNamespace(host="peer.example", port=443, public_only=True, timeout=1, source_address=None)
        addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))
                     for ip in ("8.8.8.8", "127.0.0.1")]
        with mock.patch("commonnet.net.client.dns.resolve", return_value="peer.example"), \
                mock.patch("socket.getaddrinfo", return_value=addresses), \
                mock.patch("socket.create_connection") as connect, self.assertRaises(OSError):
            _connect_resolved(conn, True)
        connect.assert_not_called()

    def test_reply_cannot_smuggle_hidden_tensor_or_inconsistent_verification(self):
        for reply in ({"ids": [1], "data": "hidden"}, {"ids": [1], "accepted": 2, "next": 1},
                      {"ids": [1], "accepted": 0, "next": 2}, {"ids": [True]}):
            with self.subTest(reply=reply), self.assertRaises(protocol.ProtocolError):
                chain.check_reply(reply)

    def test_nonfinite_sampling_is_rejected(self):
        for bad in (float("inf"), float("nan"), 10 ** 400):
            with self.subTest(kind=type(bad).__name__), self.assertRaises(protocol.ProtocolError):
                protocol.check_sampling({"temperature": bad, "top_p": 1, "seed": 0})


class ChainTls(unittest.TestCase):
    def test_hop_tokens_and_tls_pins_are_carried_to_each_forwarding_peer(self):
        from commonnet.mesh.records import Identity
        from commonnet.net import secure
        from urllib.parse import urlsplit
        with tempfile.TemporaryDirectory() as directory:
            blocks = [TokenBlock(0, 2, total=4), TokenBlock(2, 4, total=4)]
            servers, pins, tokens = [], {}, {}
            for i, block in enumerate(blocks):
                cert, key, fp = secure.make_peer_cert(Identity.generate(), directory + "/" + str(i))
                token = "synthetic-chain-stage-" + str(i) + "-credential"
                server = BlockServer(block, "127.0.0.1", 0, token=token, allow_chain_private=True,
                                     ssl_context=secure.server_ssl_context(cert, key)).start()
                servers.append(server)
                self.addCleanup(server.stop)
                pins[urlsplit(server.url).netloc] = fp
                tokens[str(i)] = token
            endpoints = [BlockEndpoint(str(i), *b.layers) for i, b in enumerate(blocks)]
            tr = HttpTransport({str(i): s.url for i, s in enumerate(servers)}, tokens=tokens,
                               pin=pins, retries=0, activation_codec="float")
            self.addCleanup(tr.shutdown)
            self.assertTrue(tr.chain_route(endpoints))
            self.assertEqual(tr.chain_step(endpoints, "tls", [3, 4], 0), {"ids": [5]})
            onward = next(iter(servers[0]._chain_clients.values()))
            self.assertEqual(onward._pin, pins[urlsplit(servers[1].url).netloc])
            self.assertIsNotNone(onward.token)
            # After capability negotiation, an impostor pin in the forwarded route
            # still fails TLS; it never disables verification or runs final layers.
            pins[urlsplit(servers[1].url).netloc] = "0" * 64
            before = len([c for c in blocks[1].calls if c[0] == "forward"])
            with self.assertRaises(Unreachable):
                tr.chain_step(endpoints, "wrong-pin", [9], 0)
            self.assertEqual(len([c for c in blocks[1].calls if c[0] == "forward"]), before)


if __name__ == "__main__":
    unittest.main()
