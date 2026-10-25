"""Automatic background upgrades, actual direct RPC, and transparent relay fallback."""
import threading
import time
import unittest
from unittest import mock

from commonnet.net import rtc
from commonnet.net.client import BlockClient, RemoteError
from commonnet.net.server import BlockServer, StubBlock
from commonnet.tests.test_net_server import hidden_tensor


def wait_for(predicate, timeout=8):
    until = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= until:
            raise AssertionError("background connection did not finish")
        time.sleep(0.01)


@unittest.skipUnless(rtc.available(), "aiortc unavailable")
class DirectRpc(unittest.TestCase):
    def setUp(self):
        self.block = StubBlock(0, 4)
        self.server = BlockServer(self.block, token="rtc-test-token-value", allow_rtc_private=True,
                                  rtc_stun_urls=[]).start()
        self.client = BlockClient(token="rtc-test-token-value", retries=0, rtc_allow_private=True,
                                  rtc_stun_urls=[])
        self.addCleanup(self.server.stop)
        self.addCleanup(self.client.close_connections)

    def connect(self):
        self.assertEqual(self.client.health(self.server.url), ((0, 4), "stub"))
        wait_for(lambda: any(s.get("peer") is not None for s in self.client._rtc_states.values()))
        return self.client._rtc_states[self.server.url]["peer"]

    def test_rpc_uses_direct_channel_when_http_data_calls_are_disabled(self):
        self.connect()
        with mock.patch.object(self.client, "_once", side_effect=AssertionError("HTTP data path used")):
            self.client.open(self.server.url, "job")
            value = hidden_tensor(shape=(1, 9, 4096), dtype="float32")
            self.assertEqual(self.client.forward(self.server.url, "job", value, 0), value)
            self.client.close(self.server.url, "job")
        rows = [r for r in self.client.timings if r["path"] in ("/session", "/forward", "/close")]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["transport"] == "webrtc" for r in rows))

    def test_lost_direct_reply_falls_back_to_identical_http_rpc_without_duplicate_compute(self):
        peer = self.connect()
        self.client.open(self.server.url, "job")
        original = peer.request

        def lost(*args, **kwargs):
            original(*args, **kwargs)
            raise rtc.RtcUnavailable("injected lost reply")

        value = hidden_tensor(shape=(1, 3, 32))
        with mock.patch.object(peer, "request", side_effect=lost):
            self.assertEqual(self.client.forward(self.server.url, "job", value, 0), value)
        self.assertEqual(len([c for c in self.block.calls if c[0] == "forward"]), 1)
        self.assertEqual(self.client.timings[-1]["transport"], "http")
        self.assertIsNone(self.client._rtc_states[self.server.url]["peer"])

    def test_close_releases_established_direct_peer(self):
        peer = self.connect()
        self.client.close_connections()
        self.assertFalse(peer.ready)
        self.assertEqual(self.client._rtc_states, {})


class BackgroundPolicy(unittest.TestCase):
    def client(self):
        client = BlockClient(token=None, retries=0)
        self.addCleanup(client.close_connections)
        return client

    def test_initial_request_does_not_wait_for_negotiation(self):
        started, release = threading.Event(), threading.Event()

        class Peer:
            ready = False
            def __init__(self, *args, **kwargs): pass
            def connect(self, timeout):
                started.set()
                release.wait(3)
                self.ready = True
            def close(self): self.ready = False

        client = self.client()
        health = {"ok": True, "layers": [0, 4], "device": "stub", "caps": [rtc.CAPABILITY]}
        try:
            with mock.patch.object(rtc, "available", return_value=True), \
                    mock.patch.object(rtc, "RtcClient", Peer), \
                    mock.patch.object(client, "_http_call", return_value=health):
                self.assertEqual(client.health("https://example.com"), ((0, 4), "stub"))
                self.assertTrue(started.wait(1))
                self.assertFalse(release.is_set())
                release.set()
                client._rtc_states["https://example.com"]["future"].result(timeout=2)
        finally:
            release.set()

    def test_old_peer_is_probed_once_then_uses_http_during_cooldown(self):
        client = self.client()
        health = {"ok": True, "layers": [0, 4], "device": "stub"}
        with mock.patch.object(rtc, "available", return_value=True), \
                mock.patch.object(client, "_http_call", return_value=health), \
                mock.patch.object(rtc, "RtcClient") as constructor:
            client.health("https://example.com")
            future = client._rtc_states["https://example.com"]["future"]
            future.result(timeout=2)
            for _ in range(5):
                client.health("https://example.com")
            self.assertIs(client._rtc_states["https://example.com"]["future"], future)
            constructor.assert_not_called()

    def test_disable_flag_and_private_origin_policy(self):
        with mock.patch.dict("os.environ", {"COMMONNET_RTC": "0"}):
            self.assertFalse(BlockClient(token=None)._rtc_endpoint_allowed("https://example.com"))
        client = self.client()
        for url in ("http://example.com", "https://127.0.0.1", "https://192.168.1.1", "https://localhost"):
            with self.subTest(url=url):
                self.assertIsNone(client._rtc_peer(url))

    def test_direct_application_error_is_not_downgraded_to_http(self):
        client = self.client()
        peer = mock.Mock()
        peer.request.return_value = {"status": 401, "body": {"error": "unauthorized"}, "server_ms": None}
        with mock.patch.object(client, "_rtc_peer", return_value=peer), \
                mock.patch.object(client, "_http_call") as http:
            with self.assertRaises(RemoteError) as caught:
                client.health("https://example.com")
        self.assertEqual(caught.exception.status, 401)
        http.assert_not_called()

    def test_remote_closed_peer_is_retired_before_its_state_is_replaced(self):
        client = self.client()
        old = mock.Mock(ready=False)
        def close():
            acquired = client._lock.acquire(blocking=False)
            self.assertTrue(acquired)
            if acquired:
                client._lock.release()
        old.close.side_effect = close
        finished = mock.Mock()
        finished.done.return_value = True
        client._rtc_states["https://example.com"] = {"peer": old, "future": finished,
                                                      "after": 0, "epoch": 0}
        with mock.patch.object(rtc, "available", return_value=True), \
                mock.patch.object(client, "_http_call", return_value={"ok": True, "layers": [0, 4],
                                                                    "device": "stub"}):
            client._rtc_peer("https://example.com")
            client._rtc_states["https://example.com"]["future"].result(timeout=2)
        old.close.assert_called_once_with()

    def test_pending_upgrade_cannot_resurrect_after_cleanup(self):
        started, release = threading.Event(), threading.Event()
        peers = []

        class Peer:
            ready = False
            closed = False
            def __init__(self, *args, **kwargs): peers.append(self)
            def connect(self, timeout):
                started.set()
                release.wait(3)
                self.ready = True
            def close(self):
                self.closed = True
                self.ready = False

        client = self.client()
        health = {"ok": True, "layers": [0, 4], "device": "stub", "caps": [rtc.CAPABILITY]}
        try:
            with mock.patch.object(rtc, "available", return_value=True), \
                    mock.patch.object(rtc, "RtcClient", Peer), \
                    mock.patch.object(client, "_http_call", return_value=health):
                client.health("https://example.com")
                self.assertTrue(started.wait(1))
                future = client._rtc_states["https://example.com"]["future"]
                client.close_connections()
                release.set()
                future.result(timeout=2)
            self.assertEqual(client._rtc_states, {})
            self.assertTrue(peers[0].closed)
        finally:
            release.set()


class ApiCleanup(unittest.TestCase):
    def test_generation_cancellation_closes_transport(self):
        from commonnet.serve import NetworkBackend
        backend = NetworkBackend.__new__(NetworkBackend)
        transport = mock.Mock()
        backend.network = mock.Mock()
        backend.network.plan.return_value = ([], transport)
        backend.client = mock.Mock()
        backend.client.generate_stream.return_value = iter([1, 2, 3])
        backend.drafter, backend.stop_ids = None, []
        backend.auto_draft = False
        stream = backend._generate([1], 3, 0, 1, None)
        self.assertEqual(next(stream), 1)
        stream.close()
        transport.shutdown.assert_called_once_with()

    def test_node_tunnel_only_mode_is_scoped_to_its_process_run(self):
        from commonnet import node
        def run(args):
            self.assertTrue(args.no_rtc)
            self.assertEqual(node.os.environ["COMMONNET_RTC"], "0")
            return 0
        with mock.patch.dict("os.environ", {"COMMONNET_RTC": "1"}), \
                mock.patch.object(node, "_main", side_effect=run):
            self.assertEqual(node.main(["--no-rtc"]), 0)
            self.assertEqual(node.os.environ["COMMONNET_RTC"], "1")


if __name__ == "__main__":
    unittest.main()
